"""Data preparation pipeline for trading RL."""

from __future__ import annotations

import gc
import json
import logging
import logging.handlers
import multiprocessing
import os
import random
import shutil
import tempfile
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from logger import get_logger
from trading_rl.constants import EnvBackend, EnvMode, SplitName
from trading_rl.data_loading import MemmapPaths, load_memmap_paths, load_prepared_splits, save_prepared_splits, save_symbol_memmap
from trading_rl.data.cache import (
    _feature_cache_key,
    _prepared_cache_compatible,
    _write_prepared_cache_metadata,
)
from trading_rl.data.hft import (
    _deduplicate_hft_index_single,
    _derive_close_hft_single,
    ensure_close_column_for_hft,
    ensure_unique_index_for_hft_tradingenv,
)
from trading_rl.data.loading import PreparedDataset, download_trading_data, load_trading_data
from trading_rl.data.validation import validate_prepared_data

logger = get_logger(__name__)


class _WorkerLogFilter(logging.Filter):
    """Append [worker X/N] to every log record emitted from a worker process."""

    def __init__(self, tag: str) -> None:
        super().__init__()
        self.tag = tag

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = f"{record.msg} [{self.tag}]"
        return True


def _worker_log_init(queue: multiprocessing.Queue) -> None:
    """Install a QueueHandler on the root logger in each worker process.

    Every log record emitted by the worker is forwarded to the parent's
    QueueListener, which dispatches it through the parent's handlers.
    """
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(logging.handlers.QueueHandler(queue))
    root.setLevel(logging.DEBUG)


def _resolve_symbol_index(strategy: str, n_symbols: int, memmap_dir: Path | None) -> int:
    """Return the index of the representative symbol for streaming-mode splits.

    "first"   — always 0
    "random"  — uniform random pick each call
    "rotated" — increments a per-run counter stored in memmap_dir/.eval_symbol_counter
    """
    if strategy == "random":
        idx = random.randrange(n_symbols)
        logger.info("eval_symbol_selection=random picked idx=%d of %d", idx, n_symbols)
        return idx
    if strategy == "rotated":
        counter_path = (memmap_dir / ".eval_symbol_counter") if memmap_dir else None
        count = 0
        if counter_path and counter_path.exists():
            try:
                count = int(counter_path.read_text().strip())
            except ValueError:
                count = 0
        idx = count % n_symbols
        if counter_path:
            counter_path.write_text(str(count + 1))
        logger.info("eval_symbol_selection=rotated counter=%d idx=%d of %d", count, idx, n_symbols)
        return idx
    # "first" or unrecognised
    return 0


@dataclass
class PrepareDataConfig:
    """Parameters for prepare_data, decoupled from any specific config system."""

    train_size: int
    validation_size: int | None = None
    test_size: int | None = None
    download_if_missing: bool = False
    exchange_names: list[str] | None = None
    symbols: list[str] | None = None
    timeframe: str = "1h"
    data_dir: str = "data"
    since: Any | None = None
    feature_config_path: str | None = None
    feature_cache_dir: str | None = ".cache/feature_transformation"
    filter_lob_levels: int | None = None

    @classmethod
    def from_config(cls, cfg: Any) -> PrepareDataConfig:
        """Build from an OmegaConf node or DataConfig-like object (config.data)."""
        return cls(
            train_size=cfg.train_size,
            validation_size=getattr(cfg, "validation_size", None),
            test_size=getattr(cfg, "test_size", None),
            download_if_missing=getattr(cfg, "download_data", False),
            exchange_names=getattr(cfg, "exchange_names", None),
            symbols=getattr(cfg, "symbols", None),
            timeframe=getattr(cfg, "timeframe", "1h"),
            data_dir=getattr(cfg, "data_dir", "data"),
            since=getattr(cfg, "download_since", None),
            feature_config_path=getattr(cfg, "feature_config", None),
            feature_cache_dir=getattr(cfg, "feature_cache_dir", ".cache/feature_transformation"),
            filter_lob_levels=getattr(cfg, "filter_lob_levels", None),
        )


def _resolve_feature_pipeline(config: Any, logger: Any) -> Any:
    """Return a FeaturePipeline built from feature_groups YAML, or None."""
    feature_groups = getattr(config.data, "feature_groups", None)
    if not feature_groups:
        return None
    from trading_rl.features.groups import FeatureGroupResolver
    from trading_rl.features.pipeline import FeaturePipeline

    resolver = FeatureGroupResolver.from_yaml(feature_groups)
    group_names = resolver.list_groups()
    logger.info("use feature groups n=%d source=%s", len(group_names), feature_groups)
    pipeline = FeaturePipeline(resolver.resolve(group_names))
    logger.info("build feature pipeline n_features=%d n_groups=%d", len(pipeline.features), len(group_names))
    return pipeline


def _apply_warmup_skip(
    train_df: pd.DataFrame,
    warmup_rows: int,
    logger: Any,
    label: str = "",
) -> pd.DataFrame:
    if warmup_rows <= 0:
        return train_df
    if warmup_rows >= len(train_df):
        raise ValueError(
            f"warmup_rows ({warmup_rows}) must be less than training split size ({len(train_df)})"
            + (f" [{label}]" if label else "")
        )
    logger.info("warmup skip n_rows=%d train_size_before=%d%s", warmup_rows, len(train_df), f" [{label}]" if label else "")
    return train_df.iloc[warmup_rows:]


def _finalize_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Any,
    logger: Any,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply post-processing transforms and validation."""
    train_df, val_df, test_df = ensure_close_column_for_hft(train_df, val_df, test_df, config, logger)
    train_df, val_df, test_df = ensure_unique_index_for_hft_tradingenv(train_df, val_df, test_df, config, logger)
    warmup_rows = getattr(config.data, "warmup_rows", 0)
    train_df = _apply_warmup_skip(train_df, warmup_rows, logger)
    validate_prepared_data(train_df, val_df, test_df, config)
    return train_df, val_df, test_df


def _make_dataset(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Any,
    memmap_paths: list[MemmapPaths] | None = None,
) -> PreparedDataset:
    """Detect feature/price columns and construct a PreparedDataset."""
    feature_columns = [col for col in train_df.columns if str(col).startswith("feature_")]
    configured_price_column = getattr(config.env, "price_column", None)
    price_column = (
        configured_price_column
        if isinstance(configured_price_column, str) and configured_price_column in train_df.columns
        else "close"
    )
    return PreparedDataset(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        feature_columns=feature_columns,
        price_column=price_column,
        raw_columns=list(train_df.columns),
        memmap_train_paths=memmap_paths,
    )


def _build_single_symbol_splits(
    config: Any, logger: Any
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and process one symbol through the full feature engineering pipeline."""
    pipeline = _resolve_feature_pipeline(config, logger)
    prep_cfg = PrepareDataConfig.from_config(config.data)
    train_df, val_df, test_df = prepare_data(config.data.data_path, prep_cfg, pipeline)
    return _finalize_splits(train_df, val_df, test_df, config, logger)


def _per_symbol_worker(args: tuple) -> tuple[str, list[tuple[int, Any]], dict[str, str] | None, int | None]:
    """Process one symbol: fit pipeline, transform train files, transform val file.

    Runs in a subprocess — all imports are local and config is passed as plain types.

    Returns
    -------
    symbol : str
    train_results : list of (original_train_index, memmap_paths_dict | None)
        memmap_paths_dict keys: data_path, index_path, n_rows, columns
    val_entry : {val_str: path_str, test_str: path_str} | None
    val_index : int | None
    """
    (
        symbol,
        train_indices,        # list[int] — original positions in train_paths
        train_paths_sym,      # list[str] — paths for this symbol
        val_path,             # str | None
        val_index,            # int | None
        feature_config,       # str | None
        mode,
        backend,
        filter_lob_levels,    # int | None
        warmup_rows,
        memmap_dir_str,       # str | None
        tmp_dir_str,          # str
        worker_idx,           # int — 1-based index for log tag
        n_workers_total,      # int
    ) = args

    import gc as _gc
    import json as _json
    import logging as _logging
    from pathlib import Path as _Path

    import numpy as _np
    import pandas as _pd

    from logger import get_logger as _get_logger
    from trading_rl.constants import EnvBackend, EnvMode

    # Attach worker tag to every log record from this process.
    _tag = f"worker {worker_idx}/{n_workers_total}"
    _root = _logging.getLogger()
    for _h in _root.handlers:
        _h.addFilter(_WorkerLogFilter(_tag))
    from trading_rl.data.hft import _deduplicate_hft_index_single, _derive_close_hft_single
    from trading_rl.data.loading import load_trading_data
    from trading_rl.data_loading import MemmapPaths, save_symbol_memmap
    from trading_rl.features import FeaturePipeline
    from trading_rl.features.base import NormalizationMethod

    _logger = _get_logger(__name__)
    memmap_dir = _Path(memmap_dir_str) if memmap_dir_str else None
    tmp_dir = _Path(tmp_dir_str)

    def _load(path: str) -> _pd.DataFrame:
        df = load_trading_data(path).dropna()
        if filter_lob_levels is not None:
            from trading_rl.data.lob_filters import filter_unchanged_lob
            df = filter_unchanged_lob(df, levels=filter_lob_levels)
        return df

    # ── 1. Fit pipeline ───────────────────────────────────────────────────────
    pipeline = FeaturePipeline.from_yaml(feature_config)
    needs_full_concat = any(
        fc.normalize and fc.normalization_method == NormalizationMethod.GLOBAL
        for fc in pipeline.feature_configs
    )
    _logger.info("fit pipeline symbol=%s n_days=%d incremental=%s", symbol, len(train_paths_sym), not needs_full_concat)
    if needs_full_concat:
        raw_parts = [_load(p) for p in train_paths_sym]
        combined = _pd.concat(raw_parts)
        del raw_parts
        pipeline.fit(combined)
        del combined
        _gc.collect()
    else:
        for p in train_paths_sym:
            raw_df = _load(p)
            pipeline.fit(raw_df)
            del raw_df
            _gc.collect()

    # ── 2. Transform training files ───────────────────────────────────────────
    train_results: list[tuple[int, Any]] = []
    for orig_idx, train_path in zip(train_indices, train_paths_sym):
        _logger.info("transform train path=%s", train_path)
        raw_df = _load(train_path)
        feats = pipeline.transform(raw_df)
        train_df_i = _pd.concat([raw_df, feats], axis=1)
        del raw_df, feats

        if mode == EnvMode.HFT:
            train_df_i = _derive_close_hft_single(train_df_i, f"train_{orig_idx}", _logger)
        if mode == EnvMode.HFT and backend == EnvBackend.TRADINGENV:
            train_df_i = _deduplicate_hft_index_single(train_df_i, f"train_{orig_idx}", _logger)

        if warmup_rows > 0 and warmup_rows < len(train_df_i):
            train_df_i = train_df_i.iloc[warmup_rows:]

        memmap_entry = None
        if memmap_dir:
            prefix = str(orig_idx)
            memmap_marker = memmap_dir / f"{prefix}_train_data.npy"
            expected_cols = list(train_df_i.select_dtypes(include=[_np.number]).columns)
            if memmap_marker.exists():
                cached_data = _np.load(memmap_marker, mmap_mode="r")
                cached_cols = _json.loads((memmap_dir / f"{prefix}_columns.json").read_text())
                symbol_file = memmap_dir / f"{prefix}_symbol.txt"
                if not symbol_file.exists() and symbol:
                    symbol_file.write_text(symbol)
                if cached_data.shape[0] == len(train_df_i) and cached_cols == expected_cols:
                    memmap_entry = {
                        "data_path": str(memmap_marker),
                        "index_path": str(memmap_dir / f"{prefix}_train_index.npy"),
                        "n_rows": cached_data.shape[0],
                        "columns": cached_cols,
                        "symbol": symbol,
                    }
                else:
                    mp = save_symbol_memmap(train_df_i, memmap_dir, prefix, symbol=symbol)
                    memmap_entry = {
                        "data_path": str(mp.data_path),
                        "index_path": str(mp.index_path),
                        "n_rows": mp.n_rows,
                        "columns": mp.columns,
                        "symbol": mp.symbol,
                    }
            else:
                mp = save_symbol_memmap(train_df_i, memmap_dir, prefix, symbol=symbol)
                memmap_entry = {
                    "data_path": str(mp.data_path),
                    "index_path": str(mp.index_path),
                    "n_rows": mp.n_rows,
                    "columns": mp.columns,
                    "symbol": mp.symbol,
                }
        train_results.append((orig_idx, memmap_entry))
        del train_df_i
        _gc.collect()

    # ── 3. Transform val file ─────────────────────────────────────────────────
    val_entry: dict[str, str] | None = None
    if val_path is not None and val_index is not None:
        _logger.info("transform val path=%s symbol=%s", val_path, symbol)
        raw_val = _load(val_path)
        val_feats = pipeline.transform(raw_val)
        val_df_j = _pd.concat([raw_val, val_feats], axis=1)
        del raw_val, val_feats

        if mode == EnvMode.HFT:
            val_df_j = _derive_close_hft_single(val_df_j, f"val_{val_index}", _logger)
        if mode == EnvMode.HFT and backend == EnvBackend.TRADINGENV:
            val_df_j = _deduplicate_hft_index_single(val_df_j, f"val_{val_index}", _logger)

        mid = len(val_df_j) // 2
        val_p = tmp_dir / f"{val_index}_val.parquet"
        test_p = tmp_dir / f"{val_index}_test.parquet"
        val_df_j.iloc[:mid].to_parquet(val_p)
        val_df_j.iloc[mid:].to_parquet(test_p)
        val_entry = {"val": str(val_p), "test": str(test_p)}
        del val_df_j
        _gc.collect()

    return symbol, train_results, val_entry, val_index


def _build_per_day_splits(
    config: Any,
    logger: Any,
    train_paths: list[str],
    val_paths: list[str],
    memmap_dir: Path | None,
    symbol_index: int = 0,
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[MemmapPaths] | None]:
    """Process per-(symbol, day) training files with separate validation files.

    Training files are used in full (no internal split).  One feature pipeline
    is fit per symbol from all of that symbol's training days combined, which
    gives more stable normalisation statistics than fitting on a single day.

    Validation files are transformed with their symbol's fitted pipeline then
    split 50/50 into val and test halves.

    Symbol identity is derived from the filename prefix up to the first
    underscore, e.g. ``AAPL_2026-02-25_raw_mbp-10_us_hours.parquet`` → ``AAPL``.
    Training and validation paths must be ordered alphabetically by symbol so
    that each val file maps to the matching symbol group.
    """
    from collections import defaultdict

    feature_config = getattr(config.data, "feature_config", None)
    mode = str(getattr(config.env, "mode", "mft")).lower().strip()
    backend = str(getattr(config.env, "backend", "")).lower().strip()
    filter_lob_levels = getattr(config.data, "filter_lob_levels", None)
    warmup_rows = getattr(config.data, "warmup_rows", 0)

    def _symbol_of(path: str) -> str:
        return Path(path).name.split("_")[0]

    # Group training paths by symbol, preserving original indices
    symbol_train_paths: dict[str, list[str]] = defaultdict(list)
    symbol_train_indices: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(train_paths):
        sym = _symbol_of(p)
        symbol_train_paths[sym].append(p)
        symbol_train_indices[sym].append(i)

    # Map val paths to their symbol and original index
    symbol_val_path: dict[str, tuple[str, int]] = {}
    for j, vp in enumerate(val_paths):
        sym = _symbol_of(vp)
        symbol_val_path[sym] = (vp, j)

    # Validate all val symbols have a fitted pipeline available
    all_symbols = sorted(symbol_train_paths.keys())
    for sym in [_symbol_of(vp) for vp in val_paths]:
        if sym not in symbol_train_paths:
            raise ValueError(
                f"No training paths for symbol '{sym}' (needed to fit pipeline for val). "
                f"Known symbols: {all_symbols}"
            )

    tmp_dir = Path(tempfile.mkdtemp(prefix="per_day_splits_"))
    if memmap_dir:
        memmap_dir.mkdir(parents=True, exist_ok=True)

    _cfg_workers = getattr(getattr(config, "data", None), "n_workers", 0)
    n_workers = (
        min(len(all_symbols), os.cpu_count() or 1)
        if _cfg_workers <= 0
        else min(_cfg_workers, len(all_symbols))
    )
    logger.info(
        "per-day mode: processing %d symbols with %d workers", len(all_symbols), n_workers
    )

    worker_args = []
    for worker_idx, sym in enumerate(all_symbols, start=1):
        val_p, val_j = symbol_val_path.get(sym, (None, None))
        worker_args.append((
            sym,
            symbol_train_indices[sym],
            symbol_train_paths[sym],
            val_p,
            val_j,
            feature_config,
            mode,
            backend,
            filter_lob_levels,
            warmup_rows,
            str(memmap_dir) if memmap_dir else None,
            str(tmp_dir),
            worker_idx,
            n_workers,
        ))

    # Collect results keyed by original index so ordering is preserved
    train_memmap_by_idx: dict[int, Any] = {}
    val_entries_by_idx: dict[int, dict[str, str]] = {}

    # Forward worker log records to the parent's handlers via a shared queue.
    log_queue: multiprocessing.Queue = multiprocessing.Queue()
    parent_handlers = logging.getLogger().handlers or [logging.StreamHandler()]
    listener = logging.handlers.QueueListener(log_queue, *parent_handlers, respect_handler_level=True)
    listener.start()
    try:
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_worker_log_init,
            initargs=(log_queue,),
        ) as executor:
            futures = {executor.submit(_per_symbol_worker, args): args[0] for args in worker_args}
            for future in as_completed(futures):
                sym = futures[future]
                try:
                    _sym, train_results, val_entry, val_idx = future.result()
                except Exception as exc:
                    raise RuntimeError(f"Worker for symbol '{sym}' failed: {exc}") from exc
                for orig_idx, memmap_entry in train_results:
                    train_memmap_by_idx[orig_idx] = memmap_entry
                if val_entry is not None and val_idx is not None:
                    val_entries_by_idx[val_idx] = val_entry
                if progress_callback:
                    progress_callback(f"done {sym}")
    finally:
        listener.stop()

    # Reconstruct ordered memmap list
    collected_memmap_paths: list[MemmapPaths] = []
    if memmap_dir:
        for i in range(len(train_paths)):
            entry = train_memmap_by_idx.get(i)
            if entry is None:
                raise RuntimeError(f"Missing memmap result for train_paths[{i}]")
            collected_memmap_paths.append(MemmapPaths(
                data_path=Path(entry["data_path"]),
                index_path=Path(entry["index_path"]),
                n_rows=entry["n_rows"],
                columns=entry["columns"],
                symbol=entry.get("symbol", ""),
            ))

    # Reconstruct ordered val_tmp list
    val_tmp: list[dict[str, Path]] = []
    for j in range(len(val_paths)):
        entry = val_entries_by_idx.get(j)
        if entry is None:
            raise RuntimeError(f"Missing val result for val_paths[{j}]")
        val_tmp.append({"val": Path(entry["val"]), "test": Path(entry["test"])})

    # first_train_df: load the first training file (index 0) directly from its
    # already-written memmap or re-transform from tmp parquet — cheapest is to
    # just read the first memmap back as a DataFrame if it exists, otherwise
    # read from the tmp val parquet of the first symbol.
    first_train_df: pd.DataFrame | None = None
    if memmap_dir and collected_memmap_paths:
        mp0 = collected_memmap_paths[0]
        data = np.load(mp0.data_path, mmap_mode="r")
        idx_ns = np.load(mp0.index_path, allow_pickle=True)
        # Index is stored as int64 nanosecond timestamps — restore as DatetimeIndex.
        first_train_df = pd.DataFrame(
            data,
            index=pd.DatetimeIndex(idx_ns.astype("int64"), dtype="datetime64[ns]"),
            columns=mp0.columns,
        )
    elif val_tmp:
        # Fallback: use the representative val slice as a surrogate
        first_train_df = pd.read_parquet(val_tmp[0]["val"])

    if memmap_dir:
        _si = min(symbol_index, len(val_tmp) - 1)
        val_df  = pd.read_parquet(val_tmp[_si]["val"])
        test_df = pd.read_parquet(val_tmp[_si]["test"])
        logger.info(
            "streaming mode (per-day): representative sample symbol_index=%d val=%d test=%d",
            _si, len(val_df), len(test_df),
        )
    else:
        val_df  = pd.concat([pd.read_parquet(p["val"])  for p in val_tmp])
        test_df = pd.concat([pd.read_parquet(p["test"]) for p in val_tmp])
    shutil.rmtree(tmp_dir, ignore_errors=True)

    if mode == EnvMode.HFT and backend == EnvBackend.TRADINGENV:
        val_df  = _deduplicate_hft_index_single(val_df,  "val_concat",  logger)
        test_df = _deduplicate_hft_index_single(test_df, "test_concat", logger)

    train_size_cfg  = getattr(config.data, "train_size", None)
    validation_size = getattr(config.data, "validation_size", None)
    test_size_cfg   = getattr(config.data, "test_size", None)
    if train_size_cfg is not None and first_train_df is not None:
        first_train_df = first_train_df.iloc[:train_size_cfg]
    if validation_size is not None:
        val_df = val_df.iloc[:validation_size]
    if test_size_cfg is not None:
        test_df = test_df.iloc[:test_size_cfg]

    logger.info(
        "per-day splits: n_train_memmaps=%d train_sample=%d val=%d test=%d",
        len(collected_memmap_paths),
        len(first_train_df) if first_train_df is not None else 0,
        len(val_df),
        len(test_df),
    )
    validate_prepared_data(first_train_df, val_df, test_df, config)

    return first_train_df, val_df, test_df, collected_memmap_paths or None


def _build_pooled_splits(
    config: Any,
    logger: Any,
    data_paths: list[str],
    memmap_dir: Path | None,
    symbol_index: int = 0,
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[MemmapPaths] | None]:
    """Process each symbol independently then concatenate from disk.

    When ``config.data.val_data_paths`` is set the function delegates to
    :func:`_build_per_day_splits`, which treats each training file as a
    complete training unit (no internal train/val split) and sources
    validation data from the separate val files.

    Each symbol is feature-engineered, written to a temp parquet, and freed
    before the next symbol loads.  This keeps peak memory to ~1 symbol at a
    time during feature engineering rather than accumulating all symbols.

    Resumability is provided by the per-symbol feature cache in
    prepare_data(): symbols that were fully processed before an interruption
    return immediately from cache on the next run.
    """
    val_data_paths = getattr(config.data, "val_data_paths", None)
    if val_data_paths:
        return _build_per_day_splits(config, logger, data_paths, list(val_data_paths), memmap_dir, symbol_index, progress_callback)

    pipeline = _resolve_feature_pipeline(config, logger)
    prep_cfg = PrepareDataConfig.from_config(config.data)

    logger.info("pooled training n_symbols=%d", len(data_paths))
    tmp_dir = Path(tempfile.mkdtemp(prefix="pooled_splits_"))
    tmp_paths: list[dict[str, Path]] = []
    collected_memmap_paths: list[MemmapPaths] = []

    for i, data_path in enumerate(data_paths):
        logger.info("process symbol idx=%d/%d path=%s", i + 1, len(data_paths), data_path)
        # Reset scaler state so each symbol is normalised independently.
        # Without this, RunningMeanStd accumulates statistics across symbols,
        # making symbol N's normalization order-dependent on symbols 1..N-1.
        if pipeline is not None:
            pipeline.reset()
        train_i, val_i, test_i = prepare_data(data_path, prep_cfg, pipeline)
        # Apply close-column derivation per-symbol so each memmap is self-contained.
        train_i, val_i, test_i = ensure_close_column_for_hft(train_i, val_i, test_i, config, logger)
        # Deduplicate timestamps before saving to memmap so the streaming env
        # never receives a window with duplicate indices.
        train_i, val_i, test_i = ensure_unique_index_for_hft_tradingenv(train_i, val_i, test_i, config, logger)
        warmup_rows = getattr(config.data, "warmup_rows", 0)
        train_i = _apply_warmup_skip(train_i, warmup_rows, logger, label=Path(data_path).name)

        sym_name = Path(data_path).stem.split("_")[0]
        if memmap_dir:
            memmap_marker = memmap_dir / f"{i}_train_data.npy"
            expected_columns = list(train_i.select_dtypes(include=[np.number]).columns)
            if memmap_marker.exists():
                prefix = str(i)
                cols = json.loads((memmap_dir / f"{prefix}_columns.json").read_text())
                data = np.load(memmap_marker, mmap_mode="r")
                symbol_file = memmap_dir / f"{prefix}_symbol.txt"
                cached_symbol = symbol_file.read_text().strip() if symbol_file.exists() else ""
                if data.shape[0] >= len(train_i) and cols == expected_columns:
                    collected_memmap_paths.append(
                        MemmapPaths(
                            data_path=memmap_marker,
                            index_path=memmap_dir / f"{prefix}_train_index.npy",
                            n_rows=len(train_i),  # cap to requested size
                            columns=cols,
                            symbol=cached_symbol or sym_name,
                        )
                    )
                else:
                    logger.info(
                        "memmap cache mismatch prefix=%s expected_rows=%d actual_rows=%d",
                        prefix,
                        len(train_i),
                        data.shape[0],
                    )
                    collected_memmap_paths.append(save_symbol_memmap(train_i, memmap_dir, prefix=prefix, symbol=sym_name))
            else:
                collected_memmap_paths.append(save_symbol_memmap(train_i, memmap_dir, prefix=str(i), symbol=sym_name))

        sym: dict[str, Path] = {}
        for split, df in [("train", train_i), ("val", val_i), ("test", test_i)]:
            p = tmp_dir / f"{i}_{split}.parquet"
            df.to_parquet(p)
            sym[split] = p
        tmp_paths.append(sym)
        del train_i, val_i, test_i
        gc.collect()

    # When streaming via memmap the training env never reads train_df, so
    # avoid loading all symbols — use the first symbol as a representative
    # sample for validation, logging, and train-split evaluation.
    # val/test follow the same convention: a single-symbol slice keeps the
    # price series contiguous and prevents spurious cross-symbol returns that
    # corrupt the NLV/portfolio calculation in periodic evaluation.
    if memmap_dir:
        _si = min(symbol_index, len(tmp_paths) - 1)
        train_df = pd.read_parquet(tmp_paths[_si]["train"])
        val_df   = pd.read_parquet(tmp_paths[_si]["val"])
        test_df  = pd.read_parquet(tmp_paths[_si]["test"])
        logger.info(
            "streaming mode: representative sample symbol_index=%d of %d"
            " train=%d val=%d test=%d", _si, len(tmp_paths), len(train_df), len(val_df), len(test_df),
        )
    else:
        train_df = pd.concat([pd.read_parquet(p["train"]) for p in tmp_paths])
        val_df   = pd.concat([pd.read_parquet(p["val"])   for p in tmp_paths])
        test_df  = pd.concat([pd.read_parquet(p["test"])  for p in tmp_paths])
    shutil.rmtree(tmp_dir, ignore_errors=True)

    logger.info("pooled splits train=%d val=%d test=%d", len(train_df), len(val_df), len(test_df))

    # ensure_unique_index was applied per-symbol above; re-run on concat to fix
    # cross-symbol timestamp collisions in val/test (train is first-symbol only
    # in streaming mode so also benefits from a re-check).
    train_df, val_df, test_df = ensure_unique_index_for_hft_tradingenv(train_df, val_df, test_df, config, logger)
    validate_prepared_data(train_df, val_df, test_df, config)

    return train_df, val_df, test_df, collected_memmap_paths or None


def build_prepared_dataset(
    config: Any,
    logger: Any,
    progress_callback: Callable[[str], None] | None = None,
) -> PreparedDataset:
    """Build a prepared dataset bundle for RL training and evaluation."""
    lazy_load = getattr(config.data, "lazy_load", False)
    prepared_dir = Path(d) if (d := getattr(config.data, "prepared_data_dir", None)) else None
    memmap_dir   = Path(d) if (d := getattr(config.data, "memmap_dir", None)) else None

    # Fast path: skip feature engineering only when prepared and memmap caches
    # match the current split sizes, data sources, and feature config.
    cache_ready = (
        lazy_load
        and prepared_dir
        and _prepared_cache_compatible(config, prepared_dir, memmap_dir, logger)
    )
    if cache_ready:
        logger.info("load prepared splits cache_dir=%s", prepared_dir)
        lazy_splits = load_prepared_splits(prepared_dir)
        memmap_paths = load_memmap_paths(memmap_dir) if memmap_dir and memmap_dir.exists() else None

        # In non-per-day mode the cache may have more rows than requested.
        # Cap n_rows so the streaming env only sees the configured train_size.
        if memmap_paths and not getattr(config.data, "val_data_paths", None):
            req = getattr(config.data, "train_size", None)
            if req is not None:
                req = int(req)
                memmap_paths = [
                    MemmapPaths(mp.data_path, mp.index_path, min(mp.n_rows, req), mp.columns, mp.symbol)
                    if mp.n_rows > req else mp
                    for mp in memmap_paths
                ]

        # Re-apply size slicing: a larger cache must not leak extra rows.
        train_df = lazy_splits["train"]
        val_df = lazy_splits["val"]
        test_df = lazy_splits["test"]
        train_size_cfg = getattr(config.data, "train_size", None)
        validation_size = getattr(config.data, "validation_size", None)
        test_size_cfg = getattr(config.data, "test_size", None)
        if train_size_cfg is not None:
            train_df = train_df.iloc[:train_size_cfg]
        if validation_size is not None:
            val_df = val_df.iloc[:validation_size]
        if test_size_cfg is not None:
            test_df = test_df.iloc[:test_size_cfg]

        return _make_dataset(
            train_df, val_df, test_df,
            config, memmap_paths or None,
        )

    data_paths = getattr(config.data, "data_paths", None)
    if data_paths and len(data_paths) > 1 and not memmap_dir:
        raise ValueError(
            f"Multi-symbol pooled training requires data.memmap_dir to be set. "
            f"Got {len(data_paths)} symbols in data.data_paths but memmap_dir is not configured. "
            "Use the pooled_streaming scenario configs (e.g. pooled/td3_hft_lob_state_space_pooled_streaming) "
            "which set memmap_dir and use StreamingTradingEnvXY for correct per-symbol episode resets."
        )
    if data_paths:
        _strategy = getattr(getattr(config, "data", None), "eval_symbol_selection", "first")
        _symbol_index = _resolve_symbol_index(_strategy, len(data_paths), memmap_dir)
        _val_paths = getattr(config.data, "val_data_paths", None) or data_paths
        _eval_symbol = Path(_val_paths[min(_symbol_index, len(_val_paths) - 1)]).parent.name
        if memmap_dir:
            memmap_dir.mkdir(parents=True, exist_ok=True)
            (memmap_dir / ".eval_symbol_used").write_text(_eval_symbol)
        train_df, val_df, test_df, memmap_paths = _build_pooled_splits(
            config, logger, data_paths, memmap_dir, _symbol_index, progress_callback
        )
    else:
        train_df, val_df, test_df = _build_single_symbol_splits(config, logger)
        memmap_paths = [save_symbol_memmap(train_df, memmap_dir, "0")] if memmap_dir else None

    if lazy_load and prepared_dir:
        logger.info("save prepared splits cache_dir=%s", prepared_dir)
        prepared_dir.mkdir(parents=True, exist_ok=True)
        save_prepared_splits(train_df, val_df, test_df, prepared_dir)
        _write_prepared_cache_metadata(
            config,
            prepared_dir,
            train_df,
            val_df,
            test_df,
            memmap_paths,
        )

    return _make_dataset(train_df, val_df, test_df, config, memmap_paths)


def prepare_data(
    data_path: str,
    cfg: PrepareDataConfig,
    feature_pipeline: Any | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Prepare trading data for RL training with proper train/val/test split.

    CRITICAL: This function splits data BEFORE feature engineering to prevent
    data leakage. Normalization statistics are computed only on training data.

    Args:
        data_path: Path to the data file for this symbol.
        cfg: PrepareDataConfig with split sizes, cache settings, and download params.
        feature_pipeline: Pre-built FeaturePipeline instance. When provided it
            takes priority over cfg.feature_config_path.

    Returns:
        Tuple of (train_df, val_df, test_df) with raw OHLCV plus engineered features.

    Example:
        cfg = PrepareDataConfig(train_size=450, feature_config_path="configs/feature_sets/sine_wave.yaml")
        train_df, val_df, test_df = prepare_data("data.parquet", cfg)
    """
    # Feature cache — skip the expensive transform when inputs haven't changed.
    # The full transformed dataset is cached as a single file; train/val/test
    # splits are sliced at load time so the cache is shared across configs that
    # differ only in split sizes.
    _cache_entry: Path | None = None
    _cache_key: str | None = None
    if cfg.feature_cache_dir and Path(data_path).exists():
        _cache_key = _feature_cache_key(
            data_path, cfg.feature_config_path, feature_pipeline, cfg.filter_lob_levels
        )
        _cache_entry = Path(cfg.feature_cache_dir) / _cache_key
        _full_cache = _cache_entry / "full.parquet"
        if _full_cache.exists():
            logger.info("feature cache hit key=%s path=%s", _cache_key[:8], _cache_entry)
            full_df = pd.read_parquet(_full_cache)
            # Resolve split sizes against the cached row count
            _n = len(full_df)
            _train = min(cfg.train_size, _n)
            _remaining = _n - _train
            _val = cfg.validation_size if cfg.validation_size is not None else (
                _remaining // 2 if cfg.test_size is None else max(0, _remaining - cfg.test_size)
            )
            _val_end = _train + _val
            _test_end = (_val_end + cfg.test_size) if cfg.test_size is not None else _n
            return (
                full_df.iloc[:_train],
                full_df.iloc[_train:_val_end],
                full_df.iloc[_val_end:_test_end],
            )
        logger.info("feature cache miss key=%s", _cache_key[:8])

    # Check if data exists
    if not Path(data_path).exists():
        if cfg.download_if_missing and cfg.exchange_names and cfg.symbols and cfg.since:
            download_trading_data(cfg.exchange_names, cfg.symbols, cfg.timeframe, cfg.data_dir, cfg.since)
        else:
            raise FileNotFoundError(
                f"Data file not found: {data_path}. "
                "Set download_if_missing=True to download."
            )

    # Load raw OHLCV data
    df = load_trading_data(data_path)
    df = df.dropna()
    if cfg.filter_lob_levels is not None:
        from trading_rl.data.lob_filters import filter_unchanged_lob
        df = filter_unchanged_lob(df, levels=cfg.filter_lob_levels)

    logger.info("load raw data n_rows=%d n_cols=%d", len(df), len(df.columns))

    # Split data BEFORE feature engineering (critical for preventing leakage!)
    train_size = cfg.train_size
    validation_size = cfg.validation_size
    test_size = cfg.test_size
    if train_size >= len(df):
        raise ValueError(
            f"train_size ({train_size}) must be smaller than dataset length ({len(df)})"
        )
    remaining = len(df) - train_size
    if validation_size is None:
        validation_size = remaining // 2 if test_size is None else (remaining - test_size) // 2
    if validation_size < 0:
        raise ValueError(f"validation_size must be >= 0, got {validation_size}")
    if validation_size >= remaining:
        raise ValueError(
            f"validation_size ({validation_size}) must be smaller than remaining "
            f"data after train split ({remaining})"
        )
    val_end = train_size + validation_size
    if test_size is not None:
        test_end = val_end + test_size
        if test_end > len(df):
            raise ValueError(
                f"train_size + validation_size + test_size ({test_end}) exceeds "
                f"dataset length ({len(df)})"
            )
    else:
        test_end = len(df)

    train_df_raw = df[:train_size].copy()
    val_df_raw = df[train_size:val_end].copy()
    test_df_raw = df[val_end:test_end].copy()

    logger.info(
        "split data train=%d val=%d test=%d",
        len(train_df_raw),
        len(val_df_raw),
        len(test_df_raw),
    )

    # Create feature pipeline
    from trading_rl.features import FeaturePipeline, create_default_pipeline

    if feature_pipeline is not None:
        logger.info("use pre-built feature pipeline n_features=%d", len(feature_pipeline.features))
        pipeline = feature_pipeline
    elif cfg.feature_config_path:
        logger.info("load feature pipeline path=%s", cfg.feature_config_path)
        pipeline = FeaturePipeline.from_yaml(cfg.feature_config_path)
    else:
        logger.info("use default feature pipeline")
        pipeline = create_default_pipeline()

    # Fit pipeline on training data ONLY (prevents test data leakage)
    logger.info("fit feature pipeline")
    pipeline.fit(train_df_raw)

    # Transform the full dataset in one pass. Because all normalizers are
    # session-aware causal (stats reset at each intraday session boundary),
    # the feature value at tick t depends only on prior ticks in the same
    # session — not on where the train/val/test boundary falls. Transforming
    # the full array lets us cache once and slice for any split configuration.
    full_df_raw = df.iloc[:test_end].copy()
    logger.info("transform full dataset n_rows=%d", len(full_df_raw))
    full_features = pipeline.transform(full_df_raw)
    full_df = pd.concat([full_df_raw, full_features], axis=1)
    del full_df_raw, full_features

    train_df = full_df.iloc[:train_size]
    val_df = full_df.iloc[train_size:val_end]
    test_df = full_df.iloc[val_end:test_end]

    logger.info(
        "feature engineering complete"
        " train_n_rows=%d train_n_cols=%d"
        " val_n_rows=%d val_n_cols=%d"
        " test_n_rows=%d test_n_cols=%d",
        *train_df.shape, *val_df.shape, *test_df.shape,
    )
    feature_cols = [c for c in train_df.columns if str(c).startswith("feature_")]
    logger.info("feature columns cols=%s", feature_cols)

    if _cache_entry is not None:
        _cache_entry.mkdir(parents=True, exist_ok=True)
        full_df.to_parquet(_cache_entry / "full.parquet")
        logger.info("save feature cache key=%s path=%s", _cache_key[:8], _cache_entry)

    return train_df, val_df, test_df
