"""Data loading and preprocessing utilities for trading RL."""

import gc
import hashlib
import json
import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from gym_trading_env.downloader import download
from logger import get_logger

logger = get_logger(__name__)

# Lazy loading and memmap utilities
from trading_rl.data_loading import (
    LazyDataFrame,
    MemmapPaths,
    load_memmap_paths,
    load_prepared_splits,
    save_prepared_splits,
    save_symbol_memmap,
)

_HFT_MIN_TIMESTAMP_GAP_NS = 1_000  # 1 µs — minimum distinguishable by Python datetime (microsecond precision)


@dataclass(frozen=True)
class PreparedDataset:
    """Prepared RL dataset with split dataframes and derived metadata."""

    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    feature_columns: list[str]
    price_column: str
    raw_columns: list[str]
    # Per-symbol memmap paths for StreamingTradingEnv. None when memmap_dir is
    # not configured; set by _build_pooled_dataset / build_prepared_dataset.
    memmap_train_paths: list[MemmapPaths] | None = None


def download_trading_data(
    exchange_names: list[str],
    symbols: list[str],
    timeframe: str,
    data_dir: str,
    since: Any | None = None,
) -> None:
    """Download historical trading data from exchanges.

    Args:
        exchange_names: List of exchange names (e.g., ["binance"])
        symbols: List of trading pairs (e.g., ["BTC/USDT"])
        timeframe: Timeframe for candles (e.g., "1h", "1d")
        data_dir: Directory to save downloaded data
        since: Start date for data download
    """
    if download is None:
        raise ImportError(
            "gym_trading_env package is required for data downloading. "
            "Install it with: pip install gym-trading-env"
        )

    logger.info("download data symbols=%s exchanges=%s", symbols, exchange_names)
    download(
        exchange_names=exchange_names,
        symbols=symbols,
        timeframe=timeframe,
        dir=data_dir,
        since=since,
    )
    logger.info("download data complete")


def load_trading_data(data_path: str) -> pd.DataFrame:
    """Load trading data from parquet or pickle file.

    Args:
        data_path: Path to parquet or pickle file

    Returns:
        DataFrame with OHLCV data
    """
    data_file = Path(data_path)
    logger.info("load data path=%s", data_file)
    suffix = data_file.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        df = pd.read_pickle(data_file)
    elif suffix in {".parquet"}:
        df = pd.read_parquet(data_file)
    else:
        raise ValueError(
            f"Unsupported data format '{suffix}' for file {data_file}. "
            "Supported formats: .pkl, .pickle, .parquet"
        )
    logger.info("load data n_rows=%d", len(df))
    return df



def validate_prepared_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Any,
) -> None:
    """Validate prepared split data before building environments."""
    if train_df.empty:
        raise ValueError(
            "Training data is empty. Check data_path or download settings."
        )
    if val_df.empty:
        raise ValueError("Validation data is empty. Check train/validation sizes.")
    if test_df.empty:
        raise ValueError("Test data is empty. Check train/validation size settings.")

    if "close" not in train_df.columns:
        raise ValueError(
            f"Data must contain raw 'close' column for environment pricing. "
            f"Found columns: {list(train_df.columns)}"
        )

    feature_cols = [col for col in train_df.columns if str(col).startswith("feature_")]
    if not feature_cols:
        raise ValueError(
            "No feature_* columns found in prepared data. "
            "Define features in data.feature_config."
        )

    env_feature_cols = getattr(config.env, "feature_columns", None)
    if env_feature_cols:
        non_feature_cols = [
            col for col in env_feature_cols if not str(col).startswith("feature_")
        ]
        if non_feature_cols:
            raise ValueError(
                "env.feature_columns must contain only feature_* columns. "
                f"Found: {non_feature_cols}"
            )

    if train_df.isnull().any().any():
        nan_cols = train_df.columns[train_df.isnull().any()].tolist()
        raise ValueError(
            f"Training data contains NaN values in columns: {nan_cols}. "
            f"Clean the data before training."
        )


_Splits = tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]


def _map_splits(
    fn,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> _Splits:
    """Apply fn(split_name, df) to each split and return (train, val, test)."""
    results = {name: fn(name, df) for name, df in (("train", train_df), ("val", val_df), ("test", test_df))}
    return results["train"], results["val"], results["test"]


def _derive_close_hft_single(
    df: pd.DataFrame,
    split_name: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Derive 'close' mid-price column for one HFT split if not already present."""
    if "close" in df.columns:
        return df
    required = {"ask_px_00", "bid_px_00"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "HFT mode requires a raw 'close' column or top-of-book columns "
            f"ask_px_00/bid_px_00. Missing in {split_name}: {missing}"
        )
    result = df.copy()
    mid_price = (result["ask_px_00"] + result["bid_px_00"]) / 2.0
    if "price" in result.columns:
        mid_price = mid_price.fillna(result["price"])
    result["close"] = mid_price.ffill().bfill()
    nan_ratio = float(result["close"].isna().mean())
    method = "mid_price+fallback" if "price" in result.columns else "mid_price"
    logger.info("derive close split=%s method=%s nan_ratio=%.6f", split_name, method, nan_ratio)
    return result


def _deduplicate_hft_index_single(
    df: pd.DataFrame,
    split_name: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Enforce unique, monotonic nanosecond timestamps for one HFT split."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(
            "HFT TradingEnv requires DatetimeIndex to enforce unique event ordering, "
            f"but {split_name} split has index type {type(df.index).__name__}."
        )
    min_gap_ns = _HFT_MIN_TIMESTAMP_GAP_NS
    index = df.index
    index_ns_raw = index.view("i8")
    old_min_gap_ns = int(np.diff(index_ns_raw).min()) if len(index_ns_raw) > 1 else min_gap_ns
    requires_adjustment = (
        not index.is_unique
        or not index.is_monotonic_increasing
        or old_min_gap_ns < min_gap_ns
        or index.tz is not None
    )
    if not requires_adjustment:
        return df
    if not index.is_monotonic_increasing:
        df = df.sort_index(kind="stable")
    else:
        df = df.copy(deep=False)
    index = df.index
    index_ns = index.view("i8").copy()
    positions = np.arange(len(index_ns), dtype=np.int64) * min_gap_ns
    adjusted_ns = np.maximum.accumulate(index_ns - positions) + positions
    adjusted_index = pd.to_datetime(adjusted_ns, utc=True).tz_localize(None)
    df.index = adjusted_index
    if not df.index.is_unique:
        raise ValueError(f"Failed to enforce unique index for {split_name} HFT split.")
    duplicate_count = int(index.duplicated().sum())
    max_shift_ns = int((adjusted_ns - index_ns).max()) if len(index_ns) else 0
    new_min_gap_ns = int(np.diff(adjusted_ns).min()) if len(adjusted_ns) > 1 else min_gap_ns
    logger.info(
        "adjust hft index split=%s duplicates=%d min_gap_ns=%d->%d max_shift_ns=%d",
        split_name, duplicate_count, old_min_gap_ns, new_min_gap_ns, max_shift_ns,
    )
    return df


def ensure_close_column_for_hft(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Any,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Ensure raw `close` exists in HFT mode by deriving mid-price from L1 book."""
    mode = str(getattr(config.env, "mode", "mft")).lower().strip()
    if mode != "hft":
        return train_df, val_df, test_df
    return (
        _derive_close_hft_single(train_df, "train", logger),
        _derive_close_hft_single(val_df, "val", logger),
        _derive_close_hft_single(test_df, "test", logger),
    )


def ensure_unique_index_for_hft_tradingenv(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Any,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Ensure unique, monotonic timestamps for HFT data used with TradingEnv."""
    mode = str(getattr(config.env, "mode", "mft")).lower().strip()
    backend = str(getattr(config.env, "backend", "")).lower().strip()
    if mode != "hft" or backend != "tradingenv":
        return train_df, val_df, test_df
    return (
        _deduplicate_hft_index_single(train_df, "train", logger),
        _deduplicate_hft_index_single(val_df, "val", logger),
        _deduplicate_hft_index_single(test_df, "test", logger),
    )


# ---------------------------------------------------------------------------
# Dataset-building helpers
# ---------------------------------------------------------------------------

def _parquet_cache_exists(prepared_dir: Path) -> bool:
    return all(
        (prepared_dir / f"{split}_prepared.parquet").exists()
        for split in ["train", "val", "test"]
    )


def _prepared_cache_metadata_path(prepared_dir: Path) -> Path:
    return prepared_dir / "_prepared_cache_metadata.json"


def _data_source_signature(config: Any) -> list[dict[str, Any]]:
    paths = list(config.data.data_paths or [config.data.data_path])
    val_paths = list(getattr(config.data, "val_data_paths", None) or [])
    signature = []
    for path_value in paths + val_paths:
        path = Path(path_value)
        stat = path.stat() if path.exists() else None
        signature.append(
            {
                "path": str(path),
                "mtime_ns": stat.st_mtime_ns if stat else None,
                "size": stat.st_size if stat else None,
            }
        )
    return signature


def _config_cache_signature(config: Any) -> dict[str, Any]:
    feature_config = getattr(config.data, "feature_config", None)
    feature_groups = getattr(config.data, "feature_groups", None)

    def _file_hash(path_value: str | None) -> str | None:
        if not path_value:
            return None
        path = Path(path_value)
        if not path.exists():
            return None
        return hashlib.sha256(path.read_bytes()).hexdigest()

    return {
        "version": 1,
        "data_sources": _data_source_signature(config),
        "train_size": getattr(config.data, "train_size", None),
        "validation_size": getattr(config.data, "validation_size", None),
        "test_size": getattr(config.data, "test_size", None),
        "feature_config": feature_config,
        "feature_config_hash": _file_hash(feature_config),
        "feature_groups": feature_groups,
        "feature_groups_hash": _file_hash(feature_groups),
        "env_mode": getattr(config.env, "mode", None),
        "env_backend": getattr(config.env, "backend", None),
        "price_column": getattr(config.env, "price_column", None),
        # env.feature_columns and include_position_feature are selection settings
        # applied at env build time — they do not affect how data is prepared, so
        # they must NOT be part of the cache signature.
    }


def _expected_cached_split_rows(config: Any, memmap_dir: Path | None) -> dict[str, int | None]:
    # Per-day mode: each file has a different number of rows, so we cannot
    # assert a fixed split size — return None to skip row-count validation.
    if getattr(config.data, "val_data_paths", None):
        return {"train": None, "val": None, "test": None}

    data_paths = getattr(config.data, "data_paths", None) or []
    n_symbols = len(data_paths) if data_paths else 1
    train_rows = getattr(config.data, "train_size", None)
    validation_size = getattr(config.data, "validation_size", None)
    test_size = getattr(config.data, "test_size", None)
    return {
        "train": train_rows,
        "val": validation_size * n_symbols if validation_size is not None else None,
        "test": test_size * n_symbols if test_size is not None else None,
    }


def _cached_split_rows(prepared_dir: Path) -> dict[str, int]:
    rows = {}
    for split in ["train", "val", "test"]:
        rows[split] = len(pd.read_parquet(prepared_dir / f"{split}_prepared.parquet"))
    return rows


def _memmap_cache_compatible(
    config: Any,
    memmap_dir: Path | None,
    logger: logging.Logger,
) -> bool:
    if not memmap_dir:
        return True
    if not memmap_dir.exists():
        return False
    paths = load_memmap_paths(memmap_dir)
    if not paths:
        return False

    data_paths = getattr(config.data, "data_paths", None) or []
    expected_count = len(data_paths) if data_paths else 1
    if len(paths) != expected_count:
        logger.info(
            "memmap cache mismatch expected_symbols=%d actual_symbols=%d",
            expected_count,
            len(paths),
        )
        return False

    # Per-day mode: each memmap has variable rows (one per (symbol, day) file).
    # Skip the uniform row-count check; file count is sufficient.
    if getattr(config.data, "val_data_paths", None):
        return True

    expected_train_rows = int(config.data.train_size)
    bad_rows = [p.n_rows for p in paths if p.n_rows != expected_train_rows]
    if bad_rows:
        logger.info(
            "memmap cache mismatch expected_train_rows=%d actual_rows=%s",
            expected_train_rows,
            bad_rows,
        )
        return False
    return True


def _prepared_cache_compatible(
    config: Any,
    prepared_dir: Path,
    memmap_dir: Path | None,
    logger: logging.Logger,
) -> bool:
    if not _parquet_cache_exists(prepared_dir):
        return False
    if not _memmap_cache_compatible(config, memmap_dir, logger):
        return False

    metadata_path = _prepared_cache_metadata_path(prepared_dir)
    expected_signature = _config_cache_signature(config)
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text())
        except json.JSONDecodeError:
            logger.info("prepared cache metadata invalid path=%s", metadata_path)
            return False
        if metadata.get("config_signature") != expected_signature:
            logger.info("prepared cache metadata mismatch path=%s", metadata_path)
            return False
        expected_rows = _expected_cached_split_rows(config, memmap_dir)
        metadata_rows = metadata.get("split_rows", {})
        for split, expected in expected_rows.items():
            if expected is not None and metadata_rows.get(split) != expected:
                logger.info(
                    "prepared cache metadata row mismatch split=%s expected=%s actual=%s",
                    split,
                    expected,
                    metadata_rows.get(split),
                )
                return False
        return True

    expected_rows = _expected_cached_split_rows(config, memmap_dir)
    actual_rows = _cached_split_rows(prepared_dir)
    for split, expected in expected_rows.items():
        if expected is not None and actual_rows[split] != expected:
            logger.info(
                "legacy prepared cache row mismatch split=%s expected=%s actual=%s",
                split,
                expected,
                actual_rows[split],
            )
            return False

    logger.info("legacy prepared cache accepted without metadata dir=%s", prepared_dir)
    return True


def _write_prepared_cache_metadata(
    config: Any,
    prepared_dir: Path,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    memmap_paths: list[MemmapPaths] | None,
) -> None:
    metadata = {
        "config_signature": _config_cache_signature(config),
        "split_rows": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
        "memmap_rows": [p.n_rows for p in memmap_paths or []],
        "memmap_files": [str(p.data_path) for p in memmap_paths or []],
    }
    _prepared_cache_metadata_path(prepared_dir).write_text(
        json.dumps(metadata, indent=2, sort_keys=True)
    )


def _resolve_feature_pipeline(config: Any, logger: logging.Logger) -> Any:
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
    def from_config(cls, cfg: Any) -> "PrepareDataConfig":
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


def _finalize_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Any,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply post-processing transforms and validation."""
    train_df, val_df, test_df = ensure_close_column_for_hft(train_df, val_df, test_df, config, logger)
    train_df, val_df, test_df = ensure_unique_index_for_hft_tradingenv(train_df, val_df, test_df, config, logger)
    validate_prepared_data(train_df, val_df, test_df, config)
    return train_df, val_df, test_df


def _make_dataset(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Any,
    memmap_paths: list[MemmapPaths] | None = None,
) -> "PreparedDataset":
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


# ---------------------------------------------------------------------------
# Symbol-level processing
# ---------------------------------------------------------------------------

def _build_single_symbol_splits(
    config: Any, logger: logging.Logger
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and process one symbol through the full feature engineering pipeline."""
    pipeline = _resolve_feature_pipeline(config, logger)
    prep_cfg = PrepareDataConfig.from_config(config.data)
    train_df, val_df, test_df = prepare_data(config.data.data_path, prep_cfg, pipeline)
    return _finalize_splits(train_df, val_df, test_df, config, logger)


def _build_per_day_splits(
    config: Any,
    logger: logging.Logger,
    train_paths: list[str],
    val_paths: list[str],
    memmap_dir: Path | None,
    progress_callback: "Callable[[str], None] | None" = None,
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

    from trading_rl.features import FeaturePipeline

    feature_config = getattr(config.data, "feature_config", None)
    mode = str(getattr(config.env, "mode", "mft")).lower().strip()
    backend = str(getattr(config.env, "backend", "")).lower().strip()
    filter_lob_levels = getattr(config.data, "filter_lob_levels", None)

    def _load(path: str) -> pd.DataFrame:
        df = load_trading_data(path).dropna()
        if filter_lob_levels is not None:
            from trading_rl.data.lob_filters import filter_unchanged_lob
            df = filter_unchanged_lob(df, levels=filter_lob_levels)
        return df

    def _symbol_of(path: str) -> str:
        return Path(path).name.split("_")[0]

    # Group training paths by symbol (order within each group preserved)
    symbol_train_paths: dict[str, list[str]] = defaultdict(list)
    for p in train_paths:
        symbol_train_paths[_symbol_of(p)].append(p)

    # Fit one FeaturePipeline per symbol on concatenated training days
    logger.info("per-day mode: fitting per-symbol pipelines n_symbols=%d", len(symbol_train_paths))
    symbol_pipelines: dict[str, Any] = {}
    for symbol, sym_paths in sorted(symbol_train_paths.items()):
        logger.info("fit pipeline symbol=%s n_days=%d", symbol, len(sym_paths))
        raw_parts = [_load(p) for p in sym_paths]
        combined = pd.concat(raw_parts)
        del raw_parts
        pipeline = FeaturePipeline.from_yaml(feature_config)
        pipeline.fit(combined)
        symbol_pipelines[symbol] = pipeline
        del combined
        gc.collect()
        if progress_callback:
            progress_callback(f"fit {symbol}")

    # Transform each training file → save memmap
    tmp_dir = Path(tempfile.mkdtemp(prefix="per_day_splits_"))
    collected_memmap_paths: list[MemmapPaths] = []
    first_train_df: pd.DataFrame | None = None

    for i, train_path in enumerate(train_paths):
        symbol = _symbol_of(train_path)
        pipeline = symbol_pipelines[symbol]
        logger.info("transform train idx=%d/%d path=%s", i + 1, len(train_paths), train_path)

        raw_df = _load(train_path)
        train_features = pipeline.transform(raw_df)
        train_df_i = pd.concat([raw_df, train_features], axis=1)
        del raw_df, train_features

        if mode == "hft":
            train_df_i = _derive_close_hft_single(train_df_i, f"train_{i}", logger)
        if mode == "hft" and backend == "tradingenv":
            train_df_i = _deduplicate_hft_index_single(train_df_i, f"train_{i}", logger)

        if first_train_df is None:
            first_train_df = train_df_i.copy()

        if memmap_dir:
            prefix = str(i)
            memmap_marker = memmap_dir / f"{prefix}_train_data.npy"
            expected_cols = list(train_df_i.select_dtypes(include=[np.number]).columns)
            if memmap_marker.exists():
                cached_data = np.load(memmap_marker, mmap_mode="r")
                cached_cols = json.loads((memmap_dir / f"{prefix}_columns.json").read_text())
                if cached_data.shape[0] == len(train_df_i) and cached_cols == expected_cols:
                    collected_memmap_paths.append(
                        MemmapPaths(
                            data_path=memmap_marker,
                            index_path=memmap_dir / f"{prefix}_train_index.npy",
                            n_rows=cached_data.shape[0],
                            columns=cached_cols,
                        )
                    )
                else:
                    logger.info(
                        "memmap cache mismatch prefix=%s expected_rows=%d actual_rows=%d",
                        prefix, len(train_df_i), cached_data.shape[0],
                    )
                    collected_memmap_paths.append(
                        save_symbol_memmap(train_df_i, memmap_dir, prefix)
                    )
            else:
                collected_memmap_paths.append(save_symbol_memmap(train_df_i, memmap_dir, prefix))

        del train_df_i
        gc.collect()
        if progress_callback:
            progress_callback(f"train {Path(train_path).name}")

    # Transform val files: split each 50/50 into val and test halves
    val_tmp: list[dict[str, Path]] = []
    for j, val_path in enumerate(val_paths):
        symbol = _symbol_of(val_path)
        pipeline = symbol_pipelines.get(symbol)
        if pipeline is None:
            raise ValueError(
                f"No fitted pipeline for symbol '{symbol}' (from val path {val_path}). "
                f"Known symbols: {sorted(symbol_pipelines)}"
            )
        logger.info("transform val idx=%d/%d path=%s symbol=%s", j + 1, len(val_paths), val_path, symbol)

        raw_val = _load(val_path)
        val_features = pipeline.transform(raw_val)
        val_df_j = pd.concat([raw_val, val_features], axis=1)
        del raw_val, val_features

        if mode == "hft":
            val_df_j = _derive_close_hft_single(val_df_j, f"val_{j}", logger)
        if mode == "hft" and backend == "tradingenv":
            val_df_j = _deduplicate_hft_index_single(val_df_j, f"val_{j}", logger)

        mid = len(val_df_j) // 2
        sym_paths: dict[str, Path] = {}
        for split, df_part in [("val", val_df_j.iloc[:mid]), ("test", val_df_j.iloc[mid:])]:
            p = tmp_dir / f"{j}_{split}.parquet"
            df_part.to_parquet(p)
            sym_paths[split] = p
        val_tmp.append(sym_paths)
        del val_df_j
        gc.collect()
        if progress_callback:
            progress_callback(f"val {Path(val_path).name}")

    val_df = pd.concat([pd.read_parquet(p["val"]) for p in val_tmp])
    test_df = pd.concat([pd.read_parquet(p["test"]) for p in val_tmp])
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # Re-run index deduplication on the concatenated val/test to fix any
    # cross-symbol timestamp collisions introduced by pd.concat.
    if mode == "hft" and backend == "tradingenv":
        val_df = _deduplicate_hft_index_single(val_df, "val_concat", logger)
        test_df = _deduplicate_hft_index_single(test_df, "test_concat", logger)

    logger.info(
        "per-day splits: n_train_memmaps=%d val=%d test=%d",
        len(collected_memmap_paths), len(val_df), len(test_df),
    )
    validate_prepared_data(first_train_df, val_df, test_df, config)

    return first_train_df, val_df, test_df, collected_memmap_paths or None


def _build_pooled_splits(
    config: Any,
    logger: logging.Logger,
    data_paths: list[str],
    memmap_dir: Path | None,
    progress_callback: "Callable[[str], None] | None" = None,
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
        return _build_per_day_splits(config, logger, data_paths, list(val_data_paths), memmap_dir, progress_callback)

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

        if memmap_dir:
            memmap_marker = memmap_dir / f"{i}_train_data.npy"
            expected_columns = list(train_i.select_dtypes(include=[np.number]).columns)
            if memmap_marker.exists():
                prefix = str(i)
                cols = json.loads((memmap_dir / f"{prefix}_columns.json").read_text())
                data = np.load(memmap_marker, mmap_mode="r")
                if data.shape[0] == len(train_i) and cols == expected_columns:
                    collected_memmap_paths.append(
                        MemmapPaths(
                            data_path=memmap_marker,
                            index_path=memmap_dir / f"{prefix}_train_index.npy",
                            n_rows=data.shape[0],
                            columns=cols,
                        )
                    )
                else:
                    logger.info(
                        "memmap cache mismatch prefix=%s expected_rows=%d actual_rows=%d",
                        prefix,
                        len(train_i),
                        data.shape[0],
                    )
                    collected_memmap_paths.append(save_symbol_memmap(train_i, memmap_dir, prefix=prefix))
            else:
                collected_memmap_paths.append(save_symbol_memmap(train_i, memmap_dir, prefix=str(i)))

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
    if memmap_dir:
        train_df = pd.read_parquet(tmp_paths[0]["train"])
        logger.info(
            "streaming mode: skipping full train concat, using first symbol as sample"
            " n_rows=%d", len(train_df),
        )
    else:
        train_df = pd.concat([pd.read_parquet(p["train"]) for p in tmp_paths])

    val_df  = pd.concat([pd.read_parquet(p["val"])   for p in tmp_paths])
    test_df = pd.concat([pd.read_parquet(p["test"])  for p in tmp_paths])
    shutil.rmtree(tmp_dir, ignore_errors=True)

    logger.info("pooled splits train=%d val=%d test=%d", len(train_df), len(val_df), len(test_df))

    # ensure_unique_index was applied per-symbol above; re-run on concat to fix
    # cross-symbol timestamp collisions in val/test (train is first-symbol only
    # in streaming mode so also benefits from a re-check).
    train_df, val_df, test_df = ensure_unique_index_for_hft_tradingenv(train_df, val_df, test_df, config, logger)
    validate_prepared_data(train_df, val_df, test_df, config)

    return train_df, val_df, test_df, collected_memmap_paths or None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_prepared_dataset(
    config: Any,
    logger: logging.Logger,
    progress_callback: "Callable[[str], None] | None" = None,
) -> "PreparedDataset":
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
        return _make_dataset(
            lazy_splits["train"], lazy_splits["val"], lazy_splits["test"],
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
        train_df, val_df, test_df, memmap_paths = _build_pooled_splits(
            config, logger, data_paths, memmap_dir, progress_callback
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


def _feature_cache_key(
    data_path: str,
    train_size: int,
    validation_size: int | None,
    test_size: int | None,
    feature_config_path: str | None,
    feature_pipeline: Any | None,
) -> str:
    """Compute a cache key that changes whenever inputs change."""
    file_mtime = Path(data_path).stat().st_mtime_ns

    if feature_pipeline is not None:
        pipeline_repr = [
            {"name": fc.name, "feature_type": fc.feature_type, "params": fc.params}
            for fc in feature_pipeline.feature_configs
        ]
        config_sig = hashlib.md5(
            json.dumps(pipeline_repr, sort_keys=True).encode()
        ).hexdigest()[:12]
    elif feature_config_path and Path(feature_config_path).exists():
        config_sig = hashlib.md5(Path(feature_config_path).read_bytes()).hexdigest()[:12]
    else:
        config_sig = "default"

    raw = f"{Path(data_path).name}|{file_mtime}|{train_size}|{validation_size}|{test_size}|{config_sig}"
    return hashlib.md5(raw.encode()).hexdigest()


def prepare_data(
    data_path: str,
    cfg: "PrepareDataConfig",
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
        cfg = PrepareDataConfig(train_size=450, feature_config_path="configs/features/sine_wave.yaml")
        train_df, val_df, test_df = prepare_data("data.parquet", cfg)
    """
    # Feature cache — skip the expensive transform when inputs haven't changed.
    _cache_entry: Path | None = None
    if cfg.feature_cache_dir and Path(data_path).exists():
        cache_key = _feature_cache_key(
            data_path, cfg.train_size, cfg.validation_size, cfg.test_size, cfg.feature_config_path, feature_pipeline
        )
        _cache_entry = Path(cfg.feature_cache_dir) / cache_key
        _splits = ("train", "val", "test")
        if all((_cache_entry / f"{s}.parquet").exists() for s in _splits):
            logger.info("feature cache hit key=%s path=%s", cache_key[:8], _cache_entry)
            return tuple(pd.read_parquet(_cache_entry / f"{s}.parquet") for s in _splits)  # type: ignore[return-value]
        logger.info("feature cache miss key=%s", cache_key[:8])

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

    # Transform train/validation/test using fitted parameters
    logger.info("transform split name=train")
    train_features = pipeline.transform(train_df_raw)

    logger.info("transform split name=val")
    val_features = pipeline.transform(val_df_raw)

    logger.info("transform split name=test")
    test_features = pipeline.transform(test_df_raw)

    # Keep raw OHLCV for price/info columns, append engineered features
    train_df = pd.concat([train_df_raw, train_features], axis=1)
    del train_df_raw, train_features
    val_df = pd.concat([val_df_raw, val_features], axis=1)
    del val_df_raw, val_features
    test_df = pd.concat([test_df_raw, test_features], axis=1)
    del test_df_raw, test_features

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
        train_df.to_parquet(_cache_entry / "train.parquet")
        val_df.to_parquet(_cache_entry / "val.parquet")
        test_df.to_parquet(_cache_entry / "test.parquet")
        logger.info("save feature cache key=%s path=%s", cache_key[:8], _cache_entry)

    return train_df, val_df, test_df
