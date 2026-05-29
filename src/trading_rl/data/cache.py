"""Cache utilities for prepared trading data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from logger import get_logger
from trading_rl.constants import SplitName
from trading_rl.features.base import NormalizationMethod
from trading_rl.data_loading import MemmapPaths, load_memmap_paths

logger = get_logger(__name__)

# Split-size keys are excluded from the cache signature so that a cache built
# at a larger size (e.g. 50k) can be reused for a smaller request (e.g. 5k)
# without reprocessing — the caller slices to the requested size at load time.
_SPLIT_SIZE_KEYS: frozenset[str] = frozenset({"train_size", "validation_size", "test_size"})


def _sig_without_sizes(sig: dict) -> dict:
    return {k: v for k, v in sig.items() if k not in _SPLIT_SIZE_KEYS}


def _parquet_cache_exists(prepared_dir: Path) -> bool:
    return all(
        (prepared_dir / f"{split}_prepared.parquet").exists()
        for split in SplitName
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
        return {SplitName.TRAIN: None, SplitName.VAL: None, SplitName.TEST: None}

    data_paths = getattr(config.data, "data_paths", None) or []
    n_symbols = len(data_paths) if data_paths else 1
    # In streaming mode (memmap_dir set) the val/test parquet holds one symbol only;
    # non-streaming pooled mode concatenates all symbols so the full multiplier applies.
    val_test_multiplier = 1 if memmap_dir else n_symbols
    train_rows = getattr(config.data, "train_size", None)
    validation_size = getattr(config.data, "validation_size", None)
    test_size = getattr(config.data, "test_size", None)
    return {
        SplitName.TRAIN: train_rows,
        SplitName.VAL: validation_size * val_test_multiplier if validation_size is not None else None,
        SplitName.TEST: test_size * val_test_multiplier if test_size is not None else None,
    }


def _cached_split_rows(prepared_dir: Path) -> dict[str, int]:
    rows = {}
    for split in SplitName:
        rows[split] = len(pd.read_parquet(prepared_dir / f"{split}_prepared.parquet"))
    return rows


def _memmap_cache_compatible(
    config: Any,
    memmap_dir: Path | None,
    logger: Any,
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
    bad_rows = [p.n_rows for p in paths if p.n_rows < expected_train_rows]
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
    logger: Any,
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
            logger.info("prepared cache metadata invalid path={}", metadata_path)
            return False
        cached_sig = metadata.get("config_signature", {})
        if _sig_without_sizes(cached_sig) != _sig_without_sizes(expected_signature):
            logger.info("prepared cache metadata mismatch path={}", metadata_path)
            return False
        expected_rows = _expected_cached_split_rows(config, memmap_dir)
        metadata_rows = metadata.get("split_rows", {})
        for split, expected in expected_rows.items():
            cached = metadata_rows.get(split)
            if expected is not None and (cached is None or cached < expected):
                logger.info(
                    "prepared cache metadata row mismatch split=%s expected=%s actual=%s",
                    split,
                    expected,
                    cached,
                )
                return False
        return True

    expected_rows = _expected_cached_split_rows(config, memmap_dir)
    actual_rows = _cached_split_rows(prepared_dir)
    for split, expected in expected_rows.items():
        if expected is not None and actual_rows[split] < expected:
            logger.info(
                "legacy prepared cache row mismatch split=%s expected=%s actual=%s",
                split,
                expected,
                actual_rows[split],
            )
            return False

    logger.info("legacy prepared cache accepted without metadata dir={}", prepared_dir)
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
            SplitName.TRAIN: len(train_df),
            SplitName.VAL: len(val_df),
            SplitName.TEST: len(test_df),
        },
        "memmap_rows": [p.n_rows for p in memmap_paths or []],
        "memmap_files": [str(p.data_path) for p in memmap_paths or []],
    }
    _prepared_cache_metadata_path(prepared_dir).write_text(
        json.dumps(metadata, indent=2, sort_keys=True)
    )


def _pipeline_uses_global(feature_pipeline: Any) -> bool:
    """Return True if any normalised feature in the pipeline uses GLOBAL normalization."""
    return any(
        fc.normalize and fc.normalization_method == NormalizationMethod.GLOBAL
        for fc in feature_pipeline.feature_configs
    )


def _yaml_uses_global(feature_config_path: str) -> bool:
    """Return True if any normalised feature in the yaml config uses GLOBAL normalization."""
    try:
        with open(feature_config_path) as f:
            cfg = yaml.safe_load(f)
        global_default = cfg.get("normalization_method", NormalizationMethod.RUNNING) == NormalizationMethod.GLOBAL
        for fc in cfg.get("features", []):
            if not fc.get("normalize", False):
                continue
            method = fc.get("normalization_method", NormalizationMethod.GLOBAL if global_default else NormalizationMethod.RUNNING)
            if method == NormalizationMethod.GLOBAL:
                return True
    except Exception as exc:
        logger.debug("could not parse feature config for GLOBAL detection: {}", exc)
    return False


def _feature_cache_key(
    data_path: str,
    feature_config_path: str | None,
    feature_pipeline: Any | None,
    filter_lob_levels: int | None = None,
    train_size: int | None = None,
) -> str:
    """Compute a cache key that changes whenever feature inputs change.

    For session-aware causal normalizers (RUNNING with reset_on_session_break=True),
    split sizes are excluded: the feature value at tick t depends only on prior ticks
    within the same session, not on where the train/val boundary falls. The full
    transformed dataset is cached once and sliced at load time.

    For GLOBAL normalization (StandardScaler fit on training data only), train_size
    is included in the key because the scaler statistics change when the training
    boundary moves — omitting it would cause stale cached features to be returned
    after a train_size change, encoding lookahead bias into val/test rows.

    filter_lob_levels is included because it changes the raw row set before
    feature computation.
    """
    file_mtime = Path(data_path).stat().st_mtime_ns
    uses_global = False

    if feature_pipeline is not None:
        pipeline_repr = [
            {
                "name": fc.name,
                "feature_type": fc.feature_type,
                "params": fc.params,
                "normalize": fc.normalize,
                "normalization_method": fc.normalization_method,
                "rolling_window": fc.rolling_window,
                "reset_on_session_break": fc.reset_on_session_break,
                "session_break_threshold_hours": fc.session_break_threshold_hours,
                "use_time_weights": fc.use_time_weights,
                "output_name": fc.output_name,
                "domain": str(fc.domain),
            }
            for fc in feature_pipeline.feature_configs
        ]
        config_sig = hashlib.md5(
            json.dumps(pipeline_repr, sort_keys=True).encode()
        ).hexdigest()[:12]
        uses_global = _pipeline_uses_global(feature_pipeline)
    elif feature_config_path and Path(feature_config_path).exists():
        config_sig = hashlib.md5(Path(feature_config_path).read_bytes()).hexdigest()[:12]
        uses_global = _yaml_uses_global(feature_config_path)
    else:
        config_sig = "default"

    train_suffix = f"|train{train_size}" if (uses_global and train_size is not None) else ""
    raw = f"{Path(data_path).name}|{file_mtime}|lob{filter_lob_levels}|{config_sig}{train_suffix}"
    return hashlib.md5(raw.encode()).hexdigest()
