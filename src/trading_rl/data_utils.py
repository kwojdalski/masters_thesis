"""Data loading and preprocessing utilities for trading RL."""

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from gym_trading_env.downloader import download
from joblib import Memory

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

_HFT_MIN_TIMESTAMP_GAP_NS = 1_000_000_000

# Setup joblib memory for caching expensive operations
memory = Memory(location=".cache/joblib", verbose=1)


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


def clear_data_cache():
    """Clear data processing cache."""
    memory.clear(warn=True)


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


@memory.cache
def load_trading_data(data_path: str, cache_bust: float | None = None) -> pd.DataFrame:
    """Load trading data from pickle file.

    Args:
        data_path: Path to pickle file
        cache_bust: Optional timestamp or hash to bust joblib cache

    Returns:
        DataFrame with OHLCV data
    """
    data_file = Path(data_path)
    logger.info("load data path=%s", data_file)
    # cache_bust ensures cache invalidation when the file changes
    _ = cache_bust
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



def reward_function(history: dict) -> float:
    """Calculate reward based on portfolio valuation changes.

    Computes log returns of portfolio valuation.

    Args:
        history: Dictionary containing trading history

    Returns:
        Log return reward
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log(
            history["portfolio_valuation", -1] / history["portfolio_valuation", -2]
        )
    return float(np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0))


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

    required_cols = {"ask_px_00", "bid_px_00"}
    dataframes = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }
    updated: dict[str, pd.DataFrame] = {}

    for split_name, df in dataframes.items():
        if "close" in df.columns:
            updated[split_name] = df
            continue

        missing = sorted(required_cols - set(df.columns))
        if missing:
            raise ValueError(
                "HFT mode requires a raw 'close' column or top-of-book columns "
                f"ask_px_00/bid_px_00 to derive it. Missing columns in {split_name}: {missing}"
            )

        derived_df = df.copy()
        mid_price = (derived_df["ask_px_00"] + derived_df["bid_px_00"]) / 2.0
        if "price" in derived_df.columns:
            mid_price = mid_price.fillna(derived_df["price"])

        mid_price = mid_price.ffill().bfill()
        derived_df["close"] = mid_price
        updated[split_name] = derived_df

        nan_ratio = float(derived_df["close"].isna().mean())
        logger.info(
            "derive close split=%s method=mid_price%s nan_ratio=%.6f",
            split_name,
            "+fallback" if "price" in derived_df.columns else "",
            nan_ratio,
        )

    return updated["train"], updated["val"], updated["test"]


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

    dataframes = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }
    updated: dict[str, pd.DataFrame] = {}
    min_gap_ns = _HFT_MIN_TIMESTAMP_GAP_NS

    for split_name, df in dataframes.items():
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(
                "HFT TradingEnv requires DatetimeIndex to enforce unique event ordering, "
                f"but {split_name} split has index type {type(df.index).__name__}."
            )

        index = df.index
        index_ns_raw = index.view("i8")
        old_min_gap_ns = (
            int(np.diff(index_ns_raw).min()) if len(index_ns_raw) > 1 else min_gap_ns
        )
        requires_adjustment = (
            not index.is_unique
            or not index.is_monotonic_increasing
            or old_min_gap_ns < min_gap_ns
            or index.tz is not None
        )

        if not requires_adjustment:
            updated[split_name] = df
            continue

        adjusted_df = df.sort_index(kind="stable").copy()
        index = adjusted_df.index
        index_ns = index.view("i8")
        positions = np.arange(len(index_ns), dtype=np.int64) * min_gap_ns
        adjusted_ns = np.maximum.accumulate(index_ns - positions) + positions
        adjusted_index = pd.to_datetime(adjusted_ns, utc=True).tz_localize(None)

        adjusted_df.index = adjusted_index
        if not adjusted_df.index.is_unique:
            raise ValueError(
                f"Failed to enforce unique index for {split_name} HFT split."
            )

        duplicate_count = int(index.duplicated().sum())
        max_shift_ns = int((adjusted_ns - index_ns).max()) if len(index_ns) else 0
        new_min_gap_ns = (
            int(np.diff(adjusted_ns).min()) if len(adjusted_ns) > 1 else min_gap_ns
        )
        logger.info(
            "adjust hft index split=%s duplicates=%d min_gap_ns=%d->%d max_shift_ns=%d",
            split_name,
            duplicate_count,
            old_min_gap_ns,
            new_min_gap_ns,
            max_shift_ns,
        )

        updated[split_name] = adjusted_df

    return updated["train"], updated["val"], updated["test"]


# ---------------------------------------------------------------------------
# Dataset-building helpers
# ---------------------------------------------------------------------------

def _parquet_cache_exists(prepared_dir: Path) -> bool:
    return all(
        (prepared_dir / f"{split}_prepared.parquet").exists()
        for split in ["train", "val", "test"]
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


def _call_prepare_data(
    data_path: str,
    config: Any,
    feature_pipeline: Any = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Call prepare_data with all standard config fields."""
    return prepare_data(
        data_path=data_path,
        train_size=config.data.train_size,
        validation_size=getattr(config.data, "validation_size", None),
        download_if_missing=config.data.download_data,
        exchange_names=getattr(config.data, "exchange_names", None),
        symbols=getattr(config.data, "symbols", None),
        timeframe=getattr(config.data, "timeframe", "1h"),
        data_dir=getattr(config.data, "data_dir", "data"),
        since=getattr(config.data, "download_since", None),
        feature_config_path=getattr(config.data, "feature_config", None),
        feature_pipeline=feature_pipeline,
        feature_cache_dir=getattr(config.data, "feature_cache_dir", "data/.feature_cache"),
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
    train_df, val_df, test_df = _call_prepare_data(config.data.data_path, config, pipeline)
    return _finalize_splits(train_df, val_df, test_df, config, logger)


def _build_pooled_splits(
    config: Any,
    logger: logging.Logger,
    data_paths: list[str],
    memmap_dir: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[MemmapPaths] | None]:
    """Process each symbol independently then concatenate from disk.

    Each symbol is feature-engineered, written to a temp parquet, and freed
    before the next symbol loads.  This keeps peak memory to ~1 symbol at a
    time during feature engineering rather than accumulating all symbols.
    """
    import gc
    import shutil
    import tempfile

    pipeline = _resolve_feature_pipeline(config, logger)

    logger.info("pooled training n_symbols=%d", len(data_paths))
    tmp_dir = Path(tempfile.mkdtemp(prefix="pooled_splits_"))
    tmp_paths: list[dict[str, Path]] = []
    collected_memmap_paths: list[MemmapPaths] = []

    for i, data_path in enumerate(data_paths):
        logger.info("process symbol idx=%d/%d path=%s", i + 1, len(data_paths), data_path)
        train_i, val_i, test_i = _call_prepare_data(data_path, config, pipeline)
        # Apply close-column derivation per-symbol so each memmap is self-contained.
        train_i, val_i, test_i = ensure_close_column_for_hft(train_i, val_i, test_i, config, logger)

        if memmap_dir:
            collected_memmap_paths.append(save_symbol_memmap(train_i, memmap_dir, prefix=str(i)))

        sym: dict[str, Path] = {}
        for split, df in [("train", train_i), ("val", val_i), ("test", test_i)]:
            p = tmp_dir / f"{i}_{split}.parquet"
            df.to_parquet(p)
            sym[split] = p
        tmp_paths.append(sym)
        del train_i, val_i, test_i
        gc.collect()

    train_df = pd.concat([pd.read_parquet(p["train"]) for p in tmp_paths], ignore_index=True)
    val_df   = pd.concat([pd.read_parquet(p["val"])   for p in tmp_paths], ignore_index=True)
    test_df  = pd.concat([pd.read_parquet(p["test"])  for p in tmp_paths], ignore_index=True)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    logger.info("pooled splits train=%d val=%d test=%d", len(train_df), len(val_df), len(test_df))

    # ensure_close was applied per-symbol above; only run unique-index + validate on concat.
    train_df, val_df, test_df = ensure_unique_index_for_hft_tradingenv(train_df, val_df, test_df, config, logger)
    validate_prepared_data(train_df, val_df, test_df, config)

    return train_df, val_df, test_df, collected_memmap_paths or None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_prepared_dataset(config: Any, logger: logging.Logger) -> "PreparedDataset":
    """Build a prepared dataset bundle for RL training and evaluation."""
    lazy_load = getattr(config.data, "lazy_load", False)
    prepared_dir = Path(d) if (d := getattr(config.data, "prepared_data_dir", None)) else None
    memmap_dir   = Path(d) if (d := getattr(config.data, "memmap_dir", None)) else None

    # Fast path: skip all feature engineering when the parquet cache exists.
    if lazy_load and prepared_dir and _parquet_cache_exists(prepared_dir):
        logger.info("load prepared splits cache_dir=%s", prepared_dir)
        lazy_splits = load_prepared_splits(prepared_dir)
        memmap_paths = load_memmap_paths(memmap_dir) if memmap_dir and memmap_dir.exists() else None
        return _make_dataset(
            lazy_splits["train"], lazy_splits["val"], lazy_splits["test"],
            config, memmap_paths or None,
        )

    data_paths = getattr(config.data, "data_paths", None)
    if data_paths:
        train_df, val_df, test_df, memmap_paths = _build_pooled_splits(
            config, logger, data_paths, memmap_dir
        )
    else:
        train_df, val_df, test_df = _build_single_symbol_splits(config, logger)
        memmap_paths = [save_symbol_memmap(train_df, memmap_dir, "0")] if memmap_dir else None

    if lazy_load and prepared_dir:
        logger.info("save prepared splits cache_dir=%s", prepared_dir)
        prepared_dir.mkdir(parents=True, exist_ok=True)
        save_prepared_splits(train_df, val_df, test_df, prepared_dir)

    return _make_dataset(train_df, val_df, test_df, config, memmap_paths)


def _feature_cache_key(
    data_path: str,
    train_size: int,
    validation_size: int | None,
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

    raw = f"{Path(data_path).name}|{file_mtime}|{train_size}|{validation_size}|{config_sig}"
    return hashlib.md5(raw.encode()).hexdigest()


def prepare_data(
    data_path: str,
    train_size: int,
    validation_size: int | None = None,
    download_if_missing: bool = False,
    exchange_names: list[str] | None = None,
    symbols: list[str] | None = None,
    timeframe: str = "1h",
    data_dir: str = "data",
    since: Any | None = None,
    feature_config_path: str | None = None,
    feature_pipeline: Any | None = None,
    feature_cache_dir: str | None = "data/.feature_cache",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Prepare trading data for RL training with proper train/val/test split.

    CRITICAL: This function splits data BEFORE feature engineering to prevent
    data leakage. Normalization statistics are computed only on training data.

    Args:
        data_path: Path to data file
        train_size: Number of samples to use for training
        validation_size: Number of samples for validation. If None, split the
            remaining data equally between validation and test.
        download_if_missing: Whether to download data if file doesn't exist
        exchange_names: Exchange names for download
        symbols: Trading symbols for download
        timeframe: Data timeframe
        data_dir: Directory for downloaded data
        since: Start date for download
        feature_config_path: Path to YAML config for features. If None, uses default pipeline.
        feature_pipeline: Pre-built FeaturePipeline instance. If provided, takes priority
            over feature_config_path.

    Returns:
        Tuple of (train_df, val_df, test_df) with raw OHLCV plus engineered features.

    Example:
        train_df, val_df, test_df = prepare_data(
            data_path="data.parquet",
            train_size=450,
            feature_config_path="configs/features/sine_wave.yaml"
        )
    """
    # Feature cache — skip the expensive transform when inputs haven't changed.
    _cache_entry: Path | None = None
    if feature_cache_dir and Path(data_path).exists():
        cache_key = _feature_cache_key(
            data_path, train_size, validation_size, feature_config_path, feature_pipeline
        )
        _cache_entry = Path(feature_cache_dir) / cache_key
        if (_cache_entry / "train.parquet").exists():
            logger.info("feature cache hit key=%s path=%s", cache_key[:8], _cache_entry)
            return (
                pd.read_parquet(_cache_entry / "train.parquet"),
                pd.read_parquet(_cache_entry / "val.parquet"),
                pd.read_parquet(_cache_entry / "test.parquet"),
            )
        logger.info("feature cache miss key=%s", cache_key[:8])

    # Check if data exists
    if not Path(data_path).exists():
        if download_if_missing and exchange_names and symbols and since:
            download_trading_data(exchange_names, symbols, timeframe, data_dir, since)
        else:
            raise FileNotFoundError(
                f"Data file not found: {data_path}. "
                "Set download_if_missing=True to download."
            )

    # Load raw OHLCV data
    file_signature = Path(data_path).stat().st_mtime_ns
    df = load_trading_data(data_path, cache_bust=file_signature)
    df = df.dropna()

    logger.info("load raw data n_rows=%d n_cols=%d", len(df), len(df.columns))

    # Split data BEFORE feature engineering (critical for preventing leakage!)
    if train_size >= len(df):
        raise ValueError(
            f"train_size ({train_size}) must be smaller than dataset length ({len(df)})"
        )
    remaining = len(df) - train_size
    if validation_size is None:
        validation_size = remaining // 2
    if validation_size < 0:
        raise ValueError(f"validation_size must be >= 0, got {validation_size}")
    if validation_size >= remaining:
        raise ValueError(
            f"validation_size ({validation_size}) must be smaller than remaining "
            f"data after train split ({remaining})"
        )

    val_end = train_size + validation_size
    train_df_raw = df[:train_size].copy()
    val_df_raw = df[train_size:val_end].copy()
    test_df_raw = df[val_end:].copy()

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
    elif feature_config_path:
        logger.info("load feature pipeline path=%s", feature_config_path)
        pipeline = FeaturePipeline.from_yaml(feature_config_path)
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
    val_df = pd.concat([val_df_raw, val_features], axis=1)
    test_df = pd.concat([test_df_raw, test_features], axis=1)

    logger.info(
        "feature engineering complete train_shape=%s val_shape=%s test_shape=%s",
        train_df.shape,
        val_df.shape,
        test_df.shape,
    )
    logger.info("feature columns cols=%s", list(train_features.columns))

    if _cache_entry is not None:
        _cache_entry.mkdir(parents=True, exist_ok=True)
        train_df.to_parquet(_cache_entry / "train.parquet")
        val_df.to_parquet(_cache_entry / "val.parquet")
        test_df.to_parquet(_cache_entry / "test.parquet")
        logger.info("save feature cache key=%s path=%s", cache_key[:8], _cache_entry)

    return train_df, val_df, test_df
