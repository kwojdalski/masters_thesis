"""Data loading and preprocessing utilities for trading RL."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from gym_trading_env.downloader import download
from joblib import Memory

from logger import get_logger

logger = get_logger(__name__)

# Setup joblib memory for caching expensive operations
memory = Memory(location=".cache/joblib", verbose=1)


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

    logger.info(f"Downloading data for {symbols} from {exchange_names}")
    download(
        exchange_names=exchange_names,
        symbols=symbols,
        timeframe=timeframe,
        dir=data_dir,
        since=since,
    )
    logger.info("Data download complete")


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
    logger.info(f"Loading data from {data_file}")
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
    logger.info(f"Loaded {len(df)} rows of data")
    return df


@memory.cache
def create_features(df: pd.DataFrame, data_path: str = "") -> pd.DataFrame:
    """Create technical features from OHLCV data.

    Features are normalized using z-score normalization.
    All features must have 'feature' in their name for gym_trading_env.

    Args:
        df: DataFrame with OHLCV columns (open, high, low, close, volume)
        data_path: Path to data file (used to detect upward drift data)

    Returns:
        DataFrame with added feature columns
    """
    logger.info("Creating features")
    df = df.copy()

    # Check if this is upward drift data
    is_upward_drift = "upward_drift" in data_path.lower()

    if is_upward_drift:
        logger.info(
            "Detected upward drift data - skipping feature_high and feature_low"
        )

    # Price return feature (log return)
    df["feature_return"] = (df["close"] / df["close"].shift(1) - 1).fillna(0)
    df["feature_return"] = (df["feature_return"] - df["feature_return"].mean()) / df[
        "feature_return"
    ].std()

    # Percentage change feature
    df["feature_pct_chng"] = df["close"].pct_change().fillna(0)
    df["feature_pct_chng"] = (
        df["feature_pct_chng"] - df["feature_pct_chng"].mean()
    ) / df["feature_pct_chng"].std()

    # High relative to close (skip for upward drift)
    if not is_upward_drift:
        df["feature_high"] = (df["high"] / df["close"] - 1).fillna(0)
        df["feature_high"] = (df["feature_high"] - df["feature_high"].mean()) / df[
            "feature_high"
        ].std()

    # Low relative to close (skip for upward drift)
    if not is_upward_drift:
        df["feature_low"] = (df["low"] / df["close"] - 1).fillna(0)
        df["feature_low"] = (df["feature_low"] - df["feature_low"].mean()) / df[
            "feature_low"
        ].std()

    # Volume feature (log-normalized to handle skewness)
    # Log transform handles the heavy right tail of volume distribution
    df["feature_volume"] = np.log1p(df["volume"])  # log1p = log(1 + x) handles zero volumes
    df["feature_volume"] = (df["feature_volume"] - df["feature_volume"].mean()) / df[
        "feature_volume"
    ].std()

    # Trend feature: price relative to initial price (NOT z-scored to preserve trend)
    # This feature captures the cumulative price movement
    # For upward drift: will increase from 1.0 to ~2.1 (110% gain)
    # Scaled to [0, 1] range for stability
    df["feature_trend"] = df["close"] / df["close"].iloc[0]
    # Min-max normalization to [0, 1] instead of z-score
    feature_trend_min = df["feature_trend"].min()
    feature_trend_max = df["feature_trend"].max()
    df["feature_trend"] = (df["feature_trend"] - feature_trend_min) / (
        feature_trend_max - feature_trend_min + 1e-8
    )

    # Drop any rows with NaN values
    df = df.dropna()

    logger.info(f"Created {len([c for c in df.columns if 'feature' in c])} features")
    logger.info(f"Dataset has {len(df)} rows after dropping NaNs")

    return df


def reward_function(history: dict) -> float:
    """Calculate reward based on portfolio valuation changes.

    Computes log returns of portfolio valuation.

    Args:
        history: Dictionary containing trading history

    Returns:
        Log return reward
    """
    returns = np.log(
        history["portfolio_valuation", -1] / history["portfolio_valuation", -2]
    )
    return returns


def prepare_data(
    data_path: str,
    download_if_missing: bool = False,
    exchange_names: list[str] | None = None,
    symbols: list[str] | None = None,
    timeframe: str = "1h",
    data_dir: str = "data",
    since: Any | None = None,
    no_features: bool = False,
) -> pd.DataFrame:
    """Prepare trading data for RL training.

    Args:
        data_path: Path to data file
        download_if_missing: Whether to download data if file doesn't exist
        exchange_names: Exchange names for download
        symbols: Trading symbols for download
        timeframe: Data timeframe
        data_dir: Directory for downloaded data
        since: Start date for download
        no_features: If True, skip feature engineering and return only OHLCV data

    Returns:
        DataFrame with features (or just OHLCV if no_features=True)
    """
    # Check if data exists
    if not Path(data_path).exists():
        if download_if_missing and exchange_names and symbols and since:
            download_trading_data(exchange_names, symbols, timeframe, data_dir, since)
        else:
            raise FileNotFoundError(
                f"Data file not found: {data_path}. "
                "Set download_if_missing=True to download."
            )

    # Load and process data
    file_signature = Path(data_path).stat().st_mtime_ns
    df = load_trading_data(data_path, cache_bust=file_signature)

    if no_features:
        # Skip feature creation and return raw OHLCV data
        df = df.dropna()
        logger.info(
            f"Loaded data without features: {len(df)} rows, {len(df.columns)} columns"
        )
        logger.info(f"Columns: {list(df.columns)}")
    else:
        # Create features as usual
        df = create_features(df, data_path=data_path)

    return df
