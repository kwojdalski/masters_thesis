"""Data loading utilities for trading RL."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from gym_trading_env.downloader import download

from logger import get_logger
from trading_rl.data_loading import MemmapPaths

logger = get_logger(__name__)


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
