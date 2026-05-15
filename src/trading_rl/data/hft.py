"""HFT-specific data utilities for trading RL."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from logger import get_logger
from trading_rl.constants import EnvMode, SplitName

logger = get_logger(__name__)

_HFT_MIN_TIMESTAMP_GAP_NS = 1_000  # 1 µs — minimum distinguishable by Python datetime (microsecond precision)

_Splits = tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]


def _map_splits(
    fn,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> _Splits:
    """Apply fn(split_name, df) to each split and return (train, val, test)."""
    results = {name: fn(name, df) for name, df in (
        (SplitName.TRAIN, train_df), (SplitName.VAL, val_df), (SplitName.TEST, test_df)
    )}
    return results[SplitName.TRAIN], results[SplitName.VAL], results[SplitName.TEST]


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
    if mode != EnvMode.HFT:
        return train_df, val_df, test_df
    return (
        _derive_close_hft_single(train_df, SplitName.TRAIN, logger),
        _derive_close_hft_single(val_df, SplitName.VAL, logger),
        _derive_close_hft_single(test_df, SplitName.TEST, logger),
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
    if mode != EnvMode.HFT or backend != "tradingenv":
        return train_df, val_df, test_df
    return (
        _deduplicate_hft_index_single(train_df, SplitName.TRAIN, logger),
        _deduplicate_hft_index_single(val_df, SplitName.VAL, logger),
        _deduplicate_hft_index_single(test_df, SplitName.TEST, logger),
    )
