"""LOB (Limit Order Book) data filtering utilities.

Functions for filtering order book data to remove:
- Stale/unchanged data
- Periods of no activity
- Invalid or corrupted LOB snapshots
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


def filter_unchanged_lob(
    df: pd.DataFrame,
    levels: int = 5,
    bid_px_prefix: str = "bid_px_",
    ask_px_prefix: str = "ask_px_",
    bid_sz_prefix: str = "bid_sz_",
    ask_sz_prefix: str = "ask_sz_",
    keep_first: bool = True,
) -> pd.DataFrame:
    """Filter out rows where LOB hasn't changed from previous tick.

    Removes rows where all bid/ask prices and sizes for the specified levels
    are identical to the previous row. This is useful for:
    - Removing stale data during low-activity periods
    - Reducing dataset size without losing information
    - Focusing on active trading periods

    Args:
        df: DataFrame with LOB data
        levels: Number of LOB levels to check (default: 5 for L0-L4)
        bid_px_prefix: Column prefix for bid prices (default: "bid_px_")
        ask_px_prefix: Column prefix for ask prices (default: "ask_px_")
        bid_sz_prefix: Column prefix for bid sizes (default: "bid_sz_")
        ask_sz_prefix: Column prefix for ask sizes (default: "ask_sz_")
        keep_first: Whether to keep the first row (default: True)

    Returns:
        Filtered DataFrame with only rows where LOB changed

    Example:
        >>> df_active = filter_unchanged_lob(df, levels=5)
        >>> print(f"Kept {len(df_active)}/{len(df)} rows ({100*len(df_active)/len(df):.1f}%)")
    """
    if len(df) == 0:
        return df.copy()

    # Build list of columns to check
    columns_to_check = []
    for i in range(levels):
        columns_to_check.extend([
            f"{bid_px_prefix}{i:02d}",
            f"{ask_px_prefix}{i:02d}",
            f"{bid_sz_prefix}{i:02d}",
            f"{ask_sz_prefix}{i:02d}",
        ])

    # Verify all columns exist
    missing_cols = [col for col in columns_to_check if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required LOB columns: {missing_cols}. "
            f"Available columns: {list(df.columns)}"
        )

    # Check which rows have changes
    # Compare each row to the previous row across all LOB columns
    subset = df[columns_to_check]

    # Row is different if ANY column changed from previous row
    changed = (subset != subset.shift(1)).any(axis=1)

    if keep_first:
        # Always keep the first row (no previous row to compare to)
        changed.iloc[0] = True

    # Filter to only changed rows
    df_filtered = df[changed].copy()

    n_original = len(df)
    n_filtered = len(df_filtered)
    n_removed = n_original - n_filtered
    pct_kept = 100 * n_filtered / n_original if n_original > 0 else 0

    logger.info(
        f"LOB change filter: kept {n_filtered:,}/{n_original:,} rows "
        f"({pct_kept:.1f}%), removed {n_removed:,} unchanged rows"
    )

    return df_filtered


def filter_valid_lob(
    df: pd.DataFrame,
    levels: int = 5,
    bid_px_prefix: str = "bid_px_",
    ask_px_prefix: str = "ask_px_",
    bid_sz_prefix: str = "bid_sz_",
    ask_sz_prefix: str = "ask_sz_",
    min_spread_bps: float = 0.1,
    max_spread_bps: float = 100.0,
    min_size: float = 0.0,
) -> pd.DataFrame:
    """Filter out invalid or corrupted LOB snapshots.

    Removes rows with:
    - Crossed book (bid >= ask)
    - Spread too tight or too wide
    - Zero or negative sizes
    - Missing/NaN values

    Args:
        df: DataFrame with LOB data
        levels: Number of LOB levels to validate
        bid_px_prefix: Column prefix for bid prices
        ask_px_prefix: Column prefix for ask prices
        bid_sz_prefix: Column prefix for bid sizes
        ask_sz_prefix: Column prefix for ask sizes
        min_spread_bps: Minimum allowed spread in basis points (default: 0.1)
        max_spread_bps: Maximum allowed spread in basis points (default: 100.0)
        min_size: Minimum allowed size (default: 0.0)

    Returns:
        Filtered DataFrame with only valid LOB snapshots

    Example:
        >>> df_valid = filter_valid_lob(df, levels=5, min_spread_bps=0.5)
    """
    if len(df) == 0:
        return df.copy()

    n_original = len(df)
    valid_mask = pd.Series(True, index=df.index)

    # Check each level
    for i in range(levels):
        bid_px_col = f"{bid_px_prefix}{i:02d}"
        ask_px_col = f"{ask_px_prefix}{i:02d}"
        bid_sz_col = f"{bid_sz_prefix}{i:02d}"
        ask_sz_col = f"{ask_sz_prefix}{i:02d}"

        # Check columns exist
        for col in [bid_px_col, ask_px_col, bid_sz_col, ask_sz_col]:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")

        bid_px = df[bid_px_col]
        ask_px = df[ask_px_col]
        bid_sz = df[bid_sz_col]
        ask_sz = df[ask_sz_col]

        # Filter criteria
        # 1. No NaN values
        valid_mask &= ~bid_px.isna()
        valid_mask &= ~ask_px.isna()
        valid_mask &= ~bid_sz.isna()
        valid_mask &= ~ask_sz.isna()

        # 2. Positive sizes
        valid_mask &= (bid_sz > min_size)
        valid_mask &= (ask_sz > min_size)

        # 3. Valid spread (for level 0 only)
        if i == 0:
            # Not crossed
            valid_mask &= (bid_px < ask_px)

            # Spread in reasonable range
            mid_price = (bid_px + ask_px) / 2
            spread_bps = ((ask_px - bid_px) / mid_price) * 10000

            valid_mask &= (spread_bps >= min_spread_bps)
            valid_mask &= (spread_bps <= max_spread_bps)

        # 4. Prices are ordered correctly within side
        if i > 0:
            prev_bid_col = f"{bid_px_prefix}{i-1:02d}"
            prev_ask_col = f"{ask_px_prefix}{i-1:02d}"

            # Bids should decrease as levels go deeper
            valid_mask &= (bid_px <= df[prev_bid_col])

            # Asks should increase as levels go deeper
            valid_mask &= (ask_px >= df[prev_ask_col])

    df_filtered = df[valid_mask].copy()

    n_filtered = len(df_filtered)
    n_removed = n_original - n_filtered
    pct_kept = 100 * n_filtered / n_original if n_original > 0 else 0

    logger.info(
        f"LOB validity filter: kept {n_filtered:,}/{n_original:,} rows "
        f"({pct_kept:.1f}%), removed {n_removed:,} invalid rows"
    )

    return df_filtered


def filter_active_lob(
    df: pd.DataFrame,
    levels: int = 5,
    remove_unchanged: bool = True,
    validate: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """Combined filter for active, valid LOB data.

    Applies both unchanged and validity filters in sequence.

    Args:
        df: DataFrame with LOB data
        levels: Number of LOB levels to check
        remove_unchanged: Whether to remove unchanged rows (default: True)
        validate: Whether to validate LOB snapshots (default: True)
        **kwargs: Additional arguments passed to filter functions

    Returns:
        Filtered DataFrame with active, valid LOB data

    Example:
        >>> df_active = filter_active_lob(
        ...     df,
        ...     levels=5,
        ...     min_spread_bps=0.5,
        ...     max_spread_bps=50.0
        ... )
    """
    n_original = len(df)

    # First validate (removes invalid data)
    if validate:
        df = filter_valid_lob(df, levels=levels, **kwargs)

    # Then remove unchanged (removes stale data)
    if remove_unchanged:
        df = filter_unchanged_lob(df, levels=levels, **kwargs)

    n_filtered = len(df)
    pct_kept = 100 * n_filtered / n_original if n_original > 0 else 0

    logger.info(
        f"Combined LOB filter: kept {n_filtered:,}/{n_original:,} rows "
        f"({pct_kept:.1f}%)"
    )

    return df


def get_lob_change_stats(
    df: pd.DataFrame,
    levels: int = 5,
    bid_px_prefix: str = "bid_px_",
    ask_px_prefix: str = "ask_px_",
    bid_sz_prefix: str = "bid_sz_",
    ask_sz_prefix: str = "ask_sz_",
) -> pd.DataFrame:
    """Calculate statistics about LOB changes.

    Returns a DataFrame with per-column change statistics:
    - n_changes: Number of times the column changed
    - pct_changes: Percentage of rows where column changed
    - mean_abs_change: Mean absolute change when changed
    - std_abs_change: Std dev of absolute changes

    Args:
        df: DataFrame with LOB data
        levels: Number of LOB levels to analyze
        bid_px_prefix: Column prefix for bid prices
        ask_px_prefix: Column prefix for ask prices
        bid_sz_prefix: Column prefix for bid sizes
        ask_sz_prefix: Column prefix for ask sizes

    Returns:
        DataFrame with change statistics per column

    Example:
        >>> stats = get_lob_change_stats(df, levels=5)
        >>> print(stats.sort_values('pct_changes', ascending=False))
    """
    if len(df) == 0:
        return pd.DataFrame()

    columns_to_check = []
    for i in range(levels):
        columns_to_check.extend([
            f"{bid_px_prefix}{i:02d}",
            f"{ask_px_prefix}{i:02d}",
            f"{bid_sz_prefix}{i:02d}",
            f"{ask_sz_prefix}{i:02d}",
        ])

    stats = []
    for col in columns_to_check:
        if col not in df.columns:
            continue

        values = df[col]
        changes = values != values.shift(1)

        # Exclude first row (no previous to compare)
        changes.iloc[0] = False

        abs_changes = (values - values.shift(1)).abs()

        stats.append({
            'column': col,
            'n_changes': changes.sum(),
            'pct_changes': 100 * changes.sum() / (len(df) - 1),
            'mean_abs_change': abs_changes[changes].mean() if changes.any() else 0,
            'std_abs_change': abs_changes[changes].std() if changes.any() else 0,
        })

    return pd.DataFrame(stats)
