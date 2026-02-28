"""Example script for filtering LOB data to remove unchanged/stale snapshots.

This demonstrates how to use the LOB filtering utilities to:
1. Load raw order book data
2. Filter out unchanged LOB snapshots
3. Validate LOB data quality
4. Analyze LOB change statistics
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import pandas as pd
import logging
from trading_rl.data.lob_filters import (
    filter_unchanged_lob,
    filter_valid_lob,
    filter_active_lob,
    get_lob_change_stats,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main example demonstrating LOB filtering."""

    # Path to your LOB data
    data_path = "./data/raw/stocks/AAPL_2026-02-25_2026-02-27_raw_mbp-10_us_hours.parquet"

    logger.info(f"Loading LOB data from: {data_path}")

    try:
        df = pd.read_parquet(data_path)
    except FileNotFoundError:
        logger.error(f"Data file not found: {data_path}")
        logger.info("Using small sample file instead...")
        data_path = "./data/raw/stocks/AAPL_2026-02-25_2026-02-27_raw_mbp-10_small.parquet"
        try:
            df = pd.read_parquet(data_path)
        except FileNotFoundError:
            logger.error(f"Small sample file also not found: {data_path}")
            logger.info("Please update the data_path in this script")
            return

    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    logger.info(f"Columns: {list(df.columns[:10])}...")

    # ==========================================================================
    # Example 1: Get LOB change statistics
    # ==========================================================================
    logger.info("\n" + "="*80)
    logger.info("Example 1: Analyzing LOB change patterns")
    logger.info("="*80)

    stats = get_lob_change_stats(df, levels=5)

    logger.info("\nTop 10 most frequently changing columns:")
    print(stats.sort_values('pct_changes', ascending=False).head(10))

    logger.info("\nLeast frequently changing columns:")
    print(stats.sort_values('pct_changes', ascending=True).head(10))

    # ==========================================================================
    # Example 2: Filter unchanged LOB data
    # ==========================================================================
    logger.info("\n" + "="*80)
    logger.info("Example 2: Filtering unchanged LOB snapshots")
    logger.info("="*80)

    df_changed = filter_unchanged_lob(df, levels=5)

    logger.info(f"\nOriginal data: {len(df):,} rows")
    logger.info(f"After filtering: {len(df_changed):,} rows")
    logger.info(f"Reduction: {100 * (1 - len(df_changed)/len(df)):.1f}%")

    # ==========================================================================
    # Example 3: Validate LOB data
    # ==========================================================================
    logger.info("\n" + "="*80)
    logger.info("Example 3: Validating LOB data quality")
    logger.info("="*80)

    df_valid = filter_valid_lob(
        df,
        levels=5,
        min_spread_bps=0.5,
        max_spread_bps=50.0,
        min_size=0.0
    )

    logger.info(f"\nOriginal data: {len(df):,} rows")
    logger.info(f"After validation: {len(df_valid):,} rows")
    logger.info(f"Invalid rows: {len(df) - len(df_valid):,}")

    # ==========================================================================
    # Example 4: Combined filtering (recommended)
    # ==========================================================================
    logger.info("\n" + "="*80)
    logger.info("Example 4: Combined active LOB filtering (recommended)")
    logger.info("="*80)

    df_active = filter_active_lob(
        df,
        levels=5,
        remove_unchanged=True,
        validate=True,
        min_spread_bps=0.5,
        max_spread_bps=50.0,
    )

    logger.info(f"\nOriginal data: {len(df):,} rows")
    logger.info(f"After combined filtering: {len(df_active):,} rows")
    logger.info(f"Data retention: {100 * len(df_active)/len(df):.1f}%")

    # ==========================================================================
    # Example 5: Save filtered data
    # ==========================================================================
    logger.info("\n" + "="*80)
    logger.info("Example 5: Saving filtered data")
    logger.info("="*80)

    output_path = data_path.replace('.parquet', '_active_l5.parquet')
    logger.info(f"Saving filtered data to: {output_path}")

    df_active.to_parquet(output_path)

    logger.info(f"Saved {len(df_active):,} rows")
    logger.info(f"Original size: {Path(data_path).stat().st_size / 1024**2:.1f} MB")
    logger.info(f"Filtered size: {Path(output_path).stat().st_size / 1024**2:.1f} MB")

    # ==========================================================================
    # Summary statistics
    # ==========================================================================
    logger.info("\n" + "="*80)
    logger.info("Summary")
    logger.info("="*80)

    logger.info(f"\nOriginal dataset: {len(df):,} rows")
    logger.info(f"Unchanged filter: {len(df_changed):,} rows ({100*len(df_changed)/len(df):.1f}%)")
    logger.info(f"Validation filter: {len(df_valid):,} rows ({100*len(df_valid)/len(df):.1f}%)")
    logger.info(f"Combined filter: {len(df_active):,} rows ({100*len(df_active)/len(df):.1f}%)")

    logger.info("\nRecommendation: Use 'filter_active_lob()' for most use cases")
    logger.info("This removes both invalid and unchanged data in one step")


if __name__ == "__main__":
    main()
