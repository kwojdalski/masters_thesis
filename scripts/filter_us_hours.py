#!/usr/bin/env python
"""
Filter stock data to US market trading hours (9:30 AM - 4:00 PM EST).

Usage:
    python scripts/filter_us_hours.py <input_file> [output_file]

If output_file is not provided, appends '_us_hours' to the input filename.

Examples:
    python scripts/filter_us_hours.py data/raw/stocks/AAPL_2024-01-01_2024-12-31_raw_mbp-10.parquet
    python scripts/filter_us_hours.py input.parquet output.parquet
"""

import sys
from pathlib import Path

import pandas as pd


def filter_us_trading_hours(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter DataFrame to US market trading hours (9:30 AM - 4:00 PM EST).

    Assumes the DataFrame index is a DatetimeIndex in UTC timezone.
    US market hours:
    - Opens: 9:30 AM EST = 14:30 UTC
    - Closes: 4:00 PM EST = 21:00 UTC

    Args:
        df: DataFrame with DatetimeIndex in UTC

    Returns:
        Filtered DataFrame with only trading hours
    """
    # Extract hour and minute from UTC timestamps
    hours = df.index.hour
    minutes = df.index.minute

    # Keep data from 14:30:00 to 20:59:59 (market closes at 21:00:00)
    mask = (((hours == 14) & (minutes >= 30)) |  # 14:30-14:59
            ((hours >= 15) & (hours <= 20)))  # 15:00-20:59

    return df[mask]


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    input_file = Path(sys.argv[1])

    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)

    # Determine output file
    if len(sys.argv) >= 3:
        output_file = Path(sys.argv[2])
    else:
        # Append '_us_hours' before the extension
        output_file = input_file.parent / f"{input_file.stem}_us_hours{input_file.suffix}"

    print(f"Loading data from: {input_file}")
    df = pd.read_parquet(input_file)

    original_size = len(df)
    print(f"Original shape: {df.shape}")

    # Check timezone
    if df.index.tz is None:
        print("Warning: Index has no timezone, assuming UTC")
    elif str(df.index.tz) != "UTC":
        print(f"Warning: Index timezone is {df.index.tz}, expected UTC")

    # Filter
    df_filtered = filter_us_trading_hours(df)

    print(f"Filtered shape: {df_filtered.shape}")
    print(f"Reduction: {100 * (1 - len(df_filtered) / original_size):.1f}%")

    if len(df_filtered) > 0:
        print(f"\nTime range:")
        print(f"  First: {df_filtered.index[0]}")
        print(f"  Last:  {df_filtered.index[-1]}")

        # Save
        df_filtered.to_parquet(output_file)
        print(f"\nSaved to: {output_file}")
    else:
        print("\nError: No data remaining after filtering!")
        sys.exit(1)


if __name__ == "__main__":
    main()
