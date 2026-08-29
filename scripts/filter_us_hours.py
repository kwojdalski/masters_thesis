#!/usr/bin/env python
"""
Filter stock data to the regular US equity session (9:30 AM - 4:00 PM ET).

Usage:
    python scripts/filter_us_hours.py <input_file> [output_file]

If output_file is not provided, appends '_us_hours' to the input filename.

Examples:
    python scripts/filter_us_hours.py data/raw/stocks/AAPL_2024-01-01_2024-12-31_raw_mbp-10.parquet
    python scripts/filter_us_hours.py input.parquet output.parquet
"""

import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

# The session is defined in exchange-local wall-clock time. Its UTC offset
# moves with US daylight saving (UTC-5 in winter, UTC-4 in summer), so the
# window must be evaluated after converting, never as fixed UTC hours.
_EXCHANGE_TZ = ZoneInfo("America/New_York")
_SESSION_OPEN_HOUR, _SESSION_OPEN_MINUTE = 9, 30
_SESSION_CLOSE_HOUR = 16  # exclusive


def filter_us_trading_hours(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter DataFrame to the regular US equity session (09:30-16:00 ET).

    The session is converted to America/New_York before masking, so the window
    stays correct across the daylight-saving boundary: 09:30 ET is 14:30 UTC
    under EST but 13:30 UTC under EDT. Masking on fixed UTC hours dropped the
    first hour of every EDT session (09:30-10:30 ET) and kept an hour of
    post-close data instead. The conversion is per-timestamp, so a dataset
    spanning a DST transition is handled correctly as well.

    Args:
        df: DataFrame with a tz-aware DatetimeIndex

    Returns:
        Filtered DataFrame containing only regular-session rows

    Raises:
        ValueError: if the index is timezone-naive, which cannot be converted
            to exchange-local time without guessing an offset.
    """
    if df.index.tz is None:
        raise ValueError(
            "filter_us_trading_hours requires a tz-aware DatetimeIndex; "
            "localize it (e.g. df.index.tz_localize('UTC')) before calling."
        )

    local = df.index.tz_convert(_EXCHANGE_TZ)
    after_open = (local.hour > _SESSION_OPEN_HOUR) | (
        (local.hour == _SESSION_OPEN_HOUR) & (local.minute >= _SESSION_OPEN_MINUTE)
    )
    before_close = local.hour < _SESSION_CLOSE_HOUR

    return df[after_open & before_close]


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
        output_file = (
            input_file.parent / f"{input_file.stem}_us_hours{input_file.suffix}"
        )

    print(f"Loading data from: {input_file}")
    df = pd.read_parquet(input_file)

    original_size = len(df)
    print(f"Original shape: {df.shape}")

    # Check timezone
    if df.index.tz is None:
        print("Warning: Index has no timezone, assuming UTC")
        df.index = df.index.tz_localize("UTC")
    elif str(df.index.tz) != "UTC":
        print(f"Warning: Index timezone is {df.index.tz}, expected UTC")

    # Filter
    df_filtered = filter_us_trading_hours(df)

    print(f"Filtered shape: {df_filtered.shape}")
    print(f"Reduction: {100 * (1 - len(df_filtered) / original_size):.1f}%")

    if len(df_filtered) > 0:
        print("\nTime range:")
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
