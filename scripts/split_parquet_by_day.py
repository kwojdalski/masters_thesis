"""Split multi-day raw MBP-10 parquet files into one file per symbol per day.

Output layout:
    data/raw/stocks/daily/{SYMBOL}/{SYMBOL}_{YYYY-MM-DD}_raw_mbp-10_us_hours.parquet

Usage:
    uv run python scripts/split_parquet_by_day.py
    uv run python scripts/split_parquet_by_day.py --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SYMBOLS = ["AAPL", "MSFT", "TSLA", "META", "AMZN", "AVGO"]
SOURCE_DIR = Path("data/raw/stocks")
OUT_DIR = Path("data/raw/stocks/daily")
SUFFIX = "_raw_mbp-10_us_hours.parquet"


def find_source(symbol: str) -> Path | None:
    """Return the widest-date-range us_hours parquet for this symbol.

    Picks the file whose end-date component (second date in the filename) is
    latest, so AAPL_2026-02-25_2026-03-03_... beats AAPL_2026-02-25_2026-02-27_...
    """
    candidates = sorted(SOURCE_DIR.glob(f"{symbol}_*_raw_mbp-10_us_hours.parquet"))
    if not candidates:
        return None

    def end_date(p: Path) -> str:
        # filename pattern: SYMBOL_START_END_raw_mbp-10_us_hours.parquet
        parts = p.stem.split("_")
        # parts[1] = start date, parts[2] = end date
        return parts[2] if len(parts) >= 3 else ""

    return sorted(candidates, key=end_date, reverse=True)[0]


def split_symbol(symbol: str, dry_run: bool = False) -> list[Path]:
    source = find_source(symbol)
    if source is None:
        print(f"  [{symbol}] no source file found — skipping")
        return []

    print(f"  [{symbol}] reading {source.name}  ...", end=" ", flush=True)
    df = pd.read_parquet(source)
    df.index = pd.to_datetime(df.index, utc=True)
    print(f"{len(df):,} rows")

    out_dir = OUT_DIR / symbol
    written: list[Path] = []

    for date, group in df.groupby(df.index.date):
        date_str = str(date)
        out_path = out_dir / f"{symbol}_{date_str}{SUFFIX}"
        print(f"    {date_str}  {len(group):>9,} rows  →  {out_path}")
        if not dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)
            group.to_parquet(out_path)
            written.append(out_path)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Split raw parquet files by day")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without writing")
    args = parser.parse_args()

    if args.dry_run:
        print("DRY RUN — no files will be written\n")

    all_written: list[Path] = []
    for symbol in SYMBOLS:
        all_written.extend(split_symbol(symbol, dry_run=args.dry_run))

    if not args.dry_run:
        print(f"\nWrote {len(all_written)} files to {OUT_DIR}/")
        print("\nSummary by symbol:")
        for symbol in SYMBOLS:
            files = sorted((OUT_DIR / symbol).glob("*.parquet"))
            if files:
                print(f"  {symbol}: {len(files)} days")
    else:
        print("\n(dry run — nothing written)")


if __name__ == "__main__":
    main()
