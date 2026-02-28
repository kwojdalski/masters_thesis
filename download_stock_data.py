"""
Quick script to download stock data using Databento.

Usage:
    python download_stock_data.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from trading_rl.data_fetchers import StockDataFetcher


def main():
    """Download stock data using Databento."""

    # Initialize the fetcher
    fetcher = StockDataFetcher(
        output_dir="data/raw/stocks",
        log_level="INFO"
    )

    print("=" * 80)
    print("Downloading stock data from Databento...")
    print("=" * 80)

    # Example 1: Download Apple stock for 2024 (1 hour bars)
    print("\n1. Downloading AAPL (Apple) - 2024 data...")
    df_aapl = fetcher.fetch_stock_data(
        symbols=["AAPL"],
        start_date="2024-01-01",
        end_date="2024-12-31",
        source="databento",
        dataset="XNAS.ITCH",  # NASDAQ
        schema="trades",       # Trade-level data
        timeframe="1h",        # Aggregate to hourly bars
        save_to_file=True,
        output_filename="AAPL_2024_1h.parquet"
    )

    print(f"✓ Downloaded {len(df_aapl)} rows")
    print(f"  Columns: {list(df_aapl.columns)}")
    print(f"  Date range: {df_aapl.index.min()} to {df_aapl.index.max()}")
    print(f"  Saved to: data/raw/stocks/AAPL_2024_1h.parquet")

    # Example 2: Download multiple tech stocks (saves one file per symbol)
    print("\n2. Downloading tech stocks (AAPL, MSFT, GOOGL)...")
    print("   Note: Each symbol will be saved to a separate file")
    df_tech = fetcher.fetch_stock_data(
        symbols=["AAPL", "MSFT", "GOOGL"],
        start_date="2024-01-01",
        end_date="2024-03-31",  # Q1 2024
        source="databento",
        dataset="XNAS.ITCH",
        timeframe="1h",
        save_to_file=True,
        # No output_filename - will auto-generate one per symbol
    )

    print(f"✓ Downloaded {len(df_tech)} total rows")
    if 'symbol' in df_tech.columns:
        print(f"  Symbols: {list(df_tech['symbol'].unique())}")
        print(f"  Saved as separate files:")
        for symbol in df_tech['symbol'].unique():
            print(f"    - {symbol}_2024-01-01_2024-03-31_1h.parquet")

    # Example 3: Download using config file
    print("\n3. Downloading using config file...")
    df_config = fetcher.fetch_from_config(
        "src/configs/data/stocks_single_symbol.yaml"
    )

    print(f"✓ Downloaded {len(df_config)} rows from config")

    print("\n" + "=" * 80)
    print("All downloads complete!")
    print("=" * 80)
    print("\nFiles saved in: data/raw/stocks/")
    print("  (One file per instrument for modularity)")
    print("\nNext steps:")
    print("1. Check the data: python -c \"import pandas as pd; print(pd.read_parquet('data/raw/stocks/AAPL_2024_1h.parquet').head())\"")
    print("2. Use in training: Update your scenario config data_path to point to a specific instrument file")
    print("   Example: data_path: './data/raw/stocks/AAPL_2024_1h.parquet'")


if __name__ == "__main__":
    main()
