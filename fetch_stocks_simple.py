#!/usr/bin/env python
"""
Simple interactive script to download stock data.

Just run: python fetch_stocks_simple.py
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def check_api_key():
    """Check if Databento API key is set."""
    api_key = os.getenv("DATABENTO_API_KEY")
    if not api_key:
        print("❌ ERROR: DATABENTO_API_KEY environment variable not set")
        print("\nTo fix this, run:")
        print("  export DATABENTO_API_KEY='your-api-key-here'")
        print("\nGet your API key from: https://databento.com")
        sys.exit(1)
    else:
        print(f"✓ API key found: {api_key[:10]}...")


def main():
    """Main function."""
    print("\n" + "=" * 80)
    print("Stock Data Downloader (Databento)")
    print("=" * 80)

    # Check API key
    check_api_key()

    from trading_rl.data_fetchers import StockDataFetcher

    # Initialize fetcher
    fetcher = StockDataFetcher(
        output_dir="data/raw/stocks",
        log_level="INFO"
    )

    print("\nAvailable datasets:")
    for dataset, desc in fetcher.list_available_datasets("databento").items():
        print(f"  • {dataset:20s} - {desc}")

    # Get user input
    print("\n" + "-" * 80)
    print("What would you like to download?")
    print("-" * 80)

    # Symbol
    symbol_input = input("Enter stock symbol (e.g., AAPL): ").strip().upper()
    if not symbol_input:
        symbol_input = "AAPL"
        print(f"  Using default: {symbol_input}")

    # Date range
    start_date = input("Start date (YYYY-MM-DD) [default: 2024-01-01]: ").strip()
    if not start_date:
        start_date = "2024-01-01"
        print(f"  Using default: {start_date}")

    end_date = input("End date (YYYY-MM-DD) [default: 2024-03-31]: ").strip()
    if not end_date:
        end_date = "2024-03-31"
        print(f"  Using default: {end_date}")

    # Dataset
    print("\nSelect dataset:")
    print("  1. XNAS.ITCH (NASDAQ)")
    print("  2. XNYS.TRADES (NYSE)")
    dataset_choice = input("Choose [1-2, default: 1]: ").strip()

    dataset_map = {
        "1": "XNAS.ITCH",
        "2": "XNYS.TRADES",
        "": "XNAS.ITCH",
    }
    dataset = dataset_map.get(dataset_choice, "XNAS.ITCH")
    print(f"  Selected: {dataset}")

    # Timeframe
    print("\nSelect timeframe:")
    print("  1. 1 hour")
    print("  2. 1 day")
    timeframe_choice = input("Choose [1-2, default: 1]: ").strip()

    timeframe_map = {
        "1": "1h",
        "2": "1d",
        "": "1h",
    }
    timeframe = timeframe_map.get(timeframe_choice, "1h")
    print(f"  Selected: {timeframe}")

    # Download
    print("\n" + "=" * 80)
    print("Downloading...")
    print("=" * 80)
    print(f"Symbol: {symbol_input}")
    print(f"Dates: {start_date} to {end_date}")
    print(f"Dataset: {dataset}")
    print(f"Timeframe: {timeframe}")
    print()

    try:
        df = fetcher.fetch_stock_data(
            symbols=[symbol_input],
            start_date=start_date,
            end_date=end_date,
            source="databento",
            dataset=dataset,
            schema="trades",
            timeframe=timeframe,
            save_to_file=True,
            output_filename=f"{symbol_input}_{start_date}_{end_date}_{timeframe}.parquet"
        )

        print("\n" + "=" * 80)
        print("✓ SUCCESS!")
        print("=" * 80)
        print(f"Downloaded {len(df)} rows")
        print(f"Columns: {list(df.columns)}")
        print(f"\nFirst few rows:")
        print(df.head())
        print(f"\nData saved to: data/raw/stocks/{symbol_input}_{start_date}_{end_date}_{timeframe}.parquet")

        print("\n" + "=" * 80)
        print("Next steps:")
        print("=" * 80)
        print("1. Inspect the data:")
        print(f"   python -c \"import pandas as pd; df = pd.read_parquet('data/raw/stocks/{symbol_input}_{start_date}_{end_date}_{timeframe}.parquet'); print(df.describe())\"")
        print("\n2. Use in training:")
        print("   Update your scenario YAML config:")
        print("   data:")
        print(f"     data_path: './data/raw/stocks/{symbol_input}_{start_date}_{end_date}_{timeframe}.parquet'")

    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ ERROR")
        print("=" * 80)
        print(f"Failed to download data: {e}")
        print("\nTroubleshooting:")
        print("1. Check your API key is correct")
        print("2. Verify the symbol exists (e.g., AAPL, MSFT, GOOGL)")
        print("3. Check date range is valid")
        print("4. Make sure weles project is at ../weles")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
