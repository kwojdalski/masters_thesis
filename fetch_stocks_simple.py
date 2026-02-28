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

    # Symbol(s)
    symbol_input = input("Enter stock symbol(s) [comma-separated, e.g., AAPL or AAPL,MSFT]: ").strip().upper()
    if not symbol_input:
        symbol_input = "AAPL"
        print(f"  Using default: {symbol_input}")

    # Parse symbols
    symbols_list = [s.strip() for s in symbol_input.split(",")]
    if len(symbols_list) > 1:
        print(f"  Will download {len(symbols_list)} symbols: {symbols_list}")
        print(f"  Note: Each symbol will be saved to a separate file")

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

    # Data format
    print("\nSelect data format:")
    print("  1. Aggregated OHLCV bars (for training)")
    print("  2. Raw tick/order book data (for analysis)")
    format_choice = input("Choose [1-2, default: 1]: ").strip()

    if format_choice == "2":
        # Raw data
        aggregate = False
        timeframe = None

        print("\nSelect schema for raw data:")
        print("  1. trades - Individual trades (tick data)")
        print("  2. tbbo - Top of book (best bid/offer)")
        print("  3. mbp-1 - Market by price level 1 (order book top)")
        print("  4. mbp-10 - Market by price level 10 (order book depth)")
        schema_choice = input("Choose [1-4, default: 1]: ").strip()

        schema_map = {
            "1": "trades",
            "2": "tbbo",
            "3": "mbp-1",
            "4": "mbp-10",
            "": "trades",
        }
        schema = schema_map.get(schema_choice, "trades")
        print(f"  Selected: {schema} (raw, no aggregation)")

    else:
        # Aggregated OHLCV
        aggregate = True
        schema = "trades"

        print("\nSelect timeframe for aggregation:")
        print("  1. 1 hour")
        print("  2. 1 day")
        print("  3. 5 minutes")
        timeframe_choice = input("Choose [1-3, default: 1]: ").strip()

        timeframe_map = {
            "1": "1h",
            "2": "1d",
            "3": "5m",
            "": "1h",
        }
        timeframe = timeframe_map.get(timeframe_choice, "1h")
        print(f"  Selected: {timeframe} bars")

    # Download
    print("\n" + "=" * 80)
    print("Downloading...")
    print("=" * 80)
    print(f"Symbol(s): {symbols_list}")
    print(f"Dates: {start_date} to {end_date}")
    print(f"Dataset: {dataset}")
    print(f"Schema: {schema}")
    if aggregate:
        print(f"Format: Aggregated OHLCV ({timeframe} bars)")
    else:
        print(f"Format: Raw {schema} data (no aggregation)")
    if len(symbols_list) > 1:
        print(f"\nNote: Will save {len(symbols_list)} separate files (one per instrument)")
    print()

    try:
        df = fetcher.fetch_stock_data(
            symbols=symbols_list,
            start_date=start_date,
            end_date=end_date,
            source="databento",
            dataset=dataset,
            schema=schema,
            timeframe=timeframe,
            aggregate=aggregate,
            save_to_file=True,
            # Let it auto-generate filenames for each symbol
        )

        print("\n" + "=" * 80)
        print("✓ SUCCESS!")
        print("=" * 80)
        print(f"Downloaded {len(df)} total rows")
        print(f"Columns: {list(df.columns)}")

        # Determine file suffix
        if aggregate and timeframe:
            file_suffix = timeframe
        else:
            file_suffix = f"raw_{schema}"

        if len(symbols_list) > 1 and 'symbol' in df.columns:
            print(f"\nData split into {len(symbols_list)} files:")
            for symbol in symbols_list:
                symbol_rows = len(df[df['symbol'] == symbol]) if symbol in df['symbol'].values else 0
                filename = f"{symbol}_{start_date}_{end_date}_{file_suffix}.parquet"
                print(f"  • {filename} ({symbol_rows} rows)")
        else:
            print(f"\nFirst few rows:")
            print(df.head())

        print(f"\nFiles saved to: data/raw/stocks/")

        if not aggregate:
            print(f"\nNote: Downloaded raw {schema} data")
            print("This is tick-level/order book data, not aggregated OHLCV")
            print("Useful for microstructure analysis, backtesting with exact fills, etc.")

        print("\n" + "=" * 80)
        print("Next steps:")
        print("=" * 80)

        example_symbol = symbols_list[0]
        if aggregate and timeframe:
            example_file = f"{example_symbol}_{start_date}_{end_date}_{timeframe}.parquet"
        else:
            example_file = f"{example_symbol}_{start_date}_{end_date}_raw_{schema}.parquet"

        print("1. Inspect the data:")
        print(f"   python -c \"import pandas as pd; df = pd.read_parquet('data/raw/stocks/{example_file}'); print(df.info()); print(df.head())\"")

        if aggregate:
            print("\n2. Use in training:")
            print("   Update your scenario YAML config to use a specific instrument:")
            print("   data:")
            print(f"     data_path: './data/raw/stocks/{example_file}'")
        else:
            print("\n2. Analyze raw data:")
            print("   Raw tick/order book data is useful for:")
            print("   - Microstructure analysis")
            print("   - Exact fill simulation")
            print("   - Order book dynamics")
            print("   - High-frequency trading research")
            print(f"\n   You can aggregate it later if needed for RL training")

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
