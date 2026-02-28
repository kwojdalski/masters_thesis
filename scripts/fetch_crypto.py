#!/usr/bin/env python
"""
Simple interactive script to download cryptocurrency data from exchanges.

Just run: python scripts/fetch_crypto.py
"""

import datetime
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def check_dependencies():
    """Check if required packages are installed."""
    try:
        from gym_trading_env.downloader import download

        return download
    except ImportError:
        print("ERROR: gym_trading_env package not found")
        print("\nTo fix this, run:")
        print("  pip install gym-trading-env")
        sys.exit(1)


def main():
    """Main function."""
    print("\n" + "=" * 80)
    print("Cryptocurrency Data Downloader (via gym_trading_env)")
    print("=" * 80)

    # Check dependencies
    download = check_dependencies()
    print("Dependencies OK")

    # Available exchanges and their characteristics
    exchanges_info = {
        "bitfinex2": "Bitfinex - High liquidity, good historical data",
        "binance": "Binance - Largest exchange by volume",
        "kraken": "Kraken - Established US-based exchange",
        "coinbasepro": "Coinbase Pro - US-regulated exchange",
    }

    print("\nAvailable exchanges:")
    for exchange, desc in exchanges_info.items():
        print(f"  • {exchange:15s} - {desc}")

    # Get user input
    print("\n" + "-" * 80)
    print("What would you like to download?")
    print("-" * 80)

    # Exchange selection
    print("\nSelect exchange:")
    print("  1. bitfinex2 (default)")
    print("  2. binance")
    print("  3. kraken")
    print("  4. coinbasepro")
    exchange_choice = input("Choose [1-4, default: 1]: ").strip()

    exchange_map = {
        "1": "bitfinex2",
        "2": "binance",
        "3": "kraken",
        "4": "coinbasepro",
        "": "bitfinex2",
    }
    exchange = exchange_map.get(exchange_choice, "bitfinex2")
    print(f"  Selected: {exchange}")

    # Symbol(s)
    symbol_input = input(
        "\nEnter trading pair(s) [comma-separated, e.g., BTC/USDT or BTC/USDT,ETH/USDT]: "
    ).strip().upper()
    if not symbol_input:
        symbol_input = "BTC/USDT"
        print(f"  Using default: {symbol_input}")

    # Parse symbols
    symbols_list = [s.strip() for s in symbol_input.split(",")]
    if len(symbols_list) > 1:
        print(f"  Will download {len(symbols_list)} symbols: {symbols_list}")

    # Timeframe
    print("\nSelect timeframe:")
    print("  1. 1 hour (1h)")
    print("  2. 1 day (1d)")
    print("  3. 5 minutes (5m)")
    print("  4. 1 minute (1m)")
    print("  5. 4 hours (4h)")
    timeframe_choice = input("Choose [1-5, default: 1]: ").strip()

    timeframe_map = {
        "1": "1h",
        "2": "1d",
        "3": "5m",
        "4": "1m",
        "5": "4h",
        "": "1h",
    }
    timeframe = timeframe_map.get(timeframe_choice, "1h")
    print(f"  Selected: {timeframe}")

    # Date range
    start_date_str = input(
        "\nStart date (YYYY-MM-DD) [default: 2024-01-01]: "
    ).strip()
    if not start_date_str:
        start_date_str = "2024-01-01"
        print(f"  Using default: {start_date_str}")

    # Parse start date
    try:
        year, month, day = map(int, start_date_str.split("-"))
        since = datetime.datetime(year=year, month=month, day=day, tzinfo=datetime.UTC)
    except ValueError:
        print(f"  ERROR: Invalid date format: {start_date_str}")
        print("  Using default: 2024-01-01")
        since = datetime.datetime(year=2024, month=1, day=1, tzinfo=datetime.UTC)

    # Output directory
    output_dir = "data/raw/crypto"
    print(f"\nOutput directory: {output_dir}")

    # Download
    print("\n" + "=" * 80)
    print("Downloading...")
    print("=" * 80)
    print(f"Exchange: {exchange}")
    print(f"Symbol(s): {symbols_list}")
    print(f"Timeframe: {timeframe}")
    print(f"Since: {since.strftime('%Y-%m-%d')}")
    print(f"Output: {output_dir}/")
    print()

    try:
        download(
            exchange_names=[exchange],
            symbols=symbols_list,
            timeframe=timeframe,
            dir=output_dir,
            since=since,
        )

        print("\n" + "=" * 80)
        print("SUCCESS!")
        print("=" * 80)

        # Show downloaded files
        print("\nDownloaded files:")
        for symbol in symbols_list:
            # gym_trading_env saves as: {exchange}-{symbol}-{timeframe}.parquet
            symbol_safe = symbol.replace("/", "")
            filename = f"{exchange}-{symbol_safe}-{timeframe}.parquet"
            print(f"  • {output_dir}/{filename}")

        print("\n" + "=" * 80)
        print("Next steps:")
        print("=" * 80)

        example_symbol = symbols_list[0].replace("/", "")
        example_file = f"{exchange}-{example_symbol}-{timeframe}.parquet"

        print("1. Inspect the data:")
        print(
            f"   python -c \"import pandas as pd; df = pd.read_parquet('{output_dir}/{example_file}'); "
            f"print(df.info()); print(df.head())\""
        )

        print("\n2. Use in training:")
        print("   Update your scenario YAML config:")
        print("   data:")
        print(f"     data_path: './{output_dir}/{example_file}'")
        print("     download_data: false")

        print("\n3. Available columns:")
        print("   OHLCV data: open, high, low, close, volume")

    except Exception as e:
        print("\n" + "=" * 80)
        print("ERROR")
        print("=" * 80)
        print(f"Failed to download data: {e}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Verify the symbol exists on the exchange (e.g., BTC/USDT, ETH/USDT)")
        print("3. Try a different exchange or timeframe")
        print("4. Check if gym_trading_env supports this exchange")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
