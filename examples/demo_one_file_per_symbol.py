#!/usr/bin/env python
"""
Demo: One file per instrument behavior

This script demonstrates that when downloading multiple symbols,
each instrument gets its own file for modularity.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trading_rl.data_fetchers import StockDataFetcher


def main():
    print("=" * 80)
    print("Demo: One File Per Instrument")
    print("=" * 80)

    fetcher = StockDataFetcher(output_dir="data/raw/stocks")

    print("\nDownloading 3 tech stocks: AAPL, MSFT, GOOGL")
    print("Expected behavior: Creates 3 separate parquet files\n")

    # Download multiple symbols
    df = fetcher.fetch_stock_data(
        symbols=["AAPL", "MSFT", "GOOGL"],
        start_date="2024-01-01",
        end_date="2024-01-31",
        source="databento",
        dataset="XNAS.ITCH",
        timeframe="1h",
        save_to_file=True,
    )

    print("\n" + "=" * 80)
    print("Result:")
    print("=" * 80)

    # Show what was created
    output_dir = Path("data/raw/stocks")
    files = sorted(output_dir.glob("*_2024-01-01_2024-01-31_1h.parquet"))

    print(f"\nCreated {len(files)} files:")
    for f in files:
        import pandas as pd

        df_file = pd.read_parquet(f)
        print(f"  • {f.name:50s} ({len(df_file):6d} rows)")

    print("\n" + "=" * 80)
    print("Why one file per instrument?")
    print("=" * 80)
    print("""
1. Modularity: Each instrument can be used independently
2. Training: RL agents typically train on single instruments
3. Storage: Easy to manage and version control individual files
4. Reusability: Mix and match instruments for different experiments
5. Clarity: Clear what each file contains

Example training config:
  data:
    data_path: './data/raw/stocks/AAPL_2024-01-01_2024-01-31_1h.parquet'
    """)

    print("\n" + "=" * 80)
    print("Accessing individual instruments:")
    print("=" * 80)

    # Show how to load individual files
    for symbol in ["AAPL", "MSFT", "GOOGL"]:
        filename = f"{symbol}_2024-01-01_2024-01-31_1h.parquet"
        filepath = output_dir / filename

        if filepath.exists():
            import pandas as pd

            df_symbol = pd.read_parquet(filepath)
            print(f"\n{symbol}:")
            print(f"  File: {filename}")
            print(f"  Rows: {len(df_symbol)}")
            print(f"  Columns: {list(df_symbol.columns)}")
            print(f"  Date range: {df_symbol.index.min()} to {df_symbol.index.max()}")


if __name__ == "__main__":
    main()
