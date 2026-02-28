#!/usr/bin/env python
"""
Example: Download raw order book data without aggregation.

This demonstrates downloading tick-level and order book data for
microstructure analysis, high-frequency trading research, or exact backtesting.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trading_rl.data_fetchers import StockDataFetcher


def main():
    print("=" * 80)
    print("Raw Order Book Data Download Examples")
    print("=" * 80)

    fetcher = StockDataFetcher(output_dir="data/raw/stocks", log_level="INFO")

    # Example 1: Raw trades (tick data)
    print("\n" + "=" * 80)
    print("Example 1: Raw Trades (Tick Data)")
    print("=" * 80)
    print("Downloading all individual trades for AAPL on 2024-01-02")
    print("This gives you exact timestamps, prices, and sizes")
    print()

    df_trades = fetcher.fetch_stock_data(
        symbols=["AAPL"],
        start_date="2024-01-02",
        end_date="2024-01-02",
        source="databento",
        dataset="XNAS.ITCH",
        schema="trades",  # Individual trades
        aggregate=False,  # No aggregation!
        save_to_file=True,
    )

    print(f"\n✓ Downloaded {len(df_trades)} individual trades")
    print(f"  Columns: {list(df_trades.columns)}")
    print(f"\n  First few trades:")
    print(df_trades.head(10))

    # Example 2: Top of book (best bid/offer)
    print("\n" + "=" * 80)
    print("Example 2: Top of Book (Best Bid/Offer)")
    print("=" * 80)
    print("Downloading best bid/ask quotes for MSFT on 2024-01-02")
    print("This gives you the order book top at every update")
    print()

    df_tbbo = fetcher.fetch_stock_data(
        symbols=["MSFT"],
        start_date="2024-01-02",
        end_date="2024-01-02",
        source="databento",
        dataset="XNAS.ITCH",
        schema="tbbo",  # Top of book
        aggregate=False,  # Keep raw quotes
        save_to_file=True,
    )

    print(f"\n✓ Downloaded {len(df_tbbo)} quote updates")
    print(f"  Columns: {list(df_tbbo.columns)}")
    if not df_tbbo.empty:
        print(f"\n  Best bid/ask at different times:")
        print(df_tbbo.head(10))

    # Example 3: Order book depth (10 levels)
    print("\n" + "=" * 80)
    print("Example 3: Order Book Depth (10 Levels)")
    print("=" * 80)
    print("Downloading full order book (10 price levels) for GOOGL")
    print("This gives you complete market depth")
    print()

    df_mbp = fetcher.fetch_stock_data(
        symbols=["GOOGL"],
        start_date="2024-01-02",
        end_date="2024-01-02",
        source="databento",
        dataset="XNAS.ITCH",
        schema="mbp-10",  # Market by price - 10 levels
        aggregate=False,  # Keep raw order book
        save_to_file=True,
    )

    print(f"\n✓ Downloaded {len(df_mbp)} order book snapshots")
    print(f"  Columns: {list(df_mbp.columns)}")
    if not df_mbp.empty:
        print(f"\n  Order book depth:")
        print(df_mbp.head(5))

    # Example 4: Compare raw vs aggregated
    print("\n" + "=" * 80)
    print("Example 4: Comparison - Raw vs Aggregated")
    print("=" * 80)

    # Raw trades
    df_raw = fetcher.fetch_stock_data(
        symbols=["AAPL"],
        start_date="2024-01-02 09:30:00",
        end_date="2024-01-02 09:31:00",  # Just 1 minute
        source="databento",
        dataset="XNAS.ITCH",
        schema="trades",
        aggregate=False,
        save_to_file=False,
    )

    # Aggregated to 1-minute bars
    df_agg = fetcher.fetch_stock_data(
        symbols=["AAPL"],
        start_date="2024-01-02 09:30:00",
        end_date="2024-01-02 09:31:00",
        source="databento",
        dataset="XNAS.ITCH",
        schema="trades",
        timeframe="1m",
        aggregate=True,
        save_to_file=False,
    )

    print(f"\nRaw trades in first minute: {len(df_raw)} rows")
    print(f"Aggregated 1-minute bars: {len(df_agg)} rows")
    print(f"\nData compression: {len(df_raw)}→{len(df_agg)} ({len(df_raw)/max(len(df_agg),1):.0f}x)")

    print("\n" + "=" * 80)
    print("Use Cases for Raw Data:")
    print("=" * 80)
    print("""
1. Microstructure Analysis
   - Analyze bid-ask spreads
   - Study order flow dynamics
   - Measure market impact

2. High-Frequency Trading Research
   - Test strategies on tick data
   - Analyze execution quality
   - Study market making

3. Exact Backtesting
   - Simulate precise fill prices
   - Account for partial fills
   - Model slippage accurately

4. Order Book Studies
   - Analyze liquidity depth
   - Study order book imbalance
   - Predict short-term price moves

5. Data Science
   - Train ML models on raw features
   - Extract microstructure signals
   - Build custom aggregations
    """)

    print("\n" + "=" * 80)
    print("Files Created:")
    print("=" * 80)

    output_dir = Path("data/raw/stocks")
    raw_files = sorted(output_dir.glob("*_raw_*.parquet"))
    for f in raw_files[-5:]:  # Show last 5
        import pandas as pd

        df_file = pd.read_parquet(f)
        print(f"  {f.name:60s} ({len(df_file):8d} rows)")

    print("\n" + "=" * 80)
    print("Comparison with Aggregated Data:")
    print("=" * 80)

    agg_files = sorted(
        [f for f in output_dir.glob("AAPL_*.parquet") if "raw" not in f.name]
    )
    if agg_files:
        agg_file = agg_files[0]
        df_agg_file = pd.read_parquet(agg_file)
        print(f"\nAggregated: {agg_file.name}")
        print(f"  Rows: {len(df_agg_file)}")
        print(f"  Columns: {list(df_agg_file.columns)}")
        print(f"  Size: {agg_file.stat().st_size / 1024:.1f} KB")

    if raw_files:
        raw_file = raw_files[0]
        df_raw_file = pd.read_parquet(raw_file)
        print(f"\nRaw: {raw_file.name}")
        print(f"  Rows: {len(df_raw_file)}")
        print(f"  Columns: {list(df_raw_file.columns)}")
        print(f"  Size: {raw_file.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
