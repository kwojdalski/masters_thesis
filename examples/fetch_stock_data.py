"""
Example script demonstrating how to use StockDataFetcher.

This script shows various ways to fetch stock market data from NYSE/NASDAQ
using the Databento or Polygon data sources through the weles market_data_fetcher.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trading_rl.data_fetchers import StockDataFetcher


def example_1_basic_fetch():
    """Example 1: Basic stock data fetch for single symbol."""
    print("\n" + "=" * 80)
    print("Example 1: Fetch single stock (AAPL) from NASDAQ")
    print("=" * 80)

    fetcher = StockDataFetcher(
        output_dir="data/raw/stocks",
        log_level="INFO",
    )

    # Fetch Apple stock data for Q1 2024
    df = fetcher.fetch_stock_data(
        symbols=["AAPL"],
        start_date="2024-01-01",
        end_date="2024-03-31",
        source="databento",
        dataset="XNAS.ITCH",  # NASDAQ
        schema="trades",
        timeframe="1h",
        save_to_file=True,
    )

    print(f"\nFetched {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nData saved to: data/raw/stocks/")


def example_2_multiple_stocks():
    """Example 2: Fetch multiple tech stocks."""
    print("\n" + "=" * 80)
    print("Example 2: Fetch multiple tech stocks from NASDAQ")
    print("=" * 80)

    fetcher = StockDataFetcher(output_dir="data/raw/stocks")

    # Fetch multiple stocks
    df = fetcher.fetch_stock_data(
        symbols=["AAPL", "MSFT", "GOOGL", "AMZN"],
        start_date="2024-01-01",
        end_date="2024-01-31",
        source="databento",
        dataset="XNAS.ITCH",
        timeframe="1h",
        save_to_file=True,
        output_filename="tech_stocks_jan2024.parquet",
    )

    print(f"\nFetched {len(df)} rows for {len(df['symbol'].unique())} symbols")
    print(f"Symbols: {df['symbol'].unique()}")


def example_3_from_config():
    """Example 3: Fetch data using YAML configuration."""
    print("\n" + "=" * 80)
    print("Example 3: Fetch using YAML configuration")
    print("=" * 80)

    fetcher = StockDataFetcher(output_dir="data/raw/stocks")

    # Use pre-defined config
    config_path = "src/configs/data/stocks_single_symbol.yaml"

    df = fetcher.fetch_from_config(config_path)

    print(f"\nFetched {len(df)} rows from config: {config_path}")
    print(f"Columns: {list(df.columns)}")


def example_4_nyse_stocks():
    """Example 4: Fetch NYSE stocks."""
    print("\n" + "=" * 80)
    print("Example 4: Fetch diversified portfolio from NYSE")
    print("=" * 80)

    fetcher = StockDataFetcher(output_dir="data/raw/stocks")

    # Fetch NYSE stocks
    df = fetcher.fetch_stock_data(
        symbols=["JPM", "JNJ", "XOM", "WMT"],
        start_date="2023-01-01",
        end_date="2023-12-31",
        source="databento",
        dataset="XNYS.TRADES",  # NYSE
        schema="trades",
        timeframe="1h",
        save_to_file=True,
        output_filename="nyse_diversified_2023.parquet",
    )

    print(f"\nFetched {len(df)} rows")
    print(f"\nPrice statistics:")
    print(df[["open", "high", "low", "close", "volume"]].describe())


def example_5_list_datasets():
    """Example 5: List available datasets."""
    print("\n" + "=" * 80)
    print("Example 5: List available datasets")
    print("=" * 80)

    fetcher = StockDataFetcher()

    print("\nAvailable Databento datasets:")
    for dataset, description in fetcher.list_available_datasets("databento").items():
        print(f"  {dataset:20s} - {description}")

    print("\nAvailable Polygon datasets:")
    for dataset, description in fetcher.list_available_datasets("polygon").items():
        print(f"  {dataset:20s} - {description}")


def example_6_integration_with_training():
    """Example 6: Integration with existing training pipeline."""
    print("\n" + "=" * 80)
    print("Example 6: Integration with training pipeline")
    print("=" * 80)

    # Step 1: Fetch stock data
    fetcher = StockDataFetcher(output_dir="data/raw/stocks")

    df = fetcher.fetch_stock_data(
        symbols=["AAPL"],
        start_date="2023-01-01",
        end_date="2023-12-31",
        source="databento",
        dataset="XNAS.ITCH",
        timeframe="1h",
        save_to_file=True,
        output_filename="AAPL_2023_training.parquet",
    )

    print(f"\nStep 1: Fetched {len(df)} rows of stock data")

    # Step 2: Use with existing data pipeline
    from trading_rl.data_utils import prepare_data

    print("\nStep 2: Preparing data for training...")

    train_df, val_df, test_df = prepare_data(
        data_path="data/raw/stocks/AAPL_2023_training.parquet",
        train_size=6000,  # ~250 days
        validation_size=1000,  # ~42 days
        feature_config_path="src/configs/features/btc_with_volume.yaml",
    )

    print(f"Training set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    print(f"\nFeatures: {[c for c in train_df.columns if 'feature' in c]}")

    print("\nData is now ready for RL training!")


def main():
    """Run all examples."""
    import argparse

    parser = argparse.ArgumentParser(description="Stock data fetcher examples")
    parser.add_argument(
        "--example",
        type=int,
        choices=range(1, 7),
        help="Run specific example (1-6). If not specified, runs all examples.",
    )
    args = parser.parse_args()

    examples = {
        1: example_1_basic_fetch,
        2: example_2_multiple_stocks,
        3: example_3_from_config,
        4: example_4_nyse_stocks,
        5: example_5_list_datasets,
        6: example_6_integration_with_training,
    }

    if args.example:
        examples[args.example]()
    else:
        # Run all examples
        for example_func in examples.values():
            try:
                example_func()
            except Exception as e:
                print(f"\nError in {example_func.__name__}: {e}")
                import traceback

                traceback.print_exc()

    print("\n" + "=" * 80)
    print("Examples complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
