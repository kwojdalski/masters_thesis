#!/usr/bin/env python
"""
Download stock data from Databento or Polygon.

Interactive mode: python scripts/fetch_stocks.py --interactive
CLI mode: python scripts/fetch_stocks.py --symbols AAPL --start-date 2024-01-01 --end-date 2024-03-31
"""

import os
import sys
from pathlib import Path

import typer
from typing_extensions import Annotated

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from logger import get_logger

logger = get_logger(__name__)

app = typer.Typer(help="Download stock data from Databento or Polygon")


def check_api_key() -> str:
    """Check if Databento API key is set."""
    api_key = os.getenv("DATABENTO_API_KEY")
    if not api_key:
        logger.error("DATABENTO_API_KEY environment variable not set")
        logger.info("\nTo fix this, run:")
        logger.info("  export DATABENTO_API_KEY='your-api-key-here'")
        logger.info("\nGet your API key from: https://databento.com")
        raise typer.Exit(code=1)
    else:
        logger.info(f"API key found: {api_key[:10]}...")
        return api_key


@app.command()
def download_stocks(
    symbols: Annotated[
        str,
        typer.Option(
            "--symbols",
            "-s",
            help="Stock symbol(s), comma-separated (e.g., AAPL or AAPL,MSFT)",
        ),
    ] = "AAPL",
    start_date: Annotated[
        str,
        typer.Option("--start-date", "-d", help="Start date (YYYY-MM-DD)"),
    ] = "2024-01-01",
    end_date: Annotated[
        str,
        typer.Option("--end-date", "-e", help="End date (YYYY-MM-DD)"),
    ] = "2024-03-31",
    dataset: Annotated[
        str,
        typer.Option(
            "--dataset", help="Dataset (XNAS.ITCH for NASDAQ, XNYS.TRADES for NYSE)"
        ),
    ] = "XNAS.ITCH",
    schema: Annotated[
        str,
        typer.Option(
            "--schema",
            help="Schema (trades, tbbo, mbp-1, mbp-10)",
        ),
    ] = "trades",
    timeframe: Annotated[
        str,
        typer.Option(
            "--timeframe",
            "-t",
            help="Timeframe for aggregation (1h, 1d, 5m) - only for aggregated data",
        ),
    ] = "1h",
    aggregate: Annotated[
        bool,
        typer.Option("--aggregate/--raw", help="Aggregate to OHLCV or keep raw data"),
    ] = True,
    output_dir: Annotated[
        str,
        typer.Option("--output-dir", "-o", help="Output directory"),
    ] = "data/raw/stocks",
    interactive: Annotated[
        bool,
        typer.Option("--interactive", "-i", help="Run in interactive mode"),
    ] = False,
):
    """Download stock data from Databento."""
    # Check API key
    check_api_key()

    logger.info("Stock Data Downloader (Databento)")

    from trading_rl.data_fetchers import StockDataFetcher

    # Initialize fetcher
    fetcher = StockDataFetcher(output_dir=output_dir, log_level="INFO")

    # Interactive mode
    if interactive:
        logger.info("Running in interactive mode")
        logger.info("\nAvailable datasets:")
        for ds, desc in fetcher.list_available_datasets("databento").items():
            logger.info(f"  {ds:20s} - {desc}")

        # Get user input
        symbols = typer.prompt(
            "Enter stock symbol(s) [comma-separated, e.g., AAPL or AAPL,MSFT]",
            default="AAPL",
        ).upper()

        start_date = typer.prompt("Start date (YYYY-MM-DD)", default="2024-01-01")
        end_date = typer.prompt("End date (YYYY-MM-DD)", default="2024-03-31")

        # Dataset selection
        typer.echo("\nSelect dataset:")
        typer.echo("  1. XNAS.ITCH (NASDAQ)")
        typer.echo("  2. XNYS.TRADES (NYSE)")
        dataset_choice = typer.prompt("Choose [1-2, default: 1]", default="1")

        dataset_map = {
            "1": "XNAS.ITCH",
            "2": "XNYS.TRADES",
        }
        dataset = dataset_map.get(dataset_choice, "XNAS.ITCH")

        # Data format
        typer.echo("\nSelect data format:")
        typer.echo("  1. Aggregated OHLCV bars (for training)")
        typer.echo("  2. Raw tick/order book data (for analysis)")
        format_choice = typer.prompt("Choose [1-2, default: 1]", default="1")

        if format_choice == "2":
            aggregate = False
            timeframe = None

            typer.echo("\nSelect schema for raw data:")
            typer.echo("  1. trades - Individual trades (tick data)")
            typer.echo("  2. tbbo - Top of book (best bid/offer)")
            typer.echo("  3. mbp-1 - Market by price level 1 (order book top)")
            typer.echo("  4. mbp-10 - Market by price level 10 (order book depth)")
            schema_choice = typer.prompt("Choose [1-4, default: 1]", default="1")

            schema_map = {
                "1": "trades",
                "2": "tbbo",
                "3": "mbp-1",
                "4": "mbp-10",
            }
            schema = schema_map.get(schema_choice, "trades")
        else:
            aggregate = True
            schema = "trades"

            typer.echo("\nSelect timeframe for aggregation:")
            typer.echo("  1. 1 hour")
            typer.echo("  2. 1 day")
            typer.echo("  3. 5 minutes")
            timeframe_choice = typer.prompt("Choose [1-3, default: 1]", default="1")

            timeframe_map = {
                "1": "1h",
                "2": "1d",
                "3": "5m",
            }
            timeframe = timeframe_map.get(timeframe_choice, "1h")

    # Parse symbols
    symbols_list = [s.strip() for s in symbols.split(",")]
    logger.info(f"Symbols to download: {symbols_list}")

    # Download
    logger.info("Starting download...")
    logger.info(f"Symbols: {symbols_list}")
    logger.info(f"Dates: {start_date} to {end_date}")
    logger.info(f"Dataset: {dataset}")
    logger.info(f"Schema: {schema}")
    if aggregate:
        logger.info(f"Format: Aggregated OHLCV ({timeframe} bars)")
    else:
        logger.info(f"Format: Raw {schema} data (no aggregation)")

    if len(symbols_list) > 1:
        logger.info(
            f"Note: Will save {len(symbols_list)} separate files (one per instrument)"
        )

    try:
        df = fetcher.fetch_stock_data(
            symbols=symbols_list,
            start_date=start_date,
            end_date=end_date,
            source="databento",
            dataset=dataset,
            schema=schema,
            timeframe=timeframe if aggregate else None,
            aggregate=aggregate,
            save_to_file=True,
        )

        logger.info("Download completed successfully")
        logger.info(f"Downloaded {len(df)} total rows")
        logger.info(f"Columns: {list(df.columns)}")

        # Determine file suffix
        if aggregate and timeframe:
            file_suffix = timeframe
        else:
            file_suffix = f"raw_{schema}"

        if len(symbols_list) > 1 and "symbol" in df.columns:
            logger.info(f"\nData split into {len(symbols_list)} files:")
            for symbol in symbols_list:
                symbol_rows = (
                    len(df[df["symbol"] == symbol])
                    if symbol in df["symbol"].values
                    else 0
                )
                filename = f"{symbol}_{start_date}_{end_date}_{file_suffix}.parquet"
                logger.info(f"  {filename} ({symbol_rows} rows)")
        else:
            logger.info(f"\nFirst few rows:")
            typer.echo(df.head())

        logger.info(f"\nFiles saved to: {output_dir}/")

        if not aggregate:
            logger.info(f"\nNote: Downloaded raw {schema} data")
            logger.info("This is tick-level/order book data, not aggregated OHLCV")
            logger.info(
                "Useful for microstructure analysis, backtesting with exact fills, etc."
            )

        logger.info("\nNext steps:")
        example_symbol = symbols_list[0]
        if aggregate and timeframe:
            example_file = f"{example_symbol}_{start_date}_{end_date}_{timeframe}.parquet"
        else:
            example_file = (
                f"{example_symbol}_{start_date}_{end_date}_raw_{schema}.parquet"
            )

        typer.echo("\n1. Inspect the data:")
        typer.echo(
            f"   python -c \"import pandas as pd; df = pd.read_parquet('{output_dir}/{example_file}'); "
            f"print(df.info()); print(df.head())\""
        )

        if aggregate:
            typer.echo("\n2. Use in training:")
            typer.echo("   Update your scenario YAML config:")
            typer.echo("   data:")
            typer.echo(f"     data_path: './{output_dir}/{example_file}'")
        else:
            typer.echo("\n2. Analyze raw data:")
            typer.echo("   Raw tick/order book data is useful for:")
            typer.echo("   - Microstructure analysis")
            typer.echo("   - Exact fill simulation")
            typer.echo("   - Order book dynamics")
            typer.echo("   - High-frequency trading research")

    except Exception as e:
        logger.error(f"Failed to download data: {e}", exc_info=True)
        logger.info("\nTroubleshooting:")
        logger.info("1. Check your API key is correct")
        logger.info("2. Verify the symbol exists (e.g., AAPL, MSFT, GOOGL)")
        logger.info("3. Check date range is valid")
        logger.info("4. Make sure weles project is at ../weles")
        raise typer.Exit(code=1)


@app.command()
def list_datasets():
    """List available datasets from Databento."""
    check_api_key()

    from trading_rl.data_fetchers import StockDataFetcher

    fetcher = StockDataFetcher(output_dir="data/raw/stocks", log_level="INFO")

    typer.echo("Available Databento datasets:\n")
    for dataset, desc in fetcher.list_available_datasets("databento").items():
        typer.echo(f"  {dataset:20s} - {desc}")


if __name__ == "__main__":
    app()
