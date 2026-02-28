#!/usr/bin/env python
"""
Download cryptocurrency data from exchanges.

Interactive mode: python scripts/fetch_crypto.py
CLI mode: python scripts/fetch_crypto.py --exchange binance --symbols BTC/USDT --timeframe 1h --start-date 2024-01-01
"""

import datetime
import sys
from pathlib import Path
from typing import Optional

import typer
from typing_extensions import Annotated

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from logger import get_logger

logger = get_logger(__name__)

app = typer.Typer(help="Download cryptocurrency data from exchanges")


def check_dependencies():
    """Check if required packages are installed."""
    try:
        from gym_trading_env.downloader import download

        return download
    except ImportError:
        logger.error("gym_trading_env package not found")
        logger.info("To fix this, run: pip install gym-trading-env")
        raise typer.Exit(code=1)


def parse_date(date_str: str) -> datetime.datetime:
    """Parse date string to datetime."""
    try:
        year, month, day = map(int, date_str.split("-"))
        return datetime.datetime(year=year, month=month, day=day, tzinfo=datetime.UTC)
    except (ValueError, AttributeError) as e:
        logger.error(f"Invalid date format: {date_str}. Expected YYYY-MM-DD")
        raise typer.BadParameter(f"Invalid date format: {date_str}") from e


@app.command()
def download_crypto(
    exchange: Annotated[
        str,
        typer.Option(
            "--exchange",
            "-e",
            help="Exchange name (bitfinex2, binance, kraken, coinbasepro)",
        ),
    ] = "bitfinex2",
    symbols: Annotated[
        str,
        typer.Option(
            "--symbols",
            "-s",
            help="Trading pair(s), comma-separated (e.g., BTC/USDT or BTC/USDT,ETH/USDT)",
        ),
    ] = "BTC/USDT",
    timeframe: Annotated[
        str,
        typer.Option(
            "--timeframe", "-t", help="Timeframe for candles (1m, 5m, 1h, 4h, 1d)"
        ),
    ] = "1h",
    start_date: Annotated[
        str,
        typer.Option("--start-date", "-d", help="Start date (YYYY-MM-DD)"),
    ] = "2024-01-01",
    output_dir: Annotated[
        str,
        typer.Option("--output-dir", "-o", help="Output directory"),
    ] = "data/raw/crypto",
    interactive: Annotated[
        bool,
        typer.Option("--interactive", "-i", help="Run in interactive mode"),
    ] = False,
):
    """Download cryptocurrency data from exchanges."""
    # Check dependencies
    download = check_dependencies()

    logger.info("Cryptocurrency Data Downloader (via gym_trading_env)")

    # Available exchanges
    exchanges_info = {
        "bitfinex2": "Bitfinex - High liquidity, good historical data",
        "binance": "Binance - Largest exchange by volume",
        "kraken": "Kraken - Established US-based exchange",
        "coinbasepro": "Coinbase Pro - US-regulated exchange",
    }

    # Interactive mode
    if interactive:
        logger.info("Running in interactive mode")
        logger.info("\nAvailable exchanges:")
        for exch, desc in exchanges_info.items():
            logger.info(f"  {exch:15s} - {desc}")

        # Exchange selection
        typer.echo("\nSelect exchange:")
        typer.echo("  1. bitfinex2 (default)")
        typer.echo("  2. binance")
        typer.echo("  3. kraken")
        typer.echo("  4. coinbasepro")
        exchange_choice = typer.prompt("Choose [1-4, default: 1]", default="1")

        exchange_map = {
            "1": "bitfinex2",
            "2": "binance",
            "3": "kraken",
            "4": "coinbasepro",
        }
        exchange = exchange_map.get(exchange_choice, "bitfinex2")

        # Symbols
        symbols = typer.prompt(
            "Enter trading pair(s) [comma-separated, e.g., BTC/USDT]",
            default="BTC/USDT",
        ).upper()

        # Timeframe
        typer.echo("\nSelect timeframe:")
        typer.echo("  1. 1 hour (1h)")
        typer.echo("  2. 1 day (1d)")
        typer.echo("  3. 5 minutes (5m)")
        typer.echo("  4. 1 minute (1m)")
        typer.echo("  5. 4 hours (4h)")
        timeframe_choice = typer.prompt("Choose [1-5, default: 1]", default="1")

        timeframe_map = {
            "1": "1h",
            "2": "1d",
            "3": "5m",
            "4": "1m",
            "5": "4h",
        }
        timeframe = timeframe_map.get(timeframe_choice, "1h")

        # Date range
        start_date = typer.prompt("Start date (YYYY-MM-DD)", default="2024-01-01")

    # Parse symbols
    symbols_list = [s.strip() for s in symbols.split(",")]
    logger.info(f"Symbols to download: {symbols_list}")

    # Parse start date
    since = parse_date(start_date)

    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_path.absolute()}")

    # Download
    logger.info("Starting download...")
    logger.info(f"Exchange: {exchange}")
    logger.info(f"Symbols: {symbols_list}")
    logger.info(f"Timeframe: {timeframe}")
    logger.info(f"Since: {since.strftime('%Y-%m-%d')}")

    try:
        download(
            exchange_names=[exchange],
            symbols=symbols_list,
            timeframe=timeframe,
            dir=str(output_path),
            since=since,
        )

        logger.info("Download completed successfully")

        # Show downloaded files
        logger.info("\nDownloaded files:")
        for symbol in symbols_list:
            # gym_trading_env saves as: {exchange}-{symbol}-{timeframe}.parquet
            symbol_safe = symbol.replace("/", "")
            filename = f"{exchange}-{symbol_safe}-{timeframe}.parquet"
            file_path = output_path / filename
            if file_path.exists():
                logger.info(f"  {filename}")
            else:
                logger.warning(f"  {filename} (not found)")

        logger.info("\nNext steps:")
        example_symbol = symbols_list[0].replace("/", "")
        example_file = f"{exchange}-{example_symbol}-{timeframe}.parquet"

        typer.echo("\n1. Inspect the data:")
        typer.echo(
            f"   python -c \"import pandas as pd; df = pd.read_parquet('{output_path}/{example_file}'); "
            f"print(df.info()); print(df.head())\""
        )

        typer.echo("\n2. Use in training:")
        typer.echo("   Update your scenario YAML config:")
        typer.echo("   data:")
        typer.echo(f"     data_path: './{output_path}/{example_file}'")
        typer.echo("     download_data: false")

        typer.echo("\n3. Available columns:")
        typer.echo("   OHLCV data: open, high, low, close, volume")

    except Exception as e:
        logger.error(f"Failed to download data: {e}", exc_info=True)
        logger.info("\nTroubleshooting:")
        logger.info("1. Check your internet connection")
        logger.info("2. Verify the symbol exists on the exchange")
        logger.info("3. Try a different exchange or timeframe")
        logger.info("4. Check if gym_trading_env supports this exchange")
        raise typer.Exit(code=1)


@app.command()
def list_exchanges():
    """List available exchanges and their characteristics."""
    exchanges_info = {
        "bitfinex2": "Bitfinex - High liquidity, good historical data",
        "binance": "Binance - Largest exchange by volume",
        "kraken": "Kraken - Established US-based exchange",
        "coinbasepro": "Coinbase Pro - US-regulated exchange",
    }

    typer.echo("Available exchanges:\n")
    for exchange, desc in exchanges_info.items():
        typer.echo(f"  {exchange:15s} - {desc}")


if __name__ == "__main__":
    app()
