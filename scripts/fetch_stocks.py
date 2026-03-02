#!/usr/bin/env python
"""
Download stock data from Databento or Polygon.

Modes:
  1. Interactive: python scripts/fetch_stocks.py download-stocks --interactive
  2. CLI: python scripts/fetch_stocks.py download-stocks --symbols AAPL --start-date 2024-01-01
  3. Batch: python scripts/fetch_stocks.py batch --config src/configs/data/batch_download.yaml

Features:
  - Automatic rate limiting (skip re-downloads within 24h)
  - Download tracking (remembers what was downloaded)
  - Parallel downloads (multiple symbols at once)
  - Batch processing from YAML config
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import typer
import yaml
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


def download_single_symbol(
    fetcher,
    tracker,
    symbol: str,
    start_date: str,
    end_date: str,
    dataset: str,
    schema: str,
    timeframe: str | None,
    aggregate: bool,
    source: str,
    force: bool = False,
) -> dict:
    """Download data for a single symbol with rate limiting.

    Args:
        fetcher: StockDataFetcher instance
        tracker: DownloadTracker instance
        symbol: Stock symbol
        start_date: Start date
        end_date: End date
        dataset: Dataset identifier
        schema: Data schema
        timeframe: Timeframe for aggregation
        aggregate: Whether to aggregate
        source: Data source
        force: Force download even if within rate limit

    Returns:
        Dict with download status and info
    """
    # Check if should download
    should_download, reason = tracker.should_download(
        symbols=[symbol],
        start_date=start_date,
        end_date=end_date,
        source=source,
        dataset=dataset,
        schema=schema,
        timeframe=timeframe,
        aggregate=aggregate,
    )

    if not should_download and not force:
        logger.info(f"[{symbol}] SKIPPED: {reason}")
        return {
            "symbol": symbol,
            "status": "skipped",
            "reason": reason,
            "rows": 0,
        }

    # Download
    try:
        logger.info(f"[{symbol}] Downloading...")
        df = fetcher.fetch_stock_data(
            symbols=[symbol],
            start_date=start_date,
            end_date=end_date,
            source=source,
            dataset=dataset,
            schema=schema,
            timeframe=timeframe if aggregate else None,
            aggregate=aggregate,
            save_to_file=True,
        )

        # Determine output file
        if aggregate and timeframe:
            file_suffix = timeframe
        else:
            file_suffix = f"raw_{schema}"

        output_file = f"{symbol}_{start_date}_{end_date}_{file_suffix}.parquet"
        output_path = str(Path(fetcher.output_dir) / output_file)

        # Record download
        tracker.record_download(
            symbols=[symbol],
            start_date=start_date,
            end_date=end_date,
            source=source,
            output_files=[output_path],
            dataset=dataset,
            schema=schema,
            timeframe=timeframe,
            aggregate=aggregate,
            rows_downloaded=len(df),
        )

        logger.info(f"[{symbol}] SUCCESS: {len(df)} rows -> {output_file}")
        return {
            "symbol": symbol,
            "status": "success",
            "rows": len(df),
            "file": output_file,
        }

    except Exception as e:
        logger.error(f"[{symbol}] FAILED: {e}")
        return {
            "symbol": symbol,
            "status": "failed",
            "error": str(e),
            "rows": 0,
        }


def download_symbols_parallel(
    fetcher,
    tracker,
    symbols: list[str],
    start_date: str,
    end_date: str,
    dataset: str,
    schema: str,
    timeframe: str | None,
    aggregate: bool,
    source: str,
    max_workers: int = 4,
    force: bool = False,
) -> list[dict]:
    """Download multiple symbols in parallel.

    Args:
        fetcher: StockDataFetcher instance
        tracker: DownloadTracker instance
        symbols: List of stock symbols
        start_date: Start date
        end_date: End date
        dataset: Dataset identifier
        schema: Data schema
        timeframe: Timeframe for aggregation
        aggregate: Whether to aggregate
        source: Data source
        max_workers: Number of parallel workers
        force: Force download even if within rate limit

    Returns:
        List of download results
    """
    results = []

    logger.info(f"Starting parallel download of {len(symbols)} symbols (workers={max_workers})")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all downloads
        future_to_symbol = {
            executor.submit(
                download_single_symbol,
                fetcher,
                tracker,
                symbol,
                start_date,
                end_date,
                dataset,
                schema,
                timeframe,
                aggregate,
                source,
                force,
            ): symbol
            for symbol in symbols
        }

        # Collect results as they complete
        for future in as_completed(future_to_symbol):
            result = future.result()
            results.append(result)

    return results


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
    parallel: Annotated[
        bool,
        typer.Option("--parallel/--sequential", help="Download symbols in parallel"),
    ] = True,
    max_workers: Annotated[
        int,
        typer.Option("--max-workers", help="Number of parallel workers"),
    ] = 4,
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Force download even if within rate limit"),
    ] = False,
    interactive: Annotated[
        bool,
        typer.Option("--interactive", "-i", help="Run in interactive mode"),
    ] = False,
):
    """Download stock data from Databento with rate limiting and parallel support."""
    # Check API key
    check_api_key()

    logger.info("Stock Data Downloader (Databento)")

    from trading_rl.data_fetchers import StockDataFetcher
    from trading_rl.data_fetchers.download_tracker import DownloadTracker

    # Initialize fetcher and tracker
    fetcher = StockDataFetcher(output_dir=output_dir, log_level="INFO")
    tracker = DownloadTracker(cache_dir="data/.download_cache", rate_limit_hours=24)

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
    symbols_list = [s.strip().upper() for s in symbols.split(",")]
    logger.info(f"Symbols to download: {symbols_list}")

    # Download
    logger.info("Starting download...")
    logger.info(f"Symbols: {symbols_list}")
    logger.info(f"Dates: {start_date} to {end_date}")
    logger.info(f"Dataset: {dataset}")
    logger.info(f"Schema: {schema}")
    logger.info(f"Parallel: {parallel} (workers={max_workers if parallel else 1})")
    logger.info(f"Force: {force}")
    if aggregate:
        logger.info(f"Format: Aggregated OHLCV ({timeframe} bars)")
    else:
        logger.info(f"Format: Raw {schema} data (no aggregation)")

    try:
        if parallel and len(symbols_list) > 1:
            # Parallel download
            results = download_symbols_parallel(
                fetcher,
                tracker,
                symbols_list,
                start_date,
                end_date,
                dataset,
                schema,
                timeframe,
                aggregate,
                "databento",
                max_workers=max_workers,
                force=force,
            )

            # Summary
            success_count = sum(1 for r in results if r["status"] == "success")
            skip_count = sum(1 for r in results if r["status"] == "skipped")
            fail_count = sum(1 for r in results if r["status"] == "failed")
            total_rows = sum(r["rows"] for r in results)

            logger.info("\n" + "="*60)
            logger.info("DOWNLOAD SUMMARY")
            logger.info("="*60)
            logger.info(f"Total symbols: {len(symbols_list)}")
            logger.info(f"Success: {success_count}")
            logger.info(f"Skipped: {skip_count}")
            logger.info(f"Failed: {fail_count}")
            logger.info(f"Total rows: {total_rows}")

            # List files
            logger.info("\nDownloaded files:")
            for result in results:
                if result["status"] == "success":
                    logger.info(f"  ✓ {result['file']} ({result['rows']} rows)")
                elif result["status"] == "skipped":
                    logger.info(f"  ⊘ {result['symbol']} - {result['reason']}")
                else:
                    logger.info(f"  ✗ {result['symbol']} - {result.get('error', 'Unknown error')}")

        else:
            # Sequential download
            results = []
            for symbol in symbols_list:
                result = download_single_symbol(
                    fetcher,
                    tracker,
                    symbol,
                    start_date,
                    end_date,
                    dataset,
                    schema,
                    timeframe,
                    aggregate,
                    "databento",
                    force=force,
                )
                results.append(result)

            # Summary for sequential
            success_count = sum(1 for r in results if r["status"] == "success")
            skip_count = sum(1 for r in results if r["status"] == "skipped")
            total_rows = sum(r["rows"] for r in results)

            logger.info("\n" + "="*60)
            logger.info(f"Downloaded {success_count}/{len(symbols_list)} symbols ({skip_count} skipped)")
            logger.info(f"Total rows: {total_rows}")

        # Next steps
        logger.info("\nNext steps:")
        logger.info(f"1. Files saved to: {output_dir}/")
        logger.info("2. View download history:")
        logger.info("     python scripts/fetch_stocks.py download-history")
        if aggregate:
            logger.info("3. Use in training config:")
            logger.info(f"     data_path: '{output_dir}/SYMBOL_START_END_{timeframe}.parquet'")

    except Exception as e:
        logger.error(f"Failed to download: {e}")
        raise typer.Exit(code=1)


@app.command()
def batch(
    config_file: Annotated[
        str,
        typer.Option("--config", "-c", help="Path to batch download YAML config"),
    ],
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Force download even if within rate limit"),
    ] = False,
):
    """Download multiple symbols/datasets from YAML config file.

    Example config: src/configs/data/batch_download_example.yaml
    """
    check_api_key()

    from trading_rl.data_fetchers import StockDataFetcher
    from trading_rl.data_fetchers.download_tracker import DownloadTracker

    logger.info(f"Loading batch config from: {config_file}")

    # Load config
    config_path = Path(config_file)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_file}")
        raise typer.Exit(code=1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Global settings
    settings = config.get("settings", {})
    output_dir = settings.get("output_dir", "data/raw/stocks")
    source = settings.get("source", "databento")
    rate_limit_hours = settings.get("rate_limit_hours", 24)
    parallel_default = settings.get("parallel_downloads", True)
    max_workers = settings.get("max_workers", 4)

    # Initialize
    fetcher = StockDataFetcher(output_dir=output_dir, log_level="INFO")
    tracker = DownloadTracker(cache_dir="data/.download_cache", rate_limit_hours=rate_limit_hours)

    # Get download jobs
    downloads = config.get("downloads", [])
    if not downloads:
        logger.error("No download jobs found in config")
        raise typer.Exit(code=1)

    logger.info(f"\nFound {len(downloads)} download jobs")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Rate limit: {rate_limit_hours}h")
    logger.info(f"Parallel: {parallel_default} (workers={max_workers})")
    logger.info("")

    all_results = []

    for i, job in enumerate(downloads, 1):
        job_name = job.get("name", f"job_{i}")
        symbols = job.get("symbols", [])
        start_date = job.get("start_date")
        end_date = job.get("end_date")
        dataset = job.get("dataset", "XNAS.ITCH")
        schema = job.get("schema", "trades")
        timeframe = job.get("timeframe", "1h")
        aggregate = job.get("aggregate", True)
        job_parallel = job.get("parallel_downloads", parallel_default)

        logger.info("="*60)
        logger.info(f"Job {i}/{len(downloads)}: {job_name}")
        logger.info(f"Symbols: {', '.join(symbols)}")
        logger.info(f"Dates: {start_date} to {end_date}")
        logger.info("="*60)

        if job_parallel and len(symbols) > 1:
            results = download_symbols_parallel(
                fetcher,
                tracker,
                symbols,
                start_date,
                end_date,
                dataset,
                schema,
                timeframe,
                aggregate,
                source,
                max_workers=max_workers,
                force=force,
            )
        else:
            results = []
            for symbol in symbols:
                result = download_single_symbol(
                    fetcher,
                    tracker,
                    symbol,
                    start_date,
                    end_date,
                    dataset,
                    schema,
                    timeframe,
                    aggregate,
                    source,
                    force=force,
                )
                results.append(result)

        all_results.extend(results)

        # Job summary
        success = sum(1 for r in results if r["status"] == "success")
        skipped = sum(1 for r in results if r["status"] == "skipped")
        failed = sum(1 for r in results if r["status"] == "failed")

        logger.info(f"\nJob complete: {success} success, {skipped} skipped, {failed} failed\n")

    # Final summary
    logger.info("\n" + "="*60)
    logger.info("BATCH DOWNLOAD COMPLETE")
    logger.info("="*60)
    total_success = sum(1 for r in all_results if r["status"] == "success")
    total_skipped = sum(1 for r in all_results if r["status"] == "skipped")
    total_failed = sum(1 for r in all_results if r["status"] == "failed")
    total_rows = sum(r["rows"] for r in all_results)

    logger.info(f"Total downloads: {len(all_results)}")
    logger.info(f"Success: {total_success}")
    logger.info(f"Skipped: {total_skipped}")
    logger.info(f"Failed: {total_failed}")
    logger.info(f"Total rows: {total_rows}")


@app.command()
def download_history(
    hours: Annotated[
        int,
        typer.Option("--hours", "-h", help="Show downloads from last N hours"),
    ] = 24,
):
    """Show recent download history and cache statistics."""
    from trading_rl.data_fetchers.download_tracker import DownloadTracker

    tracker = DownloadTracker(cache_dir="data/.download_cache")

    # Get stats
    stats = tracker.get_stats()

    logger.info("="*60)
    logger.info("DOWNLOAD CACHE STATISTICS")
    logger.info("="*60)
    logger.info(f"Total downloads: {stats['total_downloads']}")
    logger.info(f"Total symbols: {stats['total_symbols']}")
    logger.info(f"Total files: {stats['total_files']}")
    logger.info(f"Total size: {stats['total_size_mb']:.2f} MB")

    if stats["oldest_download"]:
        logger.info(f"Oldest download: {stats['oldest_download']}")
    if stats["newest_download"]:
        logger.info(f"Newest download: {stats['newest_download']}")

    # Recent downloads
    recent = tracker.get_recent_downloads(hours=hours)

    if recent:
        logger.info(f"\nRecent downloads (last {hours}h):")
        logger.info("-"*60)

        for download in recent:
            symbols_str = ", ".join(download["symbols"])
            logger.info(f"\n{symbols_str}")
            logger.info(f"  Date range: {download['start_date']} to {download['end_date']}")
            logger.info(f"  Source: {download['source']} / {download['dataset']}")
            logger.info(f"  Downloaded: {download['hours_ago']:.1f}h ago")
            logger.info(f"  Rows: {download['rows_downloaded']}")
            logger.info(f"  Files: {len(download['output_files'])}")
    else:
        logger.info(f"\nNo downloads in the last {hours}h")


@app.command()
def clear_cache():
    """Clear download cache (allows re-downloading everything)."""
    from trading_rl.data_fetchers.download_tracker import DownloadTracker

    tracker = DownloadTracker(cache_dir="data/.download_cache")

    confirm = typer.confirm("Are you sure you want to clear the download cache?")
    if confirm:
        tracker.clear_cache()
        logger.info("Download cache cleared")
    else:
        logger.info("Cancelled")


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
