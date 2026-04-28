#!/usr/bin/env python
"""Download stock data required by the thesis experiments.

This is a thin wrapper around ``scripts/fetch_stocks.py``. It downloads the
raw Databento files needed by the thesis and then derives the filtered files
that the scenario YAML files reference.

Usage:
    uv run python scripts/download_thesis_stock_data.py
    uv run python scripts/download_thesis_stock_data.py --skip-existing
    uv run python scripts/download_thesis_stock_data.py --force
    uv run python scripts/download_thesis_stock_data.py --dry-run
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = Path("data/raw/stocks")

sys.path.insert(0, str(REPO_ROOT / "src"))

from logger import get_logger, setup_logging

setup_logging(level="INFO")
logger = get_logger(__name__)

app = typer.Typer(help="Download Databento stock data required by the thesis.")


@dataclass(frozen=True)
class ThesisStockDownload:
    symbol: str
    start_date: str
    end_date: str
    dataset: str
    schema: str
    aggregate: bool
    filter_us_hours: bool

    @property
    def raw_filename(self) -> str:
        suffix = f"raw_{self.schema}" if not self.aggregate else "1h"
        return f"{self.symbol}_{self.start_date}_{self.end_date}_{suffix}.parquet"

    @property
    def filtered_filename(self) -> str:
        return f"{Path(self.raw_filename).stem}_us_hours.parquet"


THESIS_DOWNLOADS = [
    ThesisStockDownload(
        symbol="AAPL",
        start_date="2026-02-25",
        end_date="2026-03-03",
        dataset="XNAS.ITCH",
        schema="mbp-10",
        aggregate=False,
        filter_us_hours=True,
    ),
    ThesisStockDownload(
        symbol="MSFT",
        start_date="2026-02-25",
        end_date="2026-03-03",
        dataset="XNAS.ITCH",
        schema="mbp-10",
        aggregate=False,
        filter_us_hours=True,
    ),
    ThesisStockDownload(
        symbol="TSLA",
        start_date="2026-02-25",
        end_date="2026-03-03",
        dataset="XNAS.ITCH",
        schema="mbp-10",
        aggregate=False,
        filter_us_hours=True,
    ),
    ThesisStockDownload(
        symbol="GOOGL",
        start_date="2026-02-25",
        end_date="2026-03-03",
        dataset="XNAS.ITCH",
        schema="mbp-10",
        aggregate=False,
        filter_us_hours=True,
    ),
    ThesisStockDownload(
        symbol="META",
        start_date="2026-02-25",
        end_date="2026-03-03",
        dataset="XNAS.ITCH",
        schema="mbp-10",
        aggregate=False,
        filter_us_hours=True,
    ),
    ThesisStockDownload(
        symbol="AMZN",
        start_date="2026-02-25",
        end_date="2026-03-03",
        dataset="XNAS.ITCH",
        schema="mbp-10",
        aggregate=False,
        filter_us_hours=True,
    ),
    ThesisStockDownload(
        symbol="AVGO",
        start_date="2026-02-25",
        end_date="2026-03-03",
        dataset="XNAS.ITCH",
        schema="mbp-10",
        aggregate=False,
        filter_us_hours=True,
    ),
]


def run_command(command: list[str], *, dry_run: bool) -> None:
    logger.debug("run command cmd=%s", " ".join(command))
    if dry_run:
        return
    subprocess.run(command, cwd=REPO_ROOT, check=True)  # noqa: S603


def fetch_command(job: ThesisStockDownload, output_dir: Path, force: bool) -> list[str]:
    command = [
        "uv", "run", "python", "scripts/fetch_stocks.py", "download-stocks",
        "--symbols", job.symbol,
        "--start-date", job.start_date,
        "--end-date", job.end_date,
        "--dataset", job.dataset,
        "--schema", job.schema,
        "--output-dir", str(output_dir),
        "--sequential",
    ]
    command.append("--aggregate" if job.aggregate else "--raw")
    if force:
        command.append("--force")
    return command


def filter_command(input_file: Path, output_file: Path) -> list[str]:
    return [
        "uv", "run", "python", "scripts/filter_us_hours.py",
        str(input_file),
        str(output_file),
    ]


@app.command()
def main(
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", "-o", help="Output directory for downloaded files."),
    ] = DEFAULT_OUTPUT_DIR,
    skip_existing: Annotated[
        bool,
        typer.Option("--skip-existing", "-s", help="Skip symbols whose filtered *_us_hours.parquet already exists."),
    ] = False,
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Pass --force to fetch_stocks.py, ignoring the download cache."),
    ] = False,
    skip_filter: Annotated[
        bool,
        typer.Option("--skip-filter", help="Only download raw data; do not create *_us_hours.parquet files."),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Log commands without executing them."),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable debug-level logging."),
    ] = False,
) -> None:
    """Download Databento MBP-10 data for all thesis symbols and filter to US market hours."""
    if verbose:
        setup_logging(level="DEBUG")

    if not dry_run and not os.getenv("DATABENTO_API_KEY"):
        logger.error("DATABENTO_API_KEY is not set")
        raise typer.Exit(code=1)

    logger.info("start thesis data download n_symbols=%d dry_run=%s", len(THESIS_DOWNLOADS), dry_run)
    logger.debug("output_dir=%s skip_existing=%s force=%s skip_filter=%s", output_dir, skip_existing, force, skip_filter)

    skipped = 0
    downloaded = 0
    filtered = 0

    for job in THESIS_DOWNLOADS:
        filtered_file = output_dir / job.filtered_filename

        if skip_existing and filtered_file.exists():
            logger.info("skip symbol=%s reason=already exists path=%s", job.symbol, filtered_file.name)
            skipped += 1
            continue

        logger.info("download symbol=%s schema=%s dates=%s to %s", job.symbol, job.schema, job.start_date, job.end_date)
        run_command(fetch_command(job, output_dir, force), dry_run=dry_run)
        downloaded += 1

        if job.filter_us_hours and not skip_filter:
            raw_file = output_dir / job.raw_filename
            logger.info("filter us hours symbol=%s output=%s", job.symbol, filtered_file.name)
            logger.debug("filter input=%s output=%s", raw_file, filtered_file)
            run_command(filter_command(raw_file, filtered_file), dry_run=dry_run)
            filtered += 1

    logger.info("thesis data download complete downloaded=%d filtered=%d skipped=%d", downloaded, filtered, skipped)


if __name__ == "__main__":
    app()
