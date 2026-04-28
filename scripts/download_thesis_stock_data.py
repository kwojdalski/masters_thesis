#!/usr/bin/env python
"""Download stock data required by the thesis experiments.

This is a thin wrapper around ``scripts/fetch_stocks.py``. It downloads the
raw Databento files needed by the thesis and then derives the filtered files
that the scenario YAML files reference.

Usage:
    uv run python scripts/download_thesis_stock_data.py
    uv run python scripts/download_thesis_stock_data.py --force
    uv run python scripts/download_thesis_stock_data.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = Path("data/raw/stocks")


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
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the Databento stock data required by the thesis."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for downloaded files (default: data/raw/stocks).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Pass --force to fetch_stocks.py and ignore the download cache.",
    )
    parser.add_argument(
        "--skip-filter",
        action="store_true",
        help="Only download raw data; do not create *_us_hours.parquet files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    return parser.parse_args()


def run_command(command: list[str], *, dry_run: bool) -> None:
    printable = " ".join(command)
    print(printable)
    if dry_run:
        return
    subprocess.run(command, cwd=REPO_ROOT, check=True)  # noqa: S603


def ensure_api_key_present(dry_run: bool) -> None:
    if dry_run:
        return
    if not os.getenv("DATABENTO_API_KEY"):
        raise RuntimeError(
            "DATABENTO_API_KEY is not set. Export it before downloading thesis data."
        )


def fetch_command(job: ThesisStockDownload, output_dir: Path, force: bool) -> list[str]:
    command = [
        "uv",
        "run",
        "python",
        "scripts/fetch_stocks.py",
        "download-stocks",
        "--symbols",
        job.symbol,
        "--start-date",
        job.start_date,
        "--end-date",
        job.end_date,
        "--dataset",
        job.dataset,
        "--schema",
        job.schema,
        "--output-dir",
        str(output_dir),
        "--sequential",
    ]
    if job.aggregate:
        command.append("--aggregate")
    else:
        command.append("--raw")
    if force:
        command.append("--force")
    return command


def filter_command(input_file: Path, output_file: Path) -> list[str]:
    return [
        "uv",
        "run",
        "python",
        "scripts/filter_us_hours.py",
        str(input_file),
        str(output_file),
    ]


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    ensure_api_key_present(args.dry_run)

    for job in THESIS_DOWNLOADS:
        print(
            f"\nDownloading {job.symbol} {job.schema} data "
            f"from {job.start_date} to {job.end_date}"
        )
        run_command(fetch_command(job, output_dir, args.force), dry_run=args.dry_run)

        if job.filter_us_hours and not args.skip_filter:
            raw_file = output_dir / job.raw_filename
            filtered_file = output_dir / job.filtered_filename
            print(f"\nFiltering US market hours: {filtered_file}")
            run_command(filter_command(raw_file, filtered_file), dry_run=args.dry_run)

    print("\nThesis stock data workflow complete.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
