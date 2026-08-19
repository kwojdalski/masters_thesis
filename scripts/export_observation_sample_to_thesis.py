#!/usr/bin/env python3
"""Promote a trimmed `evaluate` observation-sample parquet into a committed thesis snapshot.

`evaluate` writes the full observation sample (thousands of rows) to
eval_results/evaluation_data/ (or eval_results/per_symbol/{symbol}/evaluation_data/
with --per-symbol), which is gitignored local scratch. Chapter 5's transformed-feature
table reads that file via find_observation_sample(), so a CI checkout — which never
runs `evaluate` — always renders the "not found" fallback for that table.

This script trims the source parquet down to a small row count (enough for
lob_events_table()'s 12-event window search) and copies it into
thesis/qmd/results/observation_samples/, the committed-snapshot fallback
find_observation_sample() checks last (see export_peek_to_thesis.py and
export_eval_to_thesis.py for the same pattern applied to other CLI output).

Usage:
    uv run python src/cli.py evaluate \\
        -c pooled/random_hft_lob_state_space_pooled_streaming_selected_dsr \\
        --split test --output-dir eval_results --per-symbol \\
        --config-override 'data.data_paths=[./data/raw/stocks/daily/AAPL/AAPL_2026-02-25_raw_mbp-10_us_hours.parquet]' \\
        --config-override 'data.val_data_paths=[./data/raw/stocks/daily/AAPL/AAPL_2026-03-02_raw_mbp-10_us_hours.parquet]'

    uv run python scripts/export_observation_sample_to_thesis.py \\
        --source eval_results/per_symbol/AAPL/evaluation_data/test_AAPL_observations_head_5000.parquet \\
        --split test --symbol AAPL

    git add thesis/qmd/results/observation_samples/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from logger import get_logger, setup_logging

logger = get_logger(__name__)

_DEFAULT_MAX_ROWS = 500


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Trim and copy an evaluate observation-sample parquet into a committed thesis snapshot.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--source",
        required=True,
        type=Path,
        metavar="PATH",
        help="Path to the source *_observations_head_*.parquet produced by evaluate.",
    )
    p.add_argument(
        "--split",
        required=True,
        metavar="SPLIT",
        help="Split name used in the destination filename, e.g. test.",
    )
    p.add_argument(
        "--symbol",
        metavar="SYMBOL",
        help="Symbol used in the destination filename, e.g. AAPL.",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=_DEFAULT_MAX_ROWS,
        help=f"Number of leading rows to keep (default: {_DEFAULT_MAX_ROWS}).",
    )
    p.add_argument(
        "--thesis-results-root",
        type=Path,
        metavar="DIR",
        help="Override the thesis/qmd/results root directory.",
    )
    return p.parse_args()


def main() -> int:
    setup_logging(level="INFO")
    args = _parse_args()
    repo_root = _repo_root()

    if not args.source.exists():
        logger.error(
            "source file not found: {} — run `evaluate` first to generate it",
            args.source,
        )
        return 1

    thesis_results_root = args.thesis_results_root or (
        repo_root / "thesis" / "qmd" / "results"
    )
    dest_dir = thesis_results_root / "observation_samples"
    dest_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.source)
    trimmed = df.head(args.max_rows)

    name_parts = [args.split]
    if args.symbol:
        name_parts.append(args.symbol)
    name_parts.append(f"observations_head_{len(trimmed)}")
    dest_path = dest_dir / (("_".join(name_parts)) + ".parquet")

    trimmed.to_parquet(dest_path)
    logger.info(
        "wrote {} rows ({} -> {}) to {}",
        len(trimmed),
        len(df),
        len(trimmed),
        dest_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
