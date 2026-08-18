#!/usr/bin/env python3
"""Promote `peek dataset --export` output into a committed thesis snapshot.

`peek dataset --export` writes to reports/peek/{scenario_name}/, which is
gitignored local scratch (like eval_results/ and mlruns/). Chapter 5 of the
thesis reads its data-prep tables directly from that gitignored directory,
so a CI checkout — which never runs `peek dataset` — always renders the
"not found" fallback for those tables.

This script copies the small derived summary files (a few KB each: file
inventory, correlations, split boundaries, feature stats) into
thesis/qmd/results/{experiment_name}/peek/, the directory this repo already
treats as a source of committed, reproducible snapshot data for CI Quarto
renders (see export_eval_to_thesis.py for the same pattern applied to
evaluate CLI output).

Usage:
    uv run python src/cli.py peek dataset \\
        -s pooled/td3_hft_lob_state_space_pooled_streaming_selected --export
    uv run python src/cli.py peek dataset \\
        -s pooled/td3_hft_lob_state_space_pooled_streaming_selected --corr --export
    uv run python src/cli.py peek dataset \\
        -c src/configs/scenarios/pooled/td3_hft_lob_state_space_pooled_streaming_selected/ \\
        --skip 500 --export
    uv run python scripts/export_peek_to_thesis.py \\
        --scenario pooled/td3_hft_lob_state_space_pooled_streaming_selected

    git add thesis/qmd/results/pooled_td3_hft_lob_state_space_pooled_streaming_selected/peek
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from logger import get_logger, setup_logging

logger = get_logger(__name__)

_SNAPSHOT_FILES = (
    "raw_file_inventory.json",
    "correlations.csv",
    "splits.json",
    "feature_stats.csv",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Copy peek dataset --export output into a committed thesis snapshot.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--scenario",
        required=True,
        metavar="NAME",
        help="Scenario passed to `peek dataset -s`, e.g. "
        "pooled/td3_hft_lob_state_space_pooled_streaming_selected.",
    )
    p.add_argument(
        "--peek-dir",
        type=Path,
        metavar="DIR",
        help="Override the reports/peek/{scenario_name} source directory.",
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

    scenario_name = Path(args.scenario).name
    experiment_name = args.scenario.replace("/", "_")

    source_dir = args.peek_dir or (repo_root / "reports" / "peek" / scenario_name)
    thesis_results_root = args.thesis_results_root or (
        repo_root / "thesis" / "qmd" / "results"
    )
    dest_dir = thesis_results_root / experiment_name / "peek"

    if not source_dir.exists():
        logger.error(
            "source directory not found: {} — run `peek dataset -s {} --export` "
            "(and `--corr`) first",
            source_dir,
            args.scenario,
        )
        return 1

    dest_dir.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    missing: list[str] = []
    for name in _SNAPSHOT_FILES:
        src = source_dir / name
        if not src.exists():
            missing.append(name)
            continue
        shutil.copy2(src, dest_dir / name)
        copied.append(name)

    logger.info("copied {} file(s) to {}", len(copied), dest_dir)
    for name in copied:
        logger.info("  {}", name)
    if missing:
        logger.warning(
            "missing from {} (not copied): {}", source_dir, ", ".join(missing)
        )

    return 0 if copied else 1


if __name__ == "__main__":
    raise SystemExit(main())
