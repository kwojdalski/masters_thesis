#!/usr/bin/env python3
"""Export thesis-friendly result artifacts for a tracked experiment.

This creates a stable JSON/Parquet/PNG snapshot under ``thesis/qmd/results`` so
Quarto chapters can render without querying the live MLflow SQLite database.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export thesis result snapshot from MLflow artifacts.")
    parser.add_argument(
        "--experiment",
        required=True,
        help="MLflow experiment name (e.g. btc_td3_tradingenv_reduced_features)",
    )
    parser.add_argument(
        "--output-root",
        default="thesis/qmd/results",
        help="Directory where experiment snapshots are stored (default: thesis/qmd/results)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repo_root = _repo_root()
    thesis_src = repo_root / "thesis" / "qmd" / "src"
    if str(thesis_src) not in sys.path:
        sys.path.insert(0, str(thesis_src))

    from thesis_mlflow_results import export_experiment_snapshot  # local import after sys.path setup

    output_root = (repo_root / args.output_root).resolve()
    export_dir = export_experiment_snapshot(args.experiment, output_root=output_root)

    print(f"Exported thesis snapshot for '{args.experiment}'")
    print(f"Location: {export_dir}")
    print("QMD chapters will prefer this snapshot over live MLflow DB queries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
