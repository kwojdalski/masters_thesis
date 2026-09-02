#!/usr/bin/env python3
"""Export downsampled evaluation rollout data into a thesis result snapshot.

The rollout parquets that Chapter 6's equity, reward and position figures are
drawn from live in the MLflow artifact store. Both ``mlflow.db`` and ``mlruns/``
are gitignored, so CI has neither and the figures fall back to a
"plot data not available" notice in the published PDF while rendering fine
locally.

This writes a committed copy under
``thesis/qmd/results/{experiment}/evaluation_plots/{split}/`` so the render
works with no MLflow at all. The full test-split rollout is 3.8 MB across four
frames, which is more than belongs in the repository, so the series are
downsampled on the way out. They are line charts over ~51,000 points per
series that are strided again before drawing, so a fraction of the points is
visually identical; the saving is roughly 25x.

Downsampling keeps the first and last row of every ``Run`` group. The final
point of the equity curve is the total return the results tables report, and a
plain stride would drop it whenever the series length is not a multiple of the
step.

Usage:
    # Pick the run automatically: the finished run of the experiment whose
    # plots come from the given checkpoint and cover the most observations.
    uv run python scripts/export_rollout_plots_to_thesis.py \\
        --experiment-name pooled_td3_hft_lob_state_space_pooled_streaming_selected_dsr \\
        --checkpoint-step 3000000

    # Or name the run directly.
    uv run python scripts/export_rollout_plots_to_thesis.py \\
        --experiment-name pooled_td3_hft_lob_state_space_pooled_streaming_selected_dsr \\
        --run-id d7c200923b694f1c81e0ef7186ab0c00
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "thesis" / "qmd" / "src"))

FRAMES = ("rewards", "actions", "actions_ma", "equity")
DEFAULT_STRIDE = 25


def _downsample(df: pd.DataFrame, stride: int) -> pd.DataFrame:
    """Stride each Run series, always keeping its first and last row."""
    if stride <= 1 or df.empty:
        return df
    if "Run" not in df.columns:
        keep = set(range(0, len(df), stride)) | {0, len(df) - 1}
        return df.iloc[sorted(keep)].reset_index(drop=True)

    out = []
    for _, group in df.groupby("Run", sort=False):
        keep = set(range(0, len(group), stride)) | {0, len(group) - 1}
        out.append(group.iloc[sorted(keep)])
    return pd.concat(out).reset_index(drop=True)


def _resolve_run(experiment_name: str, checkpoint_step: int | None) -> str | None:
    """Return the run id whose plots best match the requested checkpoint."""
    from thesis_mlflow_results import find_plot_run_in_experiment

    _, provenance = find_plot_run_in_experiment(
        experiment_name, require_checkpoint_step=checkpoint_step
    )
    return provenance["run_id"] if provenance else None


def _artifact_plot_dir(run_id: str) -> Path | None:
    for candidate in (REPO_ROOT / "mlruns").glob(f"*/{run_id}/artifacts"):
        plot_dir = candidate / "evaluation_plots"
        if plot_dir.is_dir():
            return plot_dir
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment-name", required=True)
    ap.add_argument("--run-id", default=None, help="skip lookup and use this run")
    ap.add_argument("--checkpoint-step", type=int, default=None)
    ap.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    ap.add_argument(
        "--splits", default="test", help="comma-separated split subdirectories"
    )
    args = ap.parse_args()

    run_id = args.run_id or _resolve_run(args.experiment_name, args.checkpoint_step)
    if run_id is None:
        print("no finished run with matching rollout plots found", file=sys.stderr)
        return 1

    plot_dir = _artifact_plot_dir(run_id)
    if plot_dir is None:
        print(f"no evaluation_plots directory for run {run_id}", file=sys.stderr)
        return 1

    dest_root = (
        REPO_ROOT
        / "thesis"
        / "qmd"
        / "results"
        / args.experiment_name
        / "evaluation_plots"
    )
    written = 0
    total_in = total_out = 0

    for split in (s.strip() for s in args.splits.split(",") if s.strip()):
        src = plot_dir / split
        if not src.is_dir():
            print(f"  {split}: absent in run {run_id[:10]}, skipped")
            continue
        dest = dest_root / split
        dest.mkdir(parents=True, exist_ok=True)

        for frame in FRAMES:
            hits = sorted(src.glob(f"*_{frame}_data.parquet"))
            if not hits:
                continue
            df = pd.read_parquet(hits[-1])
            small = _downsample(df, args.stride)
            out = dest / hits[-1].name
            small.to_parquet(out, index=False)
            total_in += len(df)
            total_out += len(small)
            written += 1
            print(
                f"  {split}/{frame}: {len(df):,} -> {len(small):,} rows "
                f"({out.stat().st_size / 1e3:.0f} kB)"
            )

        # The figures read their axis labels, symbol list and training-step
        # count from this sidecar, so it has to travel with the parquets.
        for meta in src.glob("*_plot_meta.json"):
            shutil.copy2(meta, dest / meta.name)

    if not written:
        print("nothing exported", file=sys.stderr)
        return 1

    # Record which run this came from: the figures caption themselves with it,
    # and without the MLflow database there is no other way to tell.
    provenance = {"run_id": run_id, "stride": args.stride, "source_rows": total_in}
    (dest_root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    print(
        f"exported {written} frames from run {run_id[:10]} to {dest_root.relative_to(REPO_ROOT)} "
        f"({total_in:,} -> {total_out:,} rows)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
