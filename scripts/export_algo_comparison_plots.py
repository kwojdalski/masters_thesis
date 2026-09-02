#!/usr/bin/env python3
"""Export a downsampled multi-algorithm rollout comparison for the thesis.

The rollout figures in Chapter 6 originally showed one policy against the
passive benchmarks, so they illustrated the TD3 agent's behaviour but supported
no comparison between the learners the algorithm-comparison table reports. This
builds a single set of frames in which the TD3, DDPG, PPO and random policies
appear as separate series, so the figures answer the same question the table
does.

The comparison is only meaningful if the arms are matched, so this refuses to
emit anything unless every algorithm contributes a rollout over the same split,
the same symbol and the same number of observations, from the same checkpoint
step. Random is exempt from the checkpoint check, having no trained weights.

Series are relabelled from the per-run "Deterministic" to the algorithm name
before concatenation. Passive benchmarks (Buy-and-Hold, TWAP, VWAP) are taken
from the reference algorithm alone, since every run scores the identical
market data and repeating them once per algorithm would draw four identical
lines on top of each other.

Downsampling follows export_rollout_plots_to_thesis.py: stride each series,
always keeping its first and last row, because the final point of an equity
curve is the total return the results tables report.

Usage:
    uv run python scripts/export_algo_comparison_plots.py --checkpoint-step 3000000
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
# Series a run emits for the market itself rather than for its own policy.
PASSIVE = ("Buy-and-Hold", "TWAP", "VWAP")
# The per-run label for the evaluated policy, replaced by the algorithm name.
POLICY_LABEL = "Deterministic"

ALGORITHMS = ("TD3", "DDPG", "PPO", "Random")
REFERENCE = "TD3"
DEST_EXPERIMENT = "pooled_td3_hft_lob_state_space_pooled_streaming_selected_dsr"


def _downsample(df: pd.DataFrame, stride: int) -> pd.DataFrame:
    """Stride each Run series, always keeping its first and last row."""
    if stride <= 1 or df.empty or "Run" not in df.columns:
        return df
    out = []
    for _, group in df.groupby("Run", sort=False):
        keep = set(range(0, len(group), stride)) | {0, len(group) - 1}
        out.append(group.iloc[sorted(keep)])
    return pd.concat(out).reset_index(drop=True)


def _find_rollout(algo: str, split: str, checkpoint_step: int | None):
    """Return (plot_dir, meta) for the algorithm's matching rollout, or None."""
    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{REPO_ROOT / 'mlflow.db'}")
    client = mlflow.tracking.MlflowClient()
    name = f"pooled_{algo.lower()}_hft_lob_state_space_pooled_streaming_selected_dsr"
    experiment = client.get_experiment_by_name(name)
    if experiment is None:
        return None

    best = None
    for run in client.search_runs([experiment.experiment_id], max_results=200):
        split_dir = Path(run.info.artifact_uri) / "evaluation_plots" / split
        if not split_dir.is_dir():
            continue
        for meta_path in sorted(split_dir.glob("*_plot_meta.json")):
            meta = json.loads(meta_path.read_text())
            steps = meta.get("training_steps")
            # Random has no trained weights, so its step count is 0 or absent
            # and cannot be matched against the others.
            if (
                checkpoint_step is not None
                and algo != "Random"
                and steps != checkpoint_step
            ):
                continue
            if best is None or (meta.get("n_obs") or 0) > (best[1].get("n_obs") or 0):
                best = (split_dir, meta, run.info.run_id)
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint-step", type=int, default=3_000_000)
    ap.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    found: dict[str, tuple[Path, dict, str]] = {}
    for algo in ALGORITHMS:
        hit = _find_rollout(algo, args.split, args.checkpoint_step)
        if hit is None:
            print(f"  {algo}: no rollout at the requested checkpoint", file=sys.stderr)
            return 1
        found[algo] = hit
        print(
            f"  {algo:<7} run={hit[2][:10]} steps={hit[1].get('training_steps')} "
            f"n_obs={hit[1].get('n_obs')} symbols={hit[1].get('symbols')}"
        )

    # Refuse to draw a comparison across mismatched rollouts: different symbols
    # or window lengths would put visually comparable lines on incomparable
    # data, which is worse than having no comparison at all.
    obs = {a: h[1].get("n_obs") for a, h in found.items()}
    syms = {a: tuple(h[1].get("symbols") or []) for a, h in found.items()}
    if len(set(obs.values())) != 1 or len(set(syms.values())) != 1:
        print(
            f"\nrollouts are not comparable: n_obs={obs} symbols={syms}",
            file=sys.stderr,
        )
        return 1
    # Every value is identical here -- the comparability check above returned
    # otherwise -- so take the first of each.
    matched_syms = next(iter(syms.values()))
    matched_obs = next(iter(obs.values()))
    print(f"\n  matched: {matched_syms} over {matched_obs:,} steps")

    dest = (
        REPO_ROOT
        / "thesis"
        / "qmd"
        / "results"
        / DEST_EXPERIMENT
        / "evaluation_plots_comparison"
        / args.split
    )
    dest.mkdir(parents=True, exist_ok=True)

    written = 0
    for frame in FRAMES:
        parts: list[pd.DataFrame] = []
        stem = None
        for algo in ALGORITHMS:
            split_dir = found[algo][0]
            hits = sorted(split_dir.glob(f"*_{frame}_data.parquet"))
            if not hits:
                continue
            if algo == REFERENCE:
                stem = hits[-1].name
            df = pd.read_parquet(hits[-1])
            if "Run" not in df.columns:
                continue
            policy = df[df["Run"] == POLICY_LABEL].copy()
            policy["Run"] = algo
            parts.append(policy)
            # Passive benchmarks are identical across algorithms; take one copy.
            if algo == REFERENCE:
                passive = df[df["Run"].isin(PASSIVE)]
                if not passive.empty:
                    parts.append(passive)
        if not parts or stem is None:
            continue
        merged = _downsample(pd.concat(parts, ignore_index=True), args.stride)
        out = dest / stem
        merged.to_parquet(out, index=False)
        written += 1
        series = ", ".join(merged["Run"].unique())
        print(
            f"  {frame:<11} {len(merged):>7,} rows  [{series}]  {out.stat().st_size / 1e3:.0f} kB"
        )

    # The figures read axis labels and the symbol list from this sidecar.
    for meta in found[REFERENCE][0].glob("*_plot_meta.json"):
        shutil.copy2(meta, dest / meta.name)

    (dest.parent / "provenance.json").write_text(
        json.dumps(
            {
                "runs": {a: found[a][2] for a in ALGORITHMS},
                "checkpoint_step": args.checkpoint_step,
                "split": args.split,
                "stride": args.stride,
                "n_obs": matched_obs,
                "symbols": list(matched_syms),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nexported {written} frames to {dest.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
