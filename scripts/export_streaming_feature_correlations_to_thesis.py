#!/usr/bin/env python3
"""Recompute the feature / next-return correlation table over the streamed training set.

Companion to ``export_streaming_feature_stats_to_thesis.py``. The appendix
correlation table (``@tbl-feature-correlations``) previously read a
``correlations.csv`` produced by ``peek dataset --corr``, which correlates
``dataset.train_df`` feature columns against a log-return series taken from a
single raw file. For this streaming scenario that is AAPL alone, one session,
about the first seven minutes after the open -- 50,000 rows -- and it covers only
the ten in-model features.

This script computes Pearson and Spearman correlations between every engineered
``feature_hft_*`` column and the one-step-ahead log return of the mid-price,
over all memmap shards (six instruments x three sessions), dropping the first
``--skip`` events of each shard and the last (no next step). Feature value at
event ``t`` is paired with ``log(mid[t+1] / mid[t])``.

Usage:
    uv run python scripts/export_streaming_feature_correlations_to_thesis.py \\
        --scenario pooled/td3_hft_lob_state_space_pooled_streaming_selected

    git add thesis/qmd/results/pooled_td3_hft_lob_state_space_pooled_streaming_selected/peek/correlations.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--scenario",
        default="pooled/td3_hft_lob_state_space_pooled_streaming_selected",
        metavar="NAME",
    )
    p.add_argument("--skip", type=int, default=500, metavar="N")
    p.add_argument(
        "--thesis-results-root",
        type=Path,
        default=_REPO_ROOT / "thesis" / "qmd" / "results",
        metavar="DIR",
    )
    return p.parse_args()


def _memmap_dir(scenario: str) -> Path:
    from trading_rl import ExperimentConfig

    search = [Path(scenario), _REPO_ROOT / "src" / "configs" / "scenarios" / scenario]
    config_path = next((p for p in search if p.exists()), None)
    if config_path is None:
        raise SystemExit(f"scenario not found: {scenario}")
    d = Path(ExperimentConfig.load(config_path).data.memmap_dir)
    return d if d.is_absolute() else _REPO_ROOT / d


def main() -> int:
    args = _parse_args()
    memmap_dir = _memmap_dir(args.scenario)
    ids = sorted(
        int(p.stem.split("_")[0]) for p in memmap_dir.glob("*_train_data.npy")
    )
    if not ids:
        raise SystemExit(f"no *_train_data.npy shards in {memmap_dir}")

    columns = json.loads((memmap_dir / f"{ids[0]}_columns.json").read_text())
    feat_cols = [c for c in columns if c.startswith("feature_hft_")]
    fj = {c: columns.index(c) for c in feat_cols}
    bj, aj = columns.index("bid_px_00"), columns.index("ask_px_00")

    feat_parts: dict[str, list[np.ndarray]] = {c: [] for c in feat_cols}
    ret_parts: list[np.ndarray] = []
    for i in ids:
        m = np.load(memmap_dir / f"{i}_train_data.npy", mmap_mode="r")
        mid = (
            np.asarray(m[:, bj], dtype=np.float64)
            + np.asarray(m[:, aj], dtype=np.float64)
        ) / 2.0
        logret = np.diff(np.log(mid))  # logret[t] = log(mid[t+1] / mid[t])
        hi = logret.shape[0]  # valid feature index range is [skip, hi)
        ret_parts.append(logret[args.skip : hi])
        for c in feat_cols:
            feat_parts[c].append(np.asarray(m[args.skip : hi, fj[c]], dtype=np.float64))

    y = np.concatenate(ret_parts)
    finite_y = np.isfinite(y)
    print(f"{len(ids)} shards, {finite_y.sum():,} paired events, {len(feat_cols)} features")

    rows = []
    for c in feat_cols:
        x = np.concatenate(feat_parts[c])
        mask = finite_y & np.isfinite(x)
        xm, ym = x[mask], y[mask]
        pearson = float(np.corrcoef(xm, ym)[0, 1])
        # Spearman = Pearson on ranks; rank y once for the common mask is not
        # possible (mask varies per feature), so rank per feature.
        spearman = float(np.corrcoef(rankdata(xm), rankdata(ym))[0, 1])
        rows.append({"feature": c, "pearson": pearson, "spearman": spearman})

    df = pd.DataFrame(rows).sort_values("pearson", key=lambda s: s.abs(), ascending=False)

    experiment_name = args.scenario.replace("/", "_")
    dest = (
        args.thesis_results_root / experiment_name / "peek" / "correlations.csv"
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    print(f"wrote {len(df)} rows to {dest}")

    show = df.copy()
    show["pearson"] = show["pearson"].map(lambda v: f"{v:+.5f}")
    show["spearman"] = show["spearman"].map(lambda v: f"{v:+.5f}")
    print(show.to_string(index=False))
    print(f"\nmax |Pearson| = {df['pearson'].abs().max():.5f}")
    print(f"max |Spearman| = {df['spearman'].abs().max():.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
