#!/usr/bin/env python3
"""Recompute the engineered-feature distribution table over the streamed training set.

Chapter 5's feature-statistics table (Appendix, ``@tbl-feature-stats``) previously
read a ``feature_stats.csv`` produced by ``peek dataset``, which computes the
statistics from ``dataset.train_df``. For a streaming scenario that frame is a
small staging frame bounded by ``data.train_size`` (50,000 rows here); the policy
is actually trained by ``StreamingTradingEnv`` reading the memmap shards under
``data.memmap_dir``. The committed snapshot was therefore AAPL only, on a single
session, over roughly the first seven minutes after the open -- its tail
statistics (skew, kurtosis, min, max) understate the real training set by one to
three orders of magnitude.

This script computes the table from the memmap shards directly. Each shard is one
symbol-session; the engineered ``feature_hft_*`` columns are stored already
normalized (causal running z-score, reset at session boundaries), exactly as the
env feeds them to the policy. The first ``--skip`` events of each shard are
dropped so the rolling-window features have reached steady state, matching the
note on the table.

Usage:
    uv run python scripts/export_streaming_feature_stats_to_thesis.py \\
        --scenario pooled/td3_hft_lob_state_space_pooled_streaming_selected

    git add thesis/qmd/results/pooled_td3_hft_lob_state_space_pooled_streaming_selected/peek/feature_stats.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCHEMA = [
    "feature",
    "selected",
    "mean",
    "std",
    "skew",
    "kurt",
    "q1",
    "q2",
    "q3",
    "min",
    "max",
    "nulls",
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--scenario",
        default="pooled/td3_hft_lob_state_space_pooled_streaming_selected",
        metavar="NAME",
        help="Scenario directory under src/configs/scenarios/ (default: %(default)s).",
    )
    p.add_argument(
        "--skip",
        type=int,
        default=500,
        metavar="N",
        help="Events dropped from the start of each shard (default: %(default)s).",
    )
    p.add_argument(
        "--thesis-results-root",
        type=Path,
        default=_REPO_ROOT / "thesis" / "qmd" / "results",
        metavar="DIR",
    )
    return p.parse_args()


def _load_scenario(scenario: str) -> tuple[Path, list[str]]:
    """Return (memmap_dir, in-model feature columns) for a scenario."""
    from trading_rl import ExperimentConfig

    search = [
        Path(scenario),
        _REPO_ROOT / "src" / "configs" / "scenarios" / scenario,
    ]
    config_path = next((p for p in search if p.exists()), None)
    if config_path is None:
        raise SystemExit(f"scenario not found: {scenario}")

    config = ExperimentConfig.load(config_path)
    memmap_dir = Path(config.data.memmap_dir)
    if not memmap_dir.is_absolute():
        memmap_dir = _REPO_ROOT / memmap_dir
    feature_columns = [str(c) for c in config.env.feature_columns]
    return memmap_dir, feature_columns


def _shard_ids(memmap_dir: Path) -> list[int]:
    ids = sorted(int(p.stem.split("_")[0]) for p in memmap_dir.glob("*_train_data.npy"))
    if not ids:
        raise SystemExit(f"no *_train_data.npy shards in {memmap_dir}")
    return ids


def main() -> int:
    args = _parse_args()
    memmap_dir, in_model = _load_scenario(args.scenario)
    ids = _shard_ids(memmap_dir)

    columns = json.loads((memmap_dir / f"{ids[0]}_columns.json").read_text())
    for i in ids[1:]:
        if json.loads((memmap_dir / f"{i}_columns.json").read_text()) != columns:
            raise SystemExit(f"shard {i} has a different column layout")
    feat_cols = [c for c in columns if c.startswith("feature_hft_")]
    col_idx = {c: columns.index(c) for c in feat_cols}

    mmaps = [
        np.load(memmap_dir / f"{i}_train_data.npy", mmap_mode="r") for i in ids
    ]
    total_rows = sum(m.shape[0] - args.skip for m in mmaps)
    print(
        f"{len(ids)} shards, {total_rows:,} events after skipping "
        f"{args.skip}/shard, {len(feat_cols)} engineered features"
    )

    rows = []
    for c in feat_cols:
        j = col_idx[c]
        v = np.concatenate(
            [np.asarray(m[args.skip :, j], dtype=np.float64) for m in mmaps]
        )
        finite = np.isfinite(v)
        nulls = int((~finite).sum())
        v = v[finite]
        q1, q2, q3 = (float(x) for x in np.percentile(v, [25, 50, 75]))
        rows.append(
            {
                "feature": c,
                "selected": c in in_model,
                "mean": float(v.mean()),
                "std": float(v.std(ddof=1)),
                "skew": float(skew(v, bias=False)),
                "kurt": float(kurtosis(v, fisher=True, bias=False)),
                "q1": q1,
                "q2": q2,
                "q3": q3,
                "min": float(v.min()),
                "max": float(v.max()),
                "nulls": nulls,
            }
        )

    df = pd.DataFrame(rows)[_SCHEMA]

    experiment_name = args.scenario.replace("/", "_")
    dest = args.thesis_results_root / experiment_name / "peek" / "feature_stats.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    print(f"wrote {len(df)} rows to {dest}")

    show = df.copy()
    for col in ("mean", "std", "skew", "kurt", "q2", "min", "max"):
        show[col] = show[col].map(lambda x: f"{x:+.3f}")
    print(
        show[["feature", "selected", "mean", "std", "skew", "kurt", "min", "max"]]
        .to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
