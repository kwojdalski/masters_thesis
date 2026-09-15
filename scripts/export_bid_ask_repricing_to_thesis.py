#!/usr/bin/env python3
"""Extend the single-instrument bid/ask repricing to all six instruments.

Section 7.2.1 (Execution Realism) re-prices the Hypothesis 1 TD3 action path
at the bid/ask on AAPL only: gross profit vs. the spread the frictionless
mid-price environment declined to charge. Reviewer finding (issue #858) asks
whether this can be extended to all six instruments using the half-spread
already tabulated in the appendix (Table 13, tick_breakeven.json).

The per-symbol test rollouts needed to do this already exist -- they were
produced by the "full-window TD3 re-evaluation" folded into the Hypothesis 1
results (commit 2df50cf4) -- at
``<EXPERIMENT_OUTPUT_DIR>/td3_hft_lob_state_space_pooled_streaming_selected_dsr/
per_symbol/<SYM>/test_<SYM>_rollout.parquet``. That directory is a local,
gitignored run directory, not a thesis-tracked artifact, so this script reads
from it and exports the small derived summary to
``thesis/qmd/results/.../peek/bid_ask_repricing.json`` -- the same pattern
``export_tick_breakeven_to_thesis.py`` uses for the raw DataBento files it
reads from ``data/raw/``.

Methodology (matches the prose in 07-02-limitations-and-future-research.qmd,
restated in #755/#832): the spread-crossing cost is the realised turnover of
the action path (sum of |change in position| across the test window) times
the instrument's mean relative half-spread from Table 13. The breakeven
proportional fee is the per-step gross return divided by the per-step
turnover. Both are computed directly from the raw per-step action and
simple_return columns, not from any pre-aggregated metric, so they are
unaffected by upstream fixes to summary statistics (e.g. #811/#812) that do
not touch these two raw sums.

Usage:
    uv run python scripts/export_bid_ask_repricing_to_thesis.py

Requires the per-symbol rollout parquets under the experiment output
directory, which are produced by evaluating the Hypothesis 1 TD3 checkpoint
per instrument (already run; not reproduced by this script).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from trading_rl.config import EXPERIMENT_OUTPUT_DIR

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SYMBOLS = ["AAPL", "AMZN", "AVGO", "META", "MSFT", "TSLA"]

_EXPERIMENT_DIR = (
    _REPO_ROOT
    / "thesis/qmd/results/pooled_td3_hft_lob_state_space_pooled_streaming_selected_dsr"
)
_RUN_NAME = "td3_hft_lob_state_space_pooled_streaming_selected_dsr"

# EXPERIMENT_OUTPUT_DIR moved from logs/ to output/experiments in #821. The
# rollouts these figures come from were written before that move, so resolve
# the current location first and fall back to the legacy one rather than
# silently failing on a pre-move checkout.
_ROLLOUT_DIR_CANDIDATES = (
    _REPO_ROOT / EXPERIMENT_OUTPUT_DIR / _RUN_NAME / "per_symbol",
    _REPO_ROOT / "logs" / _RUN_NAME / "per_symbol",
)


def _rollout_dir() -> Path:
    for candidate in _ROLLOUT_DIR_CANDIDATES:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "No per-symbol rollout directory found. Looked in:\n  "
        + "\n  ".join(str(c) for c in _ROLLOUT_DIR_CANDIDATES)
        + "\nRe-run `evaluate` for the pooled/"
        + _RUN_NAME
        + " scenario to produce it."
    )


_TICK_BREAKEVEN_PATH = _EXPERIMENT_DIR / "peek" / "tick_breakeven.json"
_EXPORT_PATH = _EXPERIMENT_DIR / "peek" / "bid_ask_repricing.json"


def main() -> None:
    if not _TICK_BREAKEVEN_PATH.exists():
        raise FileNotFoundError(
            f"{_TICK_BREAKEVEN_PATH} not found -- run "
            "scripts/export_tick_breakeven_to_thesis.py first."
        )
    half_spread_bp = {
        r["symbol"]: r["half_spread_bp_mean"]
        for r in json.loads(_TICK_BREAKEVEN_PATH.read_text())
    }

    rollout_dir = _rollout_dir()

    rows = []
    for symbol in _SYMBOLS:
        rollout_path = rollout_dir / symbol / f"test_{symbol}_rollout.parquet"
        if not rollout_path.exists():
            raise FileNotFoundError(
                f"{rollout_path} not found. This script reads the per-symbol "
                "Hypothesis 1 TD3 test rollouts folded in by commit 2df50cf4; "
                "re-run `evaluate` for the "
                "pooled/td3_hft_lob_state_space_pooled_streaming_selected_dsr "
                "scenario if this directory is missing."
            )
        df = pd.read_parquet(rollout_path, columns=["action", "simple_return"])
        n_steps = len(df)
        action_diff = np.abs(np.diff(df["action"].to_numpy()))
        turnover_sum = float(action_diff.sum())
        turnover_per_step = float(action_diff.mean())
        gross_pnl = float(df["simple_return"].sum())
        per_step_gross_return = float(df["simple_return"].mean())

        hs_bp = half_spread_bp[symbol]
        hs_fraction = hs_bp / 10_000.0
        spread_cost = turnover_sum * hs_fraction
        net_pnl = gross_pnl - spread_cost
        cost_to_profit_ratio = (
            spread_cost / gross_pnl if gross_pnl != 0 else float("nan")
        )
        breakeven_fee_bp = (
            (per_step_gross_return / turnover_per_step) * 10_000
            if turnover_per_step
            else float("nan")
        )

        rows.append(
            {
                "symbol": symbol,
                "n_steps": n_steps,
                "gross_pnl": gross_pnl,
                "turnover_sum": turnover_sum,
                "turnover_per_step": turnover_per_step,
                "half_spread_bp_mean": hs_bp,
                "spread_cost": spread_cost,
                "cost_to_profit_ratio": cost_to_profit_ratio,
                "net_pnl": net_pnl,
                "sign_reverses": net_pnl < 0,
                "breakeven_fee_bp": breakeven_fee_bp,
            }
        )

    rows.sort(key=lambda r: r["cost_to_profit_ratio"])

    _EXPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _EXPORT_PATH.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"exported {len(rows)} rows to {_EXPORT_PATH}")
    for r in rows:
        print(
            f"  {r['symbol']:6s} gross_pnl={r['gross_pnl']:8.3f}  "
            f"spread_cost={r['spread_cost']:8.3f}  "
            f"ratio={r['cost_to_profit_ratio']:6.2f}x  "
            f"net_pnl={r['net_pnl']:9.3f}  "
            f"sign_reverses={r['sign_reverses']}"
        )


if __name__ == "__main__":
    main()
