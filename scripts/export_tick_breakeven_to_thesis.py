#!/usr/bin/env python3
"""Compute the tick-size break-even fee per instrument and export it for the thesis.

Used by 06-02-robustness-assessment.qmd's H2 discussion (the fee level at which
the smallest possible tick-driven mid-price move stops covering the trading
fee) and by the corresponding appendix table.

The minimum price increment for these instruments is one cent, so the smallest
possible one-sided mid-price move is half a cent. At an instrument's mean price
over the evaluation window, the proportional fee that exactly consumes that
half-cent move is the break-even fee reported here.

Usage:
    uv run python scripts/export_tick_breakeven_to_thesis.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SYMBOLS = ["AAPL", "AMZN", "AVGO", "META", "MSFT", "TSLA"]
_EVAL_DATE = "2026-03-02"
_HALF_TICK = 0.005  # half of the one-cent minimum price increment

_EXPORT_PATH = (
    _REPO_ROOT
    / "thesis/qmd/results/pooled_td3_hft_lob_state_space_pooled_streaming_selected_dsr"
    / "peek"
    / "tick_breakeven.json"
)


def main() -> None:
    rows = []
    for symbol in _SYMBOLS:
        path = (
            _REPO_ROOT
            / "data/raw/stocks/daily"
            / symbol
            / f"{symbol}_{_EVAL_DATE}_raw_mbp-10_us_hours.parquet"
        )
        df = pd.read_parquet(path, columns=["bid_px_00", "ask_px_00"])
        mean_price = float(((df["bid_px_00"] + df["ask_px_00"]) / 2).mean())
        breakeven_bp = (_HALF_TICK / mean_price) * 10_000
        rows.append(
            {
                "symbol": symbol,
                "mean_price": mean_price,
                "breakeven_fee_bp": breakeven_bp,
            }
        )

    rows.sort(key=lambda r: r["mean_price"], reverse=True)

    _EXPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _EXPORT_PATH.write_text(json.dumps(rows, indent=2))
    print(f"exported {len(rows)} rows to {_EXPORT_PATH}")
    for r in rows:
        print(f"  {r['symbol']:6s} price={r['mean_price']:8.2f}  breakeven={r['breakeven_fee_bp']:.3f} bp")


if __name__ == "__main__":
    main()
