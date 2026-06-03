"""Compare equity curves for the same symbol/split across multiple scenarios.

Edit SCENARIOS below to add or remove entries.  Each entry is a tuple of
(log_dir, display_label).  The script verifies that Buy-and-Hold final values
match across scenarios — a mismatch means different underlying data.

Usage:
    uv run python scripts/compare_equity_curves.py
    uv run python scripts/compare_equity_curves.py --symbol AAPL --split test
    uv run python scripts/compare_equity_curves.py --symbols AAPL META --split val --output out.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from plotnine import aes, geom_line, ggplot, labs, scale_color_manual, scale_linetype_manual

# Import thesis visual style
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from trading_rl.evaluation.thesis_theme import (
    FIGURE_HEIGHT,
    FIGURE_WIDTH,
    LINETYPE,
    PALETTE,
    thesis_theme,
)

# ---------------------------------------------------------------------------
# Configure scenarios here
# ---------------------------------------------------------------------------
SCENARIOS: list[tuple[str, str]] = [
    ("logs/td3_hft_lob_state_space_pooled_streaming_selected_dsr", "TD3"),
    ("logs/ddpg_hft_lob_state_space_pooled_streaming_selected_dsr", "DDPG"),
    # ("logs/some_other_scenario", "Label"),
]

# Benchmark run names present in every portfolio_value_plot.csv
_BENCHMARK_RUNS: frozenset[str] = frozenset({"Buy-and-Hold", "TWAP", "VWAP", "Random"})

# Wong (2011) colorblind-safe colors for agent lines, in order
_AGENT_COLORS: list[str] = [
    "#CC0000",  # red
    "#0072B2",  # blue
    "#009E73",  # green
    "#F0E442",  # yellow
    "#CC79A7",  # pink
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _csv_path(log_dir: str | Path, split: str, symbol: str) -> Path:
    return Path(log_dir) / f"{split}_{symbol}_portfolio_value_plot.csv"


def _load_csv(log_dir: str | Path, split: str, symbol: str) -> pd.DataFrame:
    path = _csv_path(log_dir, split, symbol)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _bah_final(df: pd.DataFrame) -> float | None:
    bah = df[df["Run"] == "Buy-and-Hold"]
    return float(bah["Portfolio_Value"].iloc[-1]) if not bah.empty else None


def build_merged_df(
    scenarios: list[tuple[str, str]],
    symbol: str,
    split: str,
    stride: int,
) -> pd.DataFrame:
    """Load and merge equity data, emitting warnings when data may differ."""
    frames: list[pd.DataFrame] = []
    reference_bah: float | None = None
    reference_steps: int | None = None
    benchmarks_added = False

    for log_dir, label in scenarios:
        try:
            raw = _load_csv(log_dir, split, symbol)
        except FileNotFoundError as exc:
            print(f"WARNING: {exc} — skipping '{label}'")
            continue

        all_steps = sorted(raw["Steps"].unique())
        n_steps = len(all_steps)

        # Apples-to-apples: step count check
        if reference_steps is None:
            reference_steps = n_steps
        elif n_steps != reference_steps:
            print(
                f"WARNING: '{label}' has {n_steps} steps but reference has "
                f"{reference_steps} — may not be the same evaluation data"
            )

        # Apples-to-apples: Buy-and-Hold final value check
        bah = _bah_final(raw)
        if bah is not None:
            if reference_bah is None:
                reference_bah = bah
            elif abs(bah - reference_bah) > 0.01:
                print(
                    f"WARNING: '{label}' Buy-and-Hold final={bah:.2f} differs from "
                    f"reference={reference_bah:.2f} — scenarios may use different data"
                )

        keep_steps = set(all_steps[::stride])

        # Add benchmarks once (they are identical across scenarios on the same data)
        if not benchmarks_added:
            bench = raw[raw["Run"].isin(_BENCHMARK_RUNS) & raw["Steps"].isin(keep_steps)].copy()
            if not bench.empty:
                frames.append(bench)
            benchmarks_added = True

        # Add agent line with scenario label
        agent = raw[raw["Run"] == "Deterministic"].copy()
        if agent.empty:
            print(f"WARNING: no 'Deterministic' run in '{label}' — skipping")
            continue
        agent = agent[agent["Steps"].isin(keep_steps)].copy()
        agent["Run"] = label
        frames.append(agent)

    if not frames:
        raise SystemExit("No data loaded — check SCENARIOS paths and --symbol / --split args")

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _build_palette(scenario_labels: list[str], benchmark_runs: list[str]) -> dict[str, str]:
    palette: dict[str, str] = {}
    for i, label in enumerate(scenario_labels):
        palette[label] = _AGENT_COLORS[i % len(_AGENT_COLORS)]
    for run in benchmark_runs:
        palette[run] = PALETTE.get(run, "#888888")
    return palette


def _build_linetype(scenario_labels: list[str], benchmark_runs: list[str]) -> dict[str, str]:
    lt: dict[str, str] = {label: "solid" for label in scenario_labels}
    for run in benchmark_runs:
        lt[run] = LINETYPE.get(run, "dashed")
    return lt


def plot_equity_comparison(
    df: pd.DataFrame,
    scenario_labels: list[str],
    symbol: str,
    split: str,
) -> "ggplot":
    benchmark_runs = [r for r in df["Run"].unique() if r not in scenario_labels]
    palette = _build_palette(scenario_labels, benchmark_runs)
    lt = _build_linetype(scenario_labels, benchmark_runs)

    return (
        ggplot(df, aes(x="Steps", y="Portfolio_Value", color="Run", linetype="Run"))
        + geom_line(size=0.32)
        + labs(
            title=f"Portfolio Value — {symbol} ({split})",
            x="Steps",
            y="Portfolio Value ($)",
        )
        + scale_color_manual(values=palette, name="Strategy")
        + scale_linetype_manual(values=lt, name="Strategy")
        + thesis_theme()
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare equity curves across scenarios for the same symbol/split"
    )
    parser.add_argument("--symbols", nargs="+", default=["META"],
                        help="Symbol(s) to plot, one output file per symbol (default: META)")
    parser.add_argument("--split", default="val", choices=["val", "test", "train"],
                        help="Data split (default: val)")
    parser.add_argument("--stride", type=int, default=50,
                        help="Plot every Nth step to keep file size manageable (default: 50)")
    parser.add_argument("--output-dir", type=Path, default=Path("."),
                        help="Directory for output PNG files (default: current dir)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scenario_labels = [label for _, label in SCENARIOS]

    for symbol in args.symbols:
        print(f"\nBuilding {args.split} comparison for {symbol}...")
        df = build_merged_df(SCENARIOS, symbol, args.split, stride=args.stride)
        print(f"  Rows: {len(df)}  Runs: {sorted(df['Run'].unique())}")

        p = plot_equity_comparison(df, scenario_labels, symbol, args.split)

        out = args.output_dir / f"compare_{args.split}_{symbol}.png"
        p.save(str(out), width=FIGURE_WIDTH, height=FIGURE_HEIGHT, dpi=225, verbose=False)
        print(f"  Saved: {out}")


if __name__ == "__main__":
    main()
