"""Sensitivity-axis report.

Reads results.json from each scenario's log directory and prints a
side-by-side comparison table for every sensitivity axis defined in the
supplied config YAML. Used for the transaction-cost (H2) and reward-design
(H4) axes; see src/configs/h2_transaction_cost.yaml and
src/configs/h4_reward_design.yaml.

Usage:
    uv run python scripts/sensitivity_report.py --config src/configs/h2_transaction_cost.yaml
    uv run python scripts/sensitivity_report.py --config src/configs/h4_reward_design.yaml --split val
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml
from rich.columns import Columns
from rich.console import Console
from rich.table import Table

from trading_rl.evaluation.results_io import SplitEntry, basis_warning, load_split_entry

_METRICS: list[tuple[str, str, str]] = [
    ("sharpe_ratio", "Sharpe", ".3f"),
    ("sortino_ratio", "Sortino", ".3f"),
    ("total_return", "Return", ".2%"),
    ("max_drawdown", "Max DD", ".2%"),
    ("win_rate", "Win Rate", ".2%"),
    ("profit_factor", "PF", ".3f"),
]


def load_metrics(log_dir: Path, split: str) -> tuple[dict[str, float], SplitEntry]:
    """Return the split's metrics and the provenance they were derived from."""
    resolved = load_split_entry(log_dir / "results.json", split)
    return (resolved.entry.get("metrics") or {}), resolved


def fmt_val(key: str, val: Any, fmt: str) -> str:
    if val is None or not isinstance(val, (int, float)):
        return "—"
    try:
        return f"{val:{fmt}}"
    except (ValueError, TypeError):
        return "—"


def build_axis_table(
    axis: dict[str, Any], split: str, console: Console
) -> Table | None:
    axis_label = axis.get("label", axis.get("name", ""))
    scenarios = axis.get("scenarios", [])

    rows: list[tuple[str, dict[str, float], SplitEntry, bool]] = []
    for sc in scenarios:
        label = sc.get("label", "?")
        log_dir = Path(sc.get("log_dir", ""))
        is_baseline = sc.get("baseline", False)
        metrics, resolved = load_metrics(log_dir, split)
        rows.append((label, metrics, resolved, is_baseline))

    if not any(m for _, m, _, _ in rows):
        console.print(
            f"[yellow]  {axis_label}: no results.json found — skipping[/yellow]"
        )
        return None

    # A mixed-basis axis is not a comparison at all, so say so above the table
    # rather than letting the rows imply one.
    warning = basis_warning([r for _, _, r, _ in rows])
    if warning:
        console.print(f"[bold red]  {axis_label}: {warning}[/bold red]")

    t = Table(title=axis_label, show_header=True, header_style="bold")
    t.add_column("Variant", style="cyan", no_wrap=True)
    t.add_column("Basis", justify="right", style="dim")
    for _, display, _ in _METRICS:
        t.add_column(display, justify="right")

    for label, metrics, resolved, is_baseline in rows:
        if not metrics:
            row_vals = ["—"] * len(_METRICS)
        else:
            row_vals = [fmt_val(key, metrics.get(key), fmt) for key, _, fmt in _METRICS]

        style = "bold green" if is_baseline else None
        t.add_row(label, resolved.label, *row_vals, style=style)

    return t


def main() -> None:
    parser = argparse.ArgumentParser(description="Sensitivity-axis report")
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        required=True,
        help="Path to a sensitivity config YAML (an 'axes' list).",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Data split to report on (default: test)",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help="Read scenario result directories from this root.",
    )
    args = parser.parse_args()

    console = Console()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        console.print(f"[red]Config not found: {cfg_path}[/red]")
        sys.exit(1)

    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)

    if args.results_root is not None:
        for axis in cfg.get("axes", []):
            for scenario in axis.get("scenarios", []):
                log_dir = Path(scenario.get("log_dir", ""))
                scenario["log_dir"] = str(args.results_root / log_dir.name)

    tables: list[Table] = []
    for axis in cfg.get("axes", []):
        table = build_axis_table(axis, args.split, console)
        if table:
            tables.append(table)

    if not tables:
        console.print(
            "[yellow]No results found. Train the scenarios in the config first.[/yellow]"
        )
        sys.exit(0)

    console.print()
    console.print(
        f"[bold]Sensitivity Analysis[/bold]  "
        f"[dim]({cfg_path.stem}, split: {args.split})[/dim]"
    )
    console.print()
    console.print(Columns(tables, equal=False, expand=False))
    console.print()

    legend = [
        "[bold]Legend[/bold]",
        "[cyan]Basis[/cyan]     'pooled' = one pooled entry; 'mean(N)' = equal-weight "
        "mean over N per-symbol entries; a bare ticker = that symbol only.",
        # metrics.py stores sharpe_ratio / sortino_ratio per bar; the ×√ppy
        # variants live under the *_annualized keys, which this table does not
        # read. Calling these annualised overstated them by orders of magnitude.
        "[cyan]Sharpe[/cyan]    Per-bar Sharpe ratio at the reporting frequency, "
        "not annualised (higher is better).",
        "[cyan]Sortino[/cyan]   Per-bar Sortino ratio, not annualised "
        "(downside-only penalty).",
        "[cyan]Return[/cyan]    Cumulative portfolio return over the evaluation horizon.",
        "[cyan]Max DD[/cyan]    Maximum peak-to-trough drawdown (lower magnitude is better).",
        "[cyan]Win Rate[/cyan]  Fraction of steps with positive return.",
        "[cyan]PF[/cyan]        Profit factor: gross profit / gross loss (> 1 = profitable).",
        "[bold green]Bold green[/bold green] = baseline held constant across the other axes.",
    ]
    for line in legend:
        console.print(f"  {line}")


if __name__ == "__main__":
    main()
