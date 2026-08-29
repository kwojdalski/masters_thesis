"""H3 sensitivity analysis report.

Reads results.json from each scenario's log directory and prints a
side-by-side comparison table for every sensitivity axis defined in
src/configs/h3_sensitivity.yaml.

Usage:
    uv run python scripts/h3_sensitivity_report.py
    uv run python scripts/h3_sensitivity_report.py --split val
    uv run python scripts/h3_sensitivity_report.py --config src/configs/h3_sensitivity.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml
from rich.columns import Columns
from rich.console import Console
from rich.table import Table

_DEFAULT_CONFIG = Path("src/configs/h3_sensitivity.yaml")

_METRICS: list[tuple[str, str, str]] = [
    ("sharpe_ratio", "Sharpe", ".3f"),
    ("sortino_ratio", "Sortino", ".3f"),
    ("total_return", "Return", ".2%"),
    ("max_drawdown", "Max DD", ".2%"),
    ("win_rate", "Win Rate", ".2%"),
    ("profit_factor", "PF", ".3f"),
]


def load_metrics(log_dir: Path, split: str) -> dict[str, float]:
    results_json = log_dir / "results.json"
    if not results_json.exists():
        return {}
    try:
        with results_json.open() as f:
            results = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}

    for key in [split, *[k for k in results if k.startswith(f"{split}_")]]:
        entry = results.get(key)
        if entry:
            return entry.get("metrics") or {}
    return {}


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

    rows: list[tuple[str, dict[str, float], bool]] = []
    for sc in scenarios:
        label = sc.get("label", "?")
        log_dir = Path(sc.get("log_dir", ""))
        is_baseline = sc.get("baseline", False)
        metrics = load_metrics(log_dir, split)
        rows.append((label, metrics, is_baseline))

    if not any(m for _, m, _ in rows):
        console.print(
            f"[yellow]  {axis_label}: no results.json found — skipping[/yellow]"
        )
        return None

    t = Table(title=axis_label, show_header=True, header_style="bold")
    t.add_column("Variant", style="cyan", no_wrap=True)
    for _, display, _ in _METRICS:
        t.add_column(display, justify="right")

    for label, metrics, is_baseline in rows:
        if not metrics:
            row_vals = ["—"] * len(_METRICS)
        else:
            row_vals = [fmt_val(key, metrics.get(key), fmt) for key, _, fmt in _METRICS]

        style = "bold green" if is_baseline else None
        t.add_row(label, *row_vals, style=style)

    return t


def main() -> None:
    parser = argparse.ArgumentParser(description="H3 sensitivity analysis report")
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=_DEFAULT_CONFIG,
        help=f"Path to sensitivity config YAML (default: {_DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Data split to report on (default: test)",
    )
    args = parser.parse_args()

    console = Console()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        console.print(f"[red]Config not found: {cfg_path}[/red]")
        sys.exit(1)

    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)

    tables: list[Table] = []
    for axis in cfg.get("axes", []):
        table = build_axis_table(axis, args.split, console)
        if table:
            tables.append(table)

    if not tables:
        console.print(
            "[yellow]No results found. Train the H3 scenarios first.[/yellow]"
        )
        sys.exit(0)

    console.print()
    console.print(
        f"[bold]H3 Sensitivity Analysis[/bold]  [dim](split: {args.split})[/dim]"
    )
    console.print()
    console.print(Columns(tables, equal=False, expand=False))
    console.print()

    legend = [
        "[bold]Legend[/bold]",
        "[cyan]Sharpe[/cyan]    Annualised Sharpe ratio (higher is better).",
        "[cyan]Sortino[/cyan]   Annualised Sortino ratio (downside-only penalty).",
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
