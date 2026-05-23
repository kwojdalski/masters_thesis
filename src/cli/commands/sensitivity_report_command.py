"""H3 sensitivity analysis report command."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from rich.columns import Columns
from rich.table import Table

from .base_command import BaseCommand

_DEFAULT_CONFIG = Path("src/configs/h3_sensitivity.yaml")
_SPLIT = "test"

_METRICS: list[tuple[str, str, str]] = [
    ("sharpe_ratio",          "Sharpe",       ".3f"),
    ("sortino_ratio",         "Sortino",      ".3f"),
    ("total_return",          "Return",       ".2%"),
    ("annualized_return_cagr","CAGR",         ".2%"),
    ("max_drawdown",          "Max DD",       ".2%"),
    ("win_rate",              "Win Rate",     ".2%"),
    ("profit_factor",         "PF",           ".3f"),
]


@dataclass
class SensitivityReportParams:
    config: Path = field(default_factory=lambda: _DEFAULT_CONFIG)
    split: str = _SPLIT


class SensitivityReportCommand(BaseCommand):
    """Print H3 sensitivity tables: feature spec, reward design, and transaction cost axes."""

    def execute(self, params: SensitivityReportParams) -> None:
        try:
            cfg_path = Path(params.config)
            if not cfg_path.exists():
                raise FileNotFoundError(f"Sensitivity config not found: {cfg_path}")

            with cfg_path.open() as f:
                cfg = yaml.safe_load(f)

            split = params.split
            tables: list[Table] = []

            for axis in cfg.get("axes", []):
                table = self._build_axis_table(axis, split)
                if table:
                    tables.append(table)

            if not tables:
                self.console.print("[yellow]No results found. Train the H3 scenarios first.[/yellow]")
                return

            self.console.print()
            self.console.print(
                f"[bold]H3 Sensitivity Analysis[/bold]  "
                f"[dim](split: {split})[/dim]"
            )
            self.console.print()
            self.console.print(Columns(tables, equal=False, expand=False))
            self.console.print()
            self._print_legend()

        except Exception as e:
            self.handle_error(e, "sensitivity-report")

    def _build_axis_table(self, axis: dict[str, Any], split: str) -> Table | None:
        axis_label = axis.get("label", axis.get("name", ""))
        scenarios = axis.get("scenarios", [])

        rows: list[tuple[str, dict[str, float], bool]] = []
        for sc in scenarios:
            label = sc.get("label", "?")
            log_dir = Path(sc.get("log_dir", ""))
            is_baseline = sc.get("baseline", False)
            metrics = self._load_metrics(log_dir, split)
            rows.append((label, metrics, is_baseline))

        if not any(m for _, m, _ in rows):
            self.console.print(
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
                row_vals = []
                for key, _, fmt in _METRICS:
                    val = metrics.get(key)
                    if val is None or not isinstance(val, (int, float)):
                        row_vals.append("—")
                    elif key == "annualized_return_cagr" and abs(val) > 1000.0:
                        row_vals.append("—")
                    else:
                        try:
                            row_vals.append(f"{val:{fmt}}")
                        except (ValueError, TypeError):
                            row_vals.append("—")

            style = "bold green" if is_baseline else None
            t.add_row(label, *row_vals, style=style)

        return t

    def _load_metrics(self, log_dir: Path, split: str) -> dict[str, float]:
        results_json = log_dir / "results.json"
        if not results_json.exists():
            return {}
        try:
            with results_json.open() as f:
                results = json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

        # results.json keys: "test", "test_AAPL", "val", "val_AAPL", etc.
        # Prefer exact split key, then any key that starts with split + "_"
        for key in [split, *[k for k in results if k.startswith(f"{split}_")]]:
            entry = results.get(key)
            if entry:
                return entry.get("metrics") or {}
        return {}

    def _print_legend(self) -> None:
        lines = [
            "[bold]Legend[/bold]",
            "[cyan]Sharpe[/cyan]    Annualised Sharpe ratio (higher is better).",
            "[cyan]Sortino[/cyan]   Annualised Sortino ratio (downside-only penalty).",
            "[cyan]Return[/cyan]    Cumulative portfolio return over the test horizon.",
            "[cyan]CAGR[/cyan]      Compound annual growth rate; shown as — when horizon is too short.",
            "[cyan]Max DD[/cyan]    Maximum peak-to-trough drawdown (lower magnitude is better).",
            "[cyan]Win Rate[/cyan]  Fraction of steps with positive return.",
            "[cyan]PF[/cyan]        Profit factor: gross profit / gross loss (> 1 = profitable).",
            "[bold green]Bold green row[/bold green] = baseline variant held constant across other axes.",
        ]
        for line in lines:
            self.console.print(f"  {line}")
