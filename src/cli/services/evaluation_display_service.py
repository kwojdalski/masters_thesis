"""Display helpers for standalone policy evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.table import Table

from trading_rl.evaluation.metric_meta import METRIC_META_BY_KEY

if TYPE_CHECKING:
    import pandas as pd

PERF_ROW_KEYS = [
    "total_return",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "win_rate",
    "lose_rate",
    "profit_factor",
    "omega_ratio",
    "pct_long",
    "pct_short",
]
PERF_ROWS = [
    (key, METRIC_META_BY_KEY[key].label, METRIC_META_BY_KEY[key].fmt)
    for key in PERF_ROW_KEYS
]


class EvaluationDisplayService:
    """Render evaluation tables for the CLI."""

    def __init__(self, console: Console) -> None:
        self.console = console

    def print_metrics_table(
        self,
        split: str,
        metrics: dict[str, float],
        split_df: pd.DataFrame | None = None,
        symbols: list[str] | None = None,
    ) -> None:
        table = Table(title=f"Metrics ({split})", show_header=True, header_style="bold")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")
        if split_df is not None and not split_df.empty:

            def _fmt(ts):
                if hasattr(ts, "strftime"):
                    return ts.strftime("%Y-%m-%d %H:%M:%S UTC")
                return str(ts)[:19]

            table.add_row("Start Datetime", _fmt(split_df.index[0]))
            table.add_row("End Datetime", _fmt(split_df.index[-1]))
        if symbols:
            table.add_row("Symbols", ", ".join(symbols))
        for key, label, fmt in PERF_ROWS:
            if key in metrics:
                val = metrics[key]
                table.add_row(label, f"{val:{fmt}}")
        self.console.print(table)

    def print_benchmark_table(
        self,
        split: str,
        bench_out: dict[str, Any],
        strategy_metrics: dict[str, Any] | None = None,
    ) -> None:
        rel_cols = [
            ("alpha", "Alpha", ".4f"),
            ("beta", "Beta", ".3f"),
            ("information_ratio", "Info Ratio", ".3f"),
            ("tracking_error", "Track. Error", ".4f"),
        ]
        table = Table(
            title=f"Benchmark performance ({split})",
            show_header=True,
            header_style="bold",
        )
        table.add_column("Benchmark", style="cyan", no_wrap=True)
        for _, label, _ in PERF_ROWS:
            table.add_column(label, justify="right")
        for _, label, _ in rel_cols:
            table.add_column(label, justify="right")

        if strategy_metrics:
            row = ["[bold green]Strategy[/bold green]"]
            for key, _, fmt in PERF_ROWS:
                val = strategy_metrics.get(key)
                if val is None:
                    row.append("-")
                else:
                    row.append(f"[bold green]{val:{fmt}}[/bold green]")
            row += ["-"] * len(rel_cols)
            table.add_row(*row)

        for bench_name, entry in bench_out.items():
            bench_metrics = entry.get("benchmark_metrics", entry)
            rel_metrics = entry.get("relative_metrics", {})
            row = [bench_name]
            for key, _, fmt in PERF_ROWS:
                val = bench_metrics.get(key)
                row.append(f"{val:{fmt}}" if val is not None else "-")
            for key, _, fmt in rel_cols:
                val = rel_metrics.get(key)
                row.append(f"{val:{fmt}}" if val is not None else "-")
            table.add_row(*row)
        self.console.print(table)
