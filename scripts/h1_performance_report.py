"""H1 performance vs benchmarks report.

For each trained agent, shows test-split metrics alongside each benchmark's
metrics and the agent-vs-benchmark relative metrics (alpha, information ratio).

H1: A TD3-based agent using order-book-derived HFT features can achieve
stronger out-of-sample risk-adjusted performance than selected benchmark
strategies under the assumptions of the simulated environment.

Usage:
    uv run python scripts/h1_performance_report.py
    uv run python scripts/h1_performance_report.py --split val
    uv run python scripts/h1_performance_report.py --config src/configs/h1_performance.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.table import Table

_DEFAULT_CONFIG = Path("src/configs/h1_performance.yaml")

_AGENT_METRICS: list[tuple[str, str, str]] = [
    ("sharpe_ratio", "Sharpe", ".3f"),
    ("sortino_ratio", "Sortino", ".3f"),
    ("total_return", "Return", ".2%"),
    ("max_drawdown", "Max DD", ".2%"),
    ("win_rate", "Win Rate", ".2%"),
    ("profit_factor", "PF", ".3f"),
]

_BENCH_METRICS: list[tuple[str, str, str]] = [
    ("sharpe_ratio", "Sharpe", ".3f"),
    ("total_return", "Return", ".2%"),
    ("max_drawdown", "Max DD", ".2%"),
]

_REL_METRICS: list[tuple[str, str, str]] = [
    ("alpha", "Alpha", ".4f"),
    ("information_ratio", "IR", ".3f"),
    ("tracking_error", "Tr. Err", ".4f"),
]


def load_results(log_dir: Path, split: str) -> dict[str, Any]:
    results_json = log_dir / "results.json"
    if not results_json.exists():
        return {}
    try:
        with results_json.open() as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    for key in [split, *[k for k in data if k.startswith(f"{split}_")]]:
        entry = data.get(key)
        if entry:
            return entry
    return {}


def fmt(key: str, val: Any, fmt_str: str) -> str:
    if val is None or not isinstance(val, int | float):
        return "—"
    try:
        return f"{val:{fmt_str}}"
    except (ValueError, TypeError):
        return "—"


def beats(agent_val: Any, bench_val: Any, higher_is_better: bool = True) -> str:
    """Return a win/lose indicator."""
    if not isinstance(agent_val, int | float) or not isinstance(bench_val, int | float):
        return " "
    if higher_is_better:
        return "[green]+[/green]" if agent_val > bench_val else "[red]-[/red]"
    return "[green]+[/green]" if agent_val < bench_val else "[red]-[/red]"


def build_agent_table(
    agent_label: str,
    agent_data: dict[str, Any],
    bench_specs: list[dict[str, str]],
) -> Table:
    agent_metrics = agent_data.get("metrics") or {}
    benchmarks = agent_data.get("benchmarks") or {}

    t = Table(
        title=f"{agent_label} vs Benchmarks", show_header=True, header_style="bold"
    )
    t.add_column("", style="cyan", no_wrap=True)
    for _, col, _ in _AGENT_METRICS:
        t.add_column(col, justify="right")
    for _, col, _ in _REL_METRICS:
        t.add_column(col, justify="right")

    # Agent row
    agent_vals = [fmt(k, agent_metrics.get(k), f) for k, _, f in _AGENT_METRICS]
    rel_placeholders = ["—"] * len(_REL_METRICS)
    t.add_row(agent_label, *agent_vals, *rel_placeholders, style="bold green")

    # One row per benchmark
    # Every reported metric is better when larger on the scale it is stored on.
    # max_drawdown looks like an exception but is not: metrics.py records it as
    # np.min(equity / running_max - 1), i.e. signed-negative, so the shallower
    # drawdown is the greater value (-0.02 beats -0.10). Marking it
    # lower-is-better while comparing the raw signed values inverted the
    # win/loss verdict in both directions (#498).
    higher_better = {k: True for k, _, _ in _AGENT_METRICS}

    for spec in bench_specs:
        bench_name = spec["name"]
        bench_label = spec.get("label", bench_name)
        bench_data = benchmarks.get(bench_name) or {}
        bench_m = bench_data.get("benchmark_metrics") or {}
        rel_m = bench_data.get("relative_metrics") or {}

        if not bench_m:
            continue

        bench_vals = []
        for key, _, f in _AGENT_METRICS:
            bv = bench_m.get(key)
            av = agent_metrics.get(key)
            indicator = beats(av, bv, higher_better.get(key, True))
            bench_vals.append(f"{fmt(key, bv, f)} {indicator}")

        rel_vals = [fmt(k, rel_m.get(k), f) for k, _, f in _REL_METRICS]
        t.add_row(f"  {bench_label}", *bench_vals, *rel_vals)

    return t


def main() -> None:
    parser = argparse.ArgumentParser(description="H1 performance vs benchmarks report")
    parser.add_argument("--config", "-c", type=Path, default=_DEFAULT_CONFIG)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    console = Console()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        console.print(f"[red]Config not found: {cfg_path}[/red]")
        sys.exit(1)

    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)

    agents = cfg.get("agents", [])
    bench_specs = cfg.get("benchmarks", [])
    split = args.split

    console.print()
    console.print(f"[bold]H1 Performance Report[/bold]  [dim](split: {split})[/dim]")
    console.print(
        "[dim]H1: TD3 achieves stronger risk-adjusted performance than benchmarks.[/dim]"
    )
    console.print()

    found_any = False
    for agent in agents:
        label = agent.get("label", "?")
        log_dir = Path(agent.get("log_dir", ""))
        data = load_results(log_dir, split)
        if not data:
            console.print(f"[yellow]  {label}: no results.json — skipping[/yellow]")
            continue
        found_any = True
        table = build_agent_table(label, data, bench_specs)
        console.print(table)
        console.print()

    if not found_any:
        console.print(
            "[yellow]No results found. Run `cli.py evaluate --components metrics benchmarks` "
            "for each agent scenario first.[/yellow]"
        )
        sys.exit(0)

    console.print("[dim]Legend[/dim]")
    console.print(
        "  [dim]Agent row is [bold green]bold green[/bold green]. "
        "Benchmark rows show [green]+[/green] where agent beats benchmark, "
        "[red]-[/red] where it does not.[/dim]"
    )
    console.print(
        "  [dim]Alpha = annualised excess return vs benchmark. "
        "IR = information ratio. Tr. Err = tracking error.[/dim]"
    )


if __name__ == "__main__":
    main()
