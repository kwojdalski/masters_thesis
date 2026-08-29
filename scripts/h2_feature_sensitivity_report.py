"""H2 feature sensitivity report.

Compares out-of-sample performance across three feature-specification levels
(minimal, selected, full) to evaluate whether richer microstructure features
improve TD3 performance.

H2: Extending the state representation from basic snapshot features to a
broader microstructure-aware feature set leads to improved out-of-sample
performance under identical training and evaluation conditions.

Multi-seed support: if a log_dir contains sub-directories named seed_<N>
that each hold a results.json, metrics are aggregated as mean ± std across
seeds. A single results.json in log_dir itself is treated as a single seed.

Usage:
    uv run python scripts/h2_feature_sensitivity_report.py
    uv run python scripts/h2_feature_sensitivity_report.py --split val
    uv run python scripts/h2_feature_sensitivity_report.py --config src/configs/h2_feature_sensitivity.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from rich.console import Console
from rich.table import Table

_DEFAULT_CONFIG = Path("src/configs/h2_feature_sensitivity.yaml")

_METRICS: list[tuple[str, str, str]] = [
    ("sharpe_ratio", "Sharpe", ".3f"),
    ("sortino_ratio", "Sortino", ".3f"),
    ("total_return", "Return", ".2%"),
    ("max_drawdown", "Max DD", ".2%"),
    ("win_rate", "Win Rate", ".2%"),
    ("profit_factor", "PF", ".3f"),
]

_HIGHER_IS_BETTER = {
    "sharpe_ratio": True,
    "sortino_ratio": True,
    "total_return": True,
    # Signed-negative on the stored scale (np.min of equity/running_max - 1),
    # so a shallower drawdown is the larger value. See #498.
    "max_drawdown": True,
    "win_rate": True,
    "profit_factor": True,
}


def _load_single(results_json: Path, split: str) -> dict[str, float]:
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
            return entry.get("metrics") or {}
    return {}


def load_metrics_with_seeds(
    log_dir: Path, split: str
) -> tuple[dict[str, float], dict[str, float], int]:
    """Load metrics, aggregating across seed sub-directories when present.

    Returns:
        (mean_metrics, std_metrics, n_seeds)
        std_metrics is empty when n_seeds == 1.
    """
    # Gather results.json files: seed subdirs take priority over a flat file
    seed_dirs = (
        sorted(
            d for d in log_dir.iterdir() if d.is_dir() and d.name.startswith("seed_")
        )
        if log_dir.exists()
        else []
    )
    if seed_dirs:
        seed_metrics = [_load_single(sd / "results.json", split) for sd in seed_dirs]
        seed_metrics = [m for m in seed_metrics if m]
    else:
        single = _load_single(log_dir / "results.json", split)
        seed_metrics = [single] if single else []

    if not seed_metrics:
        return {}, {}, 0

    if len(seed_metrics) == 1:
        return seed_metrics[0], {}, 1

    # Aggregate: mean and std across seeds for each metric key
    all_keys = {k for m in seed_metrics for k in m}
    mean_m: dict[str, float] = {}
    std_m: dict[str, float] = {}
    for key in all_keys:
        vals = [
            m[key] for m in seed_metrics if key in m and isinstance(m[key], int | float)
        ]
        if vals:
            mean_m[key] = float(np.mean(vals))
            std_m[key] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return mean_m, std_m, len(seed_metrics)


def fmt_val(key: str, val: Any, fmt_str: str) -> str:
    if val is None or not isinstance(val, int | float):
        return "—"
    try:
        return f"{val:{fmt_str}}"
    except (ValueError, TypeError):
        return "—"


def fmt_delta(key: str, val: Any, baseline: Any, fmt_str: str) -> str:
    """Format the delta relative to baseline with a direction indicator."""
    if not isinstance(val, int | float) or not isinstance(baseline, int | float):
        return "—"
    delta = val - baseline
    higher_better = _HIGHER_IS_BETTER.get(key, True)
    if fmt_str.endswith("%"):
        delta_str = f"{delta:+.2%}"
    else:
        delta_str = f"{delta:+.3f}"
    if delta == 0.0:
        return f"[dim]{delta_str}[/dim]"
    wins = (delta > 0 and higher_better) or (delta < 0 and not higher_better)
    color = "green" if wins else "red"
    return f"[{color}]{delta_str}[/{color}]"


def build_report_table(
    scenarios: list[dict[str, Any]], split: str, console: Console
) -> Table | None:
    # Collect data for all scenarios
    rows: list[tuple[dict, dict[str, float], dict[str, float], int]] = []
    for sc in scenarios:
        log_dir = Path(sc.get("log_dir", ""))
        mean_m, std_m, n = load_metrics_with_seeds(log_dir, split)
        rows.append((sc, mean_m, std_m, n))

    if not any(m for _, m, _, _ in rows):
        console.print(
            "[yellow]  No results.json found for any scenario — skipping.[/yellow]"
        )
        return None

    # Find baseline metrics for delta computation
    baseline_metrics: dict[str, float] = {}
    for sc, mean_m, _, _ in rows:
        if sc.get("baseline") and mean_m:
            baseline_metrics = mean_m
            break

    t = Table(
        title="H2 Feature Specification Sensitivity",
        show_header=True,
        header_style="bold",
    )
    t.add_column("Feature Set", style="cyan", no_wrap=True)
    t.add_column("N", justify="right", style="dim")
    t.add_column("Seeds", justify="right", style="dim")
    for _, display, _ in _METRICS:
        t.add_column(display, justify="right")
    if baseline_metrics:
        for _, display, _ in _METRICS:
            t.add_column(f"Δ {display}", justify="right")

    for sc, mean_m, std_m, n_seeds in rows:
        label = sc.get("label", "?")
        n_features = str(sc.get("n_features", "—"))
        seeds_str = str(n_seeds) if n_seeds > 0 else "—"
        is_baseline = sc.get("baseline", False)

        if not mean_m:
            empty = ["—"] * len(_METRICS)
            row_vals = empty
            delta_vals = empty if baseline_metrics else []
        else:
            row_vals = []
            for key, _, fmt_str in _METRICS:
                cell = fmt_val(key, mean_m.get(key), fmt_str)
                if std_m and key in std_m and n_seeds > 1:
                    std_val = std_m[key]
                    std_str = fmt_val(key, std_val, fmt_str)
                    cell = f"{cell} ±{std_str}"
                row_vals.append(cell)

            delta_vals = []
            if baseline_metrics:
                for key, _, fmt_str in _METRICS:
                    if is_baseline:
                        delta_vals.append("[dim]baseline[/dim]")
                    else:
                        delta_vals.append(
                            fmt_delta(
                                key, mean_m.get(key), baseline_metrics.get(key), fmt_str
                            )
                        )

        style = "bold green" if is_baseline else None
        t.add_row(label, n_features, seeds_str, *row_vals, *delta_vals, style=style)

    return t


def main() -> None:
    parser = argparse.ArgumentParser(description="H2 feature sensitivity report")
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

    scenarios = cfg.get("scenarios", [])
    split = args.split

    console.print()
    console.print(
        f"[bold]H2 Feature Sensitivity Report[/bold]  [dim](split: {split})[/dim]"
    )
    console.print(
        "[dim]H2: More microstructure features → stronger out-of-sample performance.[/dim]"
    )
    console.print()

    table = build_report_table(scenarios, split, console)
    if table is None:
        console.print(
            "[yellow]No results found. Run `cli.py train` for each H2 scenario first:[/yellow]"
        )
        for sc in scenarios:
            console.print(
                f"  [dim]uv run python src/cli.py train --scenario pooled/{Path(sc.get('log_dir', '')).name}[/dim]"
            )
        sys.exit(0)

    console.print(table)
    console.print()

    console.print("[dim]Legend[/dim]")
    console.print(
        "  [dim][bold green]Bold green[/bold green] = baseline (Selected / 10-feature) scenario.[/dim]"
    )
    console.print(
        "  [dim]Δ columns show deviation from baseline: [green]green[/green] = improvement, [red]red[/red] = regression.[/dim]"
    )
    console.print(
        "  [dim]Seeds column: number of independent runs aggregated (mean ± std when > 1).[/dim]"
    )
    console.print(
        "  [dim]Max DD: lower magnitude is better (negative values = drawdown below peak).[/dim]"
    )


if __name__ == "__main__":
    main()
