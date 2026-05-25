"""Aggregate per-algorithm evaluation results into a unified thesis results directory."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console

from .base_command import BaseCommand


@dataclass
class CollectResultsParams:
    algorithms: list[str]
    dirs: list[str]
    output_dir: Path = field(default_factory=lambda: Path("masters_thesis_results"))
    overwrite: bool = False


_METRIC_KEYS = [
    "total_return",
    "annualized_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "win_rate",
    "profit_factor",
    "pct_long",
    "pct_short",
    "var_95",
    "cvar_95",
]

_BENCHMARK_NAMES = ["buy_and_hold", "twap", "vwap", "random"]


class CollectResultsCommand(BaseCommand):
    """Merge per-algorithm evaluation results into masters_thesis_results/."""

    def __init__(self, console: Console) -> None:
        super().__init__(console)

    def execute(self, params: CollectResultsParams) -> None:
        if len(params.algorithms) != len(params.dirs):
            self.handle_error(
                ValueError(
                    f"--algorithms and --dirs must have equal length "
                    f"(got {len(params.algorithms)} vs {len(params.dirs)})"
                ),
                "collect-results",
            )

        out = params.output_dir
        if out.exists() and not params.overwrite:
            self.console.print(
                f"[yellow]Output dir already exists: {out}. Use --overwrite to replace.[/yellow]"
            )
        out.mkdir(parents=True, exist_ok=True)
        (out / "comparison").mkdir(exist_ok=True)

        all_rows: list[dict[str, Any]] = []
        all_benchmarks: list[dict[str, Any]] = []
        algo_results: dict[str, dict[str, Any]] = {}

        for algo, src_dir_str in zip(params.algorithms, params.dirs):
            src_dir = Path(src_dir_str)
            results_json = src_dir / "results.json"

            if not results_json.exists():
                self.console.print(f"[yellow]  {algo}: results.json not found in {src_dir} — skipping[/yellow]")
                continue

            self.console.print(f"[green]  Loading {algo} from {src_dir}[/green]")
            with results_json.open() as f:
                results = json.load(f)

            algo_key = algo.lower()
            algo_results[algo_key] = results

            # Copy results + plots into per-algo subdir
            algo_out = out / algo_key
            algo_out.mkdir(exist_ok=True)
            shutil.copy2(results_json, algo_out / "results.json")
            self._copy_plots(src_dir, algo_out / "plots")

            # Extract per-split rows for comparison table
            for split_key, split_data in results.items():
                symbol = _symbol_from_split_key(split_key)
                metrics = split_data.get("metrics") or {}

                row: dict[str, Any] = {
                    "algorithm": algo,
                    "symbol": symbol,
                    "split": split_key,
                    "n_steps": split_data.get("n_steps"),
                }
                for k in _METRIC_KEYS:
                    row[k] = metrics.get(k)
                all_rows.append(row)

                # Benchmark comparisons
                for bench_name in _BENCHMARK_NAMES:
                    bench = (split_data.get("benchmarks") or {}).get(bench_name)
                    if bench is None:
                        continue
                    bench_metrics = bench.get("benchmark_metrics") or {}
                    relative = bench.get("relative_metrics") or {}
                    bench_row: dict[str, Any] = {
                        "algorithm": algo,
                        "symbol": symbol,
                        "split": split_key,
                        "benchmark": bench_name,
                    }
                    for k in _METRIC_KEYS:
                        bench_row[f"strategy_{k}"] = metrics.get(k)
                        bench_row[f"benchmark_{k}"] = bench_metrics.get(k)
                    bench_row["excess_return"] = relative.get("excess_return")
                    bench_row["information_ratio"] = relative.get("information_ratio")
                    all_benchmarks.append(bench_row)

        # Write comparison tables
        comp = out / "comparison"
        _write_json(comp / "metrics_table.json", all_rows)
        _write_csv(comp / "metrics_table.csv", all_rows)
        _write_json(comp / "benchmark_table.json", all_benchmarks)
        _write_csv(comp / "benchmark_table.csv", all_benchmarks)

        # Write manifest
        manifest = {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "algorithms": list(algo_results.keys()),
            "n_splits": len(all_rows),
            "sources": {
                algo.lower(): str(Path(d).resolve())
                for algo, d in zip(params.algorithms, params.dirs)
            },
        }
        _write_json(out / "manifest.json", manifest)

        self.console.print(f"\n[bold green]Results collected → {out}[/bold green]")
        self.console.print(f"  Algorithms : {', '.join(algo_results.keys())}")
        self.console.print(f"  Splits     : {len(all_rows)}")
        self.console.print(f"  Benchmarks : {len(all_benchmarks)} rows")

    def _copy_plots(self, src_dir: Path, plots_dir: Path) -> None:
        pngs = list(src_dir.glob("*.png"))
        if not pngs:
            return
        plots_dir.mkdir(exist_ok=True)
        for p in pngs:
            shutil.copy2(p, plots_dir / p.name)


def _symbol_from_split_key(key: str) -> str:
    """'test_AAPL' → 'AAPL', 'val_MSFT' → 'MSFT', 'test' → 'test'."""
    parts = key.split("_", 1)
    return parts[1] if len(parts) == 2 else key


def _write_json(path: Path, data: Any) -> None:
    with path.open("w") as f:
        json.dump(data, f, indent=2, default=str)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    import csv
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
