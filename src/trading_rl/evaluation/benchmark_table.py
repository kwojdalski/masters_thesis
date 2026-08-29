"""Save benchmark comparison table as JSON and PNG artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_rl.evaluation.asset_meta import write_asset_meta
from trading_rl.evaluation.metric_meta import METRIC_META_BY_KEY

# Columns shown in both JSON and the rendered PNG — derived from the shared registry.
_PERF_KEYS = [
    "total_return",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "win_rate",
    "profit_factor",
    "expectancy_per_period",
    "annualized_volatility",
    "return_skewness",
    "return_kurtosis",
    "pct_long",
    "pct_short",
]
_REL_KEYS = ["alpha", "beta", "information_ratio", "tracking_error"]

# Abbreviated labels for the wide benchmark table (space is tight).
_BENCH_LABEL_OVERRIDE: dict[str, str] = {
    "sharpe_ratio": "Sharpe",
    "sortino_ratio": "Sortino",
    "max_drawdown": "Max DD",
    "expectancy_per_period": "Mean/Step",
    "annualized_volatility": "Ann. Vol",
    "tracking_error": "Track. Err",
}


def _bench_cols(keys: list[str]) -> list[tuple[str, str, str]]:
    result = []
    for key in keys:
        meta = METRIC_META_BY_KEY.get(key)
        if meta is None:
            continue
        label = _BENCH_LABEL_OVERRIDE.get(key, meta.label)
        result.append((key, label, meta.fmt))
    return result


PERF_COLS: list[tuple[str, str, str]] = _bench_cols(_PERF_KEYS)
REL_COLS: list[tuple[str, str, str]] = _bench_cols(_REL_KEYS)


def _fmt(val: Any, fmt: str) -> str:
    if val is None or (isinstance(val, float) and not math.isfinite(val)):
        return "—"
    try:
        return f"{val:{fmt}}"
    except (ValueError, TypeError):
        return str(val)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return str(obj)


def build_bench_out(
    strategy_simple_returns: np.ndarray,
    benchmarks: list[Any],
    max_steps: int,
    periods_per_year: int,
) -> dict[str, Any]:
    """Compute per-benchmark metrics and relative metrics vs strategy.

    Returns a bench_out dict compatible with save_benchmark_table_artifact
    and with _print_benchmark_table in the evaluate command.
    """
    from trading_rl.evaluation.metrics import build_metric_report

    bench_out: dict[str, Any] = {}
    for spec in benchmarks:
        bench_returns = spec.compute_returns(max_steps)
        n = min(len(strategy_simple_returns), len(bench_returns))
        bench_own = build_metric_report(
            strategy_simple_returns=bench_returns[:n],
            benchmark_simple_returns=bench_returns[:n],
            actions=None,
            periods_per_year=periods_per_year,
            risk_free_rate_annual=0.0,
        )
        bench_rel = build_metric_report(
            strategy_simple_returns=strategy_simple_returns[:n],
            benchmark_simple_returns=bench_returns[:n],
            actions=None,
            periods_per_year=periods_per_year,
            risk_free_rate_annual=0.0,
        )
        bench_out[spec.name] = {
            "benchmark_metrics": bench_own.to_dict(),
            "relative_metrics": {
                k: getattr(bench_rel, k)
                for k in ("alpha", "beta", "information_ratio", "tracking_error")
            },
        }
    return bench_out


def save_benchmark_table_artifact(
    split: str,
    split_df: pd.DataFrame,
    bench_out: dict[str, Any],
    strategy_metrics: dict[str, Any] | None,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write {split}_benchmark_table.json and {split}_benchmark_table.png."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_obs = len(split_df)
    start_dt: str | None = None
    end_dt: str | None = None
    if pd.api.types.is_datetime64_any_dtype(split_df.index) and n_obs:
        start_dt = split_df.index[0].isoformat()
        end_dt = split_df.index[-1].isoformat()

    def _row_data(name: str, perf: dict, rel: dict, is_strategy: bool) -> dict:
        d: dict[str, Any] = {"name": name, "is_strategy": is_strategy}
        for key, _, _ in PERF_COLS:
            d[key] = perf.get(key)
        for key, _, _ in REL_COLS:
            d[key] = rel.get(key)
        return d

    rows: list[dict] = []
    if strategy_metrics:
        rows.append(_row_data("Strategy", strategy_metrics, {}, is_strategy=True))
    for bench_name, entry in bench_out.items():
        rows.append(
            _row_data(
                bench_name,
                entry.get("benchmark_metrics", entry),
                entry.get("relative_metrics", {}),
                is_strategy=False,
            )
        )

    artifact: dict[str, Any] = {
        "split": split,
        "n_obs": n_obs,
        "start_datetime": start_dt,
        "end_datetime": end_dt,
        "columns": [label for _, label, _ in PERF_COLS + REL_COLS],
        "rows": rows,
    }

    json_path = output_dir / f"{split}_benchmark_table.json"
    with json_path.open("w") as f:
        json.dump(artifact, f, indent=2, default=_json_default)
    write_asset_meta(json_path, generator="evaluation/benchmark_table.py")

    # --- PNG ---
    all_cols = [("name", "Name", ""), *PERF_COLS, *REL_COLS]
    col_labels = [label for _, label, _ in all_cols]
    cell_data = [
        [
            str(row.get("name", "")) if key == "name" else _fmt(row.get(key), fmt)
            for key, _, fmt in all_cols
        ]
        for row in rows
    ]

    n_rows = len(cell_data)
    n_cols = len(col_labels)
    fig, ax = plt.subplots(
        figsize=(max(22, n_cols * 1.25), max(2.5, 0.55 * (n_rows + 2)))
    )
    ax.axis("off")

    meta_parts = [f"split: {split}", f"n_obs: {n_obs:,}"]
    if start_dt:
        meta_parts.append(f"from: {start_dt}")
    if end_dt:
        meta_parts.append(f"to: {end_dt}")
    ax.set_title("  |  ".join(meta_parts), fontsize=8, loc="left", pad=6)

    tbl = ax.table(
        cellText=cell_data, colLabels=col_labels, cellLoc="center", loc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.auto_set_column_width(col=list(range(n_cols)))

    for col_idx in range(n_cols):
        tbl[0, col_idx].set_facecolor("#343a40")
        tbl[0, col_idx].set_text_props(color="white", fontweight="bold")

    strategy_row_idx = next(
        (i + 1 for i, r in enumerate(rows) if r.get("is_strategy")), None
    )
    if strategy_row_idx is not None:
        for col_idx in range(n_cols):
            tbl[strategy_row_idx, col_idx].set_facecolor("#d4edda")

    for row_idx, row in enumerate(rows):
        if row.get("is_strategy"):
            continue
        shade = "#f8f9fa" if row_idx % 2 == 0 else "#ffffff"
        for col_idx in range(n_cols):
            tbl[row_idx + 1, col_idx].set_facecolor(shade)

    png_path = output_dir / f"{split}_benchmark_table.png"
    from trading_rl.evaluation.thesis_theme import PLOT_DPI

    fig.savefig(str(png_path), dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    write_asset_meta(png_path, generator="evaluation/benchmark_table.py")

    return json_path, png_path
