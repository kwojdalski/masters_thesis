from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pandas as pd

from cli.commands.collect_results_command import _METRIC_KEYS
from cli.commands.evaluate_command import _PERF_ROWS
from trading_rl.evaluation.benchmark_table import PERF_COLS, save_benchmark_table_artifact


def _load_script_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hft_cli_metric_lists_exclude_cagr_and_calmar() -> None:
    evaluate_keys = {key for key, _, _ in _PERF_ROWS}
    benchmark_keys = {key for key, _, _ in PERF_COLS}
    collect_keys = set(_METRIC_KEYS)

    for keys in (evaluate_keys, benchmark_keys, collect_keys):
        assert "annualized_return_cagr" not in keys
        assert "calmar_ratio" not in keys


def test_h3_report_metrics_exclude_cagr() -> None:
    h3 = _load_script_module("scripts/h3_sensitivity_report.py", "h3_sensitivity_report")

    metric_keys = {key for key, _, _ in h3._METRICS}

    assert "annualized_return_cagr" not in metric_keys


def test_benchmark_table_artifact_drops_cagr_and_calmar(tmp_path: Path) -> None:
    metrics = {
        "total_return": 1e-4,
        "annualized_return_cagr": 1e9,
        "sharpe_ratio": 1.5,
        "sortino_ratio": 2.0,
        "calmar_ratio": 1e9,
        "max_drawdown": -1e-5,
        "win_rate": 0.55,
        "profit_factor": 1.4,
        "expectancy_per_period": 1e-6,
        "annualized_volatility": 0.2,
        "return_skewness": 0.0,
        "return_kurtosis": 3.0,
        "pct_long": 0.5,
        "pct_short": 0.5,
    }
    split_df = pd.DataFrame(
        {"close": [100.0, 100.1, 100.2]},
        index=pd.date_range("2026-01-01 14:30:00", periods=3, freq="s", tz="UTC"),
    )

    json_path, _ = save_benchmark_table_artifact(
        "test",
        split_df,
        bench_out={
            "buy_and_hold": {
                "benchmark_metrics": metrics,
                "relative_metrics": {},
            }
        },
        strategy_metrics=metrics,
        output_dir=tmp_path,
    )

    rows = json.loads(json_path.read_text())["rows"]
    assert rows
    for row in rows:
        assert "annualized_return_cagr" not in row
        assert "calmar_ratio" not in row
        assert math.isfinite(row["total_return"])
