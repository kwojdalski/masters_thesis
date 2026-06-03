"""Tests for standalone evaluation display rendering."""

from __future__ import annotations

import pandas as pd
from rich.console import Console

from cli.services.evaluation_display_service import EvaluationDisplayService


def _service() -> tuple[EvaluationDisplayService, Console]:
    console = Console(record=True, width=160)
    return EvaluationDisplayService(console), console


def test_print_metrics_table_includes_time_bounds_and_symbols() -> None:
    service, console = _service()
    frame = pd.DataFrame(
        {"close": [100.0, 101.0]},
        index=pd.date_range("2024-01-01 09:30:00", periods=2, freq="1s"),
    )

    service.print_metrics_table(
        "test_AAPL",
        {"total_return": 0.1234, "sharpe_ratio": 1.25},
        split_df=frame,
        symbols=["AAPL"],
    )

    output = console.export_text()
    assert "Metrics (test_AAPL)" in output
    assert "Start Datetime" in output
    assert "AAPL" in output
    assert "Sharpe" in output


def test_print_benchmark_table_includes_strategy_and_relative_metrics() -> None:
    service, console = _service()

    service.print_benchmark_table(
        "val",
        {
            "buy_and_hold": {
                "benchmark_metrics": {
                    "total_return": 0.10,
                    "sharpe_ratio": 0.50,
                },
                "relative_metrics": {"alpha": 0.01, "beta": 0.9},
            }
        },
        strategy_metrics={"total_return": 0.20, "sharpe_ratio": 1.0},
    )

    output = console.export_text()
    assert "Benchmark performance (val)" in output
    assert "Strategy" in output
    assert "buy_and_hold" in output
    assert "Alpha" in output
