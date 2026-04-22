"""Backward-compatible facade for evaluation plotting and returns helpers."""

from trading_rl.evaluation.benchmarks import calculate_benchmark_dsr
from trading_rl.evaluation.plots import (
    compare_rollouts,
    create_actual_returns_plot,
    create_merged_comparison_plot,
)
from trading_rl.evaluation.returns import (
    calculate_actual_returns,
    extract_tradingenv_returns,
)

__all__ = [
    "compare_rollouts",
    "calculate_actual_returns",
    "create_actual_returns_plot",
    "create_merged_comparison_plot",
    "calculate_benchmark_dsr",
    "extract_tradingenv_returns",
]
