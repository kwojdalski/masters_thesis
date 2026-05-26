"""Backward-compatible facade for evaluation plotting and returns helpers."""

from trading_rl.evaluation.benchmarks import calculate_benchmark_dsr
from trading_rl.evaluation.plots import (
    compare_rollouts,
    create_equity_curve_plot,
    create_equity_progression_plot,
    create_merged_comparison_plot,
    create_train_val_progression_plot,
)
from trading_rl.evaluation.returns import (
    ReturnKind,
    ReturnSeries,
    RewardSeries,
    extract_tradingenv_return_series,
    extract_tradingenv_returns,
)

__all__ = [
    "ReturnKind",
    "ReturnSeries",
    "RewardSeries",
    "calculate_benchmark_dsr",
    "compare_rollouts",
    "create_equity_curve_plot",
    "create_equity_progression_plot",
    "create_merged_comparison_plot",
    "create_train_val_progression_plot",
    "extract_tradingenv_return_series",
    "extract_tradingenv_returns",
]
