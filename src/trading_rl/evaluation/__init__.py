"""Evaluation utilities for quantitative finance metrics."""

from trading_rl.evaluation.context import EvaluationContext
from trading_rl.evaluation.metrics import build_metric_report
from trading_rl.evaluation.report import (
    build_evaluation_report_for_trainer,
    periods_per_year_from_timeframe,
)
from trading_rl.evaluation.statistical_tests import (
    run_all_statistical_tests,
    run_statistical_tests,
    compute_buy_and_hold_returns,
    compute_random_baseline_returns,
    t_test_mean_returns,
    mann_whitney_test,
    permutation_test,
    sharpe_ratio_bootstrap_test,
    sortino_ratio_bootstrap_test,
)

__all__ = [
    "EvaluationContext",
    "build_evaluation_report_for_trainer",
    "build_metric_report",
    "periods_per_year_from_timeframe",
    "run_all_statistical_tests",
    "run_statistical_tests",
    "compute_buy_and_hold_returns",
    "compute_random_baseline_returns",
    "t_test_mean_returns",
    "mann_whitney_test",
    "permutation_test",
    "sharpe_ratio_bootstrap_test",
    "sortino_ratio_bootstrap_test",
]
