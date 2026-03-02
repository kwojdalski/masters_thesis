"""Evaluation utilities for quantitative finance metrics."""

from trading_rl.evaluation.context import EvaluationContext
from trading_rl.evaluation.metrics import build_metric_report
from trading_rl.evaluation.report import (
    build_evaluation_report_for_trainer,
    periods_per_year_from_timeframe,
)
from trading_rl.evaluation.statistical_tests import (
    # Core classes (for extensions)
    StatisticalTest,
    BootstrapTest,
    PermutationTest,
    # Concrete test classes
    TTest,
    MannWhitneyTest,
    PermutationMeanTest,
    SharpeBootstrapTest,
    SortinoBootstrapTest,
    # Factory and registry
    get_test,
    register_test,
    list_available_tests,
    TEST_REGISTRY,
    # Individual test functions
    t_test_mean_returns,
    mann_whitney_test,
    permutation_test,
    sharpe_ratio_bootstrap_test,
    sortino_ratio_bootstrap_test,
    # Baseline computation
    compute_buy_and_hold_returns,
    compute_random_baseline_returns,
    # Orchestration
    run_statistical_tests,
    run_all_statistical_tests,
)

__all__ = [
    "EvaluationContext",
    "build_evaluation_report_for_trainer",
    "build_metric_report",
    "periods_per_year_from_timeframe",
    # Statistical test classes (for extensions)
    "StatisticalTest",
    "BootstrapTest",
    "PermutationTest",
    "TTest",
    "MannWhitneyTest",
    "PermutationMeanTest",
    "SharpeBootstrapTest",
    "SortinoBootstrapTest",
    # Factory and registry
    "get_test",
    "register_test",
    "list_available_tests",
    "TEST_REGISTRY",
    # Individual test functions
    "t_test_mean_returns",
    "mann_whitney_test",
    "permutation_test",
    "sharpe_ratio_bootstrap_test",
    "sortino_ratio_bootstrap_test",
    # Baseline computation
    "compute_buy_and_hold_returns",
    "compute_random_baseline_returns",
    # Orchestration
    "run_statistical_tests",
    "run_all_statistical_tests",
]
