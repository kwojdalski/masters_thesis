"""Evaluation utilities for quantitative finance metrics."""

from trading_rl.evaluation.context import EvaluationContext
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
from trading_rl.evaluation.metrics import build_metric_report
from trading_rl.evaluation.report import (
    build_evaluation_report_for_trainer,
    periods_per_year_from_timeframe,
)
from trading_rl.evaluation.statistical_tests import (
    TEST_REGISTRY,
    BootstrapTest,
    MannWhitneyTest,
    PermutationMeanTest,
    PermutationTest,
    SharpeBootstrapTest,
    SortinoBootstrapTest,
    # Core classes (for extensions)
    StatisticalTest,
    # Concrete test classes
    TTest,
    build_benchmark_comparison_table,
    # Baseline computation
    compute_buy_and_hold_returns,
    compute_random_baseline_returns,
    compute_short_and_hold_returns,
    compute_twap_returns,
    compute_vwap_returns,
    # Factory and registry
    get_test,
    list_available_tests,
    mann_whitney_test,
    permutation_test,
    register_test,
    run_all_statistical_tests,
    # Orchestration
    run_statistical_tests,
    sharpe_ratio_bootstrap_test,
    sortino_ratio_bootstrap_test,
    # Individual test functions
    t_test_mean_returns,
)

__all__ = [
    "TEST_REGISTRY",
    "BootstrapTest",
    "calculate_actual_returns",
    "calculate_benchmark_dsr",
    "compare_rollouts",
    "create_actual_returns_plot",
    "create_merged_comparison_plot",
    "EvaluationContext",
    "extract_tradingenv_returns",
    "MannWhitneyTest",
    "PermutationMeanTest",
    "PermutationTest",
    "SharpeBootstrapTest",
    "SortinoBootstrapTest",
    # Statistical test classes (for extensions)
    "StatisticalTest",
    "TTest",
    "build_benchmark_comparison_table",
    "build_evaluation_report_for_trainer",
    "build_metric_report",
    # Baseline computation
    "compute_buy_and_hold_returns",
    "compute_random_baseline_returns",
    "compute_short_and_hold_returns",
    "compute_twap_returns",
    "compute_vwap_returns",
    # Factory and registry
    "get_test",
    "list_available_tests",
    "mann_whitney_test",
    "periods_per_year_from_timeframe",
    "permutation_test",
    "register_test",
    "run_all_statistical_tests",
    # Orchestration
    "run_statistical_tests",
    "sharpe_ratio_bootstrap_test",
    "sortino_ratio_bootstrap_test",
    # Individual test functions
    "t_test_mean_returns",
]
