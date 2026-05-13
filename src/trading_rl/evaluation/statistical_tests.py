"""Statistical significance testing orchestration for trading strategies."""

from __future__ import annotations

from typing import Any

import numpy as np

from logger import get_logger
from trading_rl.evaluation.benchmarks import BenchmarkEngine, BenchmarkSpec
from trading_rl.evaluation.statistical_benchmarks import (
    build_benchmark_comparison_table,
    compute_buy_and_hold_returns,
    compute_random_baseline_returns,
    compute_short_and_hold_returns,
    compute_twap_returns,
    compute_vwap_returns,
    summarize_random_baseline_trials,
)
from trading_rl.evaluation.statistical_test_registry import (
    TEST_REGISTRY,
    BootstrapTest,
    MannWhitneyTest,
    PermutationMeanTest,
    PermutationTest,
    SharpeBootstrapTest,
    SortinoBootstrapTest,
    StatisticalTest,
    TTest,
    get_test,
    list_available_tests,
    mann_whitney_test,
    permutation_test,
    register_test,
    run_statistical_tests,
    sharpe_ratio_bootstrap_test,
    sortino_ratio_bootstrap_test,
    t_test_mean_returns,
)

logger = get_logger(__name__)


def run_all_statistical_tests(
    strategy_returns: np.ndarray,
    benchmarks: list[BenchmarkSpec],
    max_steps: int,
    config: Any,
    *,
    random_baseline_trials: list[np.ndarray] | None = None,
    periods_per_year: int = 252,
) -> dict[str, Any]:
    """Run all configured statistical significance tests against pre-built benchmarks.

    Args:
        strategy_returns: Strategy simple returns array.
        benchmarks: Pre-built ``BenchmarkSpec`` list from ``BenchmarkEngine.build()``.
            Each spec is a pure callable — no environment objects involved.
        max_steps: Number of steps used when computing benchmark returns.
        config: ``StatisticalTestingConfig`` instance.
        random_baseline_trials: Pre-computed random-action return trials (one array
            per trial).  Pass ``None`` to skip the random baseline.
        periods_per_year: Annualisation factor for the benchmark comparison table.

    Returns:
        Dict with all test results organised by baseline.
    """
    if not config.enabled:
        return {"enabled": False}

    logger.info("run statistical significance tests")

    all_results: dict[str, Any] = {
        "enabled": True,
        "tests_configured": config.tests,
        "baselines": [],
    }
    benchmark_returns_map: dict[str, np.ndarray] = {}

    for spec in benchmarks:
        try:
            logger.info("compute %s baseline", spec.name)
            baseline_returns = spec.compute_returns(max_steps)
            benchmark_returns_map[spec.name] = baseline_returns

            logger.info("run tests baseline=%s n_samples=%d", spec.name, len(baseline_returns))
            baseline_results = run_statistical_tests(
                strategy_returns, baseline_returns, spec.name, config
            )
            baseline_results.update(spec.metadata)
            if "volume_source" in spec.metadata:
                all_results["vwap_volume_source"] = spec.metadata["volume_source"]
            all_results["baselines"].append(baseline_results)
            logger.info("%s tests complete", spec.name)
        except Exception as e:
            logger.error("%s comparison failed err=%s", spec.name, e)
            all_results["baselines"].append({"baseline": spec.name, "error": str(e)})

    if random_baseline_trials is not None:
        try:
            random_returns_mean = np.mean(random_baseline_trials, axis=0)
            logger.info("run tests baseline=random n_samples=%d", len(random_returns_mean))
            random_results = run_statistical_tests(
                strategy_returns, random_returns_mean, "random_actions", config
            )
            random_results.update(summarize_random_baseline_trials(random_baseline_trials))
            benchmark_returns_map["random_actions"] = random_returns_mean
            all_results["baselines"].append(random_results)
            logger.info("random baseline tests complete")
        except Exception as e:
            logger.error("random baseline comparison failed err=%s", e)
            all_results["baselines"].append({"baseline": "random_actions", "error": str(e)})

    all_results["benchmark_comparison_table"] = build_benchmark_comparison_table(
        strategy_returns=strategy_returns,
        benchmark_returns=benchmark_returns_map,
        periods_per_year=periods_per_year,
        risk_free_rate_annual=0.0,
    )

    logger.info("statistical significance testing complete")
    return all_results

__all__ = [
    "TEST_REGISTRY",
    # Benchmark types
    "BenchmarkEngine",
    "BenchmarkSpec",
    # Core classes (for advanced users/extensions)
    "BootstrapTest",
    "MannWhitneyTest",
    "PermutationMeanTest",
    "PermutationTest",
    "SharpeBootstrapTest",
    "SortinoBootstrapTest",
    "StatisticalTest",
    "TTest",
    "build_benchmark_comparison_table",
    # Baseline computation (kept for backward compatibility)
    "compute_buy_and_hold_returns",
    "compute_random_baseline_returns",
    "compute_short_and_hold_returns",
    "compute_twap_returns",
    "compute_vwap_returns",
    # Factory and registry
    "get_test",
    "list_available_tests",
    # Individual test functions (backward compatible)
    "mann_whitney_test",
    "permutation_test",
    "register_test",
    "run_all_statistical_tests",
    "run_statistical_tests",
    "sharpe_ratio_bootstrap_test",
    "sortino_ratio_bootstrap_test",
    "t_test_mean_returns",
]
