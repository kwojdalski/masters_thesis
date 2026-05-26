"""Statistical significance testing orchestration for trading strategies."""

from __future__ import annotations

from typing import Any

import numpy as np

from logger import get_logger
from trading_rl.constants import BenchmarkName
from trading_rl.evaluation.benchmarks import BenchmarkEngine, BenchmarkSpec
from trading_rl.evaluation.statistical_benchmarks import (
    build_benchmark_comparison_table,
    compute_buy_and_hold_returns,
    compute_random_baseline_returns,
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
    status_fn: Any | None = None,
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

    def _status(msg: str) -> None:
        logger.info(msg)
        if status_fn is not None:
            status_fn(msg)

    n_baselines = len(benchmarks) + (1 if random_baseline_trials is not None else 0)
    n_bootstrap = getattr(config, "n_bootstrap_samples", "?")
    _status(f"statistical tests: {n_baselines} baselines × {n_bootstrap:,} bootstrap samples")

    all_results: dict[str, Any] = {
        "enabled": True,
        "tests_configured": config.tests,
        "baselines": [],
    }
    benchmark_returns_map: dict[str, np.ndarray] = {}

    for i, spec in enumerate(benchmarks, 1):
        try:
            _status(f"  [{i}/{n_baselines}] {spec.name} ...")
            baseline_returns = spec.compute_returns(max_steps)
            benchmark_returns_map[spec.name] = baseline_returns

            baseline_results = run_statistical_tests(
                strategy_returns, baseline_returns, spec.name, config
            )
            baseline_results.update(spec.metadata)
            if "volume_source" in spec.metadata:
                all_results["vwap_volume_source"] = spec.metadata["volume_source"]
            all_results["baselines"].append(baseline_results)
            _status(f"  [{i}/{n_baselines}] {spec.name} done")
        except Exception as e:
            logger.error("%s comparison failed err=%s", spec.name, e)
            all_results["baselines"].append({"baseline": spec.name, "error": str(e)})

    if random_baseline_trials is not None:
        try:
            idx = n_baselines
            _status(f"  [{idx}/{n_baselines}] {BenchmarkName.RANDOM_ACTIONS} ...")
            trial_lens = [len(t) for t in random_baseline_trials]
            min_len = min(trial_lens)
            max_len = max(trial_lens)
            if min_len < max_len:
                n_short = sum(1 for n in trial_lens if n < max_len)
                logger.warning(
                    "random baseline: %d/%d trials shorter than max_len=%d; truncating to min_len=%d",
                    n_short,
                    len(random_baseline_trials),
                    max_len,
                    min_len,
                )
            random_returns_mean = np.mean(
                [t[:min_len] for t in random_baseline_trials], axis=0
            )
            random_results = run_statistical_tests(
                strategy_returns,
                random_returns_mean,
                BenchmarkName.RANDOM_ACTIONS,
                config,
            )
            random_results.update(summarize_random_baseline_trials(random_baseline_trials, periods_per_year=periods_per_year))
            benchmark_returns_map[BenchmarkName.RANDOM_ACTIONS] = random_returns_mean
            all_results["baselines"].append(random_results)
            _status(f"  [{idx}/{n_baselines}] {BenchmarkName.RANDOM_ACTIONS} done")
        except Exception as e:
            logger.error("random baseline comparison failed err=%s", e)
            all_results["baselines"].append(
                {"baseline": BenchmarkName.RANDOM_ACTIONS, "error": str(e)}
            )

    all_results["benchmark_comparison_table"] = build_benchmark_comparison_table(
        strategy_returns=strategy_returns,
        benchmark_returns=benchmark_returns_map,
        periods_per_year=periods_per_year,
        risk_free_rate_annual=0.0,
    )

    _status("statistical tests complete")
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
