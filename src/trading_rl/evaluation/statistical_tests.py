"""Statistical significance testing orchestration for trading strategies."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from logger import get_logger
from trading_rl.evaluation.statistical_benchmarks import (
    build_benchmark_comparison_table,
    compute_buy_and_hold_returns,
    compute_random_baseline_returns,
    compute_short_and_hold_returns,
    compute_twap_returns,
    compute_vwap_returns,
    resolve_vwap_volume_series,
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
    prices: pd.Series | None,
    env: Any,
    max_steps: int,
    config: Any,
    *,
    market_data: pd.DataFrame | None = None,
    periods_per_year: int = 252,
) -> dict[str, Any]:
    """Run all configured statistical significance tests.

    Args:
        strategy_returns: Strategy simple returns
        prices: Price series for buy-and-hold comparison (optional)
        env: Trading environment for random baseline
        max_steps: Number of steps for evaluation
        config: StatisticalTestingConfig

    Returns:
        Dict with all test results organized by baseline
    """
    if not config.enabled:
        return {"enabled": False}

    logger.info("run statistical significance tests")

    all_results = {
        "enabled": True,
        "tests_configured": config.tests,
        "baselines": [],
    }
    benchmark_returns_map: dict[str, np.ndarray] = {}

    if config.compare_to_buy_and_hold and prices is not None:
        try:
            logger.info("compute buy-and-hold baseline")
            bh_returns = compute_buy_and_hold_returns(prices, max_steps)
            benchmark_returns_map["buy_and_hold"] = bh_returns

            logger.info("run tests baseline=buy_and_hold n_samples=%d", len(bh_returns))
            bh_results = run_statistical_tests(
                strategy_returns, bh_returns, "buy_and_hold", config
            )
            all_results["baselines"].append(bh_results)
            logger.info("buy-and-hold tests complete")
        except Exception as e:
            logger.error("buy-and-hold comparison failed err=%s", e)
            all_results["baselines"].append(
                {"baseline": "buy_and_hold", "error": str(e)}
            )

    if getattr(config, "compare_to_short_and_hold", False) and prices is not None:
        try:
            logger.info("compute short-and-hold baseline")
            sh_returns = compute_short_and_hold_returns(prices, max_steps)
            benchmark_returns_map["short_and_hold"] = sh_returns
            sh_results = run_statistical_tests(
                strategy_returns, sh_returns, "short_and_hold", config
            )
            all_results["baselines"].append(sh_results)
            logger.info("short-and-hold tests complete")
        except Exception as e:
            logger.error("short-and-hold comparison failed err=%s", e)
            all_results["baselines"].append(
                {"baseline": "short_and_hold", "error": str(e)}
            )

    if getattr(config, "compare_to_twap", False) and prices is not None:
        try:
            logger.info("compute twap baseline")
            twap_returns = compute_twap_returns(prices, max_steps)
            benchmark_returns_map["twap"] = twap_returns
            twap_results = run_statistical_tests(
                strategy_returns, twap_returns, "twap", config
            )
            all_results["baselines"].append(twap_results)
            logger.info("twap tests complete")
        except Exception as e:
            logger.error("twap comparison failed err=%s", e)
            all_results["baselines"].append({"baseline": "twap", "error": str(e)})

    if getattr(config, "compare_to_vwap", False) and prices is not None:
        try:
            volume_series, volume_source = resolve_vwap_volume_series(market_data)
            if volume_series is None:
                msg = (
                    "VWAP baseline skipped: no usable volume column found. "
                    "Expected one of: volume, trade_volume, last_size, size, qty, "
                    "or bid_sz_00/ask_sz_00 for proxy."
                )
                logger.warning(msg)
                all_results["baselines"].append({"baseline": "vwap", "error": msg})
            else:
                if "proxy" in str(volume_source):
                    logger.warning(
                        "VWAP is using %s. This is quote-size-weighted, not true traded volume.",
                        volume_source,
                    )
                vwap_returns = compute_vwap_returns(
                    prices, volume_series, max_steps=max_steps
                )
                benchmark_returns_map["vwap"] = vwap_returns
                vwap_results = run_statistical_tests(
                    strategy_returns, vwap_returns, "vwap", config
                )
                vwap_results["volume_source"] = volume_source
                all_results["baselines"].append(vwap_results)
                all_results["vwap_volume_source"] = volume_source
                logger.info("vwap tests complete")
        except Exception as e:
            logger.error("vwap comparison failed err=%s", e)
            all_results["baselines"].append({"baseline": "vwap", "error": str(e)})

    if config.compare_to_random:
        try:
            logger.info("compute random baseline n_trials=%d", config.n_random_trials)
            random_trials = compute_random_baseline_returns(
                env, max_steps, n_trials=config.n_random_trials, seed=config.random_seed
            )

            # Aggregate random trials (use mean across trials)
            random_returns_mean = np.mean(random_trials, axis=0)

            logger.info("run tests baseline=random n_samples=%d", len(random_returns_mean))
            random_results = run_statistical_tests(
                strategy_returns, random_returns_mean, "random_actions", config
            )
            random_results.update(summarize_random_baseline_trials(random_trials))
            benchmark_returns_map["random_actions"] = random_returns_mean

            all_results["baselines"].append(random_results)
            logger.info("random baseline tests complete")
        except Exception as e:
            logger.error("random baseline comparison failed err=%s", e)
            all_results["baselines"].append(
                {"baseline": "random_actions", "error": str(e)}
            )

    all_results["benchmark_comparison_table"] = build_benchmark_comparison_table(
        strategy_returns=strategy_returns,
        benchmark_returns=benchmark_returns_map,
        periods_per_year=periods_per_year,
    )

    logger.info("statistical significance testing complete")
    return all_results

__all__ = [
    "TEST_REGISTRY",
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
    # Baseline computation
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
