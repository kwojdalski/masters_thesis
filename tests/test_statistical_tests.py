"""Regression tests for trading_rl.evaluation.statistical_tests orchestration."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from trading_rl.evaluation.statistical_tests import run_all_statistical_tests


def _config(tests: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        enabled=True,
        tests=tests,
        n_bootstrap_samples=200,
        n_permutations=200,
        confidence_level=0.95,
    )


def test_random_baseline_filters_non_finite_pairs_like_benchmark_loop():
    """The random-baseline branch must drop non-finite strategy/baseline pairs
    before testing, exactly as its comment claims it mirrors the per-benchmark
    loop — otherwise a single NaN silently propagates to p_value=nan and
    significant=False."""
    strategy_returns = np.array(
        [0.01, np.nan, 0.02, 0.015, 0.005, 0.01, 0.02, 0.015, 0.01, 0.005, 0.02]
    )
    random_trials = [np.zeros_like(strategy_returns) for _ in range(3)]

    result = run_all_statistical_tests(
        strategy_returns=strategy_returns,
        benchmarks=[],
        max_steps=len(strategy_returns),
        config=_config(["t_test"]),
        random_baseline_trials=random_trials,
    )

    random_result = result["baselines"][0]
    assert "error" not in random_result
    # The single NaN pair must be dropped, not passed through to scipy.
    assert random_result["n_strategy_samples"] == len(strategy_returns) - 1
    assert np.isfinite(random_result["t_test"]["p_value"])
