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


def test_random_baseline_significance_test_uses_full_per_trial_variance():
    """Averaging trials pointwise before testing (np.mean(trials, axis=0))
    collapses the baseline's true variance by ~sqrt(n_trials), making a
    strategy with zero real edge look "significantly" better than random
    purely from the averaging artifact (issue #474). The significance test
    must instead see the flat concatenation of all trials, preserving their
    real per-trial variance."""
    rng = np.random.default_rng(0)
    n_steps = 200
    n_trials = 100

    # Strategy returns and each random trial are drawn from the exact same
    # distribution -- there is no real strategy edge here at all.
    strategy_returns = rng.normal(0.0, 0.01, n_steps)
    random_trials = [rng.normal(0.0, 0.01, n_steps) for _ in range(n_trials)]

    result = run_all_statistical_tests(
        strategy_returns=strategy_returns,
        benchmarks=[],
        max_steps=n_steps,
        config=_config(["t_test"]),
        random_baseline_trials=random_trials,
    )

    random_result = result["baselines"][0]
    assert "error" not in random_result
    # The baseline sample actually used for the test must reflect genuine
    # per-observation variance, not a pointwise-averaged pseudo-trajectory
    # with variance deflated by ~1/n_trials.
    assert random_result["n_baseline_samples"] == n_trials * n_steps
    # With no real edge and correct variance, p_value should be nowhere near
    # significant on a single random draw; the pre-fix pointwise-mean
    # baseline had variance ~100x too small and could spuriously flag
    # significance here.
    assert random_result["t_test"]["p_value"] > 0.05
