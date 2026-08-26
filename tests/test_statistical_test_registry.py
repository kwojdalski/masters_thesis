from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from trading_rl.evaluation.statistical_test_registry import (
    TTest,
    _two_sided_bootstrap_p_value,
    get_test,
    list_available_tests,
    mann_whitney_test,
    permutation_test,
    run_statistical_tests,
    sharpe_ratio_bootstrap_test,
    sortino_ratio_bootstrap_test,
    t_test_mean_returns,
)

STRATEGY = np.array([0.02, 0.01, -0.005, 0.015, 0.0])
BASELINE = np.array([0.005, -0.002, 0.0, 0.004, -0.003])


def test_bootstrap_p_value_counts_both_tails() -> None:
    bootstrap_differences = np.array([-2.0, -1.0, 1.0, 2.0, 3.0])

    assert _two_sided_bootstrap_p_value(bootstrap_differences) == pytest.approx(0.8)


def test_get_test_returns_registered_test_instance() -> None:
    assert isinstance(get_test("t_test"), TTest)


def test_get_test_returns_none_for_unknown_name() -> None:
    assert get_test("not_a_test") is None


def test_list_available_tests_contains_default_registry() -> None:
    assert set(list_available_tests()) >= {
        "t_test",
        "mann_whitney",
        "permutation_test",
        "sharpe_bootstrap",
        "sortino_bootstrap",
    }


def test_t_test_wrapper_reports_sample_means() -> None:
    result = t_test_mean_returns(STRATEGY, BASELINE)

    assert result["test_name"] == "t_test"
    assert result["strategy_mean"] == pytest.approx(float(np.mean(STRATEGY)))
    assert result["baseline_mean"] == pytest.approx(float(np.mean(BASELINE)))


def test_mann_whitney_wrapper_reports_sample_medians() -> None:
    result = mann_whitney_test(STRATEGY, BASELINE)

    assert result["test_name"] == "mann_whitney"
    assert result["strategy_median"] == pytest.approx(float(np.median(STRATEGY)))
    assert result["baseline_median"] == pytest.approx(float(np.median(BASELINE)))


def test_permutation_test_is_reproducible_with_seed() -> None:
    first = permutation_test(STRATEGY, BASELINE, n_permutations=50, seed=123)
    second = permutation_test(STRATEGY, BASELINE, n_permutations=50, seed=123)

    assert first == second


def test_permutation_test_observed_statistic_is_difference_in_means() -> None:
    result = permutation_test(STRATEGY, BASELINE, n_permutations=10, seed=7)

    assert result["observed_statistic"] == pytest.approx(
        float(np.mean(STRATEGY) - np.mean(BASELINE))
    )


def test_sharpe_bootstrap_result_contains_confidence_interval_fields() -> None:
    result = sharpe_ratio_bootstrap_test(
        STRATEGY,
        BASELINE,
        n_bootstrap=25,
        confidence_level=0.90,
        seed=42,
    )

    assert result["test_name"] == "sharpe_bootstrap"
    assert result["n_bootstrap"] == 25
    assert result["confidence_level"] == 0.90
    assert "difference_ci_lower" in result
    assert "difference_ci_upper" in result


def test_sortino_bootstrap_result_uses_sortino_metric_names() -> None:
    result = sortino_ratio_bootstrap_test(
        STRATEGY,
        BASELINE,
        n_bootstrap=25,
        confidence_level=0.90,
        seed=42,
    )

    assert result["test_name"] == "sortino_bootstrap"
    assert "strategy_sortino" in result
    assert "baseline_sortino" in result
    assert "sortino_difference" in result


def test_run_statistical_tests_records_counts_and_skips_unknown_tests() -> None:
    config = SimpleNamespace(
        tests=["t_test", "unknown_test"],
        n_bootstrap_samples=10,
        n_permutations=10,
        confidence_level=0.95,
        random_seed=99,
    )

    result = run_statistical_tests(STRATEGY, BASELINE, "buy_and_hold", config)

    assert result["baseline"] == "buy_and_hold"
    assert result["n_strategy_samples"] == len(STRATEGY)
    assert result["n_baseline_samples"] == len(BASELINE)
    assert "t_test" in result
    assert "unknown_test" not in result
