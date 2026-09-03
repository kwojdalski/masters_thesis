from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from trading_rl.evaluation.statistical_test_registry import (
    MannWhitneyTest,
    PermutationMeanTest,
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


def _returns_with_p_value_between_one_and_five_percent() -> tuple[
    np.ndarray, np.ndarray
]:
    """Two samples whose t_test/mann_whitney/permutation p-values all fall
    strictly between 0.01 and 0.05, so confidence_level=0.99 (alpha=0.01) and
    confidence_level=0.95 (alpha=0.05) disagree on "significant"."""
    rng = np.random.default_rng(7)
    strategy = rng.normal(0.5, 1.0, 20)
    baseline = rng.normal(0.0, 1.0, 20)
    return strategy, baseline


class TestConfidenceLevelAffectsSignificance:
    """confidence_level must actually gate the significance threshold for
    every test, not just BootstrapTest -- issue #452."""

    def test_t_test_respects_configured_confidence_level(self) -> None:
        strategy, baseline = _returns_with_p_value_between_one_and_five_percent()
        loose = TTest().run(strategy, baseline, confidence_level=0.95)
        strict = TTest().run(strategy, baseline, confidence_level=0.99)

        assert 0.01 < loose["p_value"] < 0.05
        assert bool(loose["significant"]) is True
        assert bool(strict["significant"]) is False

    def test_mann_whitney_respects_configured_confidence_level(self) -> None:
        strategy, baseline = _returns_with_p_value_between_one_and_five_percent()
        loose = MannWhitneyTest().run(strategy, baseline, confidence_level=0.95)
        strict = MannWhitneyTest().run(strategy, baseline, confidence_level=0.99)

        assert 0.01 < loose["p_value"] < 0.05
        assert bool(loose["significant"]) is True
        assert bool(strict["significant"]) is False

    def test_permutation_test_respects_configured_confidence_level(self) -> None:
        strategy, baseline = _returns_with_p_value_between_one_and_five_percent()
        loose = PermutationMeanTest().run(
            strategy, baseline, n_permutations=2000, seed=123, confidence_level=0.95
        )
        strict = PermutationMeanTest().run(
            strategy, baseline, n_permutations=2000, seed=123, confidence_level=0.99
        )

        assert 0.01 < loose["p_value"] < 0.05
        assert bool(loose["significant"]) is True
        assert bool(strict["significant"]) is False


def test_healthy_bootstrap_reports_no_degenerate_resamples() -> None:
    # ~100 losing bars out of 200: a resample with zero downside is
    # astronomically unlikely, so no draw is dropped.
    series = np.where(np.arange(200) % 2 == 0, 0.01, -0.008)
    result = sortino_ratio_bootstrap_test(
        series, series, n_bootstrap=500, confidence_level=0.95, seed=1
    )

    assert result["n_bootstrap"] == 500
    assert result["n_bootstrap_valid"] == 500
    assert result["bootstrap_degenerate_fraction"] == 0.0


def test_near_lossless_series_exposes_dropped_sortino_resamples() -> None:
    # Issue #669 regime: 2 losing bars out of 237. Many resamples contain no
    # loss -> zero downside deviation -> NaN Sortino -> dropped.
    strategy = np.full(237, 0.001)
    strategy[[10, 50]] = -0.002
    baseline = np.where(np.arange(237) % 2 == 0, 0.001, -0.001)

    result = sortino_ratio_bootstrap_test(
        strategy, baseline, n_bootstrap=2000, confidence_level=0.95, seed=0
    )

    assert result["n_bootstrap"] == 2000
    assert result["n_bootstrap_valid"] < 2000
    assert result["bootstrap_degenerate_fraction"] > 0.01
    # A CI built from a heavily conditioned resample set must not read as a
    # positive significance result.
    assert result["significant"] is False


def test_all_degenerate_bootstrap_returns_nan_ci_without_crashing() -> None:
    # No losing bar anywhere: every Sortino resample is undefined.
    strategy = np.full(50, 0.001)
    baseline = np.full(50, 0.0005)

    result = sortino_ratio_bootstrap_test(
        strategy, baseline, n_bootstrap=200, confidence_level=0.95, seed=0
    )

    assert result["n_bootstrap_valid"] == 0
    assert result["bootstrap_degenerate_fraction"] == 1.0
    assert np.isnan(result["difference_ci_lower"])
    assert np.isnan(result["difference_ci_upper"])
    assert np.isnan(result["p_value"])
    assert result["significant"] is False


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
