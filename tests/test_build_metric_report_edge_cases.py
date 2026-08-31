"""Edge case tests for build_metric_report function."""

import numpy as np
import pytest

from trading_rl.evaluation.metrics import build_metric_report


def _report(returns, benchmark=None, ppy=252, rf=0.0, actions=None):
    """Helper to call build_metric_report with defaults."""
    return build_metric_report(
        strategy_simple_returns=np.asarray(returns, dtype=float),
        benchmark_simple_returns=np.asarray(benchmark, dtype=float)
        if benchmark is not None
        else None,
        actions=actions,
        periods_per_year=ppy,
        risk_free_rate_annual=rf,
    )


# ---------------------------------------------------------------------------
# build_metric_report — specific edge cases
# ---------------------------------------------------------------------------


class TestBuildMetricReportEdgeCases:
    """Test edge cases and numerical stability for build_metric_report."""

    def test_empty_returns_nan_dict(self):
        """Empty returns should return all NaN."""
        report = _report([])

        assert all(np.isnan(v) for v in report.to_dict().values())
        assert len(report.to_dict()) == 46  # All metric keys should be present

    def test_single_return(self):
        """Single return should work correctly."""
        returns = [0.05]
        report = _report(returns, ppy=1)  # ppy=1 so CAGR = simple total return

        assert report["total_return"] == pytest.approx(0.05, rel=1e-9)
        assert report["annualized_return_cagr"] == pytest.approx(0.05, rel=1e-9)
        assert report["max_drawdown"] == pytest.approx(0.0, abs=1e-10)

    def test_all_losses_max_drawdown_correct(self):
        """All-losses series should have max_drawdown = total_return."""
        returns = [-0.01] * 10  # Constant -1% each period
        report = _report(returns)

        total = report["total_return"]
        max_dd = report["max_drawdown"]

        # For monotone decline, max_dd should equal total_return
        assert abs(max_dd - total) < 1e-9

    def test_all_zeros_all_metrics_zero(self):
        """All-zero returns should produce zero returns; ratios are NaN (0/0)."""
        returns = [0.0] * 100
        report = _report(returns)

        assert report["total_return"] == pytest.approx(0.0, abs=1e-10)
        assert report["annualized_return_cagr"] == pytest.approx(0.0, abs=1e-10)
        assert np.isnan(report["sharpe_ratio"]) or report[
            "sharpe_ratio"
        ] == pytest.approx(0.0, abs=1e-10)
        assert np.isnan(report["sortino_ratio"]) or report[
            "sortino_ratio"
        ] == pytest.approx(0.0, abs=1e-10)
        assert report["max_drawdown"] == pytest.approx(0.0, abs=1e-10)
        assert np.isnan(report["calmar_ratio"]) or report[
            "calmar_ratio"
        ] == pytest.approx(0.0, abs=1e-10)

    def test_infinite_return_filtered(self):
        """Infinite returns should be filtered out."""
        returns = [0.01, np.inf, 0.02, -np.inf, 0.03]
        report = _report(returns)

        # Should filter infinite values and compute on finite ones only
        assert np.isfinite(report["total_return"])
        assert np.isfinite(report["annualized_return_cagr"])
        assert (
            report["max_drawdown"] >= -1.0
        )  # Should be finite and not worse than -100%

    def test_nan_return_filtered(self):
        """NaN returns should be filtered out."""
        returns = [0.01, np.nan, 0.02, np.nan, 0.03]
        report = _report(returns)

        # Should filter NaN values and compute on finite ones only
        assert np.isfinite(report["total_return"])
        assert np.isfinite(report["annualized_return_cagr"])
        assert report["max_drawdown"] >= -1.0

    def test_mixed_signs_sharpe_correct(self):
        """Mixed positive/negative returns should produce reasonable Sharpe."""
        # Returns with mean ~0 but positive risk-adjusted performance
        returns = [0.10, -0.05, 0.08, -0.03, 0.12, -0.02, 0.06]
        report = _report(returns)

        # Sharpe should be positive (returns compensate for risk)
        assert report["sharpe_ratio"] > 0.5  # Reasonable positive Sharpe

    def test_zero_std_returns_natural_sharpe(self):
        """Near-constant returns → annualised vol below minimum threshold → Sharpe NaN.

        Previously this test expected a large finite Sharpe, but the minimum-vol
        guard (_MIN_ANNUAL_VOL = 1e-3) intentionally returns NaN when the equity
        curve is essentially flat — which prevents absurdly inflated values like
        4000+ from appearing in training output tables.
        """
        returns = [0.01] * 50
        rng = np.random.default_rng(42)
        noisy_returns = [r + rng.normal(0, 1e-10) for r in returns]

        report = _report(noisy_returns)

        # annual_vol ≈ 1e-10 * sqrt(252) << 1e-3 threshold → Sharpe must be NaN
        assert np.isnan(report["sharpe_ratio"]), (
            "near-constant returns should produce NaN Sharpe (flat equity guard)"
        )

    def test_with_risk_free_rate_excess_correct(self):
        """With positive risk-free rate, excess returns should be calculated."""
        # Use stochastic returns with sufficient variance to pass the min-vol gate.
        rng = np.random.default_rng(7)
        returns = rng.normal(loc=0.05, scale=0.02, size=200).tolist()
        rf_annual = 0.02  # 2% annual risk-free rate

        report = _report(returns, rf=rf_annual)

        # Sharpe with RF should be positive (mean ~5% > 2% RF)
        assert report["sharpe_ratio"] > 0

        # Sharpe with RF should be lower than without RF (positive RF reduces excess)
        report_no_rf = _report(returns, rf=0.0)
        assert report["sharpe_ratio"] < report_no_rf["sharpe_ratio"]

    def test_negative_risk_free_rate_excess_correct(self):
        """Negative risk-free rate should increase excess returns vs RF=0."""
        rng = np.random.default_rng(11)
        returns = rng.normal(loc=0.05, scale=0.02, size=200).tolist()
        rf_annual = -0.01  # -1% annual risk-free rate

        report = _report(returns, rf=rf_annual)

        # With negative RF, excess return = 0.05 - (-0.01) = 0.06 > 0.05 → higher Sharpe
        report_rf_zero = _report(returns, rf=0.0)
        assert report["sharpe_ratio"] > report_rf_zero["sharpe_ratio"]

    def test_high_volatility_low_sharpe(self):
        """High volatility with moderate returns should produce low Sharpe."""
        rng = np.random.default_rng(123)
        # High vol (0.10), moderate mean (0.01)
        returns = rng.normal(loc=0.01, scale=0.10, size=100).tolist()

        report = _report(returns)

        # Sharpe should be positive (positive mean) and finite
        assert np.isfinite(report["sharpe_ratio"])
        assert report["sharpe_ratio"] > 0

    def test_low_volatility_high_sharpe(self):
        """Low volatility with good returns should produce high Sharpe."""
        rng = np.random.default_rng(456)
        # Low vol (0.01), good mean (0.02)
        returns = rng.normal(loc=0.02, scale=0.01, size=100).tolist()

        report = _report(returns)

        # Low vol should boost Sharpe ratio
        assert report["sharpe_ratio"] > 2.0  # Should be high due to low vol

    def test_recovery_time_from_max_drawdown(self):
        """Recovery time should be calculated correctly."""
        # Series with clear peak, trough, and recovery
        returns = [
            0.10,
            -0.05,
            0.03,
            0.02,
            0.04,
        ]  # Peak at 0.10, trough at -0.05, recovers
        report = _report(returns)

        # Recovery time should be number of steps from trough to recovery
        # Trough is at step 1 (after peak at 0), recovery at step 4
        assert report["recovery_time_from_max_drawdown"] == pytest.approx(3, abs=0.1)

    def test_no_recovery_recovery_time_is_nan(self):
        """If no recovery from max DD, recovery_time should be NaN."""
        # Monotone decline - never recovers
        returns = [0.05] + [-0.01] * 10
        report = _report(returns)

        # Never recovers from max drawdown
        assert np.isnan(report["recovery_time_from_max_drawdown"])

    def test_calmar_with_zero_drawdown(self):
        """Calmar with zero drawdown should be NaN or handled."""
        returns = [0.01] * 10  # All positive returns
        report = _report(returns)

        # Calmar = CAGR / abs(max_dd), with max_dd=0 this is division by zero
        assert np.isnan(report["calmar_ratio"]) or np.isinf(report["calmar_ratio"])

    def test_hit_rate_edge_cases(self):
        """Test hit_rate calculation edge cases."""
        # All wins
        assert _report([0.01] * 10)["win_rate"] == pytest.approx(1.0, abs=1e-10)
        # All losses
        assert _report([-0.01] * 10)["win_rate"] == pytest.approx(0.0, abs=1e-10)
        # Exactly half wins
        mixed = [0.01] * 5 + [-0.01] * 5
        assert _report(mixed)["win_rate"] == pytest.approx(0.5, abs=1e-9)

    def test_turnover_no_actions(self):
        """Turnover with no/empty actions should handle gracefully."""
        report = _report([0.01, 0.02, 0.03], actions=[])

        # Turnover should be NaN for empty actions
        assert np.isnan(report["average_holding_period"])
        assert np.isnan(report["turnover"])

    def test_single_action_turnover(self):
        """Turnover with single action should be 0."""
        # Single action: no changes
        report = _report([0.01, 0.02, 0.03], actions=[1, 1, 1])

        assert report["turnover"] == pytest.approx(0.0, abs=1e-10)

    def test_constant_actions_holding_period(self):
        """Constant actions should produce holding period equal to series length."""
        constant_actions = [1] * 50  # Always hold position 1
        report = _report([0.01] * 50, actions=constant_actions)

        # No changes, so holding period = full series length
        assert report["average_holding_period"] == pytest.approx(50.0, abs=1e-9)

    def test_very_short_series_metrics(self):
        """Very short return series should work correctly."""
        # Just 2 returns
        returns = [0.05, 0.03]
        report = _report(returns)

        # Should compute basic metrics correctly
        assert np.isfinite(report["total_return"])
        assert np.isfinite(report["annualized_return_cagr"])
        assert np.isfinite(report["sharpe_ratio"])

    def test_very_long_series_metrics(self):
        """Very long return series should work correctly."""
        # 1000 returns
        rng = np.random.default_rng(789)
        returns = rng.normal(loc=0.0005, scale=0.01, size=1000).tolist()
        report = _report(returns)

        # Should handle long series without performance issues
        assert np.isfinite(report["total_return"])
        assert np.isfinite(report["sharpe_ratio"])
        assert np.isfinite(report["sortino_ratio"])

    def test_negative_returns_sortino_vs_sharpe(self):
        """Negative returns should produce Sortino >= Sharpe."""
        # Predominantly negative returns
        returns = [-0.05, -0.02, -0.08, 0.01, -0.03, -0.04, 0.02]
        report = _report(returns)

        # Sortino uses only downside deviation, should be >= Sharpe for negative skew
        if np.isfinite(report["sortino_ratio"]) and np.isfinite(report["sharpe_ratio"]):
            assert report["sortino_ratio"] >= report["sharpe_ratio"]

    def test_positive_skew_sortino_vs_sharpe(self):
        """Positive skew returns should produce Sortino <= Sharpe."""
        # Predominantly positive returns with some small losses
        returns = [0.08, 0.12, 0.05, -0.01, 0.10, 0.07, -0.02]
        report = _report(returns)

        # With positive skew, Sharpe (uses all volatility) should be <= Sortino
        if np.isfinite(report["sortino_ratio"]) and np.isfinite(report["sharpe_ratio"]):
            assert report["sharpe_ratio"] <= report["sortino_ratio"]

    def test_skewness_and_kurtosis(self):
        """Test skewness and kurtosis calculations."""
        rng = np.random.default_rng(42)

        # Normal distribution: skewness ≈ 0, excess kurtosis ≈ 0
        normal_returns = rng.normal(loc=0.01, scale=0.02, size=200).tolist()
        report_normal = _report(normal_returns)

        assert (
            abs(report_normal["return_skewness"]) < 0.5
        )  # Near-zero skewness for normal
        assert (
            abs(report_normal["return_kurtosis"]) < 1.0
        )  # Near-zero excess kurtosis for normal

        # Highly positive skewed: many small gains, few big losses
        skewed_returns = [0.01] * 95 + [-0.5] * 5
        report_skewed = _report(skewed_returns)

        assert report_skewed["return_skewness"] < -1.0  # Strong negative skew

    def test_all_metric_keys_present(self):
        """All expected metric keys should be in the report."""
        returns = [0.01, -0.02, 0.03]
        report = _report(returns)

        expected_keys = {
            "n_periods",
            "n_bars",
            "periods_per_year_used",
            "total_return",
            "annualized_return_cagr",
            "annualized_volatility",
            "mean_return",
            "std_return",
            "sharpe_ratio",
            "sharpe_ratio_annualized",
            "sharpe_expost",
            "sortino_ratio",
            "sortino_ratio_annualized",
            "calmar_ratio",
            "omega_ratio",
            "max_drawdown",
            "average_drawdown",
            "max_drawdown_duration",
            "recovery_time_from_max_drawdown",
            "ulcer_index",
            "var_95",
            "cvar_95",
            "var_99",
            "cvar_99",
            "downside_deviation",
            "downside_deviation_annualized",
            "tail_ratio",
            "return_skewness",
            "return_kurtosis",
            "win_rate",
            "lose_rate",
            "profit_factor",
            "payoff_ratio",
            "expectancy_per_period",
            "gross_profit",
            "gross_loss",
            "turnover",
            "average_holding_period",
            "n_trades",
            "pct_long",
            "pct_short",
            "pct_neutral",
            "beta",
            "alpha",
            "information_ratio",
            "tracking_error",
        }

        assert set(report.to_dict().keys()) == expected_keys

    def test_no_benchmark_alpha_beta_are_nan(self):
        """Without benchmark, alpha and beta should be NaN."""
        returns = [0.01, -0.02, 0.03]
        report = _report(returns, benchmark=None)

        assert np.isnan(report["alpha"])
        assert np.isnan(report["beta"])
        assert np.isnan(report["information_ratio"])


class TestDownsideDeviationMatchesSortinoDenominator:
    """downside_deviation must actually be the value sortino_ratio divides by
    (issue #477) -- annualized_volatility/std_return already follow this
    raw-vs-annualized split, downside_deviation previously did not."""

    def test_downside_deviation_is_the_actual_sortino_denominator(self):
        rng = np.random.default_rng(0)
        returns = rng.normal(0.0005, 0.01, 300)
        report = _report(returns, ppy=252)

        recomputed_sortino = report["mean_return"] / report["downside_deviation"]
        assert recomputed_sortino == pytest.approx(report["sortino_ratio"], rel=1e-9)

    def test_downside_deviation_annualized_is_the_sqrt_ppy_scaled_value(self):
        rng = np.random.default_rng(1)
        returns = rng.normal(0.0005, 0.01, 300)
        report = _report(returns, ppy=252)

        expected = report["downside_deviation"] * np.sqrt(252)
        assert report["downside_deviation_annualized"] == pytest.approx(
            expected, rel=1e-9
        )
