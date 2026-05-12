"""Unit tests for trading_rl.evaluation.metrics."""

from __future__ import annotations

import numpy as np
import pytest

from trading_rl.evaluation.metrics import (
    _drawdown_series,
    _equity_curve,
    build_metric_report,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PPY = 252  # periods per year for simple daily tests


def _report(returns, benchmark=None, ppy=PPY):
    return build_metric_report(
        strategy_simple_returns=np.asarray(returns, dtype=float),
        benchmark_simple_returns=np.asarray(benchmark, dtype=float) if benchmark is not None else None,
        actions=None,
        periods_per_year=ppy,
    )


# ---------------------------------------------------------------------------
# _equity_curve — must start from 1.0
# ---------------------------------------------------------------------------

class TestEquityCurve:
    def test_starts_at_one(self):
        curve = _equity_curve(np.array([0.01, -0.02, 0.03]))
        assert curve[0] == pytest.approx(1.0)

    def test_length_is_n_plus_one(self):
        r = np.array([0.01, 0.02, -0.01])
        assert len(_equity_curve(r)) == len(r) + 1

    def test_final_value_equals_compound_return(self):
        r = np.array([0.10, -0.05, 0.08])
        curve = _equity_curve(r)
        expected = (1.10) * (0.95) * (1.08)
        assert curve[-1] == pytest.approx(expected, rel=1e-10)

    def test_all_zeros_stays_flat(self):
        r = np.zeros(5)
        curve = _equity_curve(r)
        np.testing.assert_allclose(curve, 1.0)


# ---------------------------------------------------------------------------
# _drawdown_series — initial decline must be captured
# ---------------------------------------------------------------------------

class TestDrawdownSeries:
    def test_immediate_decline_is_captured(self):
        """First period decline from 1.0 must not be zero-masked."""
        r = np.array([-0.10, -0.05])
        equity = _equity_curve(r)
        dd = _drawdown_series(equity)
        # First drawdown (after period 1) should be -10 %
        assert dd[1] == pytest.approx(-0.10, rel=1e-6)

    def test_no_drawdown_all_ones(self):
        r = np.zeros(10)
        equity = _equity_curve(r)
        dd = _drawdown_series(equity)
        np.testing.assert_allclose(dd, 0.0, atol=1e-12)

    def test_recovery_resets_drawdown(self):
        r = np.array([-0.10, 0.20])       # -10 % then +20 % → above starting value
        equity = _equity_curve(r)
        dd = _drawdown_series(equity)
        assert dd[-1] == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Total return
# ---------------------------------------------------------------------------

class TestTotalReturn:
    def test_zero_returns(self):
        r = [_report([0.0, 0.0, 0.0])["total_return"]]
        assert r[0] == pytest.approx(0.0, abs=1e-12)

    def test_single_positive_period(self):
        assert _report([0.05])["total_return"] == pytest.approx(0.05, rel=1e-9)

    def test_single_negative_period(self):
        assert _report([-0.05])["total_return"] == pytest.approx(-0.05, rel=1e-9)

    def test_compounding(self):
        # +10 % then -10 % → not zero
        r = _report([0.10, -0.10])["total_return"]
        assert r == pytest.approx((1.10 * 0.90) - 1.0, rel=1e-9)

    def test_consistent_with_equity_curve(self):
        returns = [0.01, -0.02, 0.03, -0.01, 0.02]
        report = _report(returns)
        equity_final = np.prod([1 + x for x in returns])
        assert report["total_return"] == pytest.approx(equity_final - 1.0, rel=1e-9)


# ---------------------------------------------------------------------------
# CAGR
# ---------------------------------------------------------------------------

class TestCAGR:
    def test_flat_returns_cagr_is_zero(self):
        r = [0.0] * PPY
        assert _report(r)["annualized_return_cagr"] == pytest.approx(0.0, abs=1e-10)

    def test_one_full_year_cagr_equals_total_return(self):
        # Exactly PPY periods of constant return r_p → CAGR ≈ (1+r_p)^PPY - 1
        r_p = 0.001
        returns = [r_p] * PPY
        report = _report(returns)
        expected_total = (1 + r_p) ** PPY - 1
        assert report["total_return"] == pytest.approx(expected_total, rel=1e-6)
        # CAGR over exactly one year equals total return
        assert report["annualized_return_cagr"] == pytest.approx(expected_total, rel=1e-6)

    def test_cagr_annualizes_correctly_over_two_years(self):
        # Two years of 5 % annual growth: total_return ≈ 10.25 %, CAGR = 5 %
        r_per_period = (1.05) ** (1 / PPY) - 1
        returns = [r_per_period] * (2 * PPY)
        report = _report(returns)
        assert report["annualized_return_cagr"] == pytest.approx(0.05, rel=1e-4)


# ---------------------------------------------------------------------------
# Max drawdown
# ---------------------------------------------------------------------------

class TestMaxDrawdown:
    def test_monotone_decline_equals_total_return(self):
        returns = [-0.01] * 10
        report = _report(returns)
        # max drawdown should approximately equal total return in a monotone decline
        total = report["total_return"]
        assert report["max_drawdown"] == pytest.approx(total, rel=1e-4)

    def test_immediate_decline_is_captured(self):
        """Regression: equity curve starting at (1+r[0]) caused drawdown[0]=0."""
        report = _report([-0.10])
        assert report["max_drawdown"] == pytest.approx(-0.10, rel=1e-6)

    def test_no_drawdown_for_rising_curve(self):
        returns = [0.01] * 20
        assert _report(returns)["max_drawdown"] == pytest.approx(0.0, abs=1e-10)

    def test_max_dd_negative_or_zero(self):
        returns = [0.01, -0.05, 0.02, -0.03, 0.04]
        assert _report(returns)["max_drawdown"] <= 0.0

    def test_max_dd_peak_to_trough(self):
        # +50 % then -40 % → drawdown from peak = -40 %
        report = _report([0.50, -0.40])
        assert report["max_drawdown"] == pytest.approx(-0.40, rel=1e-6)


# ---------------------------------------------------------------------------
# Sharpe ratio
# ---------------------------------------------------------------------------

class TestSharpeRatio:
    def test_zero_std_returns_extreme(self):
        # Constant returns → std ≈ 0 (floating-point noise), Sharpe blows up.
        # _safe_div only guards exact zero; near-zero sigma is not caught.
        returns = [0.001] * 100
        sr = _report(returns)["sharpe_ratio"]
        # Either NaN (exact zero sigma) or a very large finite number
        assert np.isnan(sr) or abs(sr) > 1e10

    def test_sign_positive_for_positive_excess_return(self):
        returns = [0.005] * 50 + [-0.001] * 50
        assert _report(returns)["sharpe_ratio"] > 0

    def test_sign_negative_for_negative_mean(self):
        # Varying negative returns → negative Sharpe
        rng = np.random.default_rng(5)
        returns = rng.normal(loc=-0.005, scale=0.01, size=100).tolist()
        assert _report(returns)["sharpe_ratio"] < 0

    def test_known_value(self):
        # Sharpe = mean_excess * sqrt(ppy) / std; rf = 0
        rng = np.random.default_rng(42)
        r = rng.normal(loc=0.001, scale=0.01, size=500)
        mu = np.mean(r)
        sigma = np.std(r, ddof=1)
        expected = mu * np.sqrt(PPY) / sigma
        assert _report(r)["sharpe_ratio"] == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Sortino ratio
# ---------------------------------------------------------------------------

class TestSortinoRatio:
    def test_sortino_geq_sharpe_for_positive_skew(self):
        """With no downside returns Sortino is infinite (or NaN), otherwise >= Sharpe."""
        rng = np.random.default_rng(7)
        r = rng.normal(loc=0.002, scale=0.005, size=300)
        report = _report(r)
        sharpe = report["sharpe_ratio"]
        sortino = report["sortino_ratio"]
        if np.isfinite(sortino) and np.isfinite(sharpe):
            assert sortino >= sharpe

    def test_sign_matches_sharpe(self):
        rng = np.random.default_rng(99)
        r = rng.normal(loc=-0.001, scale=0.01, size=200)
        report = _report(r)
        sharpe = report["sharpe_ratio"]
        sortino = report["sortino_ratio"]
        if np.isfinite(sharpe) and np.isfinite(sortino):
            assert np.sign(sharpe) == np.sign(sortino)


# ---------------------------------------------------------------------------
# Win rate
# ---------------------------------------------------------------------------

class TestWinRate:
    def test_all_positive(self):
        assert _report([0.01] * 10)["win_rate"] == pytest.approx(1.0)

    def test_all_negative(self):
        assert _report([-0.01] * 10)["win_rate"] == pytest.approx(0.0)

    def test_half_half(self):
        returns = [0.01, -0.01] * 50
        assert _report(returns)["win_rate"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Profit factor
# ---------------------------------------------------------------------------

class TestProfitFactor:
    def test_all_wins_is_nan(self):
        # No losses → division by zero → NaN
        pf = _report([0.01] * 10)["profit_factor"]
        assert np.isnan(pf)

    def test_all_losses_is_zero(self):
        pf = _report([-0.01] * 10)["profit_factor"]
        assert pf == pytest.approx(0.0)

    def test_known_value(self):
        # 5 wins of 0.02 each, 5 losses of 0.01 each → PF = 0.10 / 0.05 = 2.0
        returns = [0.02] * 5 + [-0.01] * 5
        assert _report(returns)["profit_factor"] == pytest.approx(2.0, rel=1e-9)

    def test_above_one_means_profitable(self):
        returns = [0.03, -0.01, 0.02, -0.01, 0.04, -0.01]
        report = _report(returns)
        assert report["profit_factor"] > 1.0
        assert report["total_return"] > 0.0


# ---------------------------------------------------------------------------
# Beta / alpha / information ratio (benchmark metrics)
# ---------------------------------------------------------------------------

class TestBenchmarkMetrics:
    def test_beta_one_when_identical(self):
        rng = np.random.default_rng(0)
        b = rng.normal(0, 0.01, 200)
        report = _report(b, benchmark=b)
        assert report["beta"] == pytest.approx(1.0, abs=1e-9)

    def test_alpha_zero_when_identical(self):
        rng = np.random.default_rng(0)
        b = rng.normal(0, 0.01, 200)
        report = _report(b, benchmark=b)
        assert report["alpha"] == pytest.approx(0.0, abs=1e-9)

    def test_info_ratio_nan_when_no_benchmark(self):
        assert np.isnan(_report([0.01] * 10)["information_ratio"])

    def test_info_ratio_nan_when_identical(self):
        # When strategy == benchmark, active returns are all zero → std(active)=0
        # → _safe_div returns NaN (0/0 case)
        rng = np.random.default_rng(1)
        b = rng.normal(0, 0.01, 200)
        report = _report(b, benchmark=b)
        assert np.isnan(report["information_ratio"])


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_returns_all_nan(self):
        report = _report([])
        assert all(np.isnan(v) for v in report.values())

    def test_single_return(self):
        report = _report([0.05])
        assert report["total_return"] == pytest.approx(0.05)
        assert report["max_drawdown"] == pytest.approx(0.0, abs=1e-10)

    def test_single_negative_return_max_dd(self):
        report = _report([-0.05])
        assert report["max_drawdown"] == pytest.approx(-0.05, rel=1e-6)

    def test_non_finite_returns_filtered(self):
        returns = [0.01, np.nan, 0.02, np.inf, -0.01]
        report = _report(returns)
        assert np.isfinite(report["total_return"])
