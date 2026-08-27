"""Direct tests for metrics.py private helpers and build_metric_report with actions."""

from __future__ import annotations

import numpy as np
import pytest

from trading_rl.evaluation.metrics import (
    _drawdown_stats,
    _holding_period,
    _tail_risk,
    _turnover,
    build_metric_report,
)

# ---------------------------------------------------------------------------
# _tail_risk
# ---------------------------------------------------------------------------


class TestTailRisk:
    def test_var_is_5th_percentile(self):
        rng = np.random.default_rng(0)
        r = rng.normal(0, 1, 1000)
        var, _ = _tail_risk(r, alpha=0.05)
        assert var == pytest.approx(np.quantile(r, 0.05), rel=1e-9)

    def test_cvar_is_mean_of_tail(self):
        rng = np.random.default_rng(1)
        r = rng.normal(0, 1, 1000)
        var, cvar = _tail_risk(r, alpha=0.05)
        tail = r[r <= var]
        assert cvar == pytest.approx(np.mean(tail), rel=1e-9)

    def test_cvar_leq_var(self):
        rng = np.random.default_rng(2)
        r = rng.normal(0, 1, 500)
        var, cvar = _tail_risk(r)
        assert cvar <= var

    def test_empty_returns_nan_tuple(self):
        var, cvar = _tail_risk(np.array([]))
        assert np.isnan(var)
        assert np.isnan(cvar)

    def test_all_equal_values(self):
        r = np.ones(100)
        var, cvar = _tail_risk(r)
        assert var == pytest.approx(1.0)
        assert cvar == pytest.approx(1.0)

    def test_known_sorted_sequence(self):
        r = np.arange(1.0, 21.0)  # 1..20
        var, _cvar = _tail_risk(r, alpha=0.05)
        assert var == pytest.approx(np.quantile(r, 0.05), rel=1e-9)

    def test_var_alpha_01_is_1st_percentile(self):
        r = np.linspace(0, 1, 1000)
        var, _ = _tail_risk(r, alpha=0.01)
        assert var == pytest.approx(np.quantile(r, 0.01), rel=1e-6)


# ---------------------------------------------------------------------------
# _turnover
# ---------------------------------------------------------------------------


class TestTurnover:
    def test_empty_actions_returns_nan(self):
        assert np.isnan(_turnover(np.array([])))

    def test_constant_actions_returns_zero(self):
        assert _turnover(np.ones(10)) == pytest.approx(0.0)

    def test_alternating_actions_exact(self):
        actions = np.array([0.0, 1.0, 0.0, 1.0])
        assert _turnover(actions) == pytest.approx(1.0, rel=1e-9)

    def test_single_action_returns_zero(self):
        assert _turnover(np.array([1.0])) == pytest.approx(0.0)

    def test_two_actions_with_known_diff(self):
        actions = np.array([0.0, 0.5])
        assert _turnover(actions) == pytest.approx(0.5)

    def test_2d_actions_sums_over_assets(self):
        # Each of 2 assets changes by 1.0 → sum per step = 2.0; two transitions → mean = 2.0
        actions = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
        to = _turnover(actions)
        assert to == pytest.approx(2.0, rel=1e-9)

    def test_build_metric_report_includes_turnover_with_actions(self):
        returns = np.random.default_rng(42).normal(0, 0.01, 100)
        actions = np.zeros(100)
        actions[::10] = 1.0
        report = build_metric_report(returns, None, actions, periods_per_year=252)
        assert np.isfinite(report["turnover"])
        assert report["turnover"] >= 0.0


# ---------------------------------------------------------------------------
# _holding_period
# ---------------------------------------------------------------------------


class TestHoldingPeriod:
    def test_empty_actions_returns_nan(self):
        assert np.isnan(_holding_period(np.array([])))

    def test_constant_actions_returns_full_length(self):
        actions = np.array([1, 1, 1, 1, 1], dtype=float)
        assert _holding_period(actions) == pytest.approx(5.0)

    def test_alternating_actions_returns_one(self):
        actions = np.array([0, 1, 0, 1, 0, 1], dtype=float)
        result = _holding_period(actions)
        assert result == pytest.approx(1.0, rel=1e-9)

    def test_single_switch_midpoint(self):
        actions = np.array([0, 0, 0, 1, 1, 1], dtype=float)
        result = _holding_period(actions)
        assert result == pytest.approx(3.0, rel=1e-9)

    def test_2d_actions_uses_argmax(self):
        # Two-column one-hot: first 3 rows pick col 0, next 3 pick col 1
        actions = np.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ]
        )
        result = _holding_period(actions)
        assert result == pytest.approx(3.0, rel=1e-9)

    def test_single_action_returns_one(self):
        assert _holding_period(np.array([1.0])) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _drawdown_stats — recovery_time and avg_dd
# ---------------------------------------------------------------------------


class TestDrawdownStats:
    def _dd(self, returns):
        from trading_rl.evaluation.metrics import _drawdown_series, _equity_curve

        equity = _equity_curve(np.asarray(returns, dtype=float))
        return _drawdown_series(equity)

    def test_no_drawdown_max_duration_is_zero(self):
        dd = self._dd([0.01, 0.02, 0.01])
        _, _, max_dur, _ = _drawdown_stats(dd)
        assert max_dur == 0

    def test_max_drawdown_duration_correct(self):
        # -5% for 4 periods then recover
        dd = self._dd([-0.05, -0.02, -0.01, -0.03, 0.20])
        _, _, max_dur, _ = _drawdown_stats(dd)
        assert max_dur == 4

    def test_average_drawdown_is_negative_when_losses_exist(self):
        dd = self._dd([-0.10, -0.05, 0.20])
        _, avg_dd, _, _ = _drawdown_stats(dd)
        assert avg_dd < 0

    def test_max_drawdown_duration_matches_the_episode_containing_max_drawdown(self):
        """max_drawdown_duration must describe the same episode as
        max_drawdown, not the longest underwater run anywhere in the series
        (issue #454). Here a shallow 6-period dip fully recovers, then a
        sharp 2-period drop becomes the actual max_drawdown -- the longest
        run (6) belongs to a different, shallower episode."""
        dd = self._dd([-0.02] * 6 + [0.30] + [-0.50] * 2)
        max_dd, _, max_dur, _ = _drawdown_stats(dd)
        assert max_dd == pytest.approx(-0.75)
        assert max_dur == 2

    def test_recovery_time_zero_after_immediate_recovery(self):
        dd = self._dd([-0.10, 0.20])
        _, _, _, rec = _drawdown_stats(dd)
        assert rec == pytest.approx(1)

    def test_no_recovery_gives_nan(self):
        dd = self._dd([-0.10, -0.05, -0.02])
        _, _, _, rec = _drawdown_stats(dd)
        assert np.isnan(rec)


# ---------------------------------------------------------------------------
# build_metric_report — position percentages via actions
# ---------------------------------------------------------------------------


class TestBuildMetricReportWithActions:
    def test_all_long_pct_long_equals_one(self):
        returns = np.random.default_rng(99).normal(0, 0.01, 50)
        actions = np.ones(50)
        report = build_metric_report(returns, None, actions, periods_per_year=252)
        assert report["pct_long"] == pytest.approx(1.0)
        assert report["pct_short"] == pytest.approx(0.0)

    def test_all_short_pct_short_equals_one(self):
        returns = np.random.default_rng(99).normal(0, 0.01, 50)
        actions = -np.ones(50)
        report = build_metric_report(returns, None, actions, periods_per_year=252)
        assert report["pct_short"] == pytest.approx(1.0)
        assert report["pct_long"] == pytest.approx(0.0)

    def test_mixed_actions_fractions_sum_correctly(self):
        returns = np.random.default_rng(7).normal(0, 0.01, 100)
        actions = np.array([1.0] * 70 + [-1.0] * 30)
        report = build_metric_report(returns, None, actions, periods_per_year=252)
        assert report["pct_long"] == pytest.approx(0.70, rel=1e-9)
        assert report["pct_short"] == pytest.approx(0.30, rel=1e-9)

    def test_none_actions_pct_long_is_nan(self):
        returns = np.random.default_rng(5).normal(0, 0.01, 50)
        report = build_metric_report(returns, None, None, periods_per_year=252)
        assert np.isnan(report["pct_long"])
        assert np.isnan(report["pct_short"])
