"""Regression guard for PR #574 / experiment-audit 2026-08-31 finding #1.

`build_metric_report` compounds a sub-daily return series into coarse reporting
"bars" before computing annualised ratios (Sharpe/Sortino/vol). The path and
frequency metrics -- win_rate, lose_rate, the drawdown block, profit_factor,
gross_profit/loss, expectancy -- must NOT use those bars: at bar resolution a
mildly-drifting session makes every bar the same sign, so win_rate saturates at
0.0/1.0, max_drawdown collapses to 0.0, and profit_factor goes NaN.

These tests use a return series long enough to actually trigger the bar
aggregation (`aggregate_to_reporting_frequency` needs >= 50 bars), so a
"simplification" that moves any of these metrics back onto the compounded bars
would fail here even though the short-array edge-case tests would still pass.
"""

from __future__ import annotations

import numpy as np
import pytest

from trading_rl.constants import ReportingFrequency
from trading_rl.evaluation.metrics import (
    _TRADE_EPS,
    aggregate_to_reporting_frequency,
    build_metric_report,
)

# 5-second bars -> ~1.18M periods/year: the regime where HFT tick evaluation
# lives and where the bar-aggregation path is exercised.
_HFT_PPY = ReportingFrequency.SEC_5.periods_per_year


def _mildly_trending_returns(
    n: int = 300_000, drift: float = 1.7e-4, noise: float = 1.0e-3, seed: int = 0
) -> np.ndarray:
    """Per-step returns with a small positive drift.

    drift/noise ~= 0.17, so ~57% of individual steps are positive, but a bar
    compounding thousands of these steps is positive with overwhelming
    probability -- exactly the h1 pathology.
    """
    rng = np.random.default_rng(seed)
    return drift + noise * rng.standard_normal(n)


def test_input_actually_triggers_the_bar_saturation_pathology():
    """Sanity check: on the compounded bars, win rate WOULD be 1.0."""
    r = _mildly_trending_returns()
    bars, ppy_used = aggregate_to_reporting_frequency(r, _HFT_PPY)

    assert bars.size >= 50, "aggregation did not produce the >=50 bars it targets"
    assert bars.size < r.size / 100, "aggregation barely compressed the series"
    assert ppy_used < _HFT_PPY, "ppy was not stepped down to a reporting frequency"
    # Every bar the same sign -- this is the bug's input condition.
    assert float(np.mean(bars > 0)) == 1.0


def test_win_and_lose_rate_are_computed_on_raw_steps_not_bars():
    r = _mildly_trending_returns()
    raw_hit = float(np.mean(r > 0))
    bars, _ = aggregate_to_reporting_frequency(r, _HFT_PPY)
    bar_hit = float(np.mean(bars > 0))

    rep = build_metric_report(r, None, None, periods_per_year=_HFT_PPY)

    # It IS the raw per-step fraction ...
    assert rep["win_rate"] == pytest.approx(raw_hit, abs=1e-9)
    assert rep["lose_rate"] == pytest.approx(float(np.mean(r < 0)), abs=1e-9)
    assert rep["win_rate"] + rep["lose_rate"] == pytest.approx(1.0, abs=1e-9)
    # ... and it is NOT the saturated bar fraction.
    assert abs(rep["win_rate"] - bar_hit) > 0.3
    assert 0.5 < rep["win_rate"] < 0.65  # the honest ~57%


def test_drawdown_block_does_not_collapse_on_a_trending_series():
    r = _mildly_trending_returns()
    rep = build_metric_report(r, None, None, periods_per_year=_HFT_PPY)

    # On the compounded (monotone) bars every one of these degenerates.
    assert rep["max_drawdown"] < 0.0
    assert rep["average_drawdown"] < 0.0
    assert rep["max_drawdown_duration"] >= 1.0
    assert np.isfinite(rep["ulcer_index"]) and rep["ulcer_index"] > 0.0


def test_profit_factor_and_gross_loss_stay_finite_and_real():
    r = _mildly_trending_returns()
    rep = build_metric_report(r, None, None, periods_per_year=_HFT_PPY)

    # gross_loss == 0.0 on the bars -> profit_factor = safe_div(x, 0) -> NaN.
    assert rep["gross_loss"] > 0.0
    assert rep["gross_profit"] > 0.0
    assert np.isfinite(rep["profit_factor"])
    assert 1.0 < rep["profit_factor"] < 5.0  # a plausible edge, not 1e6 / NaN
    assert np.isfinite(rep["payoff_ratio"]) and rep["payoff_ratio"] > 0.0
    assert rep["expectancy_per_period"] == pytest.approx(float(np.mean(r)), abs=1e-9)


def test_metadata_shows_aggregation_happened_for_annualised_metrics():
    r = _mildly_trending_returns()
    rep = build_metric_report(r, None, None, periods_per_year=_HFT_PPY)

    assert rep["n_periods"] == pytest.approx(float(r.size))
    assert rep["n_bars"] < rep["n_periods"] / 100
    assert rep["periods_per_year_used"] < _HFT_PPY


def test_total_return_is_scale_invariant_between_raw_and_bars():
    """Compounding is associative: total_return must not depend on bar size."""
    r = _mildly_trending_returns()
    raw_total = float(np.prod(1.0 + r) - 1.0)
    rep = build_metric_report(r, None, None, periods_per_year=_HFT_PPY)
    assert rep["total_return"] == pytest.approx(raw_total, rel=1e-6)


def test_per_bar_and_annualised_sharpe_are_distinct_and_consistently_scaled():
    """Finding #15: the dict must expose both scales, related by sqrt(ppy_used).

    A noisy series so the bars have both signs (a Sortino denominator exists);
    the strongly-trending helper is deliberately degenerate on that axis.
    """
    r = _mildly_trending_returns(drift=3e-5, noise=1.5e-3)
    rep = build_metric_report(r, None, None, periods_per_year=_HFT_PPY)

    assert np.isfinite(rep["sharpe_ratio"])
    assert np.isfinite(rep["sharpe_ratio_annualized"])
    assert np.isfinite(rep["sortino_ratio"])
    assert np.isfinite(rep["sortino_ratio_annualized"])

    scale = np.sqrt(rep["periods_per_year_used"])
    assert rep["sharpe_ratio_annualized"] / rep["sharpe_ratio"] == pytest.approx(
        scale, rel=1e-6
    )
    assert rep["sortino_ratio_annualized"] / rep["sortino_ratio"] == pytest.approx(
        scale, rel=1e-6
    )
    # The per-bar ratio pairs with std_return; the annualised one does not.
    assert rep["sharpe_ratio"] == pytest.approx(
        rep["mean_return"] / rep["std_return"], rel=1e-6
    )
    # Annualised != per-bar by a wide margin (ppy_used is >= 252).
    assert abs(rep["sharpe_ratio_annualized"]) > 10 * abs(rep["sharpe_ratio"])


# ---------------------------------------------------------------------------
# Finding #14: n_trades / holding period threshold on |delta position|
# ---------------------------------------------------------------------------


def test_continuous_action_drift_below_eps_is_not_counted_as_trading():
    n = 2_000
    # Position ramps by 1e-3 per step -- below _TRADE_EPS (1e-2). A raw
    # `diff != 0` test would call every step a trade.
    actions = np.arange(n, dtype=float) * (_TRADE_EPS / 20.0)
    returns = np.full(n, 1e-5)

    rep = build_metric_report(returns, None, actions, periods_per_year=252)

    assert rep["n_trades"] == 0.0
    assert rep["average_holding_period"] == pytest.approx(float(n))


def test_real_position_flips_are_counted_and_segment_the_holding_period():
    actions = np.array([1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0], dtype=float)
    returns = np.full(actions.size, 1e-4)

    rep = build_metric_report(returns, None, actions, periods_per_year=252)

    # Two crossings of magnitude 2.0 (> _TRADE_EPS): indices 2->3 and 4->5.
    assert rep["n_trades"] == 2.0
    # Segment lengths [3, 2, 2] -> mean 7/3.
    assert rep["average_holding_period"] == pytest.approx(7.0 / 3.0)


def test_tiny_jitter_around_a_flip_does_not_inflate_trade_count():
    # A single genuine flip, wrapped in sub-eps jitter on both sides.
    actions = np.array([0.001, -0.002, 0.0, 0.95, 0.951, 0.949, 0.95], dtype=float)
    returns = np.full(actions.size, 1e-4)

    rep = build_metric_report(returns, None, actions, periods_per_year=252)

    assert rep["n_trades"] == 1.0
