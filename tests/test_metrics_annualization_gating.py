"""Guards for annualisation gating and the zero-downside Sortino case.

Both defects surfaced on the h1 test split (247 one-minute bars = 4.1 hours):

- `sharpe_ratio_annualized` was 2 297 -- a per-bar Sharpe of 7.33 scaled by
  sqrt(98 280). Arithmetically correct, statistically meaningless from a
  four-hour window, and it would have been published as-is.
- `sortino_ratio` was NaN for TD3/DDPG/PPO because the frictionless equity
  curve has no losing one-minute bar, so downside deviation is exactly 0. The
  NaN was then dropped from results.json, leaving a bare "--" in the tables
  with no indication that "no downside at all" was the reason.
"""

from __future__ import annotations

import numpy as np
import pytest

from trading_rl.evaluation.metrics import (
    _MIN_ANNUALIZATION_YEARS,
    build_metric_report,
)

# 1-minute bars: 252 trading days x 390 minutes.
_MIN_1_PPY = 98_280


def _report(returns, ppy=_MIN_1_PPY, actions=None):
    arr = np.asarray(returns, dtype=float)
    if actions is None:
        actions = np.zeros_like(arr)
    return build_metric_report(arr, None, actions, periods_per_year=ppy)


# ---------------------------------------------------------------------------
# Annualisation gating
# ---------------------------------------------------------------------------


def test_short_window_suppresses_annualized_ratios_but_keeps_per_bar_ones() -> None:
    """A 4-hour window (the h1 test split) must not report annualised ratios."""
    rng = np.random.default_rng(0)
    # 247 bars, same shape as the real h1 test split: strong positive drift.
    r = rng.normal(loc=6.0e-3, scale=8.0e-4, size=247)

    rep = _report(r)

    assert rep.n_bars == 247.0
    assert rep.sample_too_short_to_annualize == 1.0
    assert rep.sample_years < _MIN_ANNUALIZATION_YEARS
    # The honest per-bar ratio survives ...
    assert np.isfinite(rep.sharpe_ratio)
    # ... the sqrt(ppy)-scaled one does not.
    assert np.isnan(rep.sharpe_ratio_annualized)


def test_long_window_still_reports_annualized_ratios() -> None:
    """The gate must not silently disable annualisation for real samples."""
    rng = np.random.default_rng(1)
    # 30 trading days of 1-minute bars clears the 20-day floor.
    n = int(_MIN_1_PPY * (30.0 / 252.0))
    r = rng.normal(loc=1e-6, scale=5e-4, size=n)

    rep = _report(r)

    assert rep.sample_too_short_to_annualize == 0.0
    assert rep.sample_years >= _MIN_ANNUALIZATION_YEARS
    assert np.isfinite(rep.sharpe_ratio_annualized)


def test_sample_years_matches_bars_over_effective_ppy() -> None:
    r = np.full(247, 1e-4)
    rep = _report(r)
    assert rep.sample_years == pytest.approx(
        rep.n_bars / rep.periods_per_year_used, rel=1e-12
    )


# ---------------------------------------------------------------------------
# Zero-downside Sortino
# ---------------------------------------------------------------------------


def test_all_positive_bars_report_zero_downside_bars_not_a_bare_nan() -> None:
    """Sortino is undefined here because there is no downside -- say so."""
    r = np.full(247, 5.0e-3)  # every bar positive, as in the frictionless run
    rep = _report(r)

    assert rep.downside_bars == 0.0
    assert rep.downside_deviation == 0.0
    # Sortino stays NaN (division by zero), but downside_bars explains why.
    assert np.isnan(rep.sortino_ratio)


def test_mixed_bars_count_downside_and_define_sortino() -> None:
    """The random-policy control has losing bars, so Sortino is computable."""
    rng = np.random.default_rng(2)
    r = rng.normal(loc=-2.0e-5, scale=3.0e-4, size=247)

    rep = _report(r)

    assert rep.downside_bars > 0
    assert rep.downside_deviation > 0
    assert np.isfinite(rep.sortino_ratio)


def test_downside_bars_never_exceeds_bar_count() -> None:
    rng = np.random.default_rng(3)
    r = rng.normal(loc=0.0, scale=1e-4, size=300)
    rep = _report(r)
    assert 0.0 <= rep.downside_bars <= rep.n_bars
