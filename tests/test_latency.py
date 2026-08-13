"""Unit tests for trading_rl.envs.latency.

No test file previously referenced LatencyModel, FixedLatency,
LogNormalTimedLatency, or _us_to_ticks anywhere in the suite, despite
this module directly controlling execution-realism timing in every
streaming-env experiment with latency configured. A regression here
(an off-by-one in the searchsorted bound, a wrong clamp, a broken
priority order in make_latency_model) would silently change fill
timing with nothing to catch it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_rl.envs.latency import (
    FixedLatency,
    FixedTimedLatency,
    LogNormalLatency,
    LogNormalTimedLatency,
    PoissonLatency,
    UniformLatency,
    ZeroLatency,
    _us_to_ticks,
    make_latency_model,
)


class _FakeRng:
    """Deterministic stand-in for np.random.Generator, returning a fixed
    value regardless of the distribution parameters passed in -- isolates
    the floor/clamp logic in each model from actual sampling variance."""

    def __init__(self, value: float) -> None:
        self.value = value
        self.calls: list[tuple] = []

    def lognormal(self, mean, sigma):
        self.calls.append(("lognormal", mean, sigma))
        return self.value

    def poisson(self, lam):
        self.calls.append(("poisson", lam))
        return self.value

    def integers(self, low, high):
        self.calls.append(("integers", low, high))
        return self.value


def _timestamps_from_offsets_us(offsets_us: list[float]) -> pd.DatetimeIndex:
    base = pd.Timestamp("2024-01-01 09:30:00")
    return pd.DatetimeIndex([base + pd.Timedelta(microseconds=o) for o in offsets_us])


# ---------------------------------------------------------------------------
# ZeroLatency / FixedLatency
# ---------------------------------------------------------------------------


def test_zero_latency_always_returns_zero() -> None:
    rng = np.random.default_rng(0)
    model = ZeroLatency()
    assert all(model.sample(rng) == 0 for _ in range(5))


def test_fixed_latency_returns_constant_ticks() -> None:
    rng = np.random.default_rng(0)
    model = FixedLatency(7)
    assert all(model.sample(rng) == 7 for _ in range(5))


def test_fixed_latency_rejects_negative_ticks() -> None:
    with pytest.raises(ValueError, match="ticks must be >= 0"):
        FixedLatency(-1)


def test_fixed_latency_allows_zero_ticks() -> None:
    assert FixedLatency(0).ticks == 0


# ---------------------------------------------------------------------------
# LatencyModel.resolve() default delegates to sample() and ignores timestamps
# ---------------------------------------------------------------------------


def test_resolve_default_delegates_to_sample_and_ignores_timestamps() -> None:
    rng = np.random.default_rng(0)
    model = FixedLatency(3)
    timestamps = _timestamps_from_offsets_us([0, 10, 20])

    assert model.resolve(rng, timestamps) == 3
    # tick-based resolve must not depend on the timestamps content at all
    other_timestamps = _timestamps_from_offsets_us([0, 100_000])
    assert model.resolve(rng, other_timestamps) == 3


def test_base_sample_raises_not_implemented() -> None:
    from trading_rl.envs.latency import LatencyModel

    with pytest.raises(NotImplementedError):
        LatencyModel().sample(np.random.default_rng(0))


# ---------------------------------------------------------------------------
# LogNormalLatency
# ---------------------------------------------------------------------------


def test_lognormal_latency_floors_to_min_ticks_when_sample_is_low() -> None:
    fake_rng = _FakeRng(value=0.2)  # rounds to 0, below min_ticks
    model = LogNormalLatency(mu=0.0, sigma=1.0, min_ticks=3)

    assert model.sample(fake_rng) == 3
    assert fake_rng.calls == [("lognormal", 0.0, 1.0)]


def test_lognormal_latency_uses_rounded_sample_when_above_floor() -> None:
    fake_rng = _FakeRng(value=8.6)
    model = LogNormalLatency(mu=0.0, sigma=1.0, min_ticks=1)

    assert model.sample(fake_rng) == 9  # round(8.6) == 9


def test_lognormal_latency_real_rng_never_below_min_ticks() -> None:
    rng = np.random.default_rng(0)
    model = LogNormalLatency(mu=0.0, sigma=2.0, min_ticks=5)

    samples = [model.sample(rng) for _ in range(200)]
    assert all(s >= 5 for s in samples)
    assert all(isinstance(s, int) for s in samples)


# ---------------------------------------------------------------------------
# PoissonLatency
# ---------------------------------------------------------------------------


def test_poisson_latency_floors_to_min_ticks_when_sample_is_low() -> None:
    fake_rng = _FakeRng(value=1)
    model = PoissonLatency(lam=3.0, min_ticks=4)

    assert model.sample(fake_rng) == 4
    assert fake_rng.calls == [("poisson", 3.0)]


def test_poisson_latency_uses_sample_when_above_floor() -> None:
    fake_rng = _FakeRng(value=10)
    model = PoissonLatency(lam=3.0, min_ticks=0)

    assert model.sample(fake_rng) == 10


def test_poisson_latency_real_rng_never_below_min_ticks() -> None:
    rng = np.random.default_rng(0)
    model = PoissonLatency(lam=1.0, min_ticks=2)

    samples = [model.sample(rng) for _ in range(200)]
    assert all(s >= 2 for s in samples)


# ---------------------------------------------------------------------------
# UniformLatency
# ---------------------------------------------------------------------------


def test_uniform_latency_rejects_low_greater_than_high() -> None:
    with pytest.raises(ValueError, match=r"low .* must be <= high"):
        UniformLatency(low=5, high=2)


def test_uniform_latency_allows_low_equal_high() -> None:
    assert UniformLatency(low=3, high=3).high == 3


def test_uniform_latency_real_rng_stays_within_inclusive_bounds() -> None:
    rng = np.random.default_rng(0)
    model = UniformLatency(low=2, high=5)

    samples = [model.sample(rng) for _ in range(500)]
    assert all(2 <= s <= 5 for s in samples)
    # both endpoints should be reachable over enough draws (np.integers'
    # upper bound is exclusive -- a common off-by-one is to forget the +1)
    assert min(samples) == 2
    assert max(samples) == 5


def test_uniform_latency_high_endpoint_is_inclusive_not_exclusive() -> None:
    """Regression guard for the classic off-by-one: rng.integers(low, high)
    excludes `high`, so the implementation must pass high + 1."""
    fake_rng = _FakeRng(value=99)  # stand-in return value, irrelevant here
    model = UniformLatency(low=1, high=10)

    model.sample(fake_rng)

    assert fake_rng.calls == [("integers", 1, 11)]


# ---------------------------------------------------------------------------
# _us_to_ticks
# ---------------------------------------------------------------------------


def test_us_to_ticks_requires_datetime_index() -> None:
    with pytest.raises(TypeError, match="requires a DatetimeIndex"):
        _us_to_ticks(100.0, pd.RangeIndex(5))


def test_us_to_ticks_zero_latency_returns_first_row() -> None:
    timestamps = _timestamps_from_offsets_us([0, 10, 20, 30])
    assert _us_to_ticks(0.0, timestamps) == 0


def test_us_to_ticks_exact_match() -> None:
    timestamps = _timestamps_from_offsets_us([0, 10, 25, 100, 500])
    assert _us_to_ticks(100.0, timestamps) == 3


def test_us_to_ticks_inexact_match_rounds_up_to_next_row() -> None:
    timestamps = _timestamps_from_offsets_us([0, 10, 25, 100, 500])
    # 15us falls strictly between row 1 (10us) and row 2 (25us); the
    # smallest row satisfying timestamps[k] - timestamps[0] >= 15us is row 2
    assert _us_to_ticks(15.0, timestamps) == 2


def test_us_to_ticks_clamps_when_latency_exceeds_window_span() -> None:
    timestamps = _timestamps_from_offsets_us([0, 10, 25, 100, 500])
    assert _us_to_ticks(10_000.0, timestamps) == len(timestamps) - 1


def test_us_to_ticks_single_row_window() -> None:
    timestamps = _timestamps_from_offsets_us([0])
    assert _us_to_ticks(50.0, timestamps) == 0


# ---------------------------------------------------------------------------
# FixedTimedLatency
# ---------------------------------------------------------------------------


def test_fixed_timed_latency_resolves_using_actual_timestamp_spacing() -> None:
    timestamps = _timestamps_from_offsets_us([0, 10, 25, 100, 500])
    model = FixedTimedLatency(latency_us=25.0)

    assert model.resolve(np.random.default_rng(0), timestamps) == 2


def test_fixed_timed_latency_rejects_negative_latency_us() -> None:
    with pytest.raises(ValueError, match="latency_us must be >= 0"):
        FixedTimedLatency(latency_us=-1.0)


def test_fixed_timed_latency_zero_latency_matches_zero_ticks() -> None:
    timestamps = _timestamps_from_offsets_us([0, 10, 20])
    model = FixedTimedLatency(latency_us=0.0)

    assert model.resolve(np.random.default_rng(0), timestamps) == 0


# ---------------------------------------------------------------------------
# LogNormalTimedLatency
# ---------------------------------------------------------------------------


def test_lognormal_timed_latency_floors_to_min_us_before_tick_conversion() -> None:
    timestamps = _timestamps_from_offsets_us([0, 10, 25, 100, 500])
    fake_rng = _FakeRng(value=5.0)  # below min_us=25.0
    model = LogNormalTimedLatency(mu_us=0.0, sigma_us=1.0, min_us=25.0)

    # floored to 25.0us -> exact match at row 2
    assert model.resolve(fake_rng, timestamps) == 2


def test_lognormal_timed_latency_uses_sample_when_above_floor() -> None:
    timestamps = _timestamps_from_offsets_us([0, 10, 25, 100, 500])
    fake_rng = _FakeRng(value=100.0)
    model = LogNormalTimedLatency(mu_us=0.0, sigma_us=1.0, min_us=0.0)

    assert model.resolve(fake_rng, timestamps) == 3


def test_lognormal_timed_latency_real_rng_produces_valid_tick_range() -> None:
    timestamps = _timestamps_from_offsets_us([0, 10, 25, 100, 500])
    rng = np.random.default_rng(0)
    model = LogNormalTimedLatency(mu_us=3.0, sigma_us=2.0, min_us=0.0)

    ticks = [model.resolve(rng, timestamps) for _ in range(100)]
    assert all(0 <= t <= len(timestamps) - 1 for t in ticks)


# ---------------------------------------------------------------------------
# make_latency_model
# ---------------------------------------------------------------------------


def test_make_latency_model_returns_none_when_both_disabled() -> None:
    assert make_latency_model(ticks=0, us=0.0) is None
    assert make_latency_model(ticks=-1, us=-5.0) is None


def test_make_latency_model_ticks_takes_priority_over_us() -> None:
    model = make_latency_model(ticks=5, us=100.0)
    assert isinstance(model, FixedLatency)
    assert model.ticks == 5


def test_make_latency_model_falls_back_to_us_when_ticks_disabled() -> None:
    model = make_latency_model(ticks=0, us=150.0)
    assert isinstance(model, FixedTimedLatency)
    assert model.latency_us == 150.0


def test_make_latency_model_negative_ticks_falls_back_to_us() -> None:
    model = make_latency_model(ticks=-3, us=150.0)
    assert isinstance(model, FixedTimedLatency)
