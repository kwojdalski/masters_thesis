"""Latency models for trading environment simulation.

Two families of models are provided:

**Tick-based** (``FixedLatency``, ``LogNormalLatency``, ``PoissonLatency``,
``UniformLatency``): latency is expressed as a number of rows (ticks).
Fast and simple, but tick spacing in the LOB data is non-uniform (~20 μs
median, P95 ~20–160 ms), so a tick count has no stable wall-clock meaning.

**Time-based** (``FixedTimedLatency``, ``LogNormalTimedLatency``): latency is
expressed in microseconds and resolved to a row count at each episode reset
using the actual timestamps in the window.  More physically meaningful for
statements like "simulate 100 μs co-location latency".

Both families share the same ``LatencyModel`` interface.  The key method is
``resolve(rng, timestamps)``; tick-based models ignore ``timestamps`` and
delegate to ``sample(rng)``.

Pass instances to ``StreamingTradingEnvXY`` via ``obs_latency`` /
``exec_latency``.  Passing ``None`` (the default) disables that component.

Usage::

    # Tick-based — fast, no timestamp dependency
    obs_latency=FixedLatency(5)
    exec_latency=LogNormalLatency(mu=1.5, sigma=0.5, min_ticks=1)

    # Time-based — resolves to ticks from actual episode timestamps
    obs_latency=FixedTimedLatency(latency_us=100.0)          # 100 μs feed delay
    exec_latency=LogNormalTimedLatency(mu_us=50.0, sigma_us=20.0)

    # YAML config (fixed latency only)
    # env:
    #   obs_latency_ticks: 3        # tick-based
    #   exec_latency_us: 150.0      # time-based (μs)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class LatencyModel(ABC):
    """Abstract base for all latency models.

    ``resolve`` is called once per episode reset; the returned tick offset is
    fixed for the duration of that episode (stochastic models vary across
    episodes).

    Tick-based subclasses implement ``sample(rng)`` and inherit the default
    ``resolve`` which delegates to it.  Time-based subclasses override
    ``resolve`` directly.
    """

    def sample(self, rng: np.random.Generator) -> int:
        """Return tick offset without timestamp context (tick-based models)."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support sample(); use resolve()."
        )

    def resolve(self, rng: np.random.Generator, timestamps: pd.DatetimeIndex) -> int:
        """Return tick offset for an episode with the given timestamps.

        Default implementation calls ``sample(rng)`` and ignores timestamps.
        Time-based subclasses override this to use the actual timestamp spacing.
        """
        return self.sample(rng)


# ---------------------------------------------------------------------------
# Tick-based models
# ---------------------------------------------------------------------------

class ZeroLatency(LatencyModel):
    """No latency — always returns 0.  Equivalent to passing ``None``."""

    def sample(self, rng: np.random.Generator) -> int:
        return 0


class FixedLatency(LatencyModel):
    """Constant k-tick latency for every episode."""

    def __init__(self, ticks: int) -> None:
        if ticks < 0:
            raise ValueError(f"ticks must be >= 0, got {ticks}")
        self.ticks = ticks

    def sample(self, rng: np.random.Generator) -> int:
        return self.ticks


class LogNormalLatency(LatencyModel):
    """k ~ max(min_ticks, round(LogNormal(mu, sigma))).

    Models heavy-tailed latency: most episodes have a modest delay but rare
    hardware or network events push the tail to much higher values.
    The median latency in ticks is approximately ``exp(mu)``.
    """

    def __init__(self, mu: float, sigma: float, min_ticks: int = 1) -> None:
        self.mu = mu
        self.sigma = sigma
        self.min_ticks = min_ticks

    def sample(self, rng: np.random.Generator) -> int:
        return max(self.min_ticks, round(float(rng.lognormal(self.mu, self.sigma))))


class PoissonLatency(LatencyModel):
    """k ~ max(min_ticks, Poisson(lam)).

    Models counting-process latency, e.g. the number of queue-position jumps
    before a limit order is filled.
    """

    def __init__(self, lam: float, min_ticks: int = 0) -> None:
        self.lam = lam
        self.min_ticks = min_ticks

    def sample(self, rng: np.random.Generator) -> int:
        return max(self.min_ticks, int(rng.poisson(self.lam)))


class UniformLatency(LatencyModel):
    """k ~ Uniform[low, high] (integer, both endpoints inclusive)."""

    def __init__(self, low: int, high: int) -> None:
        if low > high:
            raise ValueError(f"low ({low}) must be <= high ({high})")
        self.low = low
        self.high = high

    def sample(self, rng: np.random.Generator) -> int:
        return int(rng.integers(self.low, self.high + 1))


# ---------------------------------------------------------------------------
# Time-based models (latency in microseconds → resolved to ticks at reset)
# ---------------------------------------------------------------------------

def _us_to_ticks(latency_us: float, timestamps: pd.DatetimeIndex) -> int:
    """Return the smallest row offset k such that timestamps[k] - timestamps[0] >= latency_us.

    If the requested latency exceeds the span of the window, returns
    ``len(timestamps) - 1`` (the last valid row index).

    Requires a DatetimeIndex; raises TypeError for other index types.
    """
    if not isinstance(timestamps, pd.DatetimeIndex):
        raise TypeError(
            f"Time-based latency requires a DatetimeIndex but got {type(timestamps).__name__}. "
            "Ensure the memmap data has nanosecond-precision timestamps."
        )
    target = pd.Timedelta(microseconds=latency_us)
    deltas: pd.TimedeltaIndex = timestamps - timestamps[0]
    k = int(deltas.searchsorted(target))
    return min(k, len(timestamps) - 1)


class FixedTimedLatency(LatencyModel):
    """Deterministic latency of ``latency_us`` microseconds, resolved to ticks.

    At each episode reset the actual tick timestamps are used to find the
    smallest row offset k such that ``timestamps[k] - timestamps[0] >= latency_us``.
    """

    def __init__(self, latency_us: float) -> None:
        if latency_us < 0:
            raise ValueError(f"latency_us must be >= 0, got {latency_us}")
        self.latency_us = latency_us

    def resolve(self, rng: np.random.Generator, timestamps: pd.DatetimeIndex) -> int:
        return _us_to_ticks(self.latency_us, timestamps)


class LogNormalTimedLatency(LatencyModel):
    """Latency ~ max(min_us, LogNormal(mu_us, sigma_us)) microseconds, resolved to ticks.

    The sampled duration is converted to a row count using the episode's
    actual timestamps, so the distribution is in physical time units rather
    than ticks.  ``mu_us`` and ``sigma_us`` are the mean and std of the
    underlying normal in the log domain (not the mean μs directly); the
    median latency in μs is approximately ``exp(mu_us)``.
    """

    def __init__(self, mu_us: float, sigma_us: float, min_us: float = 0.0) -> None:
        self.mu_us = mu_us
        self.sigma_us = sigma_us
        self.min_us = min_us

    def resolve(self, rng: np.random.Generator, timestamps: pd.DatetimeIndex) -> int:
        latency_us = max(self.min_us, float(rng.lognormal(self.mu_us, self.sigma_us)))
        return _us_to_ticks(latency_us, timestamps)


# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------

def make_latency_model(ticks: int = 0, us: float = 0.0) -> LatencyModel | None:
    """Convenience constructor for config-driven latency.

    Priority: ``ticks`` > ``us`` > disabled (None).
    Returns ``None`` when both are zero/negative (latency disabled).

    Args:
        ticks: Fixed row-offset latency.  0 = ignore.
        us:    Fixed time-based latency in microseconds.  0.0 = ignore.
    """
    if ticks > 0:
        return FixedLatency(ticks)
    if us > 0.0:
        return FixedTimedLatency(us)
    return None
