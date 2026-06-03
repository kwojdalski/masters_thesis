"""Latency models for trading environment simulation.

Both observation latency (stale market data feed) and execution latency
(order fill delay) are modelled as a tick offset between the feature row
and the price row used for trade execution.  The two effects are additive:
``total_ticks = obs_latency.sample(rng) + exec_latency.sample(rng)``.

Pass instances to ``StreamingTradingEnvXY`` via ``obs_latency`` /
``exec_latency``.  Passing ``None`` (the default) disables that component.

Usage::

    from trading_rl.envs.latency import FixedLatency, LogNormalLatency

    env = StreamingTradingEnvXY(
        ...,
        obs_latency=FixedLatency(2),
        exec_latency=LogNormalLatency(mu=1.0, sigma=0.5, min_ticks=1),
    )
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class LatencyModel(ABC):
    """Abstract base for all latency models.

    ``sample`` is called once per episode reset so the latency is fixed
    for the duration of a single episode (but varies across episodes for
    stochastic models).
    """

    @abstractmethod
    def sample(self, rng: np.random.Generator) -> int:
        """Return the number of ticks to delay.  Must be >= 0."""


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

    Models the heavy-tailed latency distributions seen in real HFT systems:
    most episodes have a modest delay, but rare hardware or network events
    push the tail to much higher values.

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

    Models counting-process latency, e.g. the number of queue-position
    jumps before a limit order is filled.
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


def make_latency_model(ticks: int) -> LatencyModel | None:
    """Return ``FixedLatency(ticks)`` for ticks > 0, else ``None`` (disabled).

    Convenience constructor used by the environment builder to convert
    integer config values into latency model instances.
    """
    if ticks <= 0:
        return None
    return FixedLatency(ticks)
