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
    obs_latency = FixedLatency(5)
    exec_latency = LogNormalLatency(mu=1.5, sigma=0.5, min_ticks=1)

    # Time-based — resolves to ticks from actual episode timestamps
    obs_latency = FixedTimedLatency(latency_us=100.0)  # 100 μs feed delay
    exec_latency = LogNormalTimedLatency(mu_us=50.0, sigma_us=20.0)

    # YAML config (fixed latency only)
    # env:
    #   obs_latency_ticks: 3        # tick-based
    #   exec_latency_us: 150.0      # time-based (μs)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class LatencyModel:
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


def _us_to_ticks(
    latency_us: float, timestamps: pd.DatetimeIndex, n_probes: int = 16
) -> int:
    """Return the row offset an order of ``latency_us`` microseconds incurs.

    An order submitted from the book state at row ``t`` reaches the market at
    ``timestamps[t] + latency_us`` and is filled against whatever book is
    standing then. A limit order book persists between updates, so the
    prevailing state at that moment is the **last event at or before** the
    arrival time -- not the next one after it. An order that lands inside a
    quiet gap is therefore filled at the price it observed, at no cost, which
    is the correct outcome: nothing happened while it was in flight.

    Two corrections relative to the original implementation, both of which
    over-penalised latency:

    * It searched for the first event at or *after* arrival, charging a full
      event of adverse drift to any delay shorter than one inter-event gap. On
      this data the median gap is 24-69 microseconds, so a 10 microsecond
      order -- which cannot outrun the next update -- was charged as though the
      book had moved. The search is now right-sided and stepped back one row.
    * It measured every delay from ``timestamps[0]``, so a long gap at the
      start of a window absorbed the whole requested latency. One AVGO window
      opens with a 301 millisecond gap, which collapsed the 10 microsecond,
      1 millisecond and 5 millisecond arms onto the same offset: three
      nominally different experiments resolving to identical behaviour. The
      offset is now the median over ``n_probes`` start points spread across the
      window, which is representative of the episode rather than of its first
      event.

    A single offset is applied to the whole episode, so this is a
    window-level approximation to a per-decision delay; it is exact when the
    event rate is stationary within the window and close otherwise.

    Requires a DatetimeIndex; raises TypeError for other index types.
    """
    if not isinstance(timestamps, pd.DatetimeIndex):
        raise TypeError(
            f"Time-based latency requires a DatetimeIndex but got {type(timestamps).__name__}. "
            "Ensure the memmap data has nanosecond-precision timestamps."
        )
    n = len(timestamps)
    if n < 2 or latency_us <= 0:
        return 0
    ns_vals: np.ndarray = timestamps.asi8  # zero-copy int64 ns view
    target_ns = int(latency_us * 1_000)  # us -> ns

    def _offset_from(t: int) -> int:
        # side="right" then step back one: the last event at or before arrival.
        rel = ns_vals[t:] - ns_vals[t]
        return max(0, int(np.searchsorted(rel, target_ns, side="right")) - 1)

    # Probe several start points so one long gap cannot dominate, but only
    # where the window is long enough for later probes to have runway. A probe
    # starting near the end would resolve short simply because the window runs
    # out, which would bias the median down rather than make it representative.
    span_ns = int(ns_vals[-1] - ns_vals[0])
    if n < 64 or span_ns <= 2 * target_ns:
        return min(_offset_from(0), n - 1)

    # Keep every probe at least `target_ns` clear of the window end.
    usable = int(np.searchsorted(ns_vals - ns_vals[0], span_ns - target_ns, side="right"))
    starts = np.unique(
        np.linspace(0, max(0, usable - 1), num=min(n_probes, max(1, usable))).astype(int)
    )
    return int(min(np.median([_offset_from(t) for t in starts]), n - 1))


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


def resolve_total_latency_ticks(
    env_config: object,
    index: pd.Index,
    rng: np.random.Generator | None = None,
) -> int:
    """Resolve the combined observation + execution latency to a row offset.

    ``obs_latency`` (stale market data) and ``exec_latency`` (order-submission
    delay) are summed: both widen the same gap between the timestamp of the
    information the agent acted on and the timestamp of the price it is filled
    at, which is the only quantity that affects P&L.

    Returns 0 when no latency is configured, so callers can skip the shift.
    """
    obs = make_latency_model(
        int(getattr(env_config, "obs_latency_ticks", 0) or 0),
        float(getattr(env_config, "obs_latency_us", 0.0) or 0.0),
    )
    exe = make_latency_model(
        int(getattr(env_config, "exec_latency_ticks", 0) or 0),
        float(getattr(env_config, "exec_latency_us", 0.0) or 0.0),
    )
    if obs is None and exe is None:
        return 0
    rng = rng if rng is not None else np.random.default_rng(0)
    k = 0
    if obs is not None:
        k += obs.resolve(rng, index)
    if exe is not None:
        k += exe.resolve(rng, index)
    return int(k)


def split_for_latency(df: pd.DataFrame, k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split *df* into the frames an agent observes and is filled at.

    With a latency of ``k`` rows the agent sees the book at row ``t`` but its
    order reaches the market at row ``t + k``:

    * ``feature_df`` -- rows ``[0 .. N-k-1]``, what the agent observes.
    * ``price_df``   -- rows ``[k .. N-1]``, what the agent is filled at,
      re-indexed onto ``feature_df``'s timestamps so the market sees one
      consistent timeline.

    The price move across the delay window is therefore missed, which is the
    cost of being slow. This is *not* look-ahead: features are only ever
    truncated from the end, never advanced.

    ``k <= 0`` returns ``(df, df)`` unchanged so callers need no special case.

    Both the streaming training environment and the DataFrame-backed
    evaluation environment call this, so the two cannot drift apart. They
    previously implemented the shift independently, and evaluation simply
    omitted it -- every latency scenario was scored at zero latency while
    reporting a non-zero configuration.
    """
    if k <= 0:
        return df, df
    n = len(df)
    if k >= n:
        raise ValueError(
            f"latency of {k} rows >= window size {n}. Reduce obs_latency + "
            "exec_latency, or lengthen the evaluation window / episode."
        )
    feature_df = df.iloc[: n - k]
    price_df = df.iloc[k:].copy()
    price_df.index = feature_df.index
    return feature_df, price_df


class ActionThrottle:
    """Hold a position between decisions, modelling a finite decision rate.

    An agent whose round trip spans several order-book events cannot re-decide
    on every event. With ``every_n = m`` the agent's action is adopted on steps
    ``0, m, 2m, ...`` and held in between, so the position is committed for the
    duration of the flight rather than being re-chosen each tick.

    This is orthogonal to ``split_for_latency``: the shift moves *which price*
    a decision is filled at, while throttling limits *how often* a decision can
    be made. A realistic configuration needs both -- a 5 ms round trip both
    delays the fill and prevents 167 decisions being taken inside that window.

    ``every_n <= 1`` disables throttling, so callers need no special case.
    """

    def __init__(self, every_n: int = 1) -> None:
        if every_n < 1:
            raise ValueError(f"every_n must be >= 1, got {every_n}")
        self.every_n = int(every_n)
        self._step = 0
        self._held: object | None = None

    @property
    def enabled(self) -> bool:
        return self.every_n > 1

    def reset(self) -> None:
        """Clear the held action at an episode boundary."""
        self._step = 0
        self._held = None

    def filter(self, action: object) -> object:
        """Return the action actually submitted for this step.

        On a decision step the agent's action is adopted and remembered; on
        every other step the held action is replayed unchanged.
        """
        if not self.enabled:
            return action
        if self._step % self.every_n == 0 or self._held is None:
            self._held = action
        self._step += 1
        return self._held
