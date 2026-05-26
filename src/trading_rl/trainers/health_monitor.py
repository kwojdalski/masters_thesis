"""Runtime training health monitor — behavioral early stopping."""

from __future__ import annotations

from collections import deque

import numpy as np

from logger import get_logger
from trading_rl.config_guardrails import Finding, Severity

logger = get_logger(__name__)


class TrainingHealthMonitor:
    """Monitor per-episode behavioral signals and signal early stopping.

    Feed one completed episode at a time via record_episode(), then call
    check() each batch to get a stop reason (or None if training should
    continue).

    Controlled by TrainingConfig fields (all default to disabled):
      es_stale_policy_min_ratio  — minimum acceptable mean position-change
                                   ratio; 0 = disabled.
      es_stale_policy_window     — rolling window in episodes.
      es_saturation_max_rate     — maximum acceptable fraction of steps at
                                   extreme positions (|pos| > 0); 0 = disabled.
      es_saturation_window       — rolling window in episodes.
    """

    def __init__(
        self,
        *,
        stale_policy_min_ratio: float = 0.0,
        stale_policy_window: int = 20,
        saturation_max_rate: float = 0.0,
        saturation_window: int = 20,
    ) -> None:
        self._stale_ratio = stale_policy_min_ratio
        self._stale_window = max(1, stale_policy_window)
        self._saturation_max = saturation_max_rate
        self._saturation_window = max(1, saturation_window)
        self._change_ratios: deque[float] = deque(maxlen=self._stale_window)
        self._saturation_rates: deque[float] = deque(maxlen=self._saturation_window)

    @property
    def enabled(self) -> bool:
        return self._stale_ratio > 0.0 or self._saturation_max > 0.0

    def record_episode(self, actions: list, pct_extreme: float | None = None) -> None:
        """Compute and store per-episode behavioral signals.

        Args:
            actions: Flat list of per-step action values for the episode.
            pct_extreme: Fraction of steps at full exposure (|pos| > 0),
                pre-computed by EpisodeStatsTracker. Computed from actions
                when not provided.
        """
        if not actions:
            return
        arr = np.asarray(actions, dtype=float)

        if self._stale_ratio > 0.0:
            ratio = float(np.sum(np.diff(arr) != 0)) / (arr.size - 1) if arr.size > 1 else 0.0
            self._change_ratios.append(ratio)
            logger.debug(
                "health_monitor episode_change_ratio=%.4f window=%d/%d",
                ratio, len(self._change_ratios), self._stale_window,
            )

        if self._saturation_max > 0.0:
            if pct_extreme is None:
                pct_extreme = float(np.mean(arr != 0)) if arr.size > 0 else 0.0
            self._saturation_rates.append(pct_extreme)
            logger.debug(
                "health_monitor episode_saturation=%.4f window=%d/%d",
                pct_extreme, len(self._saturation_rates), self._saturation_window,
            )

    def check(self) -> Finding | None:
        """Return a Finding if a guardrail fires, else None."""
        return self._check_stale_policy() or self._check_saturation()

    def _check_stale_policy(self) -> Finding | None:
        if self._stale_ratio <= 0.0:
            return None
        if len(self._change_ratios) < self._stale_window:
            return None
        mean_ratio = float(np.mean(list(self._change_ratios)))
        if mean_ratio < self._stale_ratio:
            return Finding(
                severity=Severity.WARN,
                parameter="training.es_stale_policy_min_ratio",
                message=(
                    f"mean position-change ratio {mean_ratio:.4f} < threshold "
                    f"{self._stale_ratio} over last {self._stale_window} episodes — "
                    "agent is not changing positions; no learning signal."
                ),
                suggestion=(
                    "Increase exploration (exploration_noise_std or entropy_bonus), "
                    "lower actor_lr, or disable with es_stale_policy_min_ratio=0."
                ),
            )
        return None

    def _check_saturation(self) -> Finding | None:
        if self._saturation_max <= 0.0:
            return None
        if len(self._saturation_rates) < self._saturation_window:
            return None
        mean_sat = float(np.mean(list(self._saturation_rates)))
        if mean_sat > self._saturation_max:
            return Finding(
                severity=Severity.WARN,
                parameter="training.es_saturation_max_rate",
                message=(
                    f"mean extreme-position rate {mean_sat:.4f} > threshold "
                    f"{self._saturation_max} over last {self._saturation_window} episodes — "
                    "agent is stuck at full long/short exposure, never going flat."
                ),
                suggestion=(
                    "Increase entropy_bonus (PPO) or exploration_noise_std (TD3/DDPG) "
                    "to encourage neutrality, or disable with es_saturation_max_rate=0."
                ),
            )
        return None
