"""Runtime training health monitor — behavioral early stopping."""

from __future__ import annotations

from collections import deque

import numpy as np

from logger import get_logger

logger = get_logger(__name__)


class TrainingHealthMonitor:
    """Monitor per-episode behavioral signals and signal early stopping.

    Feed one completed episode at a time via record_episode(), then call
    check() each batch to get a stop reason (or None if training should
    continue).

    Controlled by two TrainingConfig fields (all default to disabled):
      es_stale_policy_min_ratio  — minimum acceptable mean position-change
                                   ratio over the rolling window; 0 = disabled.
      es_stale_policy_window     — number of episodes in the rolling window.
    """

    def __init__(
        self,
        *,
        stale_policy_min_ratio: float = 0.0,
        stale_policy_window: int = 20,
    ) -> None:
        self._stale_ratio = stale_policy_min_ratio
        self._stale_window = max(1, stale_policy_window)
        self._change_ratios: deque[float] = deque(maxlen=self._stale_window)

    @property
    def enabled(self) -> bool:
        return self._stale_ratio > 0.0

    def record_episode(self, actions: list) -> None:
        """Compute and store the position-change ratio for one completed episode."""
        if not self.enabled or not actions:
            return
        arr = np.asarray(actions, dtype=float)
        if arr.size <= 1:
            ratio = 0.0
        else:
            ratio = float(np.sum(np.diff(arr) != 0)) / (arr.size - 1)
        self._change_ratios.append(ratio)
        logger.debug(
            "health_monitor episode_change_ratio=%.4f window_size=%d/%d",
            ratio, len(self._change_ratios), self._stale_window,
        )

    def check(self) -> str | None:
        """Return a stop-reason string if a guardrail fires, else None."""
        if not self.enabled:
            return None
        if len(self._change_ratios) < self._stale_window:
            return None
        mean_ratio = float(np.mean(list(self._change_ratios)))
        if mean_ratio < self._stale_ratio:
            return (
                f"stale_policy: mean position-change ratio {mean_ratio:.4f} "
                f"< threshold {self._stale_ratio} "
                f"over last {self._stale_window} episodes — "
                "agent is not changing positions; no learning signal."
            )
        return None
