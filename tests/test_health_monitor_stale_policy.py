"""Regression test for the exact-float-equality stale-policy bug (issue #460)."""

from __future__ import annotations

import numpy as np

from trading_rl.trainers.health_monitor import TrainingHealthMonitor


class TestStalePolicyToleranceRatio:
    def test_near_constant_continuous_action_reports_near_zero_ratio(self):
        # A policy collapsed to ~0.5 exposure, jittering by 1e-7 between
        # steps: real movement, but well below any meaningful exposure
        # change. Bit-exact `!= 0` would report ratio ~1.0 here and mask
        # the collapse; a tolerance-based comparison must report ~0.0.
        rng = np.random.default_rng(0)
        actions = (0.5 + rng.uniform(-1e-7, 1e-7, size=50)).tolist()

        monitor = TrainingHealthMonitor(
            stale_policy_min_ratio=0.1, stale_policy_window=1
        )
        monitor.record_episode(actions)

        assert monitor._change_ratios[-1] < 0.1

    def test_genuine_alternating_action_reports_high_ratio(self):
        actions = [0.0, 1.0] * 25

        monitor = TrainingHealthMonitor(
            stale_policy_min_ratio=0.1, stale_policy_window=1
        )
        monitor.record_episode(actions)

        assert monitor._change_ratios[-1] > 0.9
