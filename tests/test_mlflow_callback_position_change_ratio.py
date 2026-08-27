"""Regression test for the position_change_ratio off-by-one (issue #457)."""

from __future__ import annotations

from trading_rl.callbacks.mlflow_callback import MLflowTrainingCallback


class TestCountPositionChanges:
    def test_single_change_over_two_steps_reports_full_ratio(self):
        # actions=[0.0, 1.0] has exactly one transition over one opportunity
        # to change (n=2 steps, n-1=1 transition slot). The ratio must be
        # able to reach 1.0, not be capped at 0.5 by dividing by n instead
        # of n-1.
        actions = [0.0, 1.0]
        changes = MLflowTrainingCallback._count_position_changes(actions)
        episode_length = len(actions)

        assert changes == 1
        ratio = changes / (episode_length - 1) if episode_length > 1 else 0.0
        assert ratio == 1.0

    def test_single_action_episode_has_zero_ratio_without_division_by_zero(self):
        actions = [1.0]
        changes = MLflowTrainingCallback._count_position_changes(actions)
        episode_length = len(actions)

        assert changes == 0
        ratio = changes / (episode_length - 1) if episode_length > 1 else 0.0
        assert ratio == 0.0
