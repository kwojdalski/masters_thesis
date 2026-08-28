"""Regression tests for position_change_ratio bugs (issues #457, #470)."""

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


class TestCountPositionChangesComparesConsecutiveSteps:
    def test_gradual_drift_under_tolerance_reports_zero_changes(self):
        # Each consecutive step's diff (0.05, 0.06) stays under tolerance
        # 0.1, so the correct count is 0. A stale-anchor implementation
        # instead compares 0.11 against the original anchor 0.0 (never
        # updated, since no single step exceeded tolerance), reporting a
        # spurious change that never happened between two consecutive
        # steps.
        actions = [0.0, 0.05, 0.11]
        changes = MLflowTrainingCallback._count_position_changes(actions, tolerance=0.1)
        assert changes == 0

    def test_vector_actions_compare_consecutive_steps_too(self):
        actions = [[0.0, 0.0], [0.05, 0.0], [0.11, 0.0]]
        changes = MLflowTrainingCallback._count_position_changes(actions, tolerance=0.1)
        assert changes == 0

    def test_still_detects_a_real_single_step_jump(self):
        actions = [0.0, 0.5, 0.5]
        changes = MLflowTrainingCallback._count_position_changes(actions, tolerance=0.1)
        assert changes == 1
