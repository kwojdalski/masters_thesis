"""Unit tests for compare_rollouts, including mismatched rollout lengths."""

from __future__ import annotations

import torch
import pytest

from trading_rl.evaluation.plots import compare_rollouts


def _rollout(n_steps: int) -> dict:
    """Minimal rollout dict matching the keys compare_rollouts reads."""
    return {
        "action": torch.zeros(n_steps, 1),
        "next": {"reward": torch.zeros(n_steps, 1)},
    }


class TestCompareRollouts:
    def test_equal_length_does_not_crash(self):
        rollouts = [_rollout(20), _rollout(20)]
        reward_plot, action_plot = compare_rollouts(rollouts, n_obs=20)
        assert reward_plot is not None
        assert action_plot is not None

    def test_mismatched_length_does_not_crash(self):
        """Regression: RuntimeError when one rollout terminates early (shorter tensor)."""
        rollouts = [_rollout(100), _rollout(40)]
        reward_plot, action_plot = compare_rollouts(rollouts, n_obs=50)
        assert reward_plot is not None
        assert action_plot is not None

    def test_much_shorter_second_rollout(self):
        rollouts = [_rollout(200), _rollout(5)]
        reward_plot, action_plot = compare_rollouts(rollouts, n_obs=50)
        assert reward_plot is not None
        assert action_plot is not None

    def test_single_rollout(self):
        rollouts = [_rollout(30)]
        reward_plot, action_plot = compare_rollouts(rollouts, n_obs=30)
        assert reward_plot is not None
        assert action_plot is not None
