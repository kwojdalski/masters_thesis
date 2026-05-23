"""Smoke tests for random policy action sampling."""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict
from torchrl.data import Bounded
from torchrl.envs.utils import RandomPolicy


@pytest.mark.smoke
def test_random_policy_samples_finite_continuous_action() -> None:
    """RandomPolicy should emit a valid continuous portfolio action."""
    action_spec = Bounded(low=-1.0, high=1.0, shape=(1,), dtype=torch.float32)
    policy = RandomPolicy(action_spec)

    tensordict = policy(TensorDict({}, batch_size=[]))
    action = tensordict["action"]

    assert action.shape == torch.Size([1])
    assert action.dtype == torch.float32
    assert torch.isfinite(action).all()
    assert torch.all(action >= -1.0)
    assert torch.all(action <= 1.0)
