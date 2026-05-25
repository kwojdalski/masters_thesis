"""Focused tests for PPO optimization-step mechanics."""

from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from trading_rl.trainers.ppo import PPOTrainer


class _FakeOptimizer:
    def __init__(self) -> None:
        self.zero_grad_calls = 0
        self.step_calls = 0

    def zero_grad(self) -> None:
        self.zero_grad_calls += 1

    def step(self) -> None:
        self.step_calls += 1


class _FakeCallback:
    def __init__(self) -> None:
        self.steps: list[tuple[int, float, float]] = []

    def log_training_step(self, step: int, actor_loss: float, value_loss: float) -> None:
        self.steps.append((step, actor_loss, value_loss))


class _FakePpoLoss:
    def __init__(self) -> None:
        self._cached_critic_network_params_detached = {"critic": "detached"}
        self.target_critic_network_params = {"critic": "target"}
        self.value_estimator_calls: list[dict] = []
        self.loss_samples: list[dict] = []

    def value_estimator(self, batch, params=None, target_params=None) -> None:
        self.value_estimator_calls.append(
            {
                "observations": batch["observation"].reshape(-1).tolist(),
                "params": params,
                "target_params": target_params,
            }
        )
        batch.set("value_target", torch.arange(batch.numel(), dtype=torch.float32))

    def __call__(self, sample):
        self.loss_samples.append(
            {
                "observations": sorted(sample["observation"].reshape(-1).tolist()),
                "has_value_target": "value_target" in sample.keys(),
                "n": sample.numel(),
            }
        )
        return {
            "loss_objective": torch.tensor(0.10, requires_grad=True),
            "loss_critic": torch.tensor(0.20, requires_grad=True),
            "loss_entropy": torch.tensor(0.03, requires_grad=True),
        }


def _ppo_batch() -> TensorDict:
    return TensorDict(
        {
            "observation": torch.arange(4, dtype=torch.float32).reshape(4, 1),
            "action": torch.zeros(4, 1, dtype=torch.int64),
            ("next", "reward"): torch.ones(4, 1),
            ("next", "done"): torch.zeros(4, 1, dtype=torch.bool),
            ("next", "terminated"): torch.zeros(4, 1, dtype=torch.bool),
            ("next", "step_count"): torch.arange(4, dtype=torch.int64).reshape(4, 1),
        },
        batch_size=[4],
    )


def test_ppo_optimization_step_builds_targets_before_sampling_fresh_batch() -> None:
    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.config = SimpleNamespace(
        ppo_epochs=2,
        frames_per_batch=4,
        sample_size=4,
        log_interval=999,
        eval_interval=0,
    )
    trainer._current_batch = _ppo_batch()
    trainer.ppo_loss = _FakePpoLoss()
    trainer.optimizer = _FakeOptimizer()
    trainer.logs = defaultdict(list)
    trainer.callback = _FakeCallback()
    trainer._log_step_offset = 0

    trainer._optimization_step(batch_idx=1, max_length=4, buffer_len=4)

    assert trainer.ppo_loss.value_estimator_calls == [
        {
            "observations": [0.0, 1.0, 2.0, 3.0],
            "params": {"critic": "detached"},
            "target_params": {"critic": "target"},
        }
    ]
    assert trainer.ppo_loss.loss_samples == [
        {"observations": [0.0, 1.0, 2.0, 3.0], "has_value_target": True, "n": 4},
        {"observations": [0.0, 1.0, 2.0, 3.0], "has_value_target": True, "n": 4},
    ]
    assert trainer.optimizer.step_calls == 2
    assert trainer.optimizer.zero_grad_calls == 3
    assert trainer.logs["loss_actor"] == pytest.approx([0.10, 0.10])
    assert trainer.logs["loss_value"] == pytest.approx([0.20, 0.20])
    assert trainer.logs["loss_entropy"] == pytest.approx([0.03, 0.03])
    assert [step for step, _, _ in trainer.callback.steps] == [2, 3]
    assert [actor for _, actor, _ in trainer.callback.steps] == pytest.approx([0.10, 0.10])
    assert [value for _, _, value in trainer.callback.steps] == pytest.approx([0.20, 0.20])
