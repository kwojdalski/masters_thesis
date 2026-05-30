from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from trading_rl.trainers.td3 import TD3Trainer


class _FakeReplayBuffer:
    def __init__(self, sample: TensorDict) -> None:
        self.sample_template = sample
        self.sample_sizes: list[int] = []

    def sample(self, sample_size: int) -> TensorDict:
        self.sample_sizes.append(sample_size)
        return self.sample_template.clone()


class _FakeOptimizer:
    def __init__(self) -> None:
        self.zero_grad_calls = 0
        self.step_calls = 0

    def zero_grad(self) -> None:
        self.zero_grad_calls += 1

    def step(self) -> None:
        self.step_calls += 1


class _FakeUpdater:
    def __init__(self) -> None:
        self.step_calls = 0

    def step(self) -> None:
        self.step_calls += 1


class _FakeParams:
    def __init__(self) -> None:
        self.to_module_calls = 0

    def to_module(self, _module) -> None:
        self.to_module_calls += 1


class _FakeTd3Loss:
    def __init__(self) -> None:
        self.actor_network_params = _FakeParams()
        self.qvalue_network_params = _FakeParams()
        self.sample_shapes: list[dict[str, torch.Size]] = []

    def __call__(self, sample):
        self.sample_shapes.append(
            {
                "reward": sample["next", "reward"].shape,
                "done": sample["next", "done"].shape,
                "terminated": sample["next", "terminated"].shape,
            }
        )
        return {
            "loss_qvalue": torch.tensor(0.30, requires_grad=True),
            "loss_actor": torch.tensor(0.40, requires_grad=True),
        }


def _td3_sample() -> TensorDict:
    return TensorDict(
        {
            "observation": torch.arange(12, dtype=torch.float32).reshape(4, 3),
            "action": torch.zeros(4, 1),
            ("next", "reward"): torch.ones(4),
            ("next", "done"): torch.zeros(4, dtype=torch.bool),
            ("next", "terminated"): torch.zeros(4, dtype=torch.bool),
        },
        batch_size=[4],
    )


def test_td3_actor_update_waits_for_policy_delay_steps() -> None:
    trainer = TD3Trainer.__new__(TD3Trainer)
    trainer.policy_delay = 2

    assert [trainer._should_update_actor(step) for step in range(6)] == [
        False,
        False,
        True,
        False,
        True,
        False,
    ]


def test_td3_actor_update_respects_custom_policy_delay() -> None:
    trainer = TD3Trainer.__new__(TD3Trainer)
    trainer.policy_delay = 3

    assert [trainer._should_update_actor(step) for step in range(7)] == [
        False,
        False,
        False,
        True,
        False,
        False,
        True,
    ]


def test_td3_optimization_step_normalizes_shapes_and_delays_actor_update() -> None:
    trainer = TD3Trainer.__new__(TD3Trainer)
    trainer.config = SimpleNamespace(
        optim_steps_per_batch=3,
        sample_size=4,
        log_interval=999,
        eval_interval=0,
        max_grad_norm=0,
    )
    trainer.policy_delay = 2
    trainer.replay_buffer = _FakeReplayBuffer(_td3_sample())
    trainer.td3_loss = _FakeTd3Loss()
    trainer.optimizer_value = _FakeOptimizer()
    trainer.optimizer_actor = _FakeOptimizer()
    trainer.updater = _FakeUpdater()
    trainer.logs = defaultdict(list)
    trainer.callback = None
    trainer.actor = object()
    trainer.value_net = object()
    trainer.successful_batches = 0
    trainer.skipped_batches = 0
    trainer._log_step_offset = 0

    trainer._optimization_step(batch_idx=0, max_length=4, buffer_len=4)

    assert trainer.replay_buffer.sample_sizes == [4, 4, 4]
    assert trainer.td3_loss.sample_shapes == [
        {"reward": torch.Size([4, 1]), "done": torch.Size([4, 1]), "terminated": torch.Size([4, 1])},
        {"reward": torch.Size([4, 1]), "done": torch.Size([4, 1]), "terminated": torch.Size([4, 1])},
        {"reward": torch.Size([4, 1]), "done": torch.Size([4, 1]), "terminated": torch.Size([4, 1])},
        {"reward": torch.Size([4, 1]), "done": torch.Size([4, 1]), "terminated": torch.Size([4, 1])},
    ]
    assert trainer.successful_batches == 3
    assert trainer.skipped_batches == 0
    assert trainer.optimizer_value.step_calls == 3
    assert trainer.optimizer_actor.step_calls == 1
    assert trainer.updater.step_calls == 1
    assert trainer.td3_loss.qvalue_network_params.to_module_calls == 3
    assert trainer.td3_loss.actor_network_params.to_module_calls == 1
    assert trainer.logs["loss_value"] == pytest.approx([0.30, 0.30, 0.30])
    # actor loss only logged on the one delayed-update step, not on critic-only steps
    assert trainer.logs["loss_actor"] == pytest.approx([0.40])
