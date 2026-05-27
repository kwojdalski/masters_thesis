"""Tests for DDPG skipped-batch accounting."""

from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from trading_rl.trainers.ddpg import DDPGTrainer


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


class _FakeDdpgLoss:
    def __init__(self) -> None:
        self.actor_network_params = _FakeParams()
        self.value_network_params = _FakeParams()
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
            "loss_actor": torch.tensor(0.11, requires_grad=True),
            "loss_value": torch.tensor(0.22, requires_grad=True),
        }


def _bare_ddpg_trainer() -> DDPGTrainer:
    trainer = object.__new__(DDPGTrainer)
    trainer.successful_batches = 0
    trainer.skipped_batches = 0
    trainer._consecutive_skips = 0
    return trainer


def test_ddpg_record_skipped_batch_updates_skip_counters() -> None:
    trainer = _bare_ddpg_trainer()

    trainer._record_skipped_batch("done/terminated shape mismatch")

    assert trainer.skipped_batches == 1
    assert trainer._consecutive_skips == 1


def test_ddpg_record_skipped_batch_aborts_after_repeated_zero_success_skips() -> None:
    trainer = _bare_ddpg_trainer()

    for _ in range(9):
        trainer._record_skipped_batch("done/terminated shape mismatch")

    with pytest.raises(RuntimeError, match="10 consecutive optimization batches"):
        trainer._record_skipped_batch("done/terminated shape mismatch")

    assert trainer.skipped_batches == 10
    assert trainer.successful_batches == 0


def test_ddpg_record_skipped_batch_does_not_abort_after_successful_batch() -> None:
    trainer = _bare_ddpg_trainer()
    trainer.successful_batches = 1

    for _ in range(10):
        trainer._record_skipped_batch("done/terminated shape mismatch")

    assert trainer.skipped_batches == 10
    assert trainer._consecutive_skips == 10


def _ddpg_sample(
    *,
    reward: torch.Tensor | None = None,
    done: torch.Tensor | None = None,
    terminated: torch.Tensor | None = None,
) -> TensorDict:
    return TensorDict(
        {
            "observation": torch.arange(12, dtype=torch.float32).reshape(4, 3),
            "action": torch.zeros(4, 1),
            ("next", "reward"): reward if reward is not None else torch.ones(4, 1),
            ("next", "done"): done if done is not None else torch.zeros(4, 1, dtype=torch.bool),
            ("next", "terminated"): (
                terminated if terminated is not None else torch.zeros(4, 1, dtype=torch.bool)
            ),
        },
        batch_size=[4],
    )


def _optimization_trainer(sample: TensorDict) -> DDPGTrainer:
    trainer = _bare_ddpg_trainer()
    trainer.config = SimpleNamespace(
        optim_steps_per_batch=2,
        sample_size=4,
        log_interval=999,
        eval_interval=0,
        max_grad_norm=0,
    )
    trainer.replay_buffer = _FakeReplayBuffer(sample)
    trainer.ddpg_loss = _FakeDdpgLoss()
    trainer.optimizer_actor = _FakeOptimizer()
    trainer.optimizer_value = _FakeOptimizer()
    trainer.updater = _FakeUpdater()
    trainer.logs = defaultdict(list)
    trainer.callback = None
    trainer.actor = object()
    trainer.value_net = object()
    trainer._log_step_offset = 0
    return trainer


def test_ddpg_optimization_step_updates_networks_and_logs_losses() -> None:
    trainer = _optimization_trainer(_ddpg_sample())

    trainer._optimization_step(batch_idx=0, max_length=4, buffer_len=4)

    assert trainer.replay_buffer.sample_sizes == [4, 4]
    assert trainer.ddpg_loss.sample_shapes == [
        {"reward": torch.Size([4, 1]), "done": torch.Size([4, 1]), "terminated": torch.Size([4, 1])},
        {"reward": torch.Size([4, 1]), "done": torch.Size([4, 1]), "terminated": torch.Size([4, 1])},
    ]
    assert trainer.successful_batches == 2
    assert trainer.skipped_batches == 0
    assert trainer._consecutive_skips == 0
    assert trainer.optimizer_actor.step_calls == 2
    assert trainer.optimizer_value.step_calls == 2
    assert trainer.updater.step_calls == 2
    assert trainer.ddpg_loss.actor_network_params.to_module_calls == 2
    assert trainer.ddpg_loss.value_network_params.to_module_calls == 2
    assert trainer.logs["loss_actor"] == pytest.approx([0.11, 0.11])
    assert trainer.logs["loss_value"] == pytest.approx([0.22, 0.22])


def test_ddpg_optimization_step_skips_nonfinite_rewards_without_updates() -> None:
    trainer = _optimization_trainer(
        _ddpg_sample(reward=torch.tensor([[1.0], [float("nan")], [0.5], [0.2]]))
    )

    trainer._optimization_step(batch_idx=0, max_length=4, buffer_len=4)

    assert trainer.replay_buffer.sample_sizes == [4, 4]
    assert trainer.successful_batches == 0
    assert trainer.skipped_batches == 2
    assert trainer.optimizer_actor.step_calls == 0
    assert trainer.optimizer_value.step_calls == 0
    assert trainer.logs["loss_actor"] == []
    assert trainer.logs["loss_value"] == []


def test_ddpg_optimization_step_records_shape_mismatch_skip() -> None:
    trainer = _optimization_trainer(
        _ddpg_sample(
            done=torch.zeros(4, 1, dtype=torch.bool),
            terminated=torch.zeros(4, dtype=torch.bool),
        )
    )

    trainer._optimization_step(batch_idx=0, max_length=4, buffer_len=4)

    assert trainer.successful_batches == 0
    assert trainer.skipped_batches == 2
    assert trainer._consecutive_skips == 2
    assert trainer.optimizer_actor.step_calls == 0
    assert trainer.optimizer_value.step_calls == 0
