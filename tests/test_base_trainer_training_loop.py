from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from tensordict import TensorDict

import trading_rl.trainers.base as base_module
from trading_rl.trainers.base import BaseTrainer


class _LoopTrainer(BaseTrainer):
    @staticmethod
    def build_models(n_obs: int, n_act: int, config: Any, env: Any) -> None:
        raise NotImplementedError

    def _optimization_step(self, batch_idx: int, max_length: int, buffer_len: int) -> None:
        self.optimization_calls.append((batch_idx, int(max_length), int(buffer_len)))

    def _evaluate(self) -> None:
        raise NotImplementedError


class _CheckpointManager:
    def __init__(self) -> None:
        self.maybe_save_calls = 0
        self.interrupt_saved = False

    def maybe_save_checkpoint(self) -> None:
        self.maybe_save_calls += 1

    def save_interrupt_checkpoint(self) -> str:
        self.interrupt_saved = True
        return "interrupt.pt"


class _RuntimeHooks:
    def __init__(self) -> None:
        self.steps: list[int] = []

    def maybe_run(self, total_count: int) -> None:
        self.steps.append(total_count)


class _FakeHealthMonitor:
    def check(self) -> None:
        return None


def _batch(n: int, *, done: list[bool] | None = None) -> TensorDict:
    done_values = done if done is not None else [False] * n
    return TensorDict(
        {
            "next": TensorDict(
                {
                    "step_count": torch.arange(1, n + 1, dtype=torch.int64).reshape(n, 1),
                    "done": torch.tensor(done_values, dtype=torch.bool).reshape(n, 1),
                },
                batch_size=[n],
            )
        },
        batch_size=[n],
    )


def _trainer(
    batches: list[TensorDict],
    *,
    init_rand_steps: int = 0,
    max_steps: int = 10,
    max_train_seconds: float | None = None,
) -> _LoopTrainer:
    trainer = object.__new__(_LoopTrainer)
    trainer.collector = batches
    trainer.config = SimpleNamespace(
        init_rand_steps=init_rand_steps,
        max_steps=max_steps,
        max_train_seconds=max_train_seconds,
    )
    trainer.logs = defaultdict(list)
    trainer.total_count = 0
    trainer.total_episodes = 0
    trainer._use_replay_buffer = False
    trainer.checkpoint_manager = _CheckpointManager()
    trainer.runtime_hooks = _RuntimeHooks()
    trainer.health_monitor = _FakeHealthMonitor()
    trainer.optimization_calls = []
    return trainer


def test_run_training_loop_stops_at_max_steps_and_runs_hooks() -> None:
    trainer = _trainer(
        [_batch(3, done=[False, True, False]), _batch(3, done=[True, False, True])],
        init_rand_steps=2,
        max_steps=5,
    )
    starts: list[int] = []
    ends: list[int] = []
    train_end_calls: list[bool] = []

    logs = trainer._run_training_loop(
        on_batch_start=lambda i, _data: starts.append(i),
        on_batch_end=lambda i, _data: ends.append(i),
        on_train_end=lambda: train_end_calls.append(True),
    )

    assert starts == [0, 1]
    assert ends == [0, 1]
    assert trainer.optimization_calls == [(0, 3, 3), (1, 3, 3)]
    assert trainer.checkpoint_manager.maybe_save_calls == 2
    assert trainer.runtime_hooks.steps == [3, 6]
    assert trainer.total_count == 6
    assert trainer.total_episodes == 3
    assert train_end_calls == [True]
    assert len(logs["training_duration_s"]) == 1


def test_run_training_loop_stops_at_runtime_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _trainer([_batch(2), _batch(2)], max_steps=100, max_train_seconds=1.0)
    times = iter([0.0, 2.0, 2.0, 2.0])
    monkeypatch.setattr(base_module.time, "time", lambda: next(times, 2.0))

    trainer._run_training_loop()

    assert trainer.total_count == 2
    assert trainer.optimization_calls == [(0, 2, 2)]
    assert trainer.checkpoint_manager.maybe_save_calls == 1
    assert trainer.runtime_hooks.steps == [2]


def test_run_training_loop_saves_interrupt_checkpoint_and_reraises() -> None:
    class _InterruptingCollector:
        def __iter__(self):
            raise KeyboardInterrupt

    trainer = _trainer([])
    trainer.collector = _InterruptingCollector()

    with pytest.raises(KeyboardInterrupt):
        trainer._run_training_loop()

    assert trainer.checkpoint_manager.interrupt_saved is True
    assert trainer.logs["training_duration_s"] == []
