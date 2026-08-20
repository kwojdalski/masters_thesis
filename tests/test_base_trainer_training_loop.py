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

    def _optimization_step(
        self, batch_idx: int, max_length: int, buffer_len: int
    ) -> None:
        self.optimization_calls.append((batch_idx, int(max_length), int(buffer_len)))

    def _evaluate(self) -> None:
        raise NotImplementedError

    @property
    def _algo_label(self) -> str:
        return "test"

    def _get_checkpoint_network_state(self) -> dict:
        return {}

    def _load_checkpoint_network_state(self, checkpoint: dict) -> None:
        pass


class _CheckpointManager:
    def __init__(self) -> None:
        self.maybe_save_calls = 0
        self.interrupt_saved = False

    def maybe_save(self, step: int, snapshot_fn: object) -> None:
        self.maybe_save_calls += 1

    def save_interrupt(self, step: int, snapshot_fn: object) -> str:
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
                    "step_count": torch.arange(1, n + 1, dtype=torch.int64).reshape(
                        n, 1
                    ),
                    "done": torch.tensor(done_values, dtype=torch.bool).reshape(n, 1),
                },
                batch_size=[n],
            )
        },
        batch_size=[n],
    )


class _SpyReplayBuffer:
    """Replay-buffer stand-in that fails if a full-slice (`buffer[:]`) read is
    ever attempted, per issue #355's request for a spy-backed regression
    guard against the O(buffer) `replay_buffer[:]` scan this fix removed.

    `max_size`, when set, caps `__len__` the way a real bounded
    LazyTensorStorage does once it fills -- used to reproduce issue #356
    (buffer_len can never exceed buffer_size, so gating warm-up on it is
    wrong)."""

    def __init__(self, max_size: int | None = None) -> None:
        self.extended: list[TensorDict] = []
        self.max_size = max_size

    def extend(self, data: TensorDict) -> None:
        self.extended.append(data)

    def __len__(self) -> int:
        total = sum(int(d.numel()) for d in self.extended)
        return total if self.max_size is None else min(total, self.max_size)

    def __getitem__(self, index: object) -> Any:
        if index == slice(None, None, None):
            raise AssertionError(
                "replay_buffer[:] full-slice access is forbidden here -- "
                "max_length must be tracked as an incremental running max, "
                "not recomputed by scanning the whole buffer (see issue #355)"
            )
        raise NotImplementedError


def _trainer(
    batches: list[TensorDict],
    *,
    init_rand_steps: int = 0,
    max_steps: int = 10,
    max_train_seconds: float | None = None,
    use_replay_buffer: bool = False,
    replay_buffer: Any = None,
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
    trainer._use_replay_buffer = use_replay_buffer
    trainer._replay_buffer_max_step_count = 0
    trainer.replay_buffer = replay_buffer
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


def test_run_training_loop_replay_buffer_never_full_slices() -> None:
    """Regression test for issue #355: the training loop must never read
    `replay_buffer[:]` to compute max_length. _SpyReplayBuffer raises if that
    ever happens, so this test fails loudly if the O(buffer) scan comes back."""
    spy = _SpyReplayBuffer()
    trainer = _trainer(
        [_batch(3, done=[False, True, False]), _batch(2, done=[True, False])],
        init_rand_steps=0,
        max_steps=100,
        use_replay_buffer=True,
        replay_buffer=spy,
    )

    trainer._run_training_loop()

    assert len(trainer.optimization_calls) == 2


def test_run_training_loop_replay_buffer_max_length_is_a_running_max() -> None:
    """max_length must track the longest step_count ever seen so far in the
    run, and must not drop when a later batch's own max is smaller."""
    spy = _SpyReplayBuffer()
    trainer = _trainer(
        [
            _batch(3, done=[False, True, False]),  # step_count 1,2,3 -> max 3
            _batch(
                2, done=[True, False]
            ),  # step_count 1,2 -> batch max 2 < running max
        ],
        init_rand_steps=0,
        max_steps=100,
        use_replay_buffer=True,
        replay_buffer=spy,
    )

    trainer._run_training_loop()

    max_lengths = [call[1] for call in trainer.optimization_calls]
    assert max_lengths == [3, 3]
    assert trainer._replay_buffer_max_step_count == 3


def test_run_training_loop_optimization_starts_when_init_rand_steps_exceeds_buffer_size() -> (
    None
):
    """Regression test for issue #356: a bounded replay buffer's length can
    never exceed buffer_size, so gating warm-up on buffer_len makes
    `collected_steps > init_rand_steps` permanently false whenever
    init_rand_steps > buffer_size -- zero gradient updates for the whole run.
    The gate must use total_count instead, which is unbounded."""
    buffer_size = 5
    init_rand_steps = 8  # > buffer_size, the exact scenario issue #356 covers
    spy = _SpyReplayBuffer(max_size=buffer_size)
    batches = [_batch(3) for _ in range(4)]  # total_count: 3, 6, 9, 12
    trainer = _trainer(
        batches,
        init_rand_steps=init_rand_steps,
        max_steps=100,
        use_replay_buffer=True,
        replay_buffer=spy,
    )

    trainer._run_training_loop()

    # buffer_len is capped at 5 forever; total_count crosses init_rand_steps=8
    # at batch index 2 (total_count=9), so optimization must start there.
    optimized_batches = [call[0] for call in trainer.optimization_calls]
    assert optimized_batches == [2, 3]
