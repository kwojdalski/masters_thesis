from __future__ import annotations

from collections import defaultdict

import pytest
import torch
from tensordict import TensorDict

from trading_rl.config import DEFAULT_INITIAL_PORTFOLIO_VALUE
from trading_rl.constants import RewardType
from trading_rl.trainers.base import BaseTrainer


class _DummyTrainer(BaseTrainer):
    @staticmethod
    def build_models(n_obs: int, n_act: int, config, env):
        raise NotImplementedError

    def _optimization_step(self, batch_idx: int, max_length: int, buffer_len: int) -> None:
        raise NotImplementedError

    def _evaluate(self) -> None:
        raise NotImplementedError


class _Callback:
    initial_portfolio_value = DEFAULT_INITIAL_PORTFOLIO_VALUE
    reward_type = RewardType.LOG_RETURN

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def log_episode_stats(
        self,
        *,
        episode_reward: float,
        portfolio_valuation: float,
        actions: list,
        exploration_ratio: float,
    ) -> None:
        self.calls.append(
            {
                "episode_reward": episode_reward,
                "portfolio_valuation": portfolio_valuation,
                "actions": actions,
                "exploration_ratio": exploration_ratio,
            }
        )


def _trainer() -> _DummyTrainer:
    trainer = object.__new__(_DummyTrainer)
    trainer.logs = defaultdict(list)
    trainer.env = object()
    return trainer


def _batch(rewards: list[float], done: list[bool], actions: list[float]) -> TensorDict:
    n = len(rewards)
    return TensorDict(
        {
            "action": torch.tensor(actions, dtype=torch.float32).reshape(n, 1),
            "next": TensorDict(
                {
                    "reward": torch.tensor(rewards, dtype=torch.float32).reshape(n, 1),
                    "done": torch.tensor(done, dtype=torch.bool).reshape(n, 1),
                },
                batch_size=[n],
            ),
        },
        batch_size=[n],
    )


def test_episode_stats_split_multiple_completed_episodes_and_buffer_trailing() -> None:
    trainer = _trainer()
    callback = _Callback()
    data = _batch(
        rewards=[1.0, 2.0, 10.0, 20.0, 100.0],
        done=[False, True, False, True, False],
        actions=[0.0, 1.0, 0.0, 1.0, -1.0],
    )

    trainer._log_episode_stats(data, callback)

    assert [call["episode_reward"] for call in callback.calls] == pytest.approx(
        [3.0, 30.0]
    )
    assert [call["actions"] for call in callback.calls] == [[0.0, 1.0], [0.0, 1.0]]
    assert trainer._pending_episode_rewards == [100.0]
    assert trainer._pending_episode_actions == [-1.0]
    assert trainer.logs["episode_log_count"] == [1, 2]


def test_episode_stats_carry_incomplete_episode_across_batches() -> None:
    trainer = _trainer()
    callback = _Callback()

    trainer._log_episode_stats(
        _batch(
            rewards=[1.0, 2.0],
            done=[False, False],
            actions=[0.0, 0.0],
        ),
        callback,
    )
    assert callback.calls == []

    trainer._log_episode_stats(
        _batch(
            rewards=[3.0, 4.0, 5.0],
            done=[True, False, True],
            actions=[1.0, 1.0, 0.0],
        ),
        callback,
    )

    assert [call["episode_reward"] for call in callback.calls] == pytest.approx(
        [6.0, 9.0]
    )
    assert [call["actions"] for call in callback.calls] == [
        [0.0, 0.0, 1.0],
        [1.0, 0.0],
    ]
    assert trainer._pending_episode_rewards == []
    assert trainer._pending_episode_actions == []
    assert trainer.logs["episode_log_count"] == [1, 2]
