from __future__ import annotations

from types import SimpleNamespace

import torch

from trading_rl.trainers.base import BaseTrainer


class _ConcreteTrainer(BaseTrainer):
    @staticmethod
    def build_models(n_obs, n_act, config, env):
        raise NotImplementedError

    def _optimization_step(self, batch_idx, max_length, buffer_len) -> None:
        raise NotImplementedError

    def _evaluate(self) -> None:
        raise NotImplementedError


def test_extract_logged_actions_maps_one_hot_actions_to_callback_positions() -> None:
    trainer = _ConcreteTrainer.__new__(_ConcreteTrainer)
    actions = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    callback = SimpleNamespace(action_positions=[-1, 0, 1])

    logged_actions = trainer._extract_logged_actions(actions, callback)

    assert logged_actions == [-1, 0, 1]
