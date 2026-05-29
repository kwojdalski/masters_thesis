from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace

import torch

from trading_rl.trainers.base import BaseTrainer
from trading_rl.trainers.episode_stats import EpisodeStatsTracker


class _ConcreteTrainer(BaseTrainer):
    @staticmethod
    def build_models(n_obs, n_act, config, env):
        raise NotImplementedError

    def _optimization_step(self, batch_idx, max_length, buffer_len) -> None:
        raise NotImplementedError

    def _evaluate(self) -> None:
        raise NotImplementedError

    @property
    def _algo_label(self) -> str:
        return "test"

    def _get_checkpoint_network_state(self) -> dict:
        return {}

    def _load_checkpoint_network_state(self, checkpoint: dict) -> None:
        pass


def test_extract_logged_actions_maps_one_hot_actions_to_callback_positions() -> None:
    trainer = _ConcreteTrainer.__new__(_ConcreteTrainer)
    trainer.episode_stats = EpisodeStatsTracker(
        env=object(),
        logs=defaultdict(list),
        compute_exploration_ratio=lambda: 0.0,
        get_last_episode_final_nlv=lambda: (None, None),
        get_current_episode_context=lambda: (None, None, None),
    )
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
