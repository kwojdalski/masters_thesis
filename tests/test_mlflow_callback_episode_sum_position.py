"""Regression test: episode_sum_position must average, not sum, across the
time x asset axes so it stays interpretable for multi-asset actions (#471)."""

from __future__ import annotations

from unittest.mock import MagicMock

from trading_rl.callbacks import mlflow_callback as mc


def _make_callback(monkeypatch) -> mc.MLflowTrainingCallback:
    monkeypatch.setattr(mc.mlflow, "set_experiment", MagicMock())
    monkeypatch.setattr(mc.mlflow, "active_run", lambda: True)
    monkeypatch.setattr(mc.mlflow, "start_run", MagicMock())
    monkeypatch.setattr(mc.mlflow, "log_metric", MagicMock())
    return mc.MLflowTrainingCallback(start_run=False)


class TestEpisodeSumPositionIsMeanNotSum:
    def test_single_asset_matches_mean_of_scalar_actions(self, monkeypatch):
        callback = _make_callback(monkeypatch)
        actions = [0.5, -0.5, 1.0]

        callback.log_episode_stats(
            episode_reward=1.0,
            portfolio_valuation=1000.0,
            actions=actions,
            exploration_ratio=0.0,
        )

        assert callback.training_stats["sum_positions"][-1] == 1.0 / 3

    def test_multi_asset_vector_actions_average_over_time_and_assets(self, monkeypatch):
        callback = _make_callback(monkeypatch)
        # np.sum would flatten both axes into 0.8-0.1+0.3-0.5+0.2+0.9 = 1.6,
        # mixing three unrelated assets' exposure into one meaningless number.
        actions = [[0.8, -0.1, 0.3], [-0.5, 0.2, 0.9]]

        callback.log_episode_stats(
            episode_reward=1.0,
            portfolio_valuation=1000.0,
            actions=actions,
            exploration_ratio=0.0,
        )

        expected_mean = (0.8 - 0.1 + 0.3 - 0.5 + 0.2 + 0.9) / 6
        assert callback.training_stats["sum_positions"][-1] == expected_mean
        assert callback.training_stats["sum_positions"][-1] != 1.6
