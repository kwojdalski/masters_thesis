"""Checkpoint round-trip tests for trainer resume state."""

from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace

import torch

from trading_rl.trainers.ddpg import DDPGTrainer
from trading_rl.trainers.ppo import PPOTrainer
from trading_rl.trainers.td3 import TD3Trainer


class _FunctionalParams:
    def __init__(self, value: float) -> None:
        self.value = torch.tensor([value], dtype=torch.float32)
        self.loaded_state: dict[str, torch.Tensor] | None = None
        self.to_module_calls = 0

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"value": self.value.clone()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.loaded_state = {key: val.clone() for key, val in state.items()}
        self.value = self.loaded_state["value"].clone()

    def to_module(self, _module) -> None:
        self.to_module_calls += 1


class _PpoLoss:
    def __init__(self) -> None:
        self.actor_network_params = _FunctionalParams(0.5)
        self.critic_network_params = _FunctionalParams(1.5)


class _DdpgLoss:
    def __init__(self) -> None:
        self.actor_network_params = _FunctionalParams(1.1)
        self.value_network_params = _FunctionalParams(2.2)
        self.target_actor_network_params = _FunctionalParams(3.3)
        self.target_value_network_params = _FunctionalParams(4.4)


class _Td3Loss:
    def __init__(self) -> None:
        self.actor_network_params = _FunctionalParams(5.5)
        self.qvalue_network_params = _FunctionalParams(6.6)
        self.target_actor_network_params = _FunctionalParams(7.7)
        self.target_qvalue_network_params = _FunctionalParams(8.8)


def _linear(value: float) -> torch.nn.Linear:
    layer = torch.nn.Linear(2, 1)
    with torch.no_grad():
        layer.weight.fill_(value)
        layer.bias.fill_(value)
    return layer


def _assert_module_equal(left: torch.nn.Module, right: torch.nn.Module) -> None:
    for left_param, right_param in zip(left.parameters(), right.parameters(), strict=True):
        torch.testing.assert_close(left_param, right_param)


def test_ppo_checkpoint_round_trip_restores_module_optimizer_and_counters(
    tmp_path, monkeypatch
) -> None:
    import mlflow

    monkeypatch.setattr(mlflow, "active_run", lambda: None)
    monkeypatch.setattr(mlflow, "get_tracking_uri", lambda: "file://unit-test")

    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.actor = _linear(0.5)
    trainer.value_net = _linear(1.5)
    trainer.optimizer = torch.optim.Adam(
        list(trainer.actor.parameters()) + list(trainer.value_net.parameters()),
        lr=0.01,
    )
    trainer.ppo_loss = _PpoLoss()
    trainer.total_count = 123
    trainer.total_episodes = 7
    trainer.logs = defaultdict(list, {"loss_actor": [0.1], "episode_log_count": [4]})
    trainer.n_obs = 2
    trainer.n_act = 1
    trainer.actor_hidden_dims = [16]
    trainer.value_hidden_dims = [32]

    path = tmp_path / "ppo.pt"
    trainer.save_checkpoint(str(path), feature_pipeline_state={"feature_x": {"mean": 1.0}})

    restored = PPOTrainer.__new__(PPOTrainer)
    restored.actor = _linear(0.0)
    restored.value_net = _linear(0.0)
    restored.optimizer = torch.optim.Adam(
        list(restored.actor.parameters()) + list(restored.value_net.parameters()),
        lr=0.01,
    )
    restored.ppo_loss = _PpoLoss()

    restored.load_checkpoint(str(path))

    _assert_module_equal(restored.actor, trainer.actor)
    _assert_module_equal(restored.value_net, trainer.value_net)
    assert restored.total_count == 123
    assert restored.total_episodes == 7
    assert restored.logs["loss_actor"] == [0.1]
    assert restored.mlflow_tracking_uri == "file://unit-test"
    assert restored.optimizer.state_dict()["param_groups"][0]["lr"] == 0.01


def test_ddpg_checkpoint_round_trip_restores_functional_params_and_syncs_modules(
    tmp_path, monkeypatch
) -> None:
    import mlflow

    monkeypatch.setattr(mlflow, "active_run", lambda: None)
    monkeypatch.setattr(mlflow, "get_tracking_uri", lambda: "file://unit-test")

    trainer = DDPGTrainer.__new__(DDPGTrainer)
    trainer.actor = _linear(0.1)
    trainer.value_net = _linear(0.2)
    trainer.optimizer_actor = torch.optim.Adam(trainer.actor.parameters(), lr=0.01)
    trainer.optimizer_value = torch.optim.Adam(trainer.value_net.parameters(), lr=0.02)
    trainer.ddpg_loss = _DdpgLoss()
    trainer.env = SimpleNamespace(action_spec=None)
    trainer.config = SimpleNamespace(save_buffer=False)
    trainer.replay_buffer = None
    trainer.total_count = 456
    trainer.total_episodes = 8
    trainer.logs = defaultdict(list, {"loss_value": [0.2], "episode_log_count": [5]})
    trainer.n_obs = 2
    trainer.n_act = 1
    trainer.actor_hidden_dims = [8]
    trainer.value_hidden_dims = [16]

    path = tmp_path / "ddpg.pt"
    trainer.save_checkpoint(str(path), feature_pipeline_state={"feature_y": {"var": 2.0}})

    restored = DDPGTrainer.__new__(DDPGTrainer)
    restored.actor = _linear(0.0)
    restored.value_net = _linear(0.0)
    restored.optimizer_actor = torch.optim.Adam(restored.actor.parameters(), lr=0.01)
    restored.optimizer_value = torch.optim.Adam(restored.value_net.parameters(), lr=0.02)
    restored.ddpg_loss = _DdpgLoss()

    restored.load_checkpoint(str(path))

    torch.testing.assert_close(restored.ddpg_loss.actor_network_params.value, torch.tensor([1.1]))
    torch.testing.assert_close(restored.ddpg_loss.value_network_params.value, torch.tensor([2.2]))
    torch.testing.assert_close(restored.ddpg_loss.target_actor_network_params.value, torch.tensor([3.3]))
    torch.testing.assert_close(restored.ddpg_loss.target_value_network_params.value, torch.tensor([4.4]))
    assert restored.ddpg_loss.actor_network_params.to_module_calls == 1
    assert restored.ddpg_loss.value_network_params.to_module_calls == 1
    assert restored.total_count == 456
    assert restored.total_episodes == 8
    assert restored.logs["loss_value"] == [0.2]


def test_td3_checkpoint_round_trip_restores_functional_params_and_syncs_modules(
    tmp_path, monkeypatch
) -> None:
    import mlflow

    monkeypatch.setattr(mlflow, "active_run", lambda: None)
    monkeypatch.setattr(mlflow, "get_tracking_uri", lambda: "file://unit-test")

    trainer = TD3Trainer.__new__(TD3Trainer)
    trainer.actor = _linear(0.3)
    trainer.value_net = _linear(0.4)
    trainer.optimizer_actor = torch.optim.Adam(trainer.actor.parameters(), lr=0.01)
    trainer.optimizer_value = torch.optim.Adam(trainer.value_net.parameters(), lr=0.02)
    trainer.td3_loss = _Td3Loss()
    trainer.env = SimpleNamespace(action_spec=None)
    trainer.config = SimpleNamespace(save_buffer=False)
    trainer.replay_buffer = None
    trainer.total_count = 789
    trainer.total_episodes = 9
    trainer.logs = defaultdict(list, {"loss_value": [0.3], "episode_log_count": [6]})
    trainer.n_obs = 2
    trainer.n_act = 1
    trainer.actor_hidden_dims = [8]
    trainer.value_hidden_dims = [16]

    path = tmp_path / "td3.pt"
    trainer.save_checkpoint(str(path), feature_pipeline_state={"feature_z": {"std": 3.0}})

    restored = TD3Trainer.__new__(TD3Trainer)
    restored.actor = _linear(0.0)
    restored.value_net = _linear(0.0)
    restored.optimizer_actor = torch.optim.Adam(restored.actor.parameters(), lr=0.01)
    restored.optimizer_value = torch.optim.Adam(restored.value_net.parameters(), lr=0.02)
    restored.td3_loss = _Td3Loss()

    restored.load_checkpoint(str(path))

    torch.testing.assert_close(restored.td3_loss.actor_network_params.value, torch.tensor([5.5]))
    torch.testing.assert_close(restored.td3_loss.qvalue_network_params.value, torch.tensor([6.6]))
    torch.testing.assert_close(restored.td3_loss.target_actor_network_params.value, torch.tensor([7.7]))
    torch.testing.assert_close(restored.td3_loss.target_qvalue_network_params.value, torch.tensor([8.8]))
    assert restored.td3_loss.actor_network_params.to_module_calls == 1
    assert restored.td3_loss.qvalue_network_params.to_module_calls == 1
    assert restored.total_count == 789
    assert restored.total_episodes == 9
    assert restored.logs["loss_value"] == [0.3]
