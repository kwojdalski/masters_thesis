from __future__ import annotations

import pytest
import torch

from trading_rl.evaluation.policy_loader import PolicyLoader
from trading_rl.simple_continuous_wrapper import ContinuousActionWrapper


class FakeActor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))

    def forward(self, x):
        return x


def _checkpoint(**overrides) -> dict:
    actor = FakeActor()
    checkpoint = {
        "algorithm": "ppo",
        "n_obs": 3,
        "n_act": 2,
        "actor_hidden_dims": [8, 4],
        "value_hidden_dims": [4],
        "actor_state_dict": actor.state_dict(),
        "total_count": 12,
        "total_episodes": 3,
        "mlflow_run_id": "run-1",
        "mlflow_experiment_name": "exp",
    }
    checkpoint.update(overrides)
    return checkpoint


def test_policy_loader_inspect_returns_portable_metadata(tmp_path) -> None:
    path = tmp_path / "checkpoint.pt"
    torch.save(_checkpoint(action_low=[-1.0], action_high=[1.0]), path)

    meta = PolicyLoader.inspect(str(path))

    assert meta["algorithm"] == "ppo"
    assert meta["n_obs"] == 3
    assert meta["n_act"] == 2
    assert meta["actor_hidden_dims"] == [8, 4]
    assert meta["action_low"] == [-1.0]
    assert meta["action_high"] == [1.0]
    assert meta["mlflow_run_id"] == "run-1"


def test_policy_loader_build_actor_rejects_missing_algorithm() -> None:
    checkpoint = _checkpoint(algorithm=None)

    with pytest.raises(ValueError, match="algorithm"):
        PolicyLoader._build_actor(checkpoint, "cpu")


def test_policy_loader_build_actor_rejects_missing_n_obs() -> None:
    checkpoint = _checkpoint(n_obs=None)

    with pytest.raises(ValueError, match="n_obs"):
        PolicyLoader._build_actor(checkpoint, "cpu")


def test_policy_loader_build_actor_rejects_unsupported_algorithm() -> None:
    checkpoint = _checkpoint(algorithm="SAC")

    with pytest.raises(ValueError, match="Unsupported algorithm"):
        PolicyLoader._build_actor(checkpoint, "cpu")


def test_policy_loader_build_actor_uses_ppo_builder(monkeypatch) -> None:
    actor = FakeActor()
    calls = []

    def fake_builder(n_obs, n_act, hidden_dims, state_dict):
        calls.append((n_obs, n_act, hidden_dims))
        return actor

    monkeypatch.setattr(PolicyLoader, "_build_ppo_actor", staticmethod(fake_builder))

    loaded = PolicyLoader._build_actor(_checkpoint(algorithm="PPO"), "cpu")

    assert loaded is actor
    assert calls == [(3, 2, [8, 4])]
    assert loaded.training is False


def test_policy_loader_build_actor_passes_continuous_action_spec(monkeypatch) -> None:
    actor = FakeActor()
    captured = {}

    def fake_builder(n_obs, n_act, hidden_dims, spec):
        captured["n_obs"] = n_obs
        captured["n_act"] = n_act
        captured["hidden_dims"] = hidden_dims
        captured["spec"] = spec
        return actor

    monkeypatch.setattr(
        PolicyLoader,
        "_build_continuous_actor",
        staticmethod(fake_builder),
    )

    loaded = PolicyLoader._build_actor(
        _checkpoint(
            algorithm="TD3",
            n_act=1,
            action_low=[-0.5],
            action_high=[0.5],
        ),
        "cpu",
    )

    assert loaded is actor
    assert captured["n_obs"] == 3
    assert captured["n_act"] == 1
    torch.testing.assert_close(captured["spec"].low, torch.tensor([-0.5]))
    torch.testing.assert_close(captured["spec"].high, torch.tensor([0.5]))


def test_policy_loader_from_checkpoint_loads_file_and_delegates(monkeypatch, tmp_path) -> None:
    path = tmp_path / "checkpoint.pt"
    checkpoint = _checkpoint()
    torch.save(checkpoint, path)
    captured = {}

    def fake_build_actor(loaded_checkpoint, device):
        captured["checkpoint"] = loaded_checkpoint
        captured["device"] = device
        return "actor"

    monkeypatch.setattr(PolicyLoader, "_build_actor", staticmethod(fake_build_actor))

    assert PolicyLoader.from_checkpoint(str(path), device="cpu") == "actor"
    assert captured["checkpoint"]["algorithm"] == "ppo"
    assert captured["device"] == "cpu"


class _ConcreteContinuousActionWrapper(ContinuousActionWrapper):
    def _set_seed(self, seed: int | None):
        return seed


def _wrapper(thresholds=None) -> ContinuousActionWrapper:
    wrapper = object.__new__(_ConcreteContinuousActionWrapper)
    wrapper.thresholds = thresholds or [-0.33, 0.33]
    wrapper.device = "cpu"
    return wrapper


def test_continuous_action_wrapper_maps_values_to_discrete_indices() -> None:
    actions = torch.tensor([[-1.0], [-0.34], [-0.33], [0.0], [0.33], [0.34], [1.0]])

    discrete = _wrapper()._continuous_to_discrete(actions)

    torch.testing.assert_close(discrete, torch.tensor([0, 0, 1, 1, 1, 2, 2]))


def test_continuous_action_wrapper_preserves_batch_shape_without_last_action_dim() -> None:
    actions = torch.zeros((2, 3, 1), dtype=torch.float32)

    discrete = _wrapper()._continuous_to_discrete(actions)

    assert discrete.shape == (2, 3)


def test_continuous_action_wrapper_uses_custom_thresholds() -> None:
    actions = torch.tensor([[-0.75], [-0.25], [0.25], [0.75]])

    discrete = _wrapper(thresholds=[-0.5, 0.5])._continuous_to_discrete(actions)

    torch.testing.assert_close(discrete, torch.tensor([0, 1, 1, 2]))
