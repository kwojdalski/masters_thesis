"""Focused tests for off-policy trainer warmup and real loss wiring."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from tensordict import TensorDict
from torchrl.data import Bounded

from trading_rl.constants import LossFunction
from trading_rl.trainers import base as base_module
from trading_rl.trainers.ddpg import DDPGTrainer
from trading_rl.trainers.td3 import TD3Trainer
from trading_rl.trainers.warmup import WarmupController


def _action_spec(n_act: int = 1) -> Bounded:
    return Bounded(low=-0.5, high=0.5, shape=(n_act,), dtype=torch.float32)


def _network_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        network=SimpleNamespace(actor_hidden_dims=[8], value_hidden_dims=[8])
    )


def _patch_base_init(monkeypatch) -> None:
    def fake_init(self, actor, value_net, env, config, **_kwargs) -> None:
        self.actor = actor
        self.value_net = value_net
        self.env = env
        self.config = config
        self.logs = {}
        self.total_count = 0
        self.total_episodes = 0

    monkeypatch.setattr(base_module.BaseTrainer, "__init__", fake_init)


def _td3_config() -> SimpleNamespace:
    return SimpleNamespace(
        actor_lr=1e-3,
        value_lr=1e-3,
        actor_weight_decay=0.0,
        value_weight_decay=0.0,
        max_steps=100,
        tau=0.01,
        max_grad_norm=0.0,
        loss_function=LossFunction.L2,
        td3=SimpleNamespace(
            exploration_noise_std=0.1,
            policy_noise=0.2,
            noise_clip=0.5,
            delay_actor=True,
            delay_qvalue=True,
            policy_delay=2,
        ),
    )


def _ddpg_config() -> SimpleNamespace:
    return SimpleNamespace(
        actor_lr=1e-3,
        value_lr=1e-3,
        actor_weight_decay=0.0,
        value_weight_decay=0.0,
        max_steps=100,
        tau=0.01,
        max_grad_norm=0.0,
        loss_function=LossFunction.L2,
        buffer_size=100,
        td3=SimpleNamespace(exploration_noise_std=0.1),
    )


def _sample_for_loss(actor, n_obs: int, batch_size: int = 4) -> TensorDict:
    obs = torch.randn(batch_size, n_obs)
    action_td = actor(TensorDict({"observation": obs.clone()}, batch_size=[batch_size]))
    return TensorDict(
        {
            "observation": obs,
            "action": action_td["action"].detach(),
            ("next", "observation"): torch.randn(batch_size, n_obs),
            ("next", "reward"): torch.randn(batch_size, 1),
            ("next", "done"): torch.zeros(batch_size, 1, dtype=torch.bool),
            ("next", "terminated"): torch.zeros(batch_size, 1, dtype=torch.bool),
        },
        batch_size=[batch_size],
    )


def test_offpolicy_warmup_switches_only_after_threshold() -> None:
    collector = SimpleNamespace(policy=None)
    replay_buffer = [object(), object(), object()]
    exploration_policy = object()

    wc = WarmupController(
        collector=collector,
        init_rand_steps=5,
        replay_buffer=replay_buffer,
        use_replay_buffer=True,
    )
    wc.initialize(exploration_policy, _action_spec(), total_count=0, algorithm_label="DDPG")

    assert wc.random_exploration_done is False
    assert collector.policy is not exploration_policy

    wc.maybe_switch(4, algorithm_label="DDPG")
    assert collector.policy is not exploration_policy
    assert wc.random_exploration_done is False

    wc.maybe_switch(5, algorithm_label="DDPG")
    assert collector.policy is exploration_policy
    assert wc.random_exploration_done is True


def test_offpolicy_warmup_starts_with_exploration_policy_when_already_complete() -> None:
    collector = SimpleNamespace(policy=None)
    exploration_policy = object()

    wc = WarmupController(collector=collector, init_rand_steps=5)
    wc.initialize(exploration_policy, _action_spec(), total_count=5, algorithm_label="TD3")

    assert collector.policy is exploration_policy
    assert wc.random_exploration_done is True


def test_td3_build_models_and_real_loss_accept_expected_tensordict(monkeypatch) -> None:
    torch.manual_seed(0)
    _patch_base_init(monkeypatch)
    n_obs, n_act = 3, 1
    env = SimpleNamespace(action_spec=_action_spec(n_act))
    actor, value_net = TD3Trainer.build_models(n_obs, n_act, _network_cfg(), env)
    trainer = TD3Trainer(
        actor,
        value_net,
        env,
        _td3_config(),
        n_obs=n_obs,
        n_act=n_act,
        actor_hidden_dims=[8],
        value_hidden_dims=[8],
    )

    action_td = trainer.actor(
        TensorDict({"observation": torch.randn(4, n_obs)}, batch_size=[4])
    )
    assert action_td["action"].shape == torch.Size([4, n_act])
    assert torch.all(action_td["action"] <= 0.5)
    assert torch.all(action_td["action"] >= -0.5)
    assert trainer.td3_loss.qvalue_network_params.batch_size == torch.Size([2])

    losses = trainer.td3_loss(_sample_for_loss(trainer.actor, n_obs))

    assert torch.isfinite(losses["loss_actor"])
    assert torch.isfinite(losses["loss_qvalue"])
    assert losses["pred_value"].shape == torch.Size([2, 4])


def test_ddpg_build_models_and_real_loss_accept_expected_tensordict(monkeypatch) -> None:
    torch.manual_seed(1)
    _patch_base_init(monkeypatch)
    n_obs, n_act = 3, 1
    env = SimpleNamespace(action_spec=_action_spec(n_act))
    actor, value_net = DDPGTrainer.build_models(n_obs, n_act, _network_cfg(), env)
    ddpg_config = _ddpg_config()
    trainer = DDPGTrainer(
        actor,
        value_net,
        env,
        ddpg_config,
        n_obs=n_obs,
        n_act=n_act,
        actor_hidden_dims=[8],
        value_hidden_dims=[8],
    )

    assert trainer.exploration_module.sigma_init == ddpg_config.td3.exploration_noise_std
    assert trainer._compute_exploration_ratio() == ddpg_config.td3.exploration_noise_std

    action_td = trainer.actor(
        TensorDict({"observation": torch.randn(4, n_obs)}, batch_size=[4])
    )
    assert action_td["action"].shape == torch.Size([4, n_act])
    assert torch.all(action_td["action"] <= 0.5)
    assert torch.all(action_td["action"] >= -0.5)
    assert trainer.ddpg_loss.target_actor_network_params is not None
    assert trainer.ddpg_loss.target_value_network_params is not None

    losses = trainer.ddpg_loss(_sample_for_loss(trainer.actor, n_obs))

    assert torch.isfinite(losses["loss_actor"])
    assert torch.isfinite(losses["loss_value"])
    assert losses["pred_value"].shape == torch.Size([4])
