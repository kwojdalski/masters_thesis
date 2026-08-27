"""Unit tests for network architecture factories in trading_rl.models.

Covers output shapes, action bounds, and gradient flow for every
create_* factory (DQN, PPO discrete/continuous, DDPG, TD3, SAC,
recurrent PPO) so a wrong activation, wrong action-concatenation
order, or broken hidden-layer wiring fails loudly here instead of
surfacing only as "the agent underperforms" after a full training run.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict
from torchrl.data import Bounded, OneHot

from trading_rl.models import (
    ScaleFromUnitRange,
    _extract_action_bounds_from_spec,
    _FlattenObs,
    count_parameters,
    create_actor,
    create_continuous_ppo_actor,
    create_ddpg_actor,
    create_ppo_actor,
    create_ppo_value_network,
    create_recurrent_ppo_actor,
    create_recurrent_ppo_value_network,
    create_sac_actor,
    create_sac_qvalue_network,
    create_td3_actor,
    create_td3_qvalue_network,
    create_value_network,
)

N_OBS = 4
N_ACT = 2
BATCH = 8


def _obs_td(batch: int = BATCH, n_obs: int = N_OBS) -> TensorDict:
    return TensorDict({"observation": torch.randn(batch, n_obs)}, batch_size=[batch])


def _obs_action_td(
    batch: int = BATCH, n_obs: int = N_OBS, n_act: int = N_ACT
) -> TensorDict:
    return TensorDict(
        {"observation": torch.randn(batch, n_obs), "action": torch.randn(batch, n_act)},
        batch_size=[batch],
    )


def _log_prob_key(out: TensorDict) -> str:
    """Return whichever log-prob key TorchRL used for this ProbabilisticActor.

    TorchRL has a global toggle (composite log-prob aggregation mode) that
    controls whether this key is named "action_log_prob" or "sample_log_prob";
    some other test module elsewhere in the suite flips it without resetting,
    so tests here must not hardcode one name.
    """
    for key in ("action_log_prob", "sample_log_prob"):
        if key in out.keys():
            return key
    raise AssertionError(
        f"no log-prob key found in output, got keys={list(out.keys())}"
    )


def _assert_requires_grad(module: torch.nn.Module) -> None:
    params = list(module.parameters())
    assert params, "module has no parameters"
    assert all(p.requires_grad for p in params)


def _assert_backward_ok(loss: torch.Tensor, module: torch.nn.Module) -> None:
    loss.backward()
    grads = [p.grad for p in module.parameters() if p.requires_grad]
    assert any(g is not None for g in grads), "no gradients flowed to any parameter"


def _inflate_linear_weights(module: torch.nn.Module, scale: float = 50.0) -> None:
    """Force large pre-activation magnitudes so bound-checking tests actually
    exercise the saturating activation (tanh) instead of passing vacuously
    because of a small-magnitude weight init that happens to keep raw output
    inside the bound anyway."""
    with torch.no_grad():
        for m in module.modules():
            if isinstance(m, torch.nn.Linear):
                m.weight.normal_(mean=0.0, std=scale)
                m.bias.normal_(mean=0.0, std=scale)


# ---------------------------------------------------------------------------
# create_actor (discrete DQN-style)
# ---------------------------------------------------------------------------


def test_create_actor_outputs_one_hot_action_matching_n_act() -> None:
    spec = OneHot(n=3)
    actor = create_actor(N_OBS, 3, hidden_dims=[8], spec=spec)

    out = actor(_obs_td(n_obs=N_OBS))

    assert out["action"].shape == (BATCH, 3)
    assert out["action"].dtype == torch.bool
    # exactly one action selected per row (one-hot)
    assert torch.equal(out["action"].sum(dim=-1), torch.ones(BATCH, dtype=torch.int64))


def test_create_actor_requires_grad() -> None:
    spec = OneHot(n=3)
    actor = create_actor(N_OBS, 3, hidden_dims=[8], spec=spec)
    _assert_requires_grad(actor)


# ---------------------------------------------------------------------------
# create_value_network (Q(s, a) for DQN-style)
# ---------------------------------------------------------------------------


def test_create_value_network_outputs_scalar_per_row() -> None:
    value_net = create_value_network(N_OBS, N_ACT, hidden_dims=[8, 4])

    out = value_net(_obs_action_td())

    assert out["state_action_value"].shape == (BATCH, 1)


def test_create_value_network_backward() -> None:
    value_net = create_value_network(N_OBS, N_ACT, hidden_dims=[8, 4])
    out = value_net(_obs_action_td())
    _assert_backward_ok(out["state_action_value"].sum(), value_net)


# ---------------------------------------------------------------------------
# create_ppo_actor (discrete PPO)
# ---------------------------------------------------------------------------


def test_create_ppo_actor_outputs_one_hot_action_and_log_prob() -> None:
    spec = OneHot(n=4)
    actor = create_ppo_actor(N_OBS, 4, hidden_dims=[8], spec=spec)

    out = actor(_obs_td())

    assert out["action"].shape == (BATCH, 4)
    assert torch.equal(out["action"].sum(dim=-1), torch.ones(BATCH, dtype=torch.int64))
    log_prob_key = _log_prob_key(out)
    assert out[log_prob_key].shape == (BATCH,)


# ---------------------------------------------------------------------------
# create_continuous_ppo_actor (TanhNormal)
# ---------------------------------------------------------------------------


def test_create_continuous_ppo_actor_bounds_action_to_unit_range() -> None:
    spec = Bounded(low=-1.0, high=1.0, shape=(N_ACT,), dtype=torch.float32)
    actor = create_continuous_ppo_actor(N_OBS, N_ACT, hidden_dims=[8], spec=spec)

    # large-magnitude observations to push the pre-tanh mean far from 0,
    # stress-testing the tanh squash rather than relying on near-zero init
    out = actor(_obs_td(batch=200, n_obs=N_OBS) * 5.0)

    assert out["action"].shape == (200, N_ACT)
    assert torch.all(out["action"] >= -1.0)
    assert torch.all(out["action"] <= 1.0)
    _log_prob_key(out)  # raises if no log-prob key present


def test_create_continuous_ppo_actor_rescales_to_non_unit_spec_bounds() -> None:
    """A [0, 2] action spec must actually bound sampled actions -- TanhNormal
    defaults to [-1, 1] and never reads `spec` itself when safe=False."""
    spec = Bounded(low=0.0, high=2.0, shape=(N_ACT,), dtype=torch.float32)
    actor = create_continuous_ppo_actor(N_OBS, N_ACT, hidden_dims=[8], spec=spec)

    out = actor(_obs_td(batch=200, n_obs=N_OBS) * 5.0)

    assert torch.all(out["action"] >= 0.0)
    assert torch.all(out["action"] <= 2.0)
    # a real bug here clamps to [-1, 1], so assert some mass actually lands
    # outside that range to prove the wider bound is honored, not coincidental
    assert (out["action"] > 1.0).any()


def test_create_continuous_ppo_actor_requires_grad() -> None:
    spec = Bounded(low=-1.0, high=1.0, shape=(N_ACT,), dtype=torch.float32)
    actor = create_continuous_ppo_actor(N_OBS, N_ACT, hidden_dims=[8], spec=spec)
    _assert_requires_grad(actor)


# ---------------------------------------------------------------------------
# create_ppo_value_network (V(s))
# ---------------------------------------------------------------------------


def test_create_ppo_value_network_outputs_scalar_per_row() -> None:
    value_net = create_ppo_value_network(N_OBS, hidden_dims=[8, 4])

    out = value_net(_obs_td())

    assert out["state_value"].shape == (BATCH, 1)


# ---------------------------------------------------------------------------
# create_ddpg_actor / create_td3_actor (deterministic, tanh-bounded)
# ---------------------------------------------------------------------------


def test_create_ddpg_actor_outputs_action_matching_n_act() -> None:
    spec = Bounded(low=-1.0, high=1.0, shape=(N_ACT,), dtype=torch.float32)
    actor = create_ddpg_actor(N_OBS, N_ACT, hidden_dims=[8], spec=spec)

    out = actor(_obs_td())

    assert out["action"].shape == (BATCH, N_ACT)
    assert torch.all(out["action"] >= -1.0)
    assert torch.all(out["action"] <= 1.0)


def test_create_ddpg_actor_saturates_to_unit_range_under_large_activations() -> None:
    """Regression guard for a missing/removed tanh: force large pre-activation
    magnitudes (bypassing the actor's own small-init scheme) and verify the
    output is still hard-bounded to [-1, 1], not just small by coincidence."""
    spec = Bounded(low=-1.0, high=1.0, shape=(N_ACT,), dtype=torch.float32)
    actor = create_ddpg_actor(N_OBS, N_ACT, hidden_dims=[8], spec=spec)
    _inflate_linear_weights(actor)

    out = actor(_obs_td(batch=200) * 5.0)

    assert torch.all(out["action"] >= -1.0)
    assert torch.all(out["action"] <= 1.0)
    # with inflated weights the pre-tanh signal is large, so a real tanh
    # should be saturating near the bounds for most rows, not clustering at 0
    assert out["action"].abs().mean() > 0.9


def test_create_ddpg_actor_rescales_to_non_unit_spec_bounds() -> None:
    """A [0, 1] action spec (e.g. long-only allocation) must rescale the
    actor's internal tanh output, not just clip it to [-1, 1]."""
    spec = Bounded(low=0.0, high=1.0, shape=(N_ACT,), dtype=torch.float32)
    actor = create_ddpg_actor(N_OBS, N_ACT, hidden_dims=[8], spec=spec)
    _inflate_linear_weights(actor)

    out = actor(_obs_td(batch=200) * 5.0)

    assert torch.all(out["action"] >= 0.0)
    assert torch.all(out["action"] <= 1.0)
    # inflated weights should saturate the pre-scale tanh near +-1, which
    # ScaleFromUnitRange maps to the [0, 1] endpoints, not the 0.5 midpoint
    assert ((out["action"] < 0.1) | (out["action"] > 0.9)).float().mean() > 0.9


def test_create_ddpg_actor_without_spec_stays_in_unit_range() -> None:
    actor = create_ddpg_actor(N_OBS, N_ACT, hidden_dims=[8], spec=None)
    _inflate_linear_weights(actor)

    out = actor(_obs_td(batch=200) * 5.0)

    assert torch.all(out["action"] >= -1.0)
    assert torch.all(out["action"] <= 1.0)


def test_create_ddpg_actor_requires_grad_and_backward() -> None:
    actor = create_ddpg_actor(N_OBS, N_ACT, hidden_dims=[8], spec=None)
    _assert_requires_grad(actor)
    out = actor(_obs_td())
    _assert_backward_ok(out["action"].sum(), actor)


def test_create_td3_actor_matches_ddpg_actor_behavior() -> None:
    spec = Bounded(low=-1.0, high=1.0, shape=(N_ACT,), dtype=torch.float32)
    actor = create_td3_actor(N_OBS, N_ACT, hidden_dims=[8], spec=spec)

    out = actor(_obs_td())

    assert out["action"].shape == (BATCH, N_ACT)
    assert torch.all(out["action"] >= -1.0)
    assert torch.all(out["action"] <= 1.0)


# ---------------------------------------------------------------------------
# create_td3_qvalue_network / create_sac_qvalue_network
# ---------------------------------------------------------------------------


def test_create_td3_qvalue_network_outputs_scalar_per_row() -> None:
    value_net = create_td3_qvalue_network(N_OBS, N_ACT, hidden_dims=[8, 4])

    out = value_net(_obs_action_td())

    assert out["state_action_value"].shape == (BATCH, 1)


def test_create_sac_qvalue_network_matches_td3_qvalue_network_shape() -> None:
    value_net = create_sac_qvalue_network(N_OBS, N_ACT, hidden_dims=[8, 4])

    out = value_net(_obs_action_td())

    assert out["state_action_value"].shape == (BATCH, 1)


# ---------------------------------------------------------------------------
# create_sac_actor (delegates to create_continuous_ppo_actor)
# ---------------------------------------------------------------------------


def test_create_sac_actor_bounds_action_and_returns_log_prob() -> None:
    spec = Bounded(low=-1.0, high=1.0, shape=(N_ACT,), dtype=torch.float32)
    actor = create_sac_actor(N_OBS, N_ACT, hidden_dims=[8], spec=spec)

    out = actor(_obs_td(batch=200) * 5.0)

    assert out["action"].shape == (200, N_ACT)
    assert torch.all(out["action"] >= -1.0)
    assert torch.all(out["action"] <= 1.0)
    _log_prob_key(out)  # raises if no log-prob key present


def test_create_sac_actor_rescales_to_non_unit_spec_bounds() -> None:
    spec = Bounded(low=0.0, high=2.0, shape=(N_ACT,), dtype=torch.float32)
    actor = create_sac_actor(N_OBS, N_ACT, hidden_dims=[8], spec=spec)

    out = actor(_obs_td(batch=200) * 5.0)

    assert torch.all(out["action"] >= 0.0)
    assert torch.all(out["action"] <= 2.0)
    assert (out["action"] > 1.0).any()


# ---------------------------------------------------------------------------
# create_recurrent_ppo_actor / create_recurrent_ppo_value_network (GRU)
# ---------------------------------------------------------------------------


def test_create_recurrent_ppo_actor_rescales_to_non_unit_spec_bounds() -> None:
    spec = Bounded(low=0.0, high=2.0, shape=(N_ACT,), dtype=torch.float32)
    actor = create_recurrent_ppo_actor(N_OBS, N_ACT, hidden_dims=[8], spec=spec)

    td = TensorDict({"observation": torch.randn(200, N_OBS) * 5.0}, batch_size=[200])
    out = actor(td)

    assert torch.all(out["action"] >= 0.0)
    assert torch.all(out["action"] <= 2.0)
    assert (out["action"] > 1.0).any()


def test_create_recurrent_ppo_actor_handles_unbatched_1d_obs() -> None:
    spec = Bounded(low=-1.0, high=1.0, shape=(N_ACT,), dtype=torch.float32)
    actor = create_recurrent_ppo_actor(N_OBS, N_ACT, hidden_dims=[8], spec=spec)

    td = TensorDict({"observation": torch.randn(N_OBS)}, batch_size=[])
    out = actor(td)

    assert out["action"].shape == (N_ACT,)
    assert torch.all(out["action"] >= -1.0)
    assert torch.all(out["action"] <= 1.0)


def test_create_recurrent_ppo_actor_handles_2d_sequence_obs() -> None:
    spec = Bounded(low=-1.0, high=1.0, shape=(N_ACT,), dtype=torch.float32)
    actor = create_recurrent_ppo_actor(N_OBS, N_ACT, hidden_dims=[8], spec=spec)

    seq_len = 5
    td = TensorDict({"observation": torch.randn(seq_len, N_OBS)}, batch_size=[seq_len])
    out = actor(td)

    assert out["action"].shape == (seq_len, N_ACT)


def test_create_recurrent_ppo_actor_handles_batched_3d_sequence_obs() -> None:
    spec = Bounded(low=-1.0, high=1.0, shape=(N_ACT,), dtype=torch.float32)
    actor = create_recurrent_ppo_actor(N_OBS, N_ACT, hidden_dims=[8], spec=spec)

    batch, seq_len = 3, 5
    td = TensorDict(
        {"observation": torch.randn(batch, seq_len, N_OBS)}, batch_size=[batch, seq_len]
    )
    out = actor(td)

    assert out["action"].shape == (batch, seq_len, N_ACT)


def test_create_recurrent_ppo_actor_uses_hidden_dims_0_as_gru_size() -> None:
    spec = Bounded(low=-1.0, high=1.0, shape=(N_ACT,), dtype=torch.float32)
    actor = create_recurrent_ppo_actor(N_OBS, N_ACT, hidden_dims=[16, 32], spec=spec)

    gru = actor.module[0].module.gru
    assert gru.hidden_size == 16


def test_create_recurrent_ppo_value_network_outputs_scalar_per_row() -> None:
    value_net = create_recurrent_ppo_value_network(N_OBS, hidden_dims=[8, 4])

    out = value_net(_obs_td())

    assert out["state_value"].shape == (BATCH, 1)


def test_create_recurrent_ppo_value_network_uses_hidden_dims_0_as_gru_size() -> None:
    value_net = create_recurrent_ppo_value_network(N_OBS, hidden_dims=[16, 32])

    gru = value_net.module.gru
    assert gru.hidden_size == 16


# ---------------------------------------------------------------------------
# Helper classes / functions
# ---------------------------------------------------------------------------


def test_scale_from_unit_range_maps_endpoints_correctly() -> None:
    scaler = ScaleFromUnitRange(
        low=torch.tensor([0.0, -5.0]), high=torch.tensor([1.0, 5.0])
    )

    low_in = torch.tensor([[-1.0, -1.0]])
    high_in = torch.tensor([[1.0, 1.0]])
    mid_in = torch.tensor([[0.0, 0.0]])

    assert torch.allclose(scaler(low_in), torch.tensor([[0.0, -5.0]]))
    assert torch.allclose(scaler(high_in), torch.tensor([[1.0, 5.0]]))
    assert torch.allclose(scaler(mid_in), torch.tensor([[0.5, 0.0]]))


def test_extract_action_bounds_from_spec_returns_none_when_missing() -> None:
    assert _extract_action_bounds_from_spec(None) is None


def test_extract_action_bounds_from_spec_reads_bounded_spec() -> None:
    spec = Bounded(low=-2.0, high=3.0, shape=(N_ACT,), dtype=torch.float32)

    bounds = _extract_action_bounds_from_spec(spec)

    assert bounds is not None
    low, high = bounds
    assert torch.allclose(low, torch.full((N_ACT,), -2.0))
    assert torch.allclose(high, torch.full((N_ACT,), 3.0))


def test_flatten_obs_flattens_multi_dim_observation() -> None:
    flatten = _FlattenObs(obs_ndim=2)
    x = torch.randn(3, 5, 4)  # [batch, window, n_features]

    out = flatten(x)

    assert out.shape == (3, 20)


def test_flatten_obs_is_noop_for_1d_observation() -> None:
    flatten = _FlattenObs(obs_ndim=1)
    x = torch.randn(3, 4)

    out = flatten(x)

    assert torch.equal(out, x)


def test_count_parameters_matches_manual_count() -> None:
    linear = torch.nn.Linear(4, 2)  # 4*2 weights + 2 bias = 10
    assert count_parameters(linear) == 10


def test_count_parameters_excludes_frozen_params() -> None:
    linear = torch.nn.Linear(4, 2)
    linear.weight.requires_grad_(False)

    assert count_parameters(linear) == 2  # only bias counted
