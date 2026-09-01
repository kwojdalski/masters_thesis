"""Guards for the shared off-policy optimization skeleton (#495) and the
TrainerCallback null-object (#497).

#495: TD3/DDPG/SAC used to hand-roll the same sample -> guards -> critic ->
delayed-actor -> callback -> periodic-log loop. It now lives in
BaseTrainer._run_offpolicy_optimization_step; each trainer supplies hooks.

#497: the five-way-duplicated `hasattr(self.callback, "log_training_step")`
guard is gone -- BaseTrainer.callback is always a TrainerCallback (a no-op
_NullTrainerCallback stands in for None).
"""

from __future__ import annotations

import inspect

import pytest

from trading_rl.trainers.base import (
    _NULL_CALLBACK,
    BaseTrainer,
    TrainerCallback,
    _NullTrainerCallback,
)
from trading_rl.trainers.ddpg import DDPGTrainer
from trading_rl.trainers.sac import SACTrainer
from trading_rl.trainers.td3 import TD3Trainer

_OFFPOLICY = (TD3Trainer, DDPGTrainer, SACTrainer)


# ---------------------------------------------------------------------------
# #495 -- one skeleton, three hook sets
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("trainer_cls", _OFFPOLICY, ids=lambda c: c.__name__)
def test_optimization_step_delegates_to_the_shared_skeleton(trainer_cls) -> None:
    src = inspect.getsource(trainer_cls._optimization_step)
    assert "_run_offpolicy_optimization_step" in src
    # The old inlined skeleton markers must be gone from the subclass.
    assert "isnan" not in src and "optim_steps_per_batch" not in src


@pytest.mark.parametrize("trainer_cls", _OFFPOLICY, ids=lambda c: c.__name__)
def test_each_offpolicy_trainer_implements_the_three_hooks(trainer_cls) -> None:
    for hook in ("_update_critics", "_update_actor_and_targets"):
        assert trainer_cls.__dict__.get(hook) is not None, (trainer_cls.__name__, hook)
    # _should_update_actor: TD3 overrides (policy_delay), DDPG/SAC inherit the
    # "every step" default.
    assert callable(trainer_cls._should_update_actor)


def test_td3_still_delays_the_actor_update_but_ddpg_sac_do_not() -> None:
    assert "_should_update_actor" in TD3Trainer.__dict__
    assert "_should_update_actor" not in DDPGTrainer.__dict__
    assert "_should_update_actor" not in SACTrainer.__dict__
    # base default updates every step
    assert BaseTrainer._should_update_actor(object(), 5) is True


def test_normalize_batch_shapes_is_a_noop_on_base_and_overridden_only_by_td3() -> None:
    assert BaseTrainer._normalize_batch_shapes(object(), {"x": 1}) is None
    assert "_normalize_batch_shapes" in TD3Trainer.__dict__
    assert "_normalize_batch_shapes" not in DDPGTrainer.__dict__
    assert "_normalize_batch_shapes" not in SACTrainer.__dict__


# ---------------------------------------------------------------------------
# #497 -- TrainerCallback null-object
# ---------------------------------------------------------------------------


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def log_training_step(self, step, actor_loss, value_loss, *, extra_metrics=None):
        self.calls.append((step, actor_loss, value_loss, extra_metrics))


class _StubTrainer(BaseTrainer):
    """Bypass BaseTrainer.__init__ to test the callback property in isolation."""

    def __init__(self) -> None:
        self.callback = None

    @staticmethod
    def build_models(n_obs, n_act, config, env):  # pragma: no cover
        raise NotImplementedError

    def _optimization_step(self, *a):  # pragma: no cover
        raise NotImplementedError

    @property
    def _algo_label(self) -> str:  # pragma: no cover
        return "stub"

    def _get_checkpoint_network_state(self):  # pragma: no cover
        raise NotImplementedError

    def _load_checkpoint_network_state(self, checkpoint):  # pragma: no cover
        raise NotImplementedError


def test_none_callback_is_coerced_to_the_null_object() -> None:
    t = _StubTrainer()
    assert t.callback is _NULL_CALLBACK
    assert isinstance(t.callback, TrainerCallback)
    # unconditional call is safe
    assert t.callback.log_training_step(1, 0.1, 0.2, extra_metrics={"a": 1}) is None


def test_real_callback_is_stored_as_is_and_receives_calls() -> None:
    t = _StubTrainer()
    rec = _Recorder()
    t.callback = rec
    assert t.callback is rec
    t.callback.log_training_step(7, 1.0, 2.0)
    assert rec.calls == [(7, 1.0, 2.0, None)]


def test_reassigning_none_after_a_real_callback_falls_back_to_null() -> None:
    t = _StubTrainer()
    t.callback = _Recorder()
    t.callback = None
    assert t.callback is _NULL_CALLBACK


def test_null_callback_has_no_log_episode_stats_so_the_loop_still_skips_it() -> None:
    # training_loop.py gates episode-stat accumulation on
    # hasattr(callback, "log_episode_stats"); the null object must not have it.
    assert not hasattr(_NullTrainerCallback(), "log_episode_stats")


def test_no_trainer_still_hand_rolls_the_hasattr_callback_guard() -> None:
    for mod in ("td3", "ddpg", "sac", "ppo", "recurrent_ppo"):
        src = inspect.getsource(__import__(f"trading_rl.trainers.{mod}", fromlist=[""]))
        assert 'hasattr(self.callback, "log_training_step")' not in src, mod
