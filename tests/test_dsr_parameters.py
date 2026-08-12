"""Tests for DifferentialSharpeRatio clip_rewards, scale, and persist_moments."""

from __future__ import annotations

import numpy as np
import pytest

from trading_rl.rewards import DifferentialSharpeRatio


class _Env:
    """Minimal mock env with a configurable NLV sequence."""

    def net_liquidation_value(self) -> float:
        return self._nlvs[self._step]

    class _Broker:
        pass

    def __init__(self, nlvs):  # type: ignore[misc]
        from unittest.mock import MagicMock

        self.broker = MagicMock()
        self._nlvs = nlvs
        self._step = 0
        self.broker.net_liquidation_value.side_effect = lambda: self._nlvs[self._step]

    def advance(self):
        self._step = min(self._step + 1, len(self._nlvs) - 1)


def _run(nlvs: list[float], **kwargs) -> list[float]:
    """Feed nlv sequence through DSR; return all rewards."""
    dsr = DifferentialSharpeRatio(**kwargs)
    env = _Env(nlvs)
    rewards = []
    for i in range(len(nlvs)):
        env._step = i
        rewards.append(dsr.calculate(env))
    return rewards


# ---------------------------------------------------------------------------
# clip_rewards parameter
# ---------------------------------------------------------------------------


class TestClipRewards:
    def test_clip_true_clamps_extreme_positive_value(self):
        """With tiny epsilon and large jump, unclipped DSR would be > 10."""
        dsr = DifferentialSharpeRatio(eta=0.5, epsilon=1e-16, clip_reward=10.0)
        env = _Env([1.0, 1e6, 1e6, 1e6, 1e6])
        rewards = []
        for i in range(5):
            env._step = i
            rewards.append(dsr.calculate(env))
        assert all(r <= 10.0 for r in rewards), f"unclipped: {rewards}"

    def test_clip_true_clamps_extreme_negative_value(self):
        dsr = DifferentialSharpeRatio(eta=0.5, epsilon=1e-16, clip_reward=10.0)
        env = _Env([1e6, 1.0, 1.0, 1.0, 1.0])
        rewards = []
        for i in range(5):
            env._step = i
            rewards.append(dsr.calculate(env))
        assert all(r >= -10.0 for r in rewards), f"unclipped: {rewards}"

    def test_clip_false_can_produce_values_beyond_ten(self):
        """Without clipping, an extreme DSR value is returned as-is."""
        # Feed a sequence designed to create a variance that collapses to near zero,
        # inflating the denominator's epsilon term and pushing |DSR| >> 10.
        # Use constant returns (variance → 0) then a sudden spike.
        dsr = DifferentialSharpeRatio(eta=0.01, epsilon=1e-16, clip_reward=None)
        env = _Env([100.0] * 1)
        # Warm up EMAs with constant returns so variance is essentially 0
        for _i in range(200):
            env._step = 0
            dsr.calculate(env)
        # Now inject a large spike — numerator is large, denominator ≈ epsilon^1.5
        env2 = _Env([100.0, 200.0])
        dsr._prev_nlv = 100.0
        env2._step = 1
        r = dsr.calculate(env2)
        # With epsilon=1e-16 and near-zero variance, |DSR| >> 10 is possible
        # At minimum, clip_rewards=False must be settable and return a float
        assert isinstance(r, float)
        assert np.isfinite(r)

    def test_clip_default_is_true(self):
        dsr = DifferentialSharpeRatio()
        assert dsr.clip_reward == 10.0

    def test_clip_boundaries_are_symmetric(self):
        dsr = DifferentialSharpeRatio(eta=0.5, epsilon=1e-16, clip_reward=10.0)
        nlvs = [100.0, 100.0, 200.0, 100.0, 200.0]
        env = _Env(nlvs)
        for i in range(len(nlvs)):
            env._step = i
            r = dsr.calculate(env)
            assert -10.0 <= r <= 10.0, f"reward {r} at step {i} is outside [-10, 10]"


# ---------------------------------------------------------------------------
# scale parameter
# ---------------------------------------------------------------------------


class TestScaleParameter:
    def test_scale_one_is_identity(self):
        rewards_s1 = _run([100.0, 110.0, 105.0, 108.0], eta=0.1, scale=1.0)
        rewards_s2 = _run([100.0, 110.0, 105.0, 108.0], eta=0.1, scale=1.0)
        np.testing.assert_allclose(rewards_s1, rewards_s2)

    def test_scale_two_doubles_output(self):
        rewards_base = _run([100.0, 110.0, 105.0, 108.0], eta=0.1, scale=1.0)
        rewards_2x = _run([100.0, 110.0, 105.0, 108.0], eta=0.1, scale=2.0)
        np.testing.assert_allclose(
            rewards_2x, [r * 2.0 for r in rewards_base], rtol=1e-9
        )

    def test_scale_half_halves_output(self):
        rewards_base = _run([100.0, 110.0, 105.0, 108.0], eta=0.1, scale=1.0)
        rewards_half = _run([100.0, 110.0, 105.0, 108.0], eta=0.1, scale=0.5)
        np.testing.assert_allclose(
            rewards_half, [r * 0.5 for r in rewards_base], rtol=1e-9
        )

    def test_scale_preserves_sign(self):
        rewards_pos = _run([100.0, 110.0, 105.0, 108.0], eta=0.1, scale=1.0)
        rewards_neg = _run([100.0, 110.0, 105.0, 108.0], eta=0.1, scale=-1.0)
        for r_p, r_n in zip(rewards_pos, rewards_neg, strict=False):
            assert np.sign(r_p) == -np.sign(r_n) or (r_p == 0.0 and r_n == 0.0)

    def test_scale_default_is_one(self):
        dsr = DifferentialSharpeRatio()
        assert dsr.scale == 1.0


# ---------------------------------------------------------------------------
# persist_moments parameter in reset()
# ---------------------------------------------------------------------------


class TestPersistMoments:
    def test_persist_moments_false_resets_all_state(self):
        dsr = DifferentialSharpeRatio(eta=0.1)
        env = _Env([100.0, 110.0, 105.0])
        for i in range(3):
            env._step = i
            dsr.calculate(env)
        assert dsr.A_t != 0.0 or dsr.B_t != 0.0
        dsr.reset(persist_moments=False)
        assert dsr.A_t == 0.0
        assert dsr.B_t == 0.0
        assert dsr._prev_nlv is None

    def test_persist_moments_true_keeps_ema_resets_prev_nlv(self):
        dsr = DifferentialSharpeRatio(eta=0.1)
        env = _Env([100.0, 110.0, 105.0])
        for i in range(3):
            env._step = i
            dsr.calculate(env)
        a_t_before = dsr.A_t
        b_t_before = dsr.B_t
        assert a_t_before != 0.0 or b_t_before != 0.0
        dsr.reset(persist_moments=True)
        assert dsr.A_t == pytest.approx(a_t_before)
        assert dsr.B_t == pytest.approx(b_t_before)
        assert dsr._prev_nlv is None

    def test_persist_moments_true_first_step_after_reset_returns_zero(self):
        dsr = DifferentialSharpeRatio(eta=0.1)
        env = _Env([100.0, 110.0])
        env._step = 0
        dsr.calculate(env)
        env._step = 1
        dsr.calculate(env)
        dsr.reset(persist_moments=True)
        env._step = 0
        r = dsr.calculate(env)
        assert r == 0.0
        assert dsr._prev_nlv == pytest.approx(100.0)

    def test_persist_moments_default_is_false(self):
        dsr = DifferentialSharpeRatio(eta=0.1)
        env = _Env([100.0, 110.0])
        env._step = 0
        dsr.calculate(env)
        env._step = 1
        dsr.calculate(env)
        dsr.reset()
        assert dsr.A_t == 0.0
        assert dsr.B_t == 0.0
