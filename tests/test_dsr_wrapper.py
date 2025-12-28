"""Tests for StatefulRewardWrapper and DifferentialSharpeRatioAnyTrading."""

import gymnasium as gym
import numpy as np
import pytest

from trading_rl.rewards.dsr_wrapper import (
    DifferentialSharpeRatioAnyTrading,
    StatefulRewardWrapper,
)


class MockTradingEnv(gym.Env):
    """Mock trading environment for testing StatefulRewardWrapper."""

    def __init__(self):
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(5,))
        self._portfolio_values = [10000, 10100, 10050, 10200, 10150]
        self._step_count = 0

    def reset(self, **kwargs):
        self._step_count = 0
        return np.array([0.5] * 5, dtype=np.float32), {}

    def step(self, action):
        self._step_count += 1
        obs = np.array([0.5] * 5, dtype=np.float32)
        reward = 0.0  # Will be replaced by wrapper
        terminated = self._step_count >= len(self._portfolio_values) - 1
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def _get_info(self):
        """Simulate gym_anytrading history interface."""
        # Return history dict with portfolio valuations
        idx = min(self._step_count, len(self._portfolio_values) - 1)
        return {
            ("portfolio_valuation", -1): self._portfolio_values[idx],
            ("portfolio_valuation", -2): self._portfolio_values[max(0, idx - 1)],
        }


class TestDifferentialSharpeRatioAnyTrading:
    """Test DSR reward for gym_anytrading interface."""

    def test_initialization(self):
        """Test DSR initializes with correct default values."""
        dsr = DifferentialSharpeRatioAnyTrading(eta=0.01)
        assert dsr.eta == 0.01
        assert dsr.epsilon == 1e-8
        assert dsr.A_t == 0.0
        assert dsr.B_t == 0.0
        assert dsr._prev_nlv is None

    def test_reset(self):
        """Test reset clears DSR state."""
        dsr = DifferentialSharpeRatioAnyTrading(eta=0.01)

        # Simulate some steps
        history = {("portfolio_valuation", -1): 10100}
        dsr(history)
        history = {("portfolio_valuation", -1): 10050}
        dsr(history)

        # State should be non-zero
        assert dsr.A_t != 0.0
        assert dsr.B_t != 0.0
        assert dsr._prev_nlv is not None

        # Reset should clear state
        dsr.reset()
        assert dsr.A_t == 0.0
        assert dsr.B_t == 0.0
        assert dsr._prev_nlv is None

    def test_first_step_returns_zero(self):
        """Test first step (no previous value) returns 0."""
        dsr = DifferentialSharpeRatioAnyTrading(eta=0.01)
        history = {("portfolio_valuation", -1): 10000}
        reward = dsr(history)
        assert reward == 0.0
        assert dsr._prev_nlv == 10000

    def test_dsr_calculation(self):
        """Test DSR calculates values correctly."""
        dsr = DifferentialSharpeRatioAnyTrading(eta=0.1)

        # First step: store initial value
        history = {("portfolio_valuation", -1): 10000}
        r1 = dsr(history)
        assert r1 == 0.0

        # Second step: compute DSR (still 0 because EMAs are still 0)
        history = {("portfolio_valuation", -1): 10100}
        r2 = dsr(history)
        # DSR is 0 because A_{t-1}=0, B_{t-1}=0, so numerator is 0
        # But EMAs are updated after calculation
        assert r2 == 0.0 or abs(r2) < 1e-10
        assert dsr.A_t != 0.0  # EMAs should be updated
        assert dsr.B_t != 0.0

        # Third step: should use updated EMAs and produce non-zero DSR
        history = {("portfolio_valuation", -1): 10050}
        r3 = dsr(history)
        assert r3 != 0.0  # Now DSR should be non-zero
        assert abs(r3) > 1e-10

    def test_dsr_with_negative_return(self):
        """Test DSR handles negative returns correctly."""
        dsr = DifferentialSharpeRatioAnyTrading(eta=0.1)

        # First step
        history = {("portfolio_valuation", -1): 10000}
        dsr(history)

        # Positive return
        history = {("portfolio_valuation", -1): 10100}
        r_pos = dsr(history)

        # Negative return
        history = {("portfolio_valuation", -1): 9900}
        r_neg = dsr(history)

        # Both should be finite
        assert np.isfinite(r_pos)
        assert np.isfinite(r_neg)

    def test_dsr_handles_invalid_values(self):
        """Test DSR returns 0 for invalid portfolio values."""
        dsr = DifferentialSharpeRatioAnyTrading(eta=0.1)

        # First step
        history = {("portfolio_valuation", -1): 10000}
        dsr(history)

        # Zero or negative value
        history = {("portfolio_valuation", -1): 0}
        reward = dsr(history)
        assert reward == 0.0

        history = {("portfolio_valuation", -1): -100}
        reward = dsr(history)
        assert reward == 0.0


class TestStatefulRewardWrapper:
    """Test StatefulRewardWrapper for automatic reset management."""

    def test_wrapper_initialization(self):
        """Test wrapper initializes correctly."""
        dsr = DifferentialSharpeRatioAnyTrading(eta=0.01)
        env = MockTradingEnv()
        wrapped = StatefulRewardWrapper(env, reward_fn=dsr)

        assert wrapped.reward_fn is dsr
        assert wrapped.env is env

    def test_wrapper_requires_reset_method(self):
        """Test wrapper validates reward_fn has reset() method."""
        env = MockTradingEnv()

        # Function without reset() should raise error
        def simple_reward(history):
            return 0.0

        with pytest.raises(ValueError, match="must have a reset"):
            StatefulRewardWrapper(env, reward_fn=simple_reward)

    def test_wrapper_calls_reset(self):
        """Test wrapper calls reward_fn.reset() on env.reset()."""
        dsr = DifferentialSharpeRatioAnyTrading(eta=0.01)
        env = MockTradingEnv()
        wrapped = StatefulRewardWrapper(env, reward_fn=dsr)

        # Manually set state
        dsr.A_t = 0.5
        dsr.B_t = 0.3
        dsr._prev_nlv = 10000

        # Reset should clear DSR state
        wrapped.reset()
        assert dsr.A_t == 0.0
        assert dsr.B_t == 0.0
        assert dsr._prev_nlv is None

    def test_wrapper_uses_custom_reward(self):
        """Test wrapper replaces reward with custom function."""
        dsr = DifferentialSharpeRatioAnyTrading(eta=0.1)
        env = MockTradingEnv()
        wrapped = StatefulRewardWrapper(env, reward_fn=dsr)

        # Reset
        wrapped.reset()

        # Take steps
        obs, reward1, terminated, truncated, info = wrapped.step(0)
        assert reward1 == 0.0  # First DSR step is always 0

        obs, reward2, terminated, truncated, info = wrapped.step(1)
        # Second step still returns ~0 (EMAs just starting to build)
        assert abs(reward2) < 1e-6
        assert np.isfinite(reward2)

        # Third step should have non-zero DSR
        obs, reward3, terminated, truncated, info = wrapped.step(0)
        assert abs(reward3) > 1e-10
        assert np.isfinite(reward3)

    def test_wrapper_state_persists_across_steps(self):
        """Test DSR state persists within an episode."""
        dsr = DifferentialSharpeRatioAnyTrading(eta=0.1)
        env = MockTradingEnv()
        wrapped = StatefulRewardWrapper(env, reward_fn=dsr)

        wrapped.reset()

        # Take several steps
        rewards = []
        for _ in range(3):
            obs, reward, terminated, truncated, info = wrapped.step(0)
            rewards.append(reward)

        # State should have been updated
        assert dsr.A_t != 0.0
        assert dsr.B_t != 0.0

        # Rewards should differ (different returns in mock env)
        assert len(set(rewards[1:])) > 1  # Skip first (always 0)

    def test_wrapper_resets_between_episodes(self):
        """Test DSR state resets between episodes."""
        dsr = DifferentialSharpeRatioAnyTrading(eta=0.1)
        env = MockTradingEnv()
        wrapped = StatefulRewardWrapper(env, reward_fn=dsr)

        # Episode 1
        wrapped.reset()
        for _ in range(2):
            wrapped.step(0)

        state_after_ep1 = (dsr.A_t, dsr.B_t, dsr._prev_nlv)

        # Episode 2
        wrapped.reset()
        state_after_reset = (dsr.A_t, dsr.B_t, dsr._prev_nlv)

        # State should be reset
        assert state_after_reset == (0.0, 0.0, None)
        assert state_after_ep1 != state_after_reset


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
