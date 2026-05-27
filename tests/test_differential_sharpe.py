"""Unit tests for Differential Sharpe Ratio reward implementation."""

import numpy as np
import pytest


class MockBroker:
    """Mock broker for testing DSR reward."""

    def __init__(self, nlv_sequence):
        """Initialize mock broker with a sequence of NLV values.

        Args:
            nlv_sequence: List of net liquidation values over time
        """
        self.nlv_sequence = nlv_sequence
        self.current_step = 0

    def net_liquidation_value(self):
        """Return current NLV without modifying step counter."""
        return self.nlv_sequence[self.current_step]

    def step(self):
        """Advance to next step."""
        self.current_step = min(self.current_step + 1, len(self.nlv_sequence) - 1)


class MockEnv:
    """Mock environment for testing DSR reward."""

    def __init__(self, nlv_sequence):
        """Initialize mock environment with broker.

        Args:
            nlv_sequence: List of net liquidation values over time
        """
        self.broker = MockBroker(nlv_sequence)


class TestDifferentialSharpeRatio:
    """Test suite for DifferentialSharpeRatio reward."""

    def test_initialization(self):
        """Test DSR initialization with default parameters."""
        from trading_rl.rewards import DifferentialSharpeRatio

        dsr = DifferentialSharpeRatio()

        assert dsr.eta == 0.01
        assert dsr.epsilon == 1e-20
        assert dsr.A_t == 0.0
        assert dsr.B_t == 0.0
        assert dsr._prev_nlv is None

    def test_initialization_custom_params(self):
        """Test DSR initialization with custom parameters."""
        from trading_rl.rewards import DifferentialSharpeRatio

        dsr = DifferentialSharpeRatio(eta=0.05, epsilon=1e-6)

        assert dsr.eta == 0.05
        assert dsr.epsilon == 1e-6

    def test_initialization_invalid_eta(self):
        """Test DSR initialization with invalid eta."""
        from trading_rl.rewards import DifferentialSharpeRatio

        with pytest.raises(ValueError, match="eta must be in"):
            DifferentialSharpeRatio(eta=0.0)

        with pytest.raises(ValueError, match="eta must be in"):
            DifferentialSharpeRatio(eta=-0.1)

        with pytest.raises(ValueError, match="eta must be in"):
            DifferentialSharpeRatio(eta=1.5)

    def test_initialization_invalid_epsilon(self):
        """Test DSR initialization with invalid epsilon."""
        from trading_rl.rewards import DifferentialSharpeRatio

        with pytest.raises(ValueError, match="epsilon must be positive"):
            DifferentialSharpeRatio(epsilon=0.0)

        with pytest.raises(ValueError, match="epsilon must be positive"):
            DifferentialSharpeRatio(epsilon=-1e-8)

    def test_first_call_returns_zero(self):
        """Test that first calculate() call returns 0 (no previous NLV)."""
        from trading_rl.rewards import DifferentialSharpeRatio

        dsr = DifferentialSharpeRatio()
        env = MockEnv([10000, 10100])

        reward = dsr.calculate(env)

        assert reward == 0.0
        assert dsr._prev_nlv == 10000

    def test_positive_return(self):
        """Test DSR calculation with positive returns."""
        from trading_rl.rewards import DifferentialSharpeRatio

        dsr = DifferentialSharpeRatio(eta=0.1)  # Higher eta for faster update
        # Sequence: 10000 -> 11000 (10% gain)
        env = MockEnv([10000, 11000])

        # First call initializes
        reward1 = dsr.calculate(env)
        assert reward1 == 0.0

        # Advance to second value
        env.broker.step()

        # Second call calculates DSR
        reward2 = dsr.calculate(env)

        # Should have positive DSR for positive return
        assert isinstance(reward2, float)
        assert dsr.A_t > 0  # Mean should be positive for gain
        assert dsr.B_t > 0  # Second moment should be positive (always R^2)

    def test_negative_return(self):
        """Test DSR calculation with negative returns."""
        from trading_rl.rewards import DifferentialSharpeRatio

        dsr = DifferentialSharpeRatio(eta=0.1)
        # Sequence: 10000 -> 9000 (-10% loss)
        env = MockEnv([10000, 9000])

        # First call initializes
        reward1 = dsr.calculate(env)
        assert reward1 == 0.0

        # Advance to second value
        env.broker.step()

        # Second call calculates DSR
        reward2 = dsr.calculate(env)

        assert isinstance(reward2, float)
        assert dsr.A_t < 0  # Mean should be negative for loss
        assert dsr.B_t > 0  # Second moment should be positive (always R^2)

    def test_zero_return(self):
        """Test DSR calculation with zero return."""
        from trading_rl.rewards import DifferentialSharpeRatio

        dsr = DifferentialSharpeRatio(eta=0.1)
        # Sequence: 10000 -> 10000 (0% change)
        env = MockEnv([10000, 10000])

        # First call initializes
        reward1 = dsr.calculate(env)
        assert reward1 == 0.0

        # Advance to second value
        env.broker.step()

        # Second call calculates DSR
        reward2 = dsr.calculate(env)

        assert isinstance(reward2, float)
        # With zero return, both A_t (mean) and B_t (second moment) should remain near 0
        assert abs(dsr.A_t) < 1e-6  # Mean ≈ 0
        assert abs(dsr.B_t) < 1e-6  # Second moment ≈ 0

    def test_ema_updates(self):
        """Test that EMAs update correctly over multiple steps."""
        from trading_rl.rewards import DifferentialSharpeRatio

        dsr = DifferentialSharpeRatio(eta=0.1)
        # Sequence with varying returns
        nlv_sequence = [10000, 10100, 10150, 10200]
        env = MockEnv(nlv_sequence)

        rewards = []
        for i in range(len(nlv_sequence)):
            reward = dsr.calculate(env)
            rewards.append(reward)
            if i < len(nlv_sequence) - 1:
                env.broker.step()

        # First reward should be 0
        assert rewards[0] == 0.0

        # EMAs should have been updated
        assert dsr.A_t > 0  # Mean should be positive (gains)
        assert dsr.B_t > 0  # Second moment should be positive (always)

        # All rewards should be finite
        assert all(np.isfinite(r) for r in rewards)

    def test_reset(self):
        """Test that reset() clears DSR state."""
        from trading_rl.rewards import DifferentialSharpeRatio

        dsr = DifferentialSharpeRatio()
        env = MockEnv([10000, 11000])

        # Calculate some rewards to update state
        dsr.calculate(env)
        env.broker.step()
        dsr.calculate(env)

        # State should be non-zero
        assert dsr.A_t != 0.0 or dsr.B_t != 0.0
        assert dsr._prev_nlv is not None

        # Reset
        dsr.reset()

        # State should be cleared
        assert dsr.A_t == 0.0
        assert dsr.B_t == 0.0
        assert dsr._prev_nlv is None

    def test_invalid_portfolio_value(self):
        """Test DSR handling of invalid portfolio values."""
        from trading_rl.rewards import DifferentialSharpeRatio

        dsr = DifferentialSharpeRatio()

        # Test with zero NLV
        env = MockEnv([10000, 0])
        dsr.calculate(env)
        env.broker.step()
        reward = dsr.calculate(env)
        assert reward == 0.0

        # Test with negative NLV
        dsr.reset()
        env = MockEnv([10000, -1000])
        dsr.calculate(env)
        env.broker.step()
        reward = dsr.calculate(env)
        assert reward == 0.0

    def test_repr(self):
        """Test string representation of DSR."""
        from trading_rl.rewards import DifferentialSharpeRatio

        dsr = DifferentialSharpeRatio(eta=0.05)
        repr_str = repr(dsr)

        assert "DifferentialSharpeRatio" in repr_str
        assert "eta=0.05" in repr_str
        assert "A_t=" in repr_str
        assert "B_t=" in repr_str

    def test_numerical_stability(self):
        """Test DSR numerical stability with extreme values."""
        from trading_rl.rewards import DifferentialSharpeRatio

        dsr = DifferentialSharpeRatio(eta=0.01, epsilon=1e-8)

        # Very small changes
        env = MockEnv([10000.0, 10000.0001])
        dsr.calculate(env)
        env.broker.step()
        reward = dsr.calculate(env)
        assert np.isfinite(reward)

        # Very large changes
        dsr.reset()
        env = MockEnv([10000.0, 100000.0])
        dsr.calculate(env)
        env.broker.step()
        reward = dsr.calculate(env)
        assert np.isfinite(reward)

    def test_consistent_eta_behavior(self):
        """Test that different eta values produce different convergence speeds."""
        from trading_rl.rewards import DifferentialSharpeRatio

        # Same return sequence
        nlv_sequence = [10000 * (1.01 ** i) for i in range(10)]

        # Fast learning (high eta)
        dsr_fast = DifferentialSharpeRatio(eta=0.5)
        env_fast = MockEnv(nlv_sequence.copy())
        for i in range(len(nlv_sequence)):
            dsr_fast.calculate(env_fast)
            if i < len(nlv_sequence) - 1:
                env_fast.broker.step()

        # Slow learning (low eta)
        dsr_slow = DifferentialSharpeRatio(eta=0.001)
        env_slow = MockEnv(nlv_sequence.copy())
        for i in range(len(nlv_sequence)):
            dsr_slow.calculate(env_slow)
            if i < len(nlv_sequence) - 1:
                env_slow.broker.step()

        # Fast learning should have larger EMA values (closer to recent data)
        assert abs(dsr_fast.B_t) > abs(dsr_slow.B_t)
        assert dsr_fast.A_t > dsr_slow.A_t


def _run_dsr_any_episode(dsr, nlv_sequence):
    """Drive one episode through DifferentialSharpeRatioAnyTrading."""
    for nlv in nlv_sequence:
        dsr({"portfolio_valuation": [nlv]})


class TestDifferentialSharpeRatioAnyTrading:
    """Test suite for DifferentialSharpeRatioAnyTrading reward."""

    def test_reset_persist_moments_true_keeps_ema(self):
        """reset(persist_moments=True) clears _prev_nlv but retains A_t and B_t."""
        from trading_rl.rewards.dsr_wrapper import DifferentialSharpeRatioAnyTrading

        dsr = DifferentialSharpeRatioAnyTrading(eta=0.1, clip_reward=None)
        _run_dsr_any_episode(dsr, [10000, 10100, 10200, 10300])

        a_before = dsr.A_t
        b_before = dsr.B_t
        assert a_before != 0.0, "A_t must be non-zero after episode steps"
        assert b_before != 0.0, "B_t must be non-zero after episode steps"

        dsr.reset(persist_moments=True)

        assert dsr._prev_nlv is None
        assert dsr.A_t == a_before
        assert dsr.B_t == b_before

    def test_reset_persist_moments_false_clears_all(self):
        """reset(persist_moments=False) clears A_t, B_t, and _prev_nlv."""
        from trading_rl.rewards.dsr_wrapper import DifferentialSharpeRatioAnyTrading

        dsr = DifferentialSharpeRatioAnyTrading(eta=0.1, clip_reward=None)
        _run_dsr_any_episode(dsr, [10000, 10100, 10200, 10300])

        assert dsr.A_t != 0.0
        assert dsr.B_t != 0.0

        dsr.reset(persist_moments=False)

        assert dsr.A_t == 0.0
        assert dsr.B_t == 0.0
        assert dsr._prev_nlv is None

    def test_reset_none_uses_constructor_default_false(self):
        """reset() with no argument respects persist_moments=False set at construction."""
        from trading_rl.rewards.dsr_wrapper import DifferentialSharpeRatioAnyTrading

        dsr = DifferentialSharpeRatioAnyTrading(eta=0.1, clip_reward=None, persist_moments=False)
        _run_dsr_any_episode(dsr, [10000, 10100, 10200])

        dsr.reset()

        assert dsr.A_t == 0.0
        assert dsr.B_t == 0.0
        assert dsr._prev_nlv is None

    def test_reset_none_uses_constructor_default_true(self):
        """reset() with no argument respects persist_moments=True set at construction."""
        from trading_rl.rewards.dsr_wrapper import DifferentialSharpeRatioAnyTrading

        dsr = DifferentialSharpeRatioAnyTrading(eta=0.1, clip_reward=None, persist_moments=True)
        _run_dsr_any_episode(dsr, [10000, 10100, 10200])

        a_before = dsr.A_t
        b_before = dsr.B_t

        dsr.reset()

        assert dsr._prev_nlv is None
        assert dsr.A_t == a_before
        assert dsr.B_t == b_before

    def test_persist_moments_ema_continuity_across_episodes(self):
        """With persist_moments=True, EMA warms up across episodes instead of restarting."""
        from trading_rl.rewards.dsr_wrapper import DifferentialSharpeRatioAnyTrading

        dsr_persist = DifferentialSharpeRatioAnyTrading(eta=0.1, clip_reward=None, persist_moments=True)
        dsr_reset = DifferentialSharpeRatioAnyTrading(eta=0.1, clip_reward=None, persist_moments=False)

        nlv_ep1 = [10000, 10100, 10200]
        nlv_ep2 = [10200, 10250, 10300]

        _run_dsr_any_episode(dsr_persist, nlv_ep1)
        dsr_persist.reset()
        _run_dsr_any_episode(dsr_persist, nlv_ep2)

        _run_dsr_any_episode(dsr_reset, nlv_ep1)
        dsr_reset.reset()
        _run_dsr_any_episode(dsr_reset, nlv_ep2)

        # With persist, EMA carries prior history so A_t should differ from cold-start
        assert dsr_persist.A_t != dsr_reset.A_t or dsr_persist.B_t != dsr_reset.B_t
