"""Gymnasium wrapper for stateful reward functions with automatic reset."""

import gymnasium as gym
import numpy as np


class StatefulRewardWrapper(gym.Wrapper):
    """Wrapper that manages stateful reward functions and calls reset() on env reset.

    This allows using stateful reward functions (like DSR) with gym_anytrading
    or gym_trading_env, which normally only support simple functions.

    The wrapper:
    1. Intercepts env.reset() calls and resets the reward function
    2. Intercepts env.step() calls and uses the custom reward function
    3. Works with any Gymnasium-compatible environment

    Example:
        >>> dsr_reward = DifferentialSharpeRatioAnyTrading(eta=0.01)
        >>> base_env = gym.make("forex-v0", df=df)
        >>> env = StatefulRewardWrapper(base_env, reward_fn=dsr_reward)
        >>> # Now dsr_reward.reset() is called automatically on env.reset()
    """

    def __init__(self, env: gym.Env, reward_fn):
        """Initialize wrapper with stateful reward function.

        Args:
            env: Base Gymnasium environment
            reward_fn: Callable reward function that:
                - Has a reset() method for state initialization
                - Can be called with history dict to compute reward
                - Example: DifferentialSharpeRatioAnyTrading instance
        """
        super().__init__(env)
        self.reward_fn = reward_fn

        # Validate that reward_fn has reset() method
        if not hasattr(reward_fn, 'reset'):
            raise ValueError(
                f"reward_fn must have a reset() method. "
                f"Got {type(reward_fn).__name__} without reset()"
            )

    def reset(self, **kwargs):
        """Reset environment and reward function state."""
        obs, info = self.env.reset(**kwargs)

        # Reset reward function state
        self.reward_fn.reset()

        return obs, info

    def step(self, action):
        """Take step and use custom reward function."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Replace reward with custom reward function
        # Most gym_anytrading/gym_trading_env expose history in info
        if hasattr(self.env.unwrapped, '_get_info'):
            # gym_anytrading pattern
            history = self.env.unwrapped._get_info()
            custom_reward = self.reward_fn(history)
        elif 'history' in info:
            # Direct history in info
            custom_reward = self.reward_fn(info['history'])
        else:
            # Fallback: try to access history directly
            try:
                history = self.env.unwrapped.history
                custom_reward = self.reward_fn(history)
            except AttributeError:
                # Cannot access history - use original reward
                custom_reward = reward

        return obs, custom_reward, terminated, truncated, info


class DifferentialSharpeRatioAnyTrading:
    """DSR reward function compatible with gym_anytrading and gym_trading_env.

    This is a callable class that:
    - Maintains state (A_t, B_t, prev_nlv) across steps
    - Has reset() method for automatic reset via StatefulRewardWrapper
    - Computes DSR using the Moody & Saffell (2001) formula

    Usage with StatefulRewardWrapper:
        >>> dsr = DifferentialSharpeRatioAnyTrading(eta=0.01)
        >>> env = gym.make("forex-v0", df=df)
        >>> env = StatefulRewardWrapper(env, reward_fn=dsr)
        >>> # Now dsr.reset() is called automatically on env.reset()
    """

    def __init__(self, eta: float = 0.01, epsilon: float = 1e-8):
        """Initialize DSR reward function.

        Args:
            eta: EMA learning rate (controls adaptation speed)
            epsilon: Small constant for numerical stability
        """
        self.eta = eta
        self.epsilon = epsilon

        # Internal state - reset on each episode
        self.A_t = 0.0  # EMA of returns (mean)
        self.B_t = 0.0  # EMA of squared returns (second moment)
        self._prev_nlv = None

    def reset(self):
        """Reset state for new episode."""
        self.A_t = 0.0
        self.B_t = 0.0
        self._prev_nlv = None

    def __call__(self, history: dict) -> float:
        """Calculate DSR reward from trading history.

        Args:
            history: Dictionary with portfolio history
                Must contain 'portfolio_valuation' with current and previous NLV

        Returns:
            Differential Sharpe Ratio for current step
        """
        # Extract current portfolio value
        # gym_anytrading uses 'portfolio_valuation' key with tuple indexing
        try:
            nlv_now = history["portfolio_valuation", -1]
        except (KeyError, TypeError):
            # Fallback for different history formats
            if "portfolio_valuation" in history:
                nlv_now = history["portfolio_valuation"][-1]
            else:
                # Cannot compute DSR without portfolio value
                return 0.0

        # First step: just store initial value
        if self._prev_nlv is None:
            self._prev_nlv = nlv_now
            return 0.0

        # Calculate log return
        if self._prev_nlv > 0 and nlv_now > 0:
            R_t = float(np.log(nlv_now / self._prev_nlv))
        else:
            # Invalid values - return 0
            self._prev_nlv = nlv_now
            return 0.0

        # Calculate DSR using OLD EMA values (t-1)
        # This matches Moody & Saffell (2001) formula
        delta_A = R_t - self.A_t  # ΔA_t = R_t - A_{t-1}
        delta_B = R_t ** 2 - self.B_t  # ΔB_t = R_t^2 - B_{t-1}

        # D_t = (B_{t-1} * ΔA_t - A_{t-1} * ΔB_t / 2) / (B_{t-1} - A_{t-1}^2)^(3/2)
        variance = self.B_t - self.A_t ** 2  # Var = B_{t-1} - A_{t-1}^2
        denominator = variance ** 1.5 + self.epsilon
        dsr = (self.B_t * delta_A - self.A_t * delta_B / 2) / denominator

        # NOW update EMAs for next step
        self.A_t = (1 - self.eta) * self.A_t + self.eta * R_t
        self.B_t = (1 - self.eta) * self.B_t + self.eta * (R_t ** 2)

        # Update prev_nlv for next step
        self._prev_nlv = nlv_now

        return float(dsr)
