"""Differential Sharpe Ratio reward implementation.

This module implements the online Differential Sharpe Ratio (DSR) as described in:
Moody & Saffell (2001) - "Reinforcement Learning for Trading"
http://papers.nips.cc/paper/1551-reinforcement-learning-for-trading.pdf

The DSR provides a risk-adjusted reward that adapts online to changing market conditions,
making it more suitable for RL training than traditional Sharpe ratio calculations.
"""

import numpy as np
from tradingenv.rewards import AbstractReward

from logger import get_logger

logger = get_logger(__name__)


class DifferentialSharpeRatio(AbstractReward):
    """Online Differential Sharpe Ratio reward calculator.

    The DSR uses exponential moving averages to compute an online estimate of the
    Sharpe ratio's derivative, providing a stationary reward signal suitable for RL.

    Formula:
        DSR_t = (A_t * ΔR_t - B_t * R_t) / (A_t^1.5 + ε)

        where:
        - A_t = (1 - η) * A_{t-1} + η * R_t^2  (EMA of squared returns)
        - B_t = (1 - η) * B_{t-1} + η * R_t    (EMA of returns)
        - ΔR_t = R_t - B_{t-1}                  (return minus previous mean)
        - R_t = log(V_t / V_{t-1})              (log return of portfolio value)
        - η = learning rate (typically 0.001 to 0.1)
        - ε = small constant for numerical stability

    Args:
        eta: Learning rate for exponential moving averages (default: 0.01)
            - Higher values (0.1): Fast adaptation, higher variance
            - Lower values (0.001): Slow adaptation, smoother signal
        epsilon: Small constant for numerical stability (default: 1e-8)

    Attributes:
        A_t: Current EMA of squared returns (variance estimate)
        B_t: Current EMA of returns (mean estimate)
    """

    def __init__(self, eta: float = 0.01, epsilon: float = 1e-8):
        """Initialize DSR reward calculator.

        Args:
            eta: Learning rate for EMAs (0.001 to 0.1, default: 0.01)
            epsilon: Stability constant (default: 1e-8)
        """
        if not 0 < eta <= 1:
            raise ValueError(f"eta must be in (0, 1], got {eta}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

        self.eta = eta
        self.epsilon = epsilon

        # Initialize EMAs
        self.A_t = 0.0  # EMA of R^2
        self.B_t = 0.0  # EMA of R

        # Track previous portfolio value for return calculation
        self._prev_nlv = None

        logger.debug(f"Initialized DSR with eta={eta}, epsilon={epsilon}")

    def calculate(self, env) -> float:
        """Calculate DSR reward for the current step.

        Args:
            env: TradingEnv instance with broker.net_liquidation_value() method

        Returns:
            DSR reward value (float)
        """
        # Get current net liquidation value
        nlv_now = env.broker.net_liquidation_value()

        # On first call, initialize previous NLV and return 0
        if self._prev_nlv is None:
            self._prev_nlv = nlv_now
            return 0.0

        # Calculate log return
        if self._prev_nlv <= 0 or nlv_now <= 0:
            logger.warning(
                f"Invalid portfolio value: prev={self._prev_nlv}, now={nlv_now}. "
                "Returning 0 reward."
            )
            self._prev_nlv = nlv_now
            return 0.0

        R_t = float(np.log(nlv_now / self._prev_nlv))

        # Calculate ΔR_t = R_t - B_{t-1} (return minus previous mean estimate)
        delta_R = R_t - self.B_t

        # Update EMAs
        self.A_t = (1 - self.eta) * self.A_t + self.eta * (R_t ** 2)
        self.B_t = (1 - self.eta) * self.B_t + self.eta * R_t

        # Calculate DSR
        # Note: A_t approximates variance, so A_t^1.5 scales with std^3
        denominator = self.A_t ** 1.5 + self.epsilon
        dsr = (self.A_t * delta_R - self.B_t * R_t) / denominator

        # Update previous NLV for next step
        self._prev_nlv = nlv_now

        return float(dsr)

    def reset(self) -> None:
        """Reset DSR state for new episode.

        This should be called at the start of each new episode to clear
        the EMA state and previous portfolio value.
        """
        self.A_t = 0.0
        self.B_t = 0.0
        self._prev_nlv = None
        logger.debug("DSR state reset")

    def __repr__(self) -> str:
        """String representation of DSR state."""
        return (
            f"DifferentialSharpeRatio(eta={self.eta}, "
            f"A_t={self.A_t:.6f}, B_t={self.B_t:.6f})"
        )
