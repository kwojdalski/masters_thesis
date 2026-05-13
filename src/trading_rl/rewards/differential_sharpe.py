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

    Formula (from Moody & Saffell 2001):
        D_t = (B_{t-1} * ΔA_t - A_{t-1} * ΔB_t / 2) / ((B_{t-1} - A_{t-1}^2)^1.5 + ε)

        where:
        - A_t = (1 - η) * A_{t-1} + η * R_t    (EMA of returns - first moment)
        - B_t = (1 - η) * B_{t-1} + η * R_t^2  (EMA of squared returns - second moment)
        - ΔA_t = R_t - A_{t-1}                  (change in mean estimate)
        - ΔB_t = R_t^2 - B_{t-1}                (change in second moment)
        - Var = B_{t-1} - A_{t-1}^2             (variance estimate)
        - R_t = log(V_t / V_{t-1})              (log return of portfolio value)
        - η = learning rate (typically 0.001 to 0.1)
        - ε = small constant for numerical stability

    
    Workflow Diagram:
    ──────────────────────────────────────────────────────────────────────────────────────────

    Step 1: Initialization
    ┌─────────────────────────────────────────────────────────────────────────────────────┐
    │  • Set EMA learning rate (η) and stability constant (ε)                   │
    │  • Initialize first moment EMA: A_t = 0.0 (mean estimate)               │
    │  • Initialize second moment EMA: B_t = 0.0 (squared returns)             │
    │  • Set previous net liquidation value: _prev_nlv = None                  │
    └─────────────────────────────────────────────────────────────────────────────────────┘

    Step 2: First Call Handling (when _prev_nlv is None)
    ┌─────────────────────────────────────────────────────────────────────────────────────┐
    │  • Get current net liquidation value: nlv_now                            │
    │  • Store as previous: _prev_nlv = nlv_now                                 │
    │  • Return reward: 0.0 (no calculation yet)                                │
    └─────────────────────────────────────────────────────────────────────────────────────┘

    Step 3: Normal Call Handling
    ┌─────────────────────────────────────────────────────────────────────────────────────┐
    │  INPUT: nlv_now (current portfolio value)                                  │
    │  INPUT: _prev_nlv (previous portfolio value)                                  │
    │  INPUT: A_t, B_t (previous EMA estimates at t-1)                           │
    │                                                                              │
    │  3.1 Calculate Log Return:                                                   │
    │      R_t = log(nlv_now / _prev_nlv)                                         │
    │                                                                              │
    │  3.2 Calculate Deltas (using OLD EMA values at t-1):                        │
    │      ΔA_t = R_t - A_t                    (change in mean estimate)            │
    │      ΔB_t = R_t² - B_t                   (change in second moment)         │
    │                                                                              │
    │  3.3 Calculate Variance:                                                     │
    │      Var = B_t - A_t²             (variance estimate)                      │
    │                                                                              │
    │  3.4 Calculate DSR (using OLD EMA values):                                     │
    │      numerator = B_t * ΔA_t - A_t * ΔB_t / 2                                  │
    │      denominator = max(Var, 0.0)^1.5 + ε                                        │
    │      D_t = numerator / denominator                                                   │
    │                                                                              │
    │  3.5 Update EMAs for NEXT step (t):                                          │
    │      A_t = (1 - η) * A_t + η * R_t        (update first moment)   │
    │      B_t = (1 - η) * B_t + η * R_t²        (update second moment)  │
    │                                                                              │
    │  3.6 Update Previous NLV:                                                    │
    │      _prev_nlv = nlv_now                                                        │
    │                                                                              │
    │  OUTPUT: D_t (Differential Sharpe Ratio reward)                                │
    └─────────────────────────────────────────────────────────────────────────────────────┘

    Step 4: Episode Reset Handling
    ┌─────────────────────────────────────────────────────────────────────────────────────┐
    │  • Reset first moment EMA: A_t = 0.0                                          │
    │  • Reset second moment EMA: B_t = 0.0                                        │
    │  • Clear previous NLV: _prev_nlv = None                                        │
    │                                                                              │
    │  Ready for new episode with fresh EMA state                                    │
    └─────────────────────────────────────────────────────────────────────────────────────┘

    Key Design Decisions:
    ┌─────────────────────────────────────────────────────────────────────────────────────┐
    │  • Calculate DSR BEFORE updating EMAs (uses t-1 values)                    │
    │  • Use max(Var, 0.0) to prevent numerical issues with negative variance    │
    │  • Add ε (epsilon) to denominator for numerical stability                       │
    │  • EMA update formula: new = (1 - η) * old + η * current                      │
    │  • Initialize _prev_nlv to None to detect first call                              │
    └─────────────────────────────────────────────────────────────────────────────────────┘

    Performance Characteristics:
    ┌─────────────────────────────────────────────────────────────────────────────────────┐
    │  η (eta) = 0.01  → Moderate adaptation (balanced variance/smoothness)          │
    │  η (eta) = 0.1   → Fast adaptation (higher variance, quicker response)       │
    │  η (eta) = 0.001 → Slow adaptation (smoother signal, stable variance)        │
    │                                                                              │
    │  Computational cost per step: O(1) constant time                               │
    │  Memory footprint: O(1) - stores only A_t, B_t, _prev_nlv                    │
    └─────────────────────────────────────────────────────────────────────────────────────┘

    ──────────────────────────────────────────────────────────────────────────────────────────

    Workflow Diagram:
    ──────────────────────────────────────────────────────────────────────────────────────────

    Step 1: Initialization
    ┌─────────────────────────────────────────────────────────────────────────────────────┐
    │  • Set EMA learning rate (η) and stability constant (ε)                   │
    │  • Initialize first moment EMA: A_t = 0.0 (mean estimate)               │
    │  • Initialize second moment EMA: B_t = 0.0 (squared returns)             │
    │  • Set previous net liquidation value: _prev_nlv = None                  │
    └─────────────────────────────────────────────────────────────────────────────────────┘

    Step 2: First Call Handling (when _prev_nlv is None)
    ┌─────────────────────────────────────────────────────────────────────────────────────┐
    │  • Get current net liquidation value: nlv_now                            │
    │  • Store as previous: _prev_nlv = nlv_now                                 │
    │  • Return reward: 0.0 (no calculation yet)                                │
    └─────────────────────────────────────────────────────────────────────────────────────┘

    Step 3: Normal Call Handling
    ┌─────────────────────────────────────────────────────────────────────────────────────┐
    │  INPUT: nlv_now (current portfolio value)                                  │
    │  INPUT: _prev_nlv (previous portfolio value)                                  │
    │  INPUT: A_t, B_t (previous EMA estimates at t-1)                           │
    │                                                                              │
    │  3.1 Calculate Log Return:                                                   │
    │      R_t = log(nlv_now / _prev_nlv)                                         │
    │                                                                              │
    │  3.2 Calculate Deltas (using OLD EMA values at t-1):                        │
    │      ΔA_t = R_t - A_t                    (change in mean estimate)            │
    │      ΔB_t = R_t² - B_t                   (change in second moment)         │
    │                                                                              │
    │  3.3 Calculate Variance:                                                     │
    │      Var = B_t - A_t²             (variance estimate)                      │
    │                                                                              │
    │  3.4 Calculate DSR (using OLD EMA values):                                     │
    │      numerator = B_t * ΔA_t - A_t * ΔB_t / 2                                  │
    │      denominator = max(Var, 0.0)^1.5 + ε                                        │
    │      D_t = numerator / denominator                                                   │
    │                                                                              │
    │  3.5 Update EMAs for NEXT step (t):                                          │
    │      A_t = (1 - η) * A_t + η * R_t        (update first moment)   │
    │      B_t = (1 - η) * B_t + η * R_t²        (update second moment)  │
    │                                                                              │
    │  3.6 Update Previous NLV:                                                    │
    │      _prev_nlv = nlv_now                                                        │
    │                                                                              │
    │  OUTPUT: D_t (Differential Sharpe Ratio reward)                                │
    └─────────────────────────────────────────────────────────────────────────────────────┘

    Step 4: Episode Reset Handling
    ┌─────────────────────────────────────────────────────────────────────────────────────┐
    │  • Reset first moment EMA: A_t = 0.0                                          │
    │  • Reset second moment EMA: B_t = 0.0                                        │
    │  • Clear previous NLV: _prev_nlv = None                                        │
    │                                                                              │
    │  Ready for new episode with fresh EMA state                                    │
    └─────────────────────────────────────────────────────────────────────────────────────┘

    Step 4: Episode Reset Handling
    ┌─────────────────────────────────────────────────────────────────────────────────────┐
    │  • Reset first moment EMA: A_t = 0.0                                          │
    │  • Reset second moment EMA: B_t = 0.0                                        │
    │  • Clear previous NLV: _prev_nlv = None                                        │
    │                                                                              │
    │  Ready for new episode with fresh EMA state                                    │
    └─────────────────────────────────────────────────────────────────────────────────────┘

    Step 5: Artifact Consumption by Documentation
    ┌─────────────────────────────────────────────────────────────────────────────────────┐
    │  • Training artifacts (checkpoints, replay buffers) logged to MLflow                 │
    │  • Evaluation plots (rewards, actions, returns) saved as artifacts            │
    │  • Training logs (trading_env_debug.log) contain DSR calculation traces          │
    │  • Final metrics and evaluation reports consumed for documentation                       │
    │  • MLflow tracking provides artifact links for paper/dataset integration             │
    │                                                                              │
    │  Documentation Process:                                                       │
    │      1. Run training with DSR reward → MLflow artifacts                        │
    │      2. Generate evaluation plots → artifact storage                                 │
    │      3. Export metrics/reports → documentation consumption                           │
    │      4. Link artifacts in paper via MLflow UI or programmatic access               │
    └─────────────────────────────────────────────────────────────────────────────────────┘

    Performance Characteristics:
    ┌─────────────────────────────────────────────────────────────────────────────────────┐
    │  η (eta) = 0.01  → Moderate adaptation (balanced variance/smoothness)          │
    │  η (eta) = 0.1   → Fast adaptation (higher variance, quicker response)       │
    │  η (eta) = 0.001 → Slow adaptation (smoother signal, stable variance)        │
    │                                                                              │
    │  Computational cost per step: O(1) constant time                               │
    │  Memory footprint: O(1) - stores only A_t, B_t, _prev_nlv                    │
    └─────────────────────────────────────────────────────────────────────────────────────┘

    ──────────────────────────────────────────────────────────────────────────────────────────
Note: DSR must be calculated BEFORE updating the EMAs to use old values (t-1).

    Args:
        eta: Learning rate for exponential moving averages (default: 0.01)
            - Higher values (0.1): Fast adaptation, higher variance
            - Lower values (0.001): Slow adaptation, smoother signal
        epsilon: Small constant for numerical stability (default: 1e-8)

    Attributes:
        A_t: Current EMA of returns (mean estimate)
        B_t: Current EMA of squared returns (second moment estimate)
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
        self.A_t = 0.0  # EMA of R (mean)
        self.B_t = 0.0  # EMA of R^2 (second moment)

        # Track previous portfolio value for return calculation
        self._prev_nlv = None

        logger.debug("init dsr eta=%s epsilon=%s", eta, epsilon)

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

        # Calculate deltas using OLD EMA values (t-1)
        # Following Moody & Saffell (2001) and github.com/AchillesJJ/DSR
        delta_A = R_t - self.A_t  # ΔA_t = R_t - A_{t-1}
        delta_B = R_t ** 2 - self.B_t  # ΔB_t = R_t^2 - B_{t-1}

        # Calculate DSR using old EMA values
        # D_t = (B_{t-1} * ΔA_t - A_{t-1} * ΔB_t / 2) / (B_{t-1} - A_{t-1}^2)^(3/2)
        variance = self.B_t - self.A_t ** 2  # Var = B_{t-1} - A_{t-1}^2
        # Clamp variance to non-negative: mathematically E[X^2] >= (E[X])^2, but
        # floating-point drift can produce a slightly negative value, which would
        # crash on fractional exponentiation (Python: ValueError on negative ** 1.5).
        denominator = max(variance, 0.0) ** 1.5 + self.epsilon
        dsr = (self.B_t * delta_A - self.A_t * delta_B / 2) / denominator

        # NOW update EMAs for next step
        self.A_t = (1 - self.eta) * self.A_t + self.eta * R_t
        self.B_t = (1 - self.eta) * self.B_t + self.eta * (R_t ** 2)

        # Update previous NLV for next step
        self._prev_nlv = nlv_now

        # Clamp to prevent gradient explosion when variance collapses near zero
        return float(np.clip(dsr, -10.0, 10.0))

    def reset(self) -> None:
        """Reset DSR state for new episode.

        This should be called at the start of each new episode to clear
        the EMA state and previous portfolio value.
        """
        self.A_t = 0.0
        self.B_t = 0.0
        self._prev_nlv = None
        logger.debug("dsr state reset")

    def __repr__(self) -> str:
        """String representation of DSR state."""
        return (
            f"DifferentialSharpeRatio(eta={self.eta}, "
            f"A_t={self.A_t:.6f}, B_t={self.B_t:.6f})"
        )
