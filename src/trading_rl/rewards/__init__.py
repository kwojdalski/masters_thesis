"""Reward functions for trading environments."""

import numpy as np

from logger import get_logger
from trading_rl.rewards.differential_sharpe import DifferentialSharpeRatio

logger = get_logger(__name__)

_INVALID_VALUATION_PENALTY = -1.0


def reward_function(history: dict) -> float:
    """Log return of portfolio valuation between the last two steps."""
    prev = float(history["portfolio_valuation", -2])
    curr = float(history["portfolio_valuation", -1])
    if not (np.isfinite(prev) and prev > 0):
        logger.warning(
            "reward_function: invalid previous portfolio valuation %.6g; "
            "returning penalty %.2f",
            prev,
            _INVALID_VALUATION_PENALTY,
        )
        return _INVALID_VALUATION_PENALTY
    if not (np.isfinite(curr) and curr > 0):
        logger.warning(
            "reward_function: invalid current portfolio valuation %.6g; "
            "returning penalty %.2f",
            curr,
            _INVALID_VALUATION_PENALTY,
        )
        return _INVALID_VALUATION_PENALTY
    return float(np.log(curr / prev))


__all__ = ["DifferentialSharpeRatio", "reward_function"]
