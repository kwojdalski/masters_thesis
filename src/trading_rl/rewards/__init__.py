"""Reward functions for trading environments."""

import numpy as np

from logger import get_logger
from trading_rl.rewards.differential_sharpe import DifferentialSharpeRatio
from trading_rl.rewards.registry import RewardRegistry, register_reward

logger = get_logger(__name__)

_INVALID_VALUATION_PENALTY = -1.0


def reward_function(history: dict) -> float:
    """Log return of portfolio valuation between the last two steps.

    The penalty is latched to the first invalid step of a streak: it fires
    only on the valid-to-invalid transition, not on every step of the
    streak. Without latching, a run of N invalid steps (e.g. a data
    glitch) would accumulate N * _INVALID_VALUATION_PENALTY, dwarfing
    normal step rewards (~1e-4 in magnitude) and dominating the training
    signal. Once `prev` itself is invalid we are already inside (or
    recovering from) a streak that already emitted the one-time penalty,
    so subsequent steps return 0.0 -- mirroring the skip-and-rebase
    treatment used for invalid values in the DSR rewards.
    """
    prev = float(history["portfolio_valuation", -2])
    curr = float(history["portfolio_valuation", -1])
    prev_valid = np.isfinite(prev) and prev > 0
    curr_valid = np.isfinite(curr) and curr > 0

    if prev_valid and curr_valid:
        return float(np.log(curr / prev))

    if not prev_valid:
        return 0.0

    logger.warning(
        "reward_function: invalid current portfolio valuation {}; "
        "returning penalty {} (latched to first invalid step of streak)",
        curr,
        _INVALID_VALUATION_PENALTY,
    )
    return _INVALID_VALUATION_PENALTY


__all__ = [
    "DifferentialSharpeRatio",
    "RewardRegistry",
    "register_reward",
    "reward_function",
]
