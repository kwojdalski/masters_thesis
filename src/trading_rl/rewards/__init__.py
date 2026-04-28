"""Reward functions for trading environments."""

import numpy as np

from trading_rl.rewards.differential_sharpe import DifferentialSharpeRatio


def reward_function(history: dict) -> float:
    """Log return of portfolio valuation between the last two steps."""
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log(
            history["portfolio_valuation", -1] / history["portfolio_valuation", -2]
        )
    return float(np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0))


__all__ = ["DifferentialSharpeRatio", "reward_function"]
