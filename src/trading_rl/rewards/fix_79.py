# Fix for Issue #79 - Log-return reward validation
import numpy as np

INVALID_PORTFOLIO_PENALTY = -10.0

def reward_function_fixed(
    current_portfolio: float,
    previous_portfolio: float,
    penalty: float = INVALID_PORTFOLIO_PENALTY,
    strict: bool = True,
) -> float:
    for name, val in [("current_portfolio", current_portfolio),
                      ("previous_portfolio", previous_portfolio)]:
        if not np.isfinite(val):
            if strict:
                raise ValueError(f"{name} is not finite: {val}")
            return penalty
        if val <= 0:
            if strict:
                raise ValueError(f"{name} must be positive, got: {val}")
            return penalty
    return float(np.log(current_portfolio / previous_portfolio))
