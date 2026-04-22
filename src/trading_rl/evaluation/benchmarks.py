"""Benchmark evaluation helpers."""

from __future__ import annotations

import numpy as np

from logger import get_logger
from trading_rl.config import DEFAULT_INITIAL_PORTFOLIO_VALUE

logger = get_logger(__name__)


def calculate_benchmark_dsr(
    df_prices,
    strategy="buy_and_hold",
    eta=0.01,
    epsilon=1e-8,
    max_steps=None,
    price_column: str = "close",
    initial_portfolio_value: float = DEFAULT_INITIAL_PORTFOLIO_VALUE,
):
    """Calculate differential Sharpe ratio for a benchmark trading strategy."""
    if price_column not in df_prices.columns:
        if "close" in df_prices.columns:
            logger.warning(
                "DSR benchmark price column '%s' not found; falling back to 'close'.",
                price_column,
            )
            price_column = "close"
        else:
            raise ValueError(
                f"DSR benchmark requires '{price_column}' or 'close' in df_prices columns."
            )

    prices = df_prices[price_column].values
    if max_steps is None:
        max_steps = len(prices) - 1
    else:
        max_steps = min(max_steps, len(prices) - 1)

    portfolio_values = [initial_portfolio_value]
    price_returns = np.diff(prices[: max_steps + 1]) / prices[:max_steps]

    if strategy == "buy_and_hold":
        positions = np.ones(len(price_returns))
    elif strategy == "max_profit":
        positions = np.sign(price_returns)
        positions[positions == 0] = 1
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    for pos, ret in zip(positions, price_returns, strict=False):
        portfolio_values.append(portfolio_values[-1] * (1 + pos * ret))

    portfolio_values = np.array(portfolio_values)
    a_t = 0.0
    b_t = 0.0
    dsr_values = [0.0]

    for i in range(1, len(portfolio_values)):
        if portfolio_values[i - 1] <= 0 or portfolio_values[i] <= 0:
            logger.warning(
                "Invalid portfolio value at step %s: %s -> %s",
                i,
                portfolio_values[i - 1],
                portfolio_values[i],
            )
            dsr_values.append(0.0)
            continue

        r_t = float(np.log(portfolio_values[i] / portfolio_values[i - 1]))
        delta_a = r_t - a_t
        delta_b = r_t**2 - b_t
        variance = b_t - a_t**2
        denominator = variance**1.5 + epsilon
        dsr = (b_t * delta_a - a_t * delta_b / 2) / denominator

        a_t = (1 - eta) * a_t + eta * r_t
        b_t = (1 - eta) * b_t + eta * (r_t**2)
        dsr_values.append(dsr)

    cumulative_dsr = np.cumsum(dsr_values)
    logger.debug(
        "Calculated %s DSR benchmark: %s steps, final DSR: %.4f, final portfolio: $%.2f",
        strategy,
        len(cumulative_dsr),
        cumulative_dsr[-1],
        portfolio_values[-1],
    )
    return cumulative_dsr, portfolio_values
