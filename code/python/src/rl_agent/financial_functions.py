"""
Financial functions for statistical arbitrage.

This module provides functions for calculating returns, evaluating strategies,
and computing financial metrics for pairs trading.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ret_asset(asset: pd.Series, previous_ret: Optional[float] = None) -> pd.Series:
    """
    Calculate returns for a buy & hold strategy on a single asset.

    Args:
        asset: Price series of the asset
        previous_ret: Optional previous return value

    Returns:
        Series of cumulative returns
    """
    ret = pd.Series(1.0, index=asset.index)

    if previous_ret is not None:
        ret.iloc[0] = previous_ret

    for i in range(1, len(asset)):
        ret.iloc[i] = (
            (asset.iloc[i] - asset.iloc[i - 1]) / asset.iloc[i - 1] + 1
        ) * ret.iloc[i - 1]

    return ret


def calculate_returns(
    pair: pd.DataFrame,
    coef: np.ndarray,
    position: np.ndarray,
    cost: float,
    previous_ret: Optional[float] = None,
) -> pd.Series:
    """
    Calculate returns for a pairs trading strategy.

    Args:
        pair: DataFrame containing price data for both assets
        coef: Cointegration coefficients
        position: Array of positions (-1: short, 0: neutral, 1: long)
        cost: Transaction cost
        previous_ret: Optional previous return value

    Returns:
        Series of cumulative returns
    """
    n = len(pair)
    additive = pd.Series(1.0, index=pair.index)

    if previous_ret is not None:
        additive.iloc[0] = previous_ret

    # Calculate position changes
    diff_pos = np.diff(position, prepend=position[0])

    for i in range(1, n):
        # Calculate spread change
        spread_change = coef[0] * (pair.iloc[i, 0] - pair.iloc[i - 1, 0]) + coef[1] * (
            pair.iloc[i, 1] - pair.iloc[i - 1, 1]
        )
        denominator = pair.iloc[i - 1, 0] + abs(coef[1]) * pair.iloc[i - 1, 1]

        if position[i] == -1:  # Short position
            if position[i - 1] != 0 and diff_pos[i - 1] != 0:  # New position
                additive.iloc[i] = (-spread_change / denominator + 1) * additive.iloc[
                    i - 1
                ]
            else:  # Existing position
                additive.iloc[i] = (-spread_change / denominator + 1) * additive.iloc[
                    i - 1
                ]

        elif position[i] == 1:  # Long position
            if position[i - 1] != 0 and diff_pos[i - 1] != 0:  # New position
                additive.iloc[i] = (spread_change / denominator + 1) * additive.iloc[
                    i - 1
                ]
            else:  # Existing position
                additive.iloc[i] = (spread_change / denominator + 1) * additive.iloc[
                    i - 1
                ]

        elif position[i] == 0:  # No position
            if position[i - 1] == 1:  # Closing long
                additive.iloc[i] = (spread_change / denominator + 1) * additive.iloc[
                    i - 1
                ]
            elif position[i - 1] == -1:  # Closing short
                additive.iloc[i] = (-spread_change / denominator + 1) * additive.iloc[
                    i - 1
                ]
            else:  # No previous position
                additive.iloc[i] = additive.iloc[i - 1]

    return additive


def plot_agent_result(
    pair: pd.DataFrame, spread: pd.Series, additive: pd.Series, strategy_name: str
) -> plt.Figure:
    """
    Plot trading strategy results.

    Args:
        pair: DataFrame containing price data for both assets
        spread: Cointegration spread
        additive: Strategy returns
        strategy_name: Name of the strategy

    Returns:
        Matplotlib figure
    """
    asset1 = ret_asset(pair.iloc[:, 0])
    asset2 = ret_asset(pair.iloc[:, 1])

    # Convert to percentage returns
    df_all = pd.DataFrame(
        {
            "additive": (additive - 1) * 100,
            "asset1": (asset1 - 1) * 100,
            "asset2": (asset2 - 1) * 100,
            "spread": spread,
        }
    )

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        df_all.index,
        df_all["additive"],
        label=strategy_name,
        color="royalblue",
        linewidth=2,
    )
    ax.plot(
        df_all.index, df_all["asset1"], label=pair.columns[0], color="gray", alpha=0.5
    )
    ax.plot(
        df_all.index,
        df_all["asset2"],
        label=pair.columns[1],
        color="darkgray",
        alpha=0.5,
    )

    ax.set_ylabel("%RoR")
    ax.set_xlabel("Time Index")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


@dataclass
class SharpeRatio:
    """Class to store Sharpe Ratio calculation results."""

    At: float  # First moment
    Bt: float  # Second moment
    DSR: float  # Differential Sharpe Ratio


def differential_sharpe_ratio(
    returns: np.ndarray, eta: float = 0.5, time: Optional[int] = None
) -> Tuple[np.ndarray, SharpeRatio]:
    """
    Calculate the Differential Sharpe Ratio.

    Args:
        returns: Array of returns
        eta: Learning rate (between 0 and 1)
        time: Optional time parameter (not used)

    Returns:
        Tuple of (DSR array, final SharpeRatio object)
    """
    if not 0 <= eta <= 1:
        raise ValueError("eta parameter must be between 0 and 1")
    if len(returns) == 0:
        raise ValueError("returns array can't be empty")

    n = len(returns)
    if n == 1:
        return np.array([0]), SharpeRatio(0, 0, 0)

    # Initialize arrays
    At = np.zeros(n)
    Bt = np.zeros(n)
    DSR = np.zeros(n)

    # Calculate DSR
    for i in range(1, n):
        if i == 1:
            At[i - 1] = 0
            Bt[i - 1] = 0
            DSR[i - 1] = 0

        At[i] = At[i - 1] + eta * (returns[i] - At[i - 1])
        Bt[i] = Bt[i - 1] + eta * (returns[i] ** 2 - Bt[i - 1])

        denominator = (Bt[i - 1] - At[i - 1] ** 2) ** (3 / 2)
        if denominator != 0:
            DSR[i] = (
                Bt[i - 1] * (At[i] - At[i - 1]) - 0.5 * At[i - 1] * (Bt[i] - Bt[i - 1])
            ) / denominator
        else:
            DSR[i] = 0

    return DSR, SharpeRatio(At[-1], Bt[-1], DSR[-1])
