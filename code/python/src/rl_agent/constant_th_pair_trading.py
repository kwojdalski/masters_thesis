"""
Constant threshold pair trading strategies.

This module implements traditional statistical arbitrage strategies
using constant thresholds for entry and exit signals.
"""

from dataclasses import dataclass
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.rl_agent_python.financial_functions import calculate_returns


@dataclass
class SignalResult:
    """Class to store signal generation results."""

    plot: plt.Figure
    position: np.ndarray


def generate_signals(
    spread: pd.Series,
    coint_pair: pd.Series,
    mean: float,
    std: float,
    th_enter: float,
    th_exit: float,
) -> SignalResult:
    """
    Generate trading signals based on spread deviations.

    Args:
        spread: Cointegration spread series
        coint_pair: Cointegrated pair series
        mean: Spread mean
        std: Spread standard deviation
        th_enter: Entry threshold
        th_exit: Exit threshold

    Returns:
        SignalResult containing plot and position array
    """
    n = len(spread)
    position = np.zeros(n)
    signal = 0

    # Generate signals
    for i in range(n):
        if (spread.iloc[i] - mean) > (th_enter * std) and signal == 0:
            # Take short position
            signal = -1
        elif (mean - spread.iloc[i]) > (th_enter * std) and signal == 0:
            # Take long position
            signal = 1
        elif (spread.iloc[i] - mean) < (th_exit * std) and signal == -1:
            # Exit short position
            signal = 0
        elif (mean - spread.iloc[i]) < (th_exit * std) and signal == 1:
            # Exit long position
            signal = 0

        position[i] = signal

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot spread
    ax.plot(spread.index, spread - mean, color="black", label="Spread", linewidth=0.5)

    # Plot positions
    position_scaled = position * max(abs(spread - mean))
    ax.bar(
        spread.index,
        position_scaled,
        color=[
            "grey" if x == -1 else "white" if x == 0 else "lightgrey" for x in position
        ],
        width=1,
        alpha=1,
        label="Position",
    )

    # Add mean and threshold lines
    ax.axhline(y=0, color="grey20", linestyle="-", alpha=0.5)
    ax.axhline(y=th_enter * std, color="grey60", linestyle="--", alpha=0.5)
    ax.axhline(y=-th_enter * std, color="grey60", linestyle="--", alpha=0.5)
    ax.axhline(y=th_exit * std, color="grey60", linestyle=":", alpha=0.5)
    ax.axhline(y=-th_exit * std, color="grey60", linestyle=":", alpha=0.5)

    # Customize plot
    ax.set_title(f"Pair Trading of\n{coint_pair.iloc[0]}+{coint_pair.iloc[1]}")
    ax.set_ylabel("Spread value and position")
    ax.legend(["Spread", "Short", "Neutral", "Long"])
    ax.grid(True, alpha=0.3)

    return SignalResult(plot=fig, position=position)


def quarantine_positions(
    pair: pd.DataFrame,
    coint_vec: np.ndarray,
    position: np.ndarray,
    jump: float = 0.1,
    quar_period: int = 5,
) -> np.ndarray:
    """
    Set positions to neutral after large price movements.

    Args:
        pair: DataFrame of pair prices
        coint_vec: Cointegration coefficients
        position: Position array
        jump: Maximum allowed return before quarantine
        quar_period: Number of periods to quarantine

    Returns:
        Modified position array
    """
    n = len(position)
    spread = coint_vec[0] * pair.iloc[:, 0] + coint_vec[1] * pair.iloc[:, 1]
    ret_spread = np.zeros(n)

    # Calculate spread returns
    for i in range(1, n):
        ret_spread[i] = (
            (spread.iloc[i] - spread.iloc[i - 1]) / abs(spread.iloc[i - 1])
        ) + 1

    # Apply quarantine
    for i in range(1, n):
        if abs(1 - ret_spread[i]) > jump:
            end_idx = min(i + quar_period, n)
            position[i:end_idx] = 0

    return position


def backtest_constant_threshold(
    pair: pd.DataFrame,
    spread: pd.Series,
    coef: np.ndarray,
    th_enter: float = 1.0,
    th_exit: float = 0.5,
    cost: float = 0.0,
) -> Tuple[SignalResult, pd.Series]:
    """
    Backtest constant threshold strategy.

    Args:
        pair: DataFrame of pair prices
        spread: Cointegration spread series
        coef: Cointegration coefficients
        th_enter: Entry threshold
        th_exit: Exit threshold
        cost: Transaction cost

    Returns:
        Tuple of (SignalResult, returns series)
    """
    # Calculate spread statistics
    mean = spread.mean()
    std = spread.std()

    # Generate signals
    result = generate_signals(spread, pair.columns, mean, std, th_enter, th_exit)

    # Calculate returns
    returns = calculate_returns(pair, coef, result.position, cost)

    return result, returns
