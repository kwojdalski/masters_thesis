"""
Full backtesting script for reinforcement learning trading strategy.

This script implements a rolling window backtesting approach to evaluate
the performance of reinforcement learning agents on cointegrated pairs.
"""

from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.rl_agent_python.attributes import attributes_all
from src.rl_agent_python.cointegration import (
    cointegration_spreads,
    get_cointegrated_pairs,
)
from src.rl_agent_python.financial_functions import calculate_returns, ret_asset
from src.rl_agent_python.mcc import mcc_bootstrap
from src.rl_agent_python.pre_run import pre_run


@dataclass
class BacktestResult:
    """Class to store backtesting results."""

    agent_returns: np.ndarray
    pair_returns: np.ndarray
    train_periods: List[Tuple[int, int]]
    test_periods: List[Tuple[int, int]]


def plot_returns(agent_returns: np.ndarray, pair_returns: np.ndarray) -> None:
    """
    Plot agent and pair returns.

    Args:
        agent_returns: Array of agent returns
        pair_returns: Array of pair returns
    """
    plt.figure(figsize=(12, 6))
    plt.plot(agent_returns, label="Agent Returns")
    plt.plot(pair_returns, label="Pair Returns")
    plt.title("Rolling Window Backtest Results")
    plt.xlabel("Time")
    plt.ylabel("Returns")
    plt.legend()
    plt.grid(True)
    plt.show()


def rolling_window_backtest(
    df_currencies: pd.DataFrame,
    currencies: List[str],
    train_length: int = 3000,
    test_length: int = 200,
    n_episodes: int = 2,
    cost: float = 0,
    delay: int = 15,
) -> BacktestResult:
    """
    Perform rolling window backtesting.

    Args:
        df_currencies: DataFrame of currency data
        currencies: List of currency codes
        train_length: Length of training window
        test_length: Length of testing window
        n_episodes: Number of episodes for training
        cost: Transaction cost
        delay: Delay for attribute calculation

    Returns:
        BacktestResult object containing returns and periods
    """
    all_agent_returns = []
    all_pair_returns = []
    train_periods = []
    test_periods = []

    previous_agent_ret = 1.0
    previous_pair_ret = 1.0

    i = 1
    while (train_length + test_length * i) < len(df_currencies):
        # Define periods for this iteration
        train_start = test_length * (i - 1) + 1
        train_end = train_length + test_length * (i - 1)
        test_start = train_length + test_length * (i - 1) + 1
        test_end = train_length + test_length * i

        train_period = (train_start, train_end)
        test_period = (test_start, test_end)

        train_periods.append(train_period)
        test_periods.append(test_period)

        # Get cointegrated pairs
        coint_pairs = get_cointegrated_pairs(
            df_currencies, currencies, train_period, critical_value=3
        )

        # Calculate spreads
        coint_spreads = cointegration_spreads(
            coint_pairs, df_currencies, train_period, test_period
        )

        train_spreads = pd.DataFrame(coint_spreads[0])
        test_spreads = pd.DataFrame(coint_spreads[1])
        all_spreads = pd.DataFrame(coint_spreads[2])
        coefs = pd.DataFrame(coint_spreads[3])

        # Create attributes
        attributes_dict = attributes_all(all_spreads, all_spreads.columns, delay)

        # Backtest each cointegrated pair
        agent_returns = {}
        pair_returns = {}

        for j in range(len(coint_pairs.columns)):
            print(f"Processing pair {j+1}/{len(coint_pairs.columns)}")

            pair_idx = j
            coint_pair = coint_pairs.iloc[:, pair_idx]
            coef = coefs.iloc[:, pair_idx]

            # Prepare training data
            train_pair = df_currencies.iloc[
                train_period[0] - 1 : train_period[1], coint_pair
            ]
            train_spread = train_spreads.iloc[:, pair_idx]
            train_features = pd.DataFrame(attributes_dict[pair_idx]).iloc[
                train_period[0] - 1 : train_period[1]
            ]

            # Prepare testing data
            test_pair = df_currencies.iloc[
                test_period[0] - 1 : test_period[1], coint_pair
            ]
            test_spread = test_spreads.iloc[:, pair_idx]
            test_features = pd.DataFrame(attributes_dict[pair_idx]).iloc[
                test_period[0] - 1 : test_period[1]
            ]

            # Train and test MCC agent
            pretrained_agent = mcc_bootstrap(
                train_features, train_pair, coef, cost, n_episodes, algorithm="mcc"
            )

            pretrained_pos = pretrained_agent.Position_track
            pretrain_return = calculate_returns(
                train_pair, coef, pretrained_pos[-len(train_pair) :], cost
            )

            tested_agent = mcc_bootstrap(
                test_features,
                test_pair,
                coef,
                cost,
                n_episodes,
                pretrained_agent=pretrained_agent,
            )

            test_pos = tested_agent.Position_track
            test_return = calculate_returns(
                test_pair,
                coef,
                test_pos[-len(test_pair) :],
                cost,
                previous_ret=previous_agent_ret,
            )

            pair_name = f"{coint_pair.iloc[0]}_{coint_pair.iloc[1]}"
            agent_returns[pair_name] = test_return

            # Calculate pair returns
            pair_ret = pd.DataFrame(
                {
                    col: ret_asset(test_pair[col], previous_ret=previous_pair_ret)
                    for col in test_pair.columns
                }
            ).mean(axis=1)

            pair_returns[pair_name] = pair_ret

        # Calculate mean returns for this window
        agent_returns_df = pd.DataFrame(agent_returns)
        pair_returns_df = pd.DataFrame(pair_returns)

        window_agent_returns = agent_returns_df.mean(axis=1)
        window_pair_returns = pair_returns_df.mean(axis=1)

        all_agent_returns.extend(window_agent_returns)
        all_pair_returns.extend(window_pair_returns)

        previous_agent_ret = window_agent_returns.iloc[-1]
        previous_pair_ret = window_pair_returns.iloc[-1]

        i += 1

    return BacktestResult(
        agent_returns=np.array(all_agent_returns),
        pair_returns=np.array(all_pair_returns),
        train_periods=train_periods,
        test_periods=test_periods,
    )


def main():
    # Initialize data
    currencies = [
        "EURGBP",
        "EURUSD",
        "USDJPY",
        "EURJPY",
        "CHFJPY",
        "GBPUSD",
        "AUDJPY",
        "EURCAD",
    ]

    # Get currency data
    df_currencies = pre_run(currencies)

    # Run backtesting
    results = rolling_window_backtest(
        df_currencies,
        currencies,
        train_length=3000,
        test_length=200,
        n_episodes=2,
        cost=0,
        delay=15,
    )

    # Plot results
    plot_returns(results.agent_returns, results.pair_returns)


if __name__ == "__main__":
    main()
