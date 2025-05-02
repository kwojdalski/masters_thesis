"""
Alternative execution script for reinforcement learning trading strategy.

This script provides an alternative implementation with additional visualization
and data handling capabilities.
"""

import matplotlib.pyplot as plt
import pandas as pd
from src.rl_agent_python.attributes import attributes_all
from src.rl_agent_python.cointegration import get_pairs
from src.rl_agent_python.financial_functions import calculate_returns, plot_agent_result
from src.rl_agent_python.mcc import mcc_bootstrap
from src.rl_agent_python.pre_run import pre_run
from src.rl_agent_python.q_learning import q_control


def plot_time_series(data: pd.DataFrame, title: str = "Time Series") -> None:
    """
    Plot time series data.

    Args:
        data: DataFrame containing time series data
        title: Plot title
    """
    plt.figure(figsize=(12, 6))
    plt.plot(data)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()


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

    # Define training and testing periods
    train_period = (1, 1000)
    test_period = (1001, 1300)  # 300 days of testing

    # Get pairs data
    pairs = get_pairs(df_currencies, currencies, train_period)
    train_pairs = pd.concat(pairs[0], axis=1)
    test_pairs = pd.concat(pairs[1], axis=1)
    all_pairs = pd.concat(pairs[2], axis=1)

    # Create attributes for all pairs
    delay = 2
    attributes_dict = attributes_all(all_pairs, all_pairs.columns, delay)

    # Select a pair for testing
    pair_idx = 0

    # Prepare training data
    train_pair = train_pairs.iloc[:, pair_idx]
    train_features = pd.DataFrame(attributes_dict[pair_idx]).iloc[
        train_period[0] - 1 : train_period[1]
    ]

    # Prepare testing data
    test_pair = test_pairs.iloc[:, pair_idx]
    test_features = pd.DataFrame(attributes_dict[pair_idx]).iloc[
        test_period[0] - 1 : test_period[1]
    ]

    # Visualize data
    plot_time_series(train_pair, "Training Data")
    plot_time_series(test_pair, "Testing Data")

    # Test Monte Carlo Control
    n_episodes = 10
    cost = 0

    # Train MCC agent
    pretrained_agent = mcc_bootstrap(
        train_features,
        train_pair,
        None,  # coef will be calculated internally
        cost,
        n_episodes,
        algorithm="mcc",
    )

    pretrained_pos = pretrained_agent.Position_track
    pretrain_return = calculate_returns(
        train_pair, None, pretrained_pos[-len(train_pair) :], cost
    )
    plot_agent_result(train_pair, None, pretrain_return, "Monte Carlo Control")

    # Test MCC agent
    tested_agent = mcc_bootstrap(
        test_features,
        test_pair,
        None,  # coef will be calculated internally
        cost,
        n_episodes,
        pretrained_agent=pretrained_agent,
    )

    test_pos = tested_agent.Position_track
    test_return = calculate_returns(test_pair, None, test_pos[-len(test_pair) :], cost)
    plot_agent_result(test_pair, None, test_return, "Monte Carlo Control")

    # Test Q-Learning
    n_episodes = 5
    cost = 0

    # Train Q-Learning agent
    pretrained_agent = q_control(
        train_pair,
        None,  # coef will be calculated internally
        train_features,
        n_episodes,
        cost,
        verbose=True,
        reward_function="dsr",
    )

    pretrained_pos = pretrained_agent[1]  # Position track is second element
    pretrain_return = calculate_returns(
        train_pair, None, pretrained_pos[-len(train_pair) :], cost
    )
    plot_agent_result(train_pair, None, pretrain_return, "Q-Learning Control")

    # Test Q-Learning agent
    tested_agent = q_control(
        test_pair,
        None,  # coef will be calculated internally
        test_features,
        1,  # Single episode for testing
        cost,
        pretrained_agent=pretrained_agent,
    )

    test_pos = tested_agent[1]
    test_return = calculate_returns(test_pair, None, test_pos[-len(test_pair) :], cost)
    plot_agent_result(test_pair, None, test_return, "Q-Learning Control")


if __name__ == "__main__":
    main()
