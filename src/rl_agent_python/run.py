"""
Main execution script for reinforcement learning trading strategy.

This script runs the complete pipeline for training and testing
reinforcement learning agents on cointegrated currency pairs.
"""

# %%
import pandas as pd

from src.rl_agent_python.attributes import attributes_all
from src.rl_agent_python.cointegration import (
    cointegration_spreads,
    get_cointegrated_pairs,
)
from src.rl_agent_python.financial_functions import calculate_returns, plot_agent_result
from src.rl_agent_python.getdata import get_quotes
from src.rl_agent_python.mcc import mcc_bootstrap
from src.rl_agent_python.q_learning import QLearning


# %%
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
    df_currencies = get_quotes(currencies, data_dir="data/raw/1min", file_format=".csv")

    # Define training and testing periods
    train_period = (1, 1000)
    test_period = (1001, 1300)  # 300 days of testing

    # Get cointegrated pairs
    coint_pairs = get_cointegrated_pairs(
        df_currencies, currencies, train_period, critical_value=3
    )

    # Calculate cointegration spreads
    coint_spreads = cointegration_spreads(
        coint_pairs, df_currencies, train_period, test_period
    )

    train_spreads = pd.DataFrame(coint_spreads[0])
    test_spreads = pd.DataFrame(coint_spreads[1])
    all_spreads = pd.DataFrame(coint_spreads[2])
    coefs = pd.DataFrame(coint_spreads[3])

    # Create attributes for all pairs
    delay = 2
    attributes_dict = attributes_all(all_spreads, all_spreads.columns, delay)
    # Select a pair for testing
    pair_idx = 0
    coint_pair = coint_pairs.iloc[:, pair_idx]
    coef = coefs.iloc[:, pair_idx]

    # Prepare training data
    train_pair = df_currencies.iloc[train_period[0] - 1 : train_period[1], coint_pair]
    train_spread = train_spreads.iloc[:, pair_idx]
    train_features = pd.DataFrame(attributes_dict[pair_idx]).iloc[
        train_period[0] - 1 : train_period[1]
    ]

    # Prepare testing data
    test_pair = df_currencies.iloc[test_period[0] - 1 : test_period[1], coint_pair]
    test_spread = test_spreads.iloc[:, pair_idx]
    test_features = pd.DataFrame(attributes_dict[pair_idx]).iloc[
        test_period[0] - 1 : test_period[1]
    ]

    # Test Monte Carlo Control
    n_episodes = 10
    cost = 0

    # Train MCC agent
    pretrained_agent = mcc_bootstrap(
        train_features, train_pair, coef, cost, n_episodes, algorithm="mcc"
    )

    pretrained_pos = pretrained_agent.Position_track
    pretrain_return = calculate_returns(
        train_pair, coef, pretrained_pos[-len(train_pair) :], cost
    )
    plot_agent_result(train_pair, train_spread, pretrain_return, "Monte Carlo Control")

    # Test MCC agent
    tested_agent = mcc_bootstrap(
        test_features,
        test_pair,
        coef,
        cost,
        n_episodes,
        pretrained_agent=pretrained_agent,
    )

    test_pos = tested_agent.Position_track
    test_return = calculate_returns(test_pair, coef, test_pos[-len(test_pair) :], cost)
    plot_agent_result(test_pair, test_spread, test_return, "Monte Carlo Control")

    # Test Q-Learning
    n_episodes = 5
    cost = 0

    # Train Q-Learning agent
    pretrained_agent = QLearning(
        train_pair,
        coef,
        train_features,
        n_episodes,
        cost,
        verbose=True,
        reward_function="dsr",
    )

    pretrained_pos = pretrained_agent[1]  # Position track is second element
    pretrain_return = calculate_returns(
        train_pair, coef, pretrained_pos[-len(train_pair) :], cost
    )
    plot_agent_result(train_pair, train_spread, pretrain_return, "Q-Learning Control")

    # Test Q-Learning agent
    tested_agent = q_control(
        test_pair,
        coef,
        test_features,
        1,  # Single episode for testing
        cost,
        pretrained_agent=pretrained_agent,
    )

    test_pos = tested_agent[1]
    test_return = calculate_returns(test_pair, coef, test_pos[-len(test_pair) :], cost)
    plot_agent_result(test_pair, test_spread, test_return, "Q-Learning Control")


if __name__ == "__main__":
    main()

# %%
