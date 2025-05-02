"""
Temporal Difference Value Approximation (TDVA) implementation.

This module implements the TDVA algorithm for reinforcement learning
in statistical arbitrage applications.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd
from src.rl_agent_python.reward import q_reward
from src.rl_agent_python.rl_utils import choose_action_td, initialize_rl_framework


def tdva_control(
    pair: pd.DataFrame,
    coef: np.ndarray,
    features: pd.DataFrame,
    n_episodes: int,
    cost: float,
    pretrained_agent: Optional[Dict] = None,
    **kwargs,
) -> Dict:
    """
    Control function for TDVA algorithm.

    Args:
        pair: DataFrame containing pair trading data
        coef: Array of cointegration coefficients
        features: DataFrame of state features
        n_episodes: Number of episodes to run
        cost: Transaction cost
        pretrained_agent: Optional dictionary containing pretrained agent parameters
        **kwargs: Additional parameters

    Returns:
        Dictionary containing:
        - Qsa: Q-values for state-action pairs
        - action_track: History of actions taken
        - nsa: Count of state-action pairs
        - ns: Count of states
        - epsilon: Final exploration rate
        - alpha: Final learning rate
        - cut_points: Discretization cut points
    """
    # Initialize RL framework
    (
        actions,
        f_params,
        alpha,
        epsilon,
        n_0,
        episode,
        action_track,
    ) = initialize_rl_framework(features, "tdva")

    # If available, continue with pretrained policy
    if pretrained_agent is not None:
        actions = pretrained_agent["actions"]
        f_params = pretrained_agent["f_params"]
        alpha = pretrained_agent["alpha"]
        epsilon = pretrained_agent["epsilon"]
        n_0 = pretrained_agent["n_0"]
        episode = pretrained_agent["episode"]
        action_track = pretrained_agent["action_track"]

    while episode <= n_episodes:
        i = 0
        while i < len(pair) - 1:
            # Get current state and choose action
            state = features.iloc[i]
            action = choose_action_td(f_params, state, epsilon, actions)
            action_dim = actions.index(action)

            # Calculate Q-value for current state-action
            qsa = np.dot(f_params[action_dim], state)
            state_action = list(state) + [action_dim]

            # Track actions and update counts
            action_track.append(action)
            action_count = pd.Series(action_track).value_counts()
            a_count = action_count.get(action, 0)

            # Update exploration and learning rates
            epsilon = n_0 / (n_0 + i)
            alpha = n_0 / (n_0 + a_count)

            # Get next state and target action
            state_next = features.iloc[i + 1]
            action_target = choose_action_td(f_params, state_next, epsilon, actions)
            action_target_dim = actions.index(action_target)
            qsa_target = np.dot(f_params[action_target_dim], state_next)
            state_action_target = list(state_next) + [action_target_dim]

            # Calculate 1-step return
            ret = q_reward(pair.iloc[i : i + 2], coef, action, cost)

            # Update the model
            delta = alpha * (ret + qsa_target - qsa) * state
            f_params[action_dim] += delta

            i += 1

        episode += 1

    return {
        "qsa": qsa,
        "action_track": action_track,
        "nsa": None,  # These would need to be tracked separately
        "ns": None,
        "epsilon": epsilon,
        "alpha": alpha,
        "cut_points": None,  # This would need to be passed in or calculated
    }
