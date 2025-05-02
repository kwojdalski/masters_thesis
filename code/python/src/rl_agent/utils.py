"""
Utility functions for reinforcement learning trading agent.
"""

from typing import List, Tuple

import numpy as np


def initialize_rl_framework(features: np.ndarray, method: str) -> Tuple:
    """
    Initialize reinforcement learning framework.

    Args:
        features: Feature matrix for state representation
        method: Learning method ('qlearning' or 'mcc')

    Returns:
        Tuple of initialized parameters
    """
    # Initialize cut points for state discretization
    cut_points = []
    for i in range(features.shape[1]):
        cut_points.append(np.percentile(features[:, i], q=np.linspace(0, 100, 10)))

    # Initialize other parameters
    actions = [-1, 0, 1]  # short, neutral, long
    alpha = 0.1  # learning rate
    epsilon = 0.1  # exploration rate
    n_0 = 100  # initial visit count
    episode = 1
    ns = np.zeros([10] * features.shape[1])  # state visit counts
    nsa = np.zeros(
        [10] * features.shape[1] + [len(actions)]
    )  # state-action visit counts
    qsa = np.zeros([10] * features.shape[1] + [len(actions)])  # Q-values
    action_track = []  # track of actions taken

    return cut_points, actions, alpha, epsilon, n_0, episode, ns, nsa, qsa, action_track


def state_indexes(features: np.ndarray, cut_points: List[np.ndarray]) -> Tuple:
    """
    Convert continuous features to discrete state indices.

    Args:
        features: Feature values
        cut_points: Cut points for discretization

    Returns:
        Tuple of state indices
    """
    state = []
    for i, feature in enumerate(features):
        state.append(np.digitize(feature, cut_points[i]) - 1)
    return tuple(state)


def choose_action(
    qsa: np.ndarray, state: Tuple, epsilon: float, actions: List[int]
) -> int:
    """
    Choose action using epsilon-greedy policy.

    Args:
        qsa: Q-value matrix
        state: Current state
        epsilon: Exploration rate
        actions: List of possible actions

    Returns:
        Chosen action
    """
    if np.random.random() < epsilon:
        return np.random.choice(actions)
    else:
        return actions[np.argmax(qsa[state])]


def update_n(
    state: Tuple, ns: np.ndarray, nsa: np.ndarray, actions: List[int], action: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Update state and state-action visit counts.

    Args:
        state: Current state
        ns: State visit counts
        nsa: State-action visit counts
        actions: List of possible actions
        action: Action taken

    Returns:
        Tuple of updated visit counts
    """
    # Update state visit count
    ns[state] += 1

    # Update state-action visit count
    action_dim = actions.index(action)
    state_action = tuple([*state, action_dim])
    nsa[state_action] += 1

    return ns, nsa
