"""
Reinforcement learning utility functions.

This module provides utility functions for reinforcement learning algorithms,
including action selection, framework initialization, and validation.
"""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from src.rl_agent_python.discretization import create_periods


@dataclass
class RLFramework:
    """Class to store reinforcement learning framework parameters."""

    cut_points: pd.DataFrame
    actions: np.ndarray
    alpha: float
    epsilon: float
    n_0: float
    episode: int
    ns: np.ndarray
    nsa: np.ndarray
    qsa: np.ndarray
    position_track: np.ndarray
    diff: int
    terminate: bool
    state_track: Optional[pd.DataFrame] = None
    action_track: Optional[np.ndarray] = None
    f_params: Optional[np.ndarray] = None


def choose_action(
    qsa: np.ndarray, state_index: int, epsilon: float, actions: np.ndarray
) -> int:
    """
    Epsilon-greedy action selection.

    Args:
        qsa: Q-value table
        state_index: Current state index
        epsilon: Exploration rate
        actions: Available actions

    Returns:
        Selected action
    """
    # Get action values for current state
    action_values = qsa[state_index]

    # Get greedy action
    greedy = actions[np.argmax(action_values)]

    # Calculate action probabilities
    probs = np.full_like(actions, epsilon / len(actions), dtype=float)
    probs[actions == greedy] += 1 - epsilon

    # Sample action
    return np.random.choice(actions, p=probs)


def choose_action_td(
    f_params: np.ndarray, state: np.ndarray, epsilon: float, actions: np.ndarray
) -> int:
    """
    Epsilon-greedy action selection for TD learning.

    Args:
        f_params: Function approximation parameters
        state: Current state
        epsilon: Exploration rate
        actions: Available actions

    Returns:
        Selected action
    """
    # Calculate action values
    action_values = f_params @ state

    # Get greedy action
    greedy = actions[np.argmax(action_values)]

    # Calculate action probabilities
    probs = np.full_like(actions, epsilon / len(actions), dtype=float)
    probs[actions == greedy] += 1 - epsilon

    # Sample action
    return np.random.choice(actions, p=probs)


def initialize_rl_framework(features: pd.DataFrame, rl_algo: str) -> RLFramework:
    """
    Initialize reinforcement learning framework.

    Args:
        features: Feature DataFrame
        rl_algo: RL algorithm ('mcc', 'qlearning', or 'tdva')

    Returns:
        Initialized RL framework
    """
    if rl_algo == "mcc":
        # Create cut points for discretization
        cut_points = pd.DataFrame(
            {
                col: create_periods(
                    features[col],
                    period_count=1,
                    method="freq",
                    multiplier=1,
                    include_extreme=False,
                )
                for col in features.columns
            }
        )

        # Initialize parameters
        n_0 = 10
        actions = np.array([-1, 0, 1])  # short, neutral, long
        dimensions = [1 + len(cut_points[col]) for col in cut_points.columns] + [
            len(actions)
        ]

        # Initialize arrays
        ns = np.zeros(dimensions[:-1])
        nsa = np.zeros(dimensions)
        qsa = np.zeros_like(nsa)
        position_track = np.array([])

        return RLFramework(
            cut_points=cut_points,
            actions=actions,
            alpha=1.0,
            epsilon=1.0,
            n_0=n_0,
            episode=1,
            ns=ns,
            nsa=nsa,
            qsa=qsa,
            position_track=position_track,
            diff=0,
            terminate=True,
        )

    elif rl_algo == "qlearning":
        # Create cut points for discretization
        cut_points = pd.DataFrame(
            {
                col: create_periods(
                    features[col],
                    period_count=1,
                    method="freq",
                    multiplier=1,
                    include_extreme=False,
                )
                for col in features.columns
            }
        )

        # Initialize parameters
        n_0 = 10
        actions = np.array([-1, 0, 1])
        dimensions = [1 + len(cut_points[col]) for col in cut_points.columns] + [
            len(actions)
        ]

        # Initialize arrays
        ns = np.zeros(dimensions[:-1])
        nsa = np.zeros(dimensions)
        qsa = np.zeros_like(nsa)

        return RLFramework(
            cut_points=cut_points,
            actions=actions,
            alpha=1.0,
            epsilon=1.0,
            n_0=n_0,
            episode=1,
            ns=ns,
            nsa=nsa,
            qsa=qsa,
            position_track=np.array([]),
            diff=0,
            terminate=True,
            state_track=pd.DataFrame(),
            action_track=np.array([]),
        )

    elif rl_algo == "tdva":
        # Initialize parameters
        n_0 = 10
        actions = np.array([-1, 0, 1])
        f_params = np.zeros((len(actions), features.shape[1]))

        return RLFramework(
            cut_points=pd.DataFrame(),
            actions=actions,
            alpha=1.0,
            epsilon=1.0,
            n_0=n_0,
            episode=1,
            ns=np.array([]),
            nsa=np.array([]),
            qsa=np.array([]),
            position_track=np.array([]),
            diff=0,
            terminate=True,
            action_track=np.array([]),
            f_params=f_params,
        )

    else:
        raise ValueError(f"Unknown RL algorithm: {rl_algo}")


def precheck_rl(obj: Any) -> None:
    """
    Validate reinforcement learning object.

    Args:
        obj: RL object to validate

    Raises:
        AssertionError: If validation fails
    """
    assert hasattr(obj, "features"), "Object must have features attribute"
    assert isinstance(obj.features, pd.DataFrame), "Features must be a DataFrame"
    assert all(
        obj.features.dtypes == float
    ), "Features must contain only numeric values"

    # Check for unique values
    unique_counts = obj.features.nunique()
    if (unique_counts == 1).any():
        cols = obj.features.columns[unique_counts == 1].tolist()
        raise AssertionError(
            f"The following columns don't have more than one unique value: {', '.join(cols)}"
        )
