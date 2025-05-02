"""
State space management for reinforcement learning.

This module provides functions for managing state spaces, including
state discretization and counting state/action occurrences.
"""

from typing import Any, List, Tuple

import numpy as np
import pandas as pd


def update_state_count(
    df_attributes: pd.DataFrame,
    state_count: np.ndarray,
    cut_points: List[List[float]],
) -> Tuple[List[int], np.ndarray]:
    """
    Update state count based on attribute values.

    Args:
        df_attributes: DataFrame containing attribute values
        state_count: Array tracking state occurrences
        cut_points: List of cut points for each attribute

    Returns:
        Tuple of (state indices, updated state count)
    """
    for i in range(len(df_attributes)):
        state = state_indexes(df_attributes.iloc[i], cut_points)
        state_count[tuple(state)] += 1

    return state, state_count


def update_state_action_count(
    df_attributes: pd.DataFrame,
    state_action_count: np.ndarray,
    actions: List[Any],
    action: List[Any],
    cut_points: List[List[float]],
) -> Tuple[List[int], np.ndarray]:
    """
    Update state-action count based on attribute values and actions.

    Args:
        df_attributes: DataFrame containing attribute values
        state_action_count: Array tracking state-action occurrences
        actions: List of all possible actions
        action: List of actions corresponding to attributes
        cut_points: List of cut points for each attribute

    Returns:
        Tuple of (state indices, updated state-action count)
    """
    for i in range(len(df_attributes)):
        action_dim = actions.index(action[i])
        state = state_indexes(df_attributes.iloc[i], cut_points)
        state = list(state) + [action_dim]
        state_action_count[tuple(state)] += 1

    return state, state_action_count


def update_n(
    state: List[int],
    ns: np.ndarray,
    nsa: np.ndarray,
    actions: List[Any],
    action: Any,
    cut_points: List[List[float]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Update state and state-action counts for a single state-action pair.

    Args:
        state: Current state indices
        ns: Array tracking state occurrences
        nsa: Array tracking state-action occurrences
        actions: List of all possible actions
        action: Current action
        cut_points: List of cut points for each attribute

    Returns:
        Tuple of (updated ns, updated nsa)
    """
    action_dim = actions.index(action)

    ns[tuple(state)] += 1
    nsa[tuple(list(state) + [action_dim])] += 1

    return ns, nsa


def state_indexes(attribute_row: pd.Series, cut_points: List[List[float]]) -> List[int]:
    """
    Discretize a row of features into state indices.

    Args:
        attribute_row: Series containing attribute values
        cut_points: List of cut points for each attribute

    Returns:
        List of state indices
    """
    state_indexes = []

    for i in range(len(attribute_row)):
        # Combine attribute value with cut points and find its position
        values = [attribute_row.iloc[i]] + cut_points[i]
        variable_order = sorted(range(len(values)), key=lambda k: values[k])[0]
        state_indexes.append(variable_order)

    return state_indexes
