"""
State space discretization for reinforcement learning.

This module provides functions for discretizing continuous state spaces
into discrete states for reinforcement learning.
"""

from typing import Tuple, Union

import numpy as np
import pandas as pd


def create_periods(
    var: Union[pd.Series, np.ndarray],
    period_count: int,
    method: str,
    multiplier: float,
    include_extreme: bool,
) -> pd.DataFrame:
    """
    Create boundaries that define different states.

    Args:
        var: Input variable to discretize
        period_count: Number of periods to create
        method: Discretization method ('SD' for standard deviation or 'freq' for frequency)
        multiplier: Multiplier for the discretization
        include_extreme: Whether to include extreme values

    Returns:
        DataFrame containing cut points
    """
    # Remove NA values and convert to numpy array
    var_clean = np.array(
        var.dropna() if isinstance(var, pd.Series) else var[~np.isnan(var)]
    )
    median = np.median(var_clean)

    if method == "SD":
        # Standard deviation based discretization
        std = multiplier * np.std(var_clean)
        mean = np.mean(var_clean)

        periods = np.concatenate(
            [
                mean + np.arange(1, period_count + 1) * std,
                [mean],
                mean - np.arange(1, period_count + 1) * std,
            ]
        )
        periods.sort()

    elif method == "freq":
        # Frequency based discretization
        below = np.sort(var_clean[var_clean <= median])[::-1]
        above = np.sort(var_clean[var_clean > median])

        below_step = len(below) // period_count
        above_step = len(above) // period_count

        periods = np.zeros(period_count * 2)
        for i in range(period_count):
            if i * below_step < len(below):
                periods[i] = multiplier * below[i * below_step]
            if i * above_step < len(above):
                periods[period_count + i] = multiplier * above[i * above_step]

        periods = np.concatenate([periods, [median]])
        periods.sort()

    # Remove extreme values if requested
    if not include_extreme and len(periods) > 2:
        periods = periods[1:-1]

    return pd.DataFrame(periods)


def discretize(
    data: Union[pd.Series, np.ndarray], periods: Union[pd.Series, np.ndarray]
) -> np.ndarray:
    """
    Discretize data using given cut points.

    Args:
        data: Input data to discretize
        periods: Cut points for discretization

    Returns:
        Array of discretized values
    """
    periods = np.sort(periods)
    count_periods = len(periods)
    result = np.zeros(len(data))
    half = (count_periods - 1) // 2

    if count_periods == 1:
        result[data <= periods[0]] = -1
        result[data > periods[0]] = 1
        result[np.isnan(data)] = np.nan
    else:
        for i in range(len(data)):
            if np.isnan(data[i]):
                result[i] = np.nan
                continue

            if data[i] <= periods[0]:
                result[i] = -(count_periods - 1) // 2 - 1
            elif data[i] > periods[-1]:
                result[i] = (count_periods - 1) // 2 + 1
            else:
                for j in range(half):
                    if periods[j] < data[i] <= periods[j + 1]:
                        result[i] = -half + j - 1
                        break

                for j in range(half + 1, count_periods - 1):
                    if periods[j] < data[i] <= periods[j + 1]:
                        result[i] = -half + j
                        break

    return result


def discretize_features(
    features: pd.DataFrame,
    method: str = "freq",
    period_count: int = 1,
    multiplier: float = 1.0,
    include_extreme: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Discretize all features in a DataFrame.

    Args:
        features: DataFrame containing features to discretize
        method: Discretization method
        period_count: Number of periods
        multiplier: Multiplier for discretization
        include_extreme: Whether to include extreme values

    Returns:
        Tuple of (discretized features, cut points)
    """
    # Create cut points for each feature
    cut_points = pd.DataFrame()
    for col in features.columns:
        cut_points[col] = create_periods(
            features[col], period_count, method, multiplier, include_extreme
        )

    # Discretize features using cut points
    discretized = pd.DataFrame(index=features.index, columns=features.columns)
    for col in features.columns:
        discretized[col] = discretize(features[col], cut_points[col])

    return discretized, cut_points
