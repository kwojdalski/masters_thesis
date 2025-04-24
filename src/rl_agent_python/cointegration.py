"""
Cointegration analysis for statistical arbitrage.

This module provides functions for finding cointegrated pairs of assets
and calculating their spreads.
"""

from typing import Dict, List, Tuple

import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen


def get_cointegrated_pairs(
    df_currencies: pd.DataFrame,
    c_currencies: List[str],
    learn_period: Tuple[int, int],
    critical_value: int = 2,
) -> pd.DataFrame:
    """
    Find cointegrated pairs of assets.

    Args:
        df_currencies: DataFrame with time series data in columns
        c_currencies: List of currency/asset names to check for cointegration
        learn_period: Tuple of (start, end) indices for the learning period
        critical_value: Critical value index (0: 10%, 1: 5%, 2: 1%)

    Returns:
        DataFrame where each column contains names of cointegrated pairs
    """
    # Cut the data to the specified learn period
    df_currencies = df_currencies.iloc[learn_period[0] : learn_period[1]]

    # Get all 2-element combinations of the given currency pair names
    from itertools import combinations

    pairs = list(combinations(c_currencies, 2))

    if not pairs:
        raise ValueError("No currency pairs to check for cointegration")

    cointegrated_pairs = []

    # Find pairs that are cointegrated with the specified p-value
    for pair in pairs:
        # Perform Johansen cointegration test
        result = coint_johansen(df_currencies[pair], det_order=0, k_ar_diff=1)

        # Check if the test statistic exceeds the critical value
        if result.lr1[0] > result.cvt[0, critical_value]:
            cointegrated_pairs.append(pair)

    # Convert to DataFrame format similar to R version
    if cointegrated_pairs:
        df_pairs = pd.DataFrame(cointegrated_pairs).T
        df_pairs.columns = [f"V{i + 1}" for i in range(len(cointegrated_pairs))]
    else:
        df_pairs = pd.DataFrame(columns=["V1"])

    return df_pairs


def cointegration_spreads(
    coint_pairs: pd.DataFrame,
    df_currencies: pd.DataFrame,
    learn_period: Tuple[int, int],
    test_period: Tuple[int, int],
) -> Dict:
    """
    Calculate cointegration spreads for pairs of known cointegrated series.

    Args:
        coint_pairs: DataFrame containing cointegrated pairs
        df_currencies: DataFrame with time series data
        learn_period: Tuple of (start, end) indices for the learning period
        test_period: Tuple of (start, end) indices for the test period

    Returns:
        Dictionary containing train, test, and all spreads, plus coefficients
    """
    l_sets = {
        "train": {},
        "test": {},
        "all": {},
        "coef": {},
    }

    df_train = df_currencies.iloc[learn_period[0] : learn_period[1]]
    df_test = df_currencies.iloc[test_period[0] : test_period[1]]
    df_all = df_currencies

    for col in coint_pairs.columns:
        pair = coint_pairs[col].tolist()
        pair_name = f"{pair[0]}+{pair[1]}"

        # Get cointegration coefficients from the learning period
        result = coint_johansen(df_train[pair], det_order=0, k_ar_diff=1)
        coint_vec = result.evec[:, 0]  # First eigenvector

        # Calculate spreads
        train_spread = (
            coint_vec[0] * df_train[pair[0]] + coint_vec[1] * df_train[pair[1]]
        )
        test_spread = coint_vec[0] * df_test[pair[0]] + coint_vec[1] * df_test[pair[1]]
        all_spread = coint_vec[0] * df_all[pair[0]] + coint_vec[1] * df_all[pair[1]]

        # Store results
        l_sets["train"][pair_name] = train_spread
        l_sets["test"][pair_name] = test_spread
        l_sets["all"][pair_name] = all_spread
        l_sets["coef"][pair_name] = coint_vec

    return l_sets


def get_pairs(
    df_currencies: pd.DataFrame,
    c_currencies: List[str],
    learn_period: Tuple[int, int],
    test_period: Tuple[int, int],
) -> Dict:
    """
    Get train/test/all data for individual currencies.

    Args:
        df_currencies: DataFrame with time series data
        c_currencies: List of currency names
        learn_period: Tuple of (start, end) indices for the learning period
        test_period: Tuple of (start, end) indices for the test period

    Returns:
        Dictionary containing train, test, and all data for each currency
    """
    df_train = df_currencies.iloc[learn_period[0] : learn_period[1]]
    df_test = df_currencies.iloc[test_period[0] : test_period[1]]
    df_all = df_currencies

    l_sets = {}
    for currency in c_currencies:
        l_sets[currency] = {
            "train": df_train[currency],
            "test": df_test[currency],
            "all": df_all[currency],
        }

    return l_sets
