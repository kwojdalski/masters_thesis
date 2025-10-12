"""
Data loading utilities for the reinforcement learning application.

This module provides functions for loading and processing financial data
from various sources and formats.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def get_single(
    asset_code: str = "", data_dir: str | Path = "", file_format: str = ".csv"
) -> pd.DataFrame:
    """
    Get data from file for a single asset.

    Args:
        asset_code: Code/name of the asset
        data_dir: Directory containing the data files
        file_format: File extension (e.g., '.csv')

    Returns:
        DataFrame containing the asset data
    """
    file_path = Path(data_dir) / f"{asset_code}{file_format}"

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    # Read the data file
    df_asset_data = pd.read_csv(
        file_path, sep="\t", header=0, na_values=[""], keep_default_na=False
    )

    return df_asset_data


def get_quotes(
    assets: list[str], data_dir: str | Path = "", file_format: str = ""
) -> pd.DataFrame:
    """
    Create a DataFrame of quotations for the given asset names.

    Args:
        assets: List of asset codes/names
        data_dir: Directory containing the data files
        file_format: File extension (e.g., '.csv')

    Returns:
        DataFrame containing quotes for all assets
    """
    # Get the first asset to determine the number of rows
    first_asset = get_single(assets[0], data_dir, file_format)
    n_rows = len(first_asset)

    # Preallocate the DataFrame
    df_quotes = pd.DataFrame(np.zeros((n_rows, len(assets))), columns=assets)

    # Fill in the data for each asset
    for asset in assets:
        df_asset = get_single(asset, data_dir, file_format)
        df_quotes[asset] = df_asset.iloc[:, 0]  # Take the first column

    return df_quotes


def random_obs(vector: list, n: int) -> list:
    """
    Get n random observations from a vector.

    Args:
        vector: List of items to sample from
        n: Number of items to sample

    Returns:
        List of n random items from the vector
    """
    return np.random.choice(vector, size=n, replace=False).tolist()


def random_rows(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Get n random rows from a DataFrame.

    Args:
        df: DataFrame to sample from
        n: Number of rows to sample

    Returns:
        DataFrame containing n random rows
    """
    return df.sample(n=n, replace=False)
