"""
Principal Component Analysis module.

This module provides functions for dimensionality reduction using
Singular Value Decomposition (SVD) and PCA.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def get_pca_coefficients(df_attributes: pd.DataFrame, threshold: float) -> np.ndarray:
    """
    Calculate PCA coefficients using SVD.

    Args:
        df_attributes: DataFrame of features to reduce
        threshold: Minimum variance to explain (0-1)

    Returns:
        PCA coefficients matrix
    """
    # Standardize data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_attributes)

    # Perform SVD
    U, s, Vh = np.linalg.svd(scaled_data)

    # Calculate variance explained
    variance = s**2
    total_variance = np.sum(variance)
    var_explained = np.cumsum(variance) / total_variance

    # Find number of components needed
    n_components = np.sum(var_explained < threshold) + 1

    # Return PCA coefficients (transpose of V matrix)
    return Vh[:n_components].T


def reduce_dimensions(
    df_attributes: pd.DataFrame, coefficients: np.ndarray
) -> pd.DataFrame:
    """
    Reduce dimensions of data using PCA coefficients.

    Args:
        df_attributes: Original feature DataFrame
        coefficients: PCA coefficients from get_pca_coefficients()

    Returns:
        DataFrame of reduced dimensions
    """
    n_components = coefficients.shape[1]
    reduced_data = df_attributes.iloc[:, :n_components] @ coefficients

    # Create DataFrame with component names
    component_names = [f"PC{i+1}" for i in range(n_components)]
    return pd.DataFrame(
        reduced_data, columns=component_names, index=df_attributes.index
    )
