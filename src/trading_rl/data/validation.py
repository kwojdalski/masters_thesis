"""Data validation utilities for trading RL."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from logger import get_logger

logger = get_logger(__name__)


class DataValidator:
    """Validates prepared split data before building RL environments.

    Each check is independently callable and can be disabled via constructor
    flags.  The :meth:`validate` method runs all enabled checks in order.

    Args:
        check_nan: Enable NaN value check (default True).
        check_inf: Enable infinite value check (default True).
        check_duplicates: Enable duplicate index check (default True).
        check_zero_variance: Enable zero-variance feature check (default True).
    """

    def __init__(
        self,
        check_nan: bool = True,
        check_inf: bool = True,
        check_duplicates: bool = True,
        check_zero_variance: bool = True,
    ) -> None:
        self.check_nan = check_nan
        self.check_inf = check_inf
        self.check_duplicates = check_duplicates
        self.check_zero_variance = check_zero_variance

    # ------------------------------------------------------------------
    # Individual checks — each raises ValueError on failure
    # ------------------------------------------------------------------

    def check_empty_splits(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> None:
        """Raise ValueError if any split is empty."""
        if train_df.empty:
            raise ValueError(
                "Training data is empty. Check data_path or download settings."
            )
        if val_df.empty:
            raise ValueError("Validation data is empty. Check train/validation sizes.")
        if test_df.empty:
            raise ValueError("Test data is empty. Check train/validation size settings.")

    def check_close_column(self, train_df: pd.DataFrame) -> None:
        """Raise ValueError if the 'close' column is missing from training data."""
        if "close" not in train_df.columns:
            raise ValueError(
                f"Data must contain raw 'close' column for environment pricing. "
                f"Found columns: {list(train_df.columns)}"
            )

    def check_feature_columns(self, train_df: pd.DataFrame) -> None:
        """Raise ValueError if no feature_* columns are present."""
        feature_cols = [col for col in train_df.columns if str(col).startswith("feature_")]
        if not feature_cols:
            raise ValueError(
                "No feature_* columns found in prepared data. "
                "Define features in data.feature_config."
            )

    def check_env_feature_columns_prefix(
        self,
        config: Any,
    ) -> None:
        """Raise ValueError if env.feature_columns contains non-feature_* names."""
        env_feature_cols = getattr(config.env, "feature_columns", None)
        if env_feature_cols:
            non_feature_cols = [
                col for col in env_feature_cols if not str(col).startswith("feature_")
            ]
            if non_feature_cols:
                raise ValueError(
                    "env.feature_columns must contain only feature_* columns. "
                    f"Found: {non_feature_cols}"
                )

    def check_nan_values(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> None:
        """Raise ValueError if any split contains NaN values."""
        for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            if split_df.isnull().any().any():
                nan_cols = split_df.columns[split_df.isnull().any()].tolist()
                raise ValueError(
                    f"{split_name} data contains NaN values in columns: {nan_cols}. "
                    f"Clean the data before training."
                )

    def check_inf_values(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> None:
        """Raise ValueError if any split contains infinite values."""
        for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            numeric_cols = split_df.select_dtypes(include="number").columns
            if np.isinf(split_df[numeric_cols].values).any():
                inf_cols = [c for c in numeric_cols if np.isinf(split_df[c].values).any()]
                raise ValueError(
                    f"{split_name} data contains infinite values in columns: {inf_cols}. "
                    f"Clean the data before training."
                )

    def check_duplicate_index(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> None:
        """Raise ValueError if any split contains duplicate index entries."""
        for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            # Duplicate row check — identical index + identical data is a data pipeline bug.
            if split_df.index.duplicated().any():
                n_dups = split_df.index.duplicated().sum()
                raise ValueError(
                    f"{split_name} data contains {n_dups} duplicate index entries. "
                    f"De-duplicate before training."
                )

    def check_zero_variance_features(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> None:
        """Raise ValueError if any split has zero-variance feature columns."""
        for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            # Zero-variance feature check — constant features break normalizers and carry no signal.
            feat_cols_in_split = [c for c in split_df.columns if str(c).startswith("feature_")]
            if feat_cols_in_split:
                stds = split_df[feat_cols_in_split].std()
                zero_var = stds[stds == 0].index.tolist()
                if zero_var:
                    raise ValueError(
                        f"{split_name} data has zero-variance feature columns: {zero_var}. "
                        f"These carry no signal and will produce NaN/inf during normalization."
                    )

    # ------------------------------------------------------------------
    # Composite entry point
    # ------------------------------------------------------------------

    def validate(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        config: Any,
    ) -> None:
        """Run all enabled checks against the prepared splits.

        Args:
            train_df: Training split DataFrame.
            val_df: Validation split DataFrame.
            test_df: Test split DataFrame.
            config: Config object exposing at least ``config.env.feature_columns``.
        """
        # Always-on structural checks
        self.check_empty_splits(train_df, val_df, test_df)
        self.check_close_column(train_df)
        self.check_feature_columns(train_df)
        self.check_env_feature_columns_prefix(config)

        # Optionally-disabled data quality checks
        if self.check_nan:
            self.check_nan_values(train_df, val_df, test_df)
        if self.check_inf:
            self.check_inf_values(train_df, val_df, test_df)
        if self.check_duplicates:
            self.check_duplicate_index(train_df, val_df, test_df)
        if self.check_zero_variance:
            self.check_zero_variance_features(train_df, val_df, test_df)


def validate_prepared_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Any,
) -> None:
    """Validate prepared split data before building environments.

    This is a thin wrapper around :class:`DataValidator` with all checks enabled.
    It exists for backwards compatibility — prefer instantiating ``DataValidator``
    directly when you need to skip individual checks.
    """
    DataValidator().validate(train_df, val_df, test_df, config)
