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
        check_lob_deltas: bool = True,
        check_temporal_order: bool = True,
        check_overlap: bool = True,
        check_sizes: bool = True,
        lob_levels: int = 5,
    ) -> None:
        self.check_nan = check_nan
        self.check_inf = check_inf
        self.check_duplicates = check_duplicates
        self.check_zero_variance = check_zero_variance
        self.check_lob_deltas = check_lob_deltas
        self.check_temporal_order = check_temporal_order
        self.check_overlap = check_overlap
        self.check_sizes = check_sizes
        self.lob_levels = lob_levels

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
        selected_columns: list[str] | None = None,
    ) -> None:
        """Raise ValueError if any split has zero-variance feature columns.

        When selected_columns is provided only those columns are checked —
        computed-but-not-selected features (e.g. hour_sin on single-day data)
        are ignored so a stale cache doesn't force a full rebuild.
        """
        for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            if selected_columns is not None:
                feat_cols_in_split = [c for c in selected_columns if c in split_df.columns]
            else:
                feat_cols_in_split = [c for c in split_df.columns if str(c).startswith("feature_")]
            if feat_cols_in_split:
                stds = split_df[feat_cols_in_split].std()
                zero_var = stds[stds == 0].index.tolist()
                if zero_var:
                    raise ValueError(
                        f"{split_name} data has zero-variance feature columns: {zero_var}. "
                        f"These carry no signal and will produce NaN/inf during normalization."
                    )

    def check_temporal_ordering(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> None:
        """Raise ValueError if splits are not in strict temporal order.

        Checks train < val < test using the min/max of each split's index.
        Only runs when all three splits have a DatetimeIndex.
        """
        frames = {"train": train_df, "val": val_df, "test": test_df}
        if not all(isinstance(df.index, pd.DatetimeIndex) for df in frames.values()):
            return
        train_max = train_df.index.max()
        val_min = val_df.index.min()
        val_max = val_df.index.max()
        test_min = test_df.index.min()
        if train_max >= val_min:
            raise ValueError(
                f"Temporal ordering violated: train ends at {train_max} but val "
                f"starts at {val_min}. Val data must come after train data."
            )
        if val_max >= test_min:
            raise ValueError(
                f"Temporal ordering violated: val ends at {val_max} but test "
                f"starts at {test_min}. Test data must come after val data."
            )

    def check_no_index_overlap(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> None:
        """Raise ValueError if any two splits share index values (data leakage)."""
        train_idx = set(train_df.index)
        val_idx = set(val_df.index)
        test_idx = set(test_df.index)
        tv = train_idx & val_idx
        if tv:
            raise ValueError(
                f"Data leakage: {len(tv)} index values shared between train and val splits."
            )
        tt = train_idx & test_idx
        if tt:
            raise ValueError(
                f"Data leakage: {len(tt)} index values shared between train and test splits."
            )
        vt = val_idx & test_idx
        if vt:
            raise ValueError(
                f"Data leakage: {len(vt)} index values shared between val and test splits."
            )

    def check_split_sizes(
        self,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        expected_val_size: int | None,
        expected_test_size: int | None,
    ) -> None:
        """Raise ValueError if val or test row count doesn't match config sizes.

        This catches stale prepared-data caches written before validation_size /
        test_size were added to the config (e.g. 969 K rows instead of 50 K).
        """
        if expected_val_size is not None and len(val_df) != expected_val_size:
            raise ValueError(
                f"Val split has {len(val_df):,} rows but config validation_size="
                f"{expected_val_size:,}. Delete the prepared-data cache and rerun."
            )
        if expected_test_size is not None and len(test_df) != expected_test_size:
            raise ValueError(
                f"Test split has {len(test_df):,} rows but config test_size="
                f"{expected_test_size:,}. Delete the prepared-data cache and rerun."
            )

    def check_lob_delta(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        levels: int | None = None,
        bid_px_prefix: str = "bid_px_",
        ask_px_prefix: str = "ask_px_",
        bid_sz_prefix: str = "bid_sz_",
        ask_sz_prefix: str = "ask_sz_",
    ) -> None:
        """Raise ValueError if any consecutive LOB rows are identical across all tracked levels.

        Checks that every row has at least one change (delta) in price or size
        across the first ``levels`` LOB levels compared to the previous row.
        A fully unchanged book between consecutive ticks indicates stale/duplicated
        data that should have been filtered by filter_unchanged_lob.
        """
        n_levels = levels if levels is not None else self.lob_levels
        for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            lob_cols = []
            for i in range(n_levels):
                for prefix in (bid_px_prefix, ask_px_prefix, bid_sz_prefix, ask_sz_prefix):
                    col = f"{prefix}{i:02d}"
                    if col in split_df.columns:
                        lob_cols.append(col)

            if not lob_cols:
                # No LOB columns present — skip silently (non-LOB dataset).
                continue

            lob = split_df[lob_cols]
            # diff() is NaN for the first row; fillna(1) so the first row is never flagged.
            unchanged = (lob.diff().fillna(1) == 0).all(axis=1)
            n_unchanged = int(unchanged.sum())
            if n_unchanged > 0:
                pct = 100.0 * n_unchanged / len(split_df)
                raise ValueError(
                    f"{split_name} data has {n_unchanged} rows ({pct:.2f}%) where the "
                    f"full order book is unchanged across all {n_levels} levels. "
                    f"Run filter_unchanged_lob() before building the dataset."
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
            selected_columns = getattr(getattr(config, "env", None), "feature_columns", None)
            self.check_zero_variance_features(train_df, val_df, test_df, selected_columns)
        if self.check_lob_deltas:
            self.check_lob_delta(train_df, val_df, test_df)
        if self.check_temporal_order:
            self.check_temporal_ordering(train_df, val_df, test_df)
        if self.check_overlap:
            self.check_no_index_overlap(train_df, val_df, test_df)
        if self.check_sizes:
            expected_val_size = getattr(getattr(config, "data", None), "validation_size", None)
            expected_test_size = getattr(getattr(config, "data", None), "test_size", None)
            self.check_split_sizes(val_df, test_df, expected_val_size, expected_test_size)


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
