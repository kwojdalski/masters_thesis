"""Base classes for feature engineering."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class FeatureConfig:
    """Configuration for a single feature.

    Attributes:
        name: Name of the feature (e.g., "log_return")
        feature_type: Type identifier for the feature class (e.g., "log_return")
        params: Additional parameters for feature creation
        normalize: Whether to apply z-score normalization
        output_name: Optional custom output column name
    """

    name: str
    feature_type: str
    params: dict[str, Any] | None = None
    normalize: bool = True
    output_name: str | None = None

    def __post_init__(self):
        if self.params is None:
            self.params = {}
        if self.output_name is None:
            self.output_name = f"feature_{self.name}"


class Feature(ABC):
    """Base class for all feature implementations.

    Features should:
    - Implement compute() to calculate raw feature values
    - Support fit/transform pattern for normalization
    - Handle missing values appropriately
    - Document what columns they require
    """

    def __init__(self, config: FeatureConfig):
        self.config = config
        self.scaler: StandardScaler | None = None
        if config.normalize:
            self.scaler = StandardScaler()

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Compute the raw feature values.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Series with raw feature values (not normalized)
        """
        pass

    @abstractmethod
    def required_columns(self) -> list[str]:
        """Return list of required columns from input DataFrame.

        Returns:
            List of column names required for this feature
        """
        pass

    def fit(self, df: pd.DataFrame) -> "Feature":
        """Fit normalization parameters on training data.

        Args:
            df: Training DataFrame

        Returns:
            self for chaining
        """
        if self.scaler is not None:
            raw_values = self.compute(df)
            # Fit scaler on non-null values
            valid_values = raw_values.dropna().values.reshape(-1, 1)
            if len(valid_values) > 0:
                self.scaler.fit(valid_values)
        return self

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Transform data using fitted normalization parameters.

        Args:
            df: DataFrame to transform

        Returns:
            Series with transformed (normalized) feature values
        """
        raw_values = self.compute(df)

        if self.scaler is not None:
            # Normalize non-null values
            result = pd.Series(index=raw_values.index, dtype=float)
            valid_mask = ~raw_values.isna()

            if valid_mask.any():
                valid_values = raw_values[valid_mask].values.reshape(-1, 1)
                normalized = self.scaler.transform(valid_values).flatten()
                result[valid_mask] = normalized

            return result
        else:
            return raw_values

    def fit_transform(self, df: pd.DataFrame) -> pd.Series:
        """Fit and transform in one step.

        Args:
            df: Training DataFrame

        Returns:
            Series with transformed feature values
        """
        return self.fit(df).transform(df)

    def get_output_name(self) -> str:
        """Get the output column name for this feature.

        Returns:
            Output column name
        """
        return self.config.output_name or f"feature_{self.config.name}"
