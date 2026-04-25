"""Base classes for feature engineering."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class FeatureDomain(StrEnum):
    """Domain tag controlling which experiment modes a feature is eligible for."""

    SHARED = "shared"
    MFT = "mft"
    HFT = "hft"


class NormalizationMethod(StrEnum):
    """Normalization method for features."""

    GLOBAL = "global"  # StandardScaler on full dataset (fast, potential lookahead bias)
    ROLLING = "rolling"  # Rolling window z-score (causal, slower)
    NONE = "none"  # No normalization (let network handle it)


class RollingWindowScaler:
    """Causal rolling window z-score normalization.

    Normalizes each value using mean/std from only the previous N values,
    preventing look-ahead bias for sequential data.
    """

    def __init__(self, window: int = 1000, min_periods: int = 1):
        """Initialize rolling window scaler.

        Args:
            window: Size of the rolling window.
            min_periods: Minimum number of observations required to compute stats.
        """
        self.window = window
        self.min_periods = min_periods
        self._fitted = False

    def fit(self, data: np.ndarray) -> "RollingWindowScaler":
        """Fit is a no-op for rolling window - stats are computed during transform."""
        self._fitted = True
        return self

    def transform(self, data: np.ndarray | pd.Series) -> np.ndarray:
        """Transform using rolling window statistics.

        Args:
            data: Input data to normalize.

        Returns:
            Normalized data with same shape as input.
        """
        if not self._fitted:
            raise RuntimeError("RollingWindowScaler must be fitted before transform.")

        if isinstance(data, pd.Series):
            s = data
        else:
            s = pd.Series(data)

        # Compute rolling mean and std (causal - only looks at past)
        rolling_mean = s.rolling(window=self.window, min_periods=self.min_periods).mean()
        rolling_std = s.rolling(window=self.window, min_periods=self.min_periods).std()

        # Normalize: (x - rolling_mean) / rolling_std
        # Fill NaNs (from insufficient window) with 0 or forward fill
        normalized = (s - rolling_mean) / rolling_std

        # Handle cases where std is 0 (constant region)
        normalized = normalized.fillna(0.0)

        # Replace inf with 0 (happens when std approaches 0)
        normalized = normalized.replace([np.inf, -np.inf], 0.0)

        return normalized.values if not isinstance(data, pd.Series) else normalized

    def fit_transform(self, data: np.ndarray | pd.Series) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(data).transform(data)


@dataclass
class FeatureConfig:
    """Configuration for a single feature.

    Attributes:
        name: Name of the feature (e.g., "log_return")
        feature_type: Type identifier for the feature class (e.g., "log_return")
        params: Additional parameters for feature creation
        normalize: Whether to apply z-score normalization
        normalization_method: How to normalize: "global" (StandardScaler), "rolling", or "none"
        rolling_window: Window size for rolling normalization (only used when method="rolling")
        output_name: Optional custom output column name
        domain: Feature domain tag used for experiment-mode validation.
            Supported values: "shared", "mft", "hft".
    """

    name: str
    feature_type: str
    params: dict[str, Any] | None = None
    normalize: bool = True
    normalization_method: str = NormalizationMethod.GLOBAL
    rolling_window: int = 1000
    output_name: str | None = None
    domain: str = FeatureDomain.SHARED

    def __post_init__(self):
        if self.params is None:
            self.params = {}
        if self.output_name is None:
            self.output_name = f"feature_{self.name}"
        self.domain = str(self.domain).lower().strip()
        self.normalization_method = str(self.normalization_method).lower().strip()
        if self.domain not in set(FeatureDomain):
            raise ValueError(
                f"Invalid feature domain '{self.domain}'. "
                f"Supported values: {sorted(FeatureDomain)}"
            )
        if self.normalization_method not in set(NormalizationMethod):
            raise ValueError(
                f"Invalid normalization_method '{self.normalization_method}'. "
                f"Supported values: {sorted(NormalizationMethod)}"
            )


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
        self.scaler: StandardScaler | RollingWindowScaler | None = None
        if config.normalize and config.normalization_method == NormalizationMethod.GLOBAL:
            self.scaler = StandardScaler()
        elif config.normalize and config.normalization_method == NormalizationMethod.ROLLING:
            self.scaler = RollingWindowScaler(window=config.rolling_window)

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
            if isinstance(self.scaler, StandardScaler):
                # StandardScaler: fit on all training data
                valid_values = raw_values.dropna().values.reshape(-1, 1)
                if len(valid_values) > 0:
                    self.scaler.fit(valid_values)
            elif isinstance(self.scaler, RollingWindowScaler):
                # RollingWindowScaler: fit is a no-op (stats computed during transform)
                self.scaler.fit(raw_values)
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
            if isinstance(self.scaler, RollingWindowScaler):
                # RollingWindowScaler handles NaNs internally and preserves index
                normalized = self.scaler.transform(raw_values)
                return pd.Series(normalized, index=raw_values.index, dtype=float)
            else:
                # StandardScaler: normalize non-null values
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
