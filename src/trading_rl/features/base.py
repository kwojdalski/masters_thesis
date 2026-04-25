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
    RUNNING = "running"  # Running mean/std (Welford's algorithm, causal, stable)
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

    def reset(self) -> "RollingWindowScaler":
        """Reset scaler (no-op for rolling window, kept for API compatibility)."""
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


class RunningMeanStd:
    """Running mean and standard deviation using Welford's algorithm.

    This maintains cumulative statistics incrementally, using O(1) updates.
    All observations contribute equally to the final statistics — no forgetting
    of past data. This provides stable normalization with no look-ahead bias.

    This is the approach used in:
    - Stable Baselines3 (VecNormalize)
    - Ray RLLib (RunningMeanStd)
    - DeepMind's RL frameworks
    """

    def __init__(self, epsilon: float = 1e-4):
        """Initialize running statistics.

        Args:
            epsilon: Small constant to prevent division by zero.
        """
        self.mean = 0.0
        self.var = 1.0
        self.count = 0
        self.epsilon = epsilon
        self._fitted = False

    def reset(self) -> "RunningMeanStd":
        """Reset running statistics to initial state.

        Use this at session boundaries (e.g., market open after overnight gap)
        to prevent cross-session contamination of normalization stats.
        """
        self.mean = 0.0
        self.var = 1.0
        self.count = 0
        self._fitted = False
        return self

    def update(self, x: np.ndarray) -> "RunningMeanStd":
        """Update running statistics with new data batch.

        Uses Welford's online algorithm for numerically stable computation.

        Args:
            x: New data batch (1D array or scalar).

        Returns:
            self for chaining.
        """
        batch_mean = np.mean(x)
        batch_var = np.var(x)
        batch_count = len(x)

        if batch_count == 0:
            return self

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        # Update mean
        new_mean = self.mean + delta * batch_count / total_count

        # Update variance using parallel algorithm
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count

        new_var = M2 / total_count if total_count > 0 else self.var

        self.mean = float(new_mean)
        self.var = float(new_var)
        self.count = total_count
        self._fitted = True

        return self

    def fit(self, data: np.ndarray | pd.Series) -> "RunningMeanStd":
        """Fit on data by updating running statistics incrementally.

        Args:
            data: Input data to fit on.

        Returns:
            self for chaining.
        """
        if isinstance(data, pd.Series):
            data = data.values
        self.update(data)
        return self

    def transform(self, data: np.ndarray | pd.Series) -> np.ndarray:
        """Transform using current running statistics.

        Args:
            data: Input data to normalize.

        Returns:
            Normalized data with same shape as input.
        """
        if not self._fitted:
            # First time: use raw data as-is, then update stats
            if isinstance(data, pd.Series):
                self.fit(data)
                return data
            else:
                self.fit(data)
                return data.copy()

        if isinstance(data, pd.Series):
            s = data
        else:
            s = pd.Series(data)

        # Normalize: (x - running_mean) / sqrt(running_var + epsilon)
        normalized = (s - self.mean) / np.sqrt(self.var + self.epsilon)

        # Handle NaNs (shouldn't happen with proper data)
        normalized = normalized.fillna(0.0)

        # Replace inf with 0
        normalized = normalized.replace([np.inf, -np.inf], 0.0)

        return normalized.values if not isinstance(data, pd.Series) else normalized

    def fit_transform(self, data: np.ndarray | pd.Series) -> np.ndarray:
        """Fit and transform in one step.

        Args:
            data: Input data.

        Returns:
            Normalized data.
        """
        return self.fit(data).transform(data)

    def state_dict(self) -> dict:
        """Get current state (for checkpointing).

        Returns:
            Dictionary with mean, var, count.
        """
        return {
            "mean": self.mean,
            "var": self.var,
            "count": self.count,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load state from dictionary.

        Args:
            state_dict: Dictionary with mean, var, count.
        """
        self.mean = state_dict["mean"]
        self.var = state_dict["var"]
        self.count = state_dict["count"]
        self._fitted = self.count > 0


class TimeWeightedRunningMeanStd:
    """Time-weighted running mean and standard deviation.

    Statistics are weighted by time deltas between observations, not by event count.
    This is more appropriate for HFT where events cluster during high activity periods.

    For each observation x_i with time weight w_i (seconds since previous event):
        mean = sum(w_i * x_i) / sum(w_i)
        var  = sum(w_i * (x_i - mean)^2) / sum(w_i)

    This gives more importance to periods with longer duration, preventing
    clustered events from dominating the statistics.
    """

    def __init__(self, epsilon: float = 1e-4):
        """Initialize time-weighted running statistics.

        Args:
            epsilon: Small constant to prevent division by zero.
        """
        self.mean = 0.0
        self.var = 1.0
        self.total_weight = 0.0  # Sum of all time weights (not event count)
        self.epsilon = epsilon
        self._fitted = False

    def reset(self) -> "TimeWeightedRunningMeanStd":
        """Reset running statistics to initial state."""
        self.mean = 0.0
        self.var = 1.0
        self.total_weight = 0.0
        self._fitted = False
        return self

    def update(self, x: np.ndarray, time_weights: np.ndarray | None = None) -> "TimeWeightedRunningMeanStd":
        """Update running statistics with new data batch.

        Args:
            x: New data batch (1D array or scalar).
            time_weights: Time weights for each observation (in seconds). If None,
                uses unit weights (degrades to event-based).

        Returns:
            self for chaining.
        """
        if len(x) == 0:
            return self

        if time_weights is None:
            # Degrade to event-based if no weights provided
            time_weights = np.ones_like(x)

        # Ensure time_weights is same length as x
        if len(time_weights) != len(x):
            raise ValueError(
                f"Length mismatch: data ({len(x)}) and time_weights ({len(time_weights)})"
            )

        batch_mean = np.average(x, weights=time_weights)
        batch_weight = np.sum(time_weights)
        batch_var = np.average((x - batch_mean) ** 2, weights=time_weights)

        if batch_weight == 0:
            return self

        delta = batch_mean - self.mean
        total_weight_new = self.total_weight + batch_weight

        # Update weighted mean
        new_mean = self.mean + delta * batch_weight / total_weight_new

        # Update weighted variance using weighted parallel algorithm
        m_a = self.var * self.total_weight
        m_b = batch_var * batch_weight
        M2 = m_a + m_b + delta**2 * self.total_weight * batch_weight / total_weight_new

        new_var = M2 / total_weight_new if total_weight_new > 0 else self.var

        self.mean = float(new_mean)
        self.var = float(new_var)
        self.total_weight = total_weight_new
        self._fitted = True

        return self

    def fit(
        self,
        data: np.ndarray | pd.Series,
        time_weights: np.ndarray | None = None,
    ) -> "TimeWeightedRunningMeanStd":
        """Fit on data by updating running statistics incrementally.

        Args:
            data: Input data to fit on.
            time_weights: Time weights for each observation. If None, computes from
                DatetimeIndex if available, otherwise uses unit weights.

        Returns:
            self for chaining.
        """
        if isinstance(data, pd.Series):
            # Compute time weights from DatetimeIndex if not provided
            if time_weights is None and isinstance(data.index, pd.DatetimeIndex):
                time_weights = data.index.to_series().diff().dt.total_seconds().fillna(1.0)
                time_weights = time_weights.values
            data = data.values

        self.update(data, time_weights)
        return self

    def transform(
        self,
        data: np.ndarray | pd.Series,
        time_weights: np.ndarray | None = None,
    ) -> np.ndarray:
        """Transform using current running statistics.

        Args:
            data: Input data to normalize.
            time_weights: Time weights for fitting if not yet fitted. Ignored during
                transform (uses fitted stats).

        Returns:
            Normalized data with same shape as input.
        """
        if not self._fitted:
            # First time: use raw data as-is, then update stats
            if isinstance(data, pd.Series):
                self.fit(data, time_weights)
                return data
            else:
                self.fit(data, time_weights)
                return data.copy()

        if isinstance(data, pd.Series):
            s = data
        else:
            s = pd.Series(data)

        # Normalize: (x - running_mean) / sqrt(running_var + epsilon)
        normalized = (s - self.mean) / np.sqrt(self.var + self.epsilon)

        # Handle NaNs
        normalized = normalized.fillna(0.0)
        normalized = normalized.replace([np.inf, -np.inf], 0.0)

        return normalized.values if not isinstance(data, pd.Series) else normalized

    def fit_transform(
        self,
        data: np.ndarray | pd.Series,
        time_weights: np.ndarray | None = None,
    ) -> np.ndarray:
        """Fit and transform in one step.

        Args:
            data: Input data.
            time_weights: Time weights for each observation.

        Returns:
            Normalized data.
        """
        return self.fit(data, time_weights).transform(data, time_weights)

    def state_dict(self) -> dict:
        """Get current state (for checkpointing).

        Returns:
            Dictionary with mean, var, total_weight.
        """
        return {
            "mean": self.mean,
            "var": self.var,
            "total_weight": self.total_weight,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load state from dictionary.

        Args:
            state_dict: Dictionary with mean, var, total_weight.
        """
        self.mean = state_dict["mean"]
        self.var = state_dict["var"]
        self.total_weight = state_dict["total_weight"]
        self._fitted = self.total_weight > 0


@dataclass
class FeatureConfig:
    """Configuration for a single feature.

    Attributes:
        name: Name of the feature (e.g., "log_return")
        feature_type: Type identifier for the feature class (e.g., "log_return")
        params: Additional parameters for feature creation
        normalize: Whether to apply z-score normalization
        normalization_method: How to normalize:
            - "global": StandardScaler on full dataset (fast, potential lookahead bias)
            - "rolling": Rolling window z-score (causal, slower, adapts to regimes)
            - "running": Running mean/std (Welford's algorithm, causal, stable, used in Stable Baselines/RLLib)
            - "none": No normalization (let network handle it)
        rolling_window: Window size for rolling normalization (only used when method="rolling")
        reset_on_session_break: For running/rolling normalization, reset stats at session boundaries
            (e.g., overnight/weekend gaps) to prevent cross-session contamination.
        session_break_threshold_hours: Minimum time gap (in hours) to consider a session break.
            Default: 1.0 hour (accounts for lunch gaps vs overnight gaps).
        use_time_weights: If True, weight observations by time delta between events
            (continuous-time normalization). If False, weight by event count (event-time).
            Only applies to "running" normalization. Default: False.
        output_name: Optional custom output column name
        domain: Feature domain tag used for experiment-mode validation.
            Supported values: "shared", "mft", "hft".
    """

    name: str
    feature_type: str
    params: dict[str, Any] | None = None
    normalize: bool = True
    normalization_method: str = NormalizationMethod.RUNNING  # Default to running (causal)
    rolling_window: int = 1000
    reset_on_session_break: bool = True  # Reset stats at overnight/weekend gaps
    session_break_threshold_hours: float = 1.0  # 1 hour gap = session break
    use_time_weights: bool = False  # Default: event-based
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
        self.scaler: (
            StandardScaler | RollingWindowScaler | RunningMeanStd | TimeWeightedRunningMeanStd | None
        ) = None
        if config.normalize and config.normalization_method == NormalizationMethod.GLOBAL:
            self.scaler = StandardScaler()
        elif config.normalize and config.normalization_method == NormalizationMethod.ROLLING:
            self.scaler = RollingWindowScaler(window=config.rolling_window)
        elif config.normalize and config.normalization_method == NormalizationMethod.RUNNING:
            if config.use_time_weights:
                self.scaler = TimeWeightedRunningMeanStd()
            else:
                self.scaler = RunningMeanStd()

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
            elif isinstance(self.scaler, (RollingWindowScaler, RunningMeanStd, TimeWeightedRunningMeanStd)):
                # RollingWindowScaler, RunningMeanStd, TimeWeightedRunningMeanStd: fit updates stats
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
            if isinstance(self.scaler, TimeWeightedRunningMeanStd):
                # Time-weighted: transform with time-aware handling
                if self.config.reset_on_session_break:
                    return self._transform_session_aware_time_weighted(raw_values)
                else:
                    normalized = self.scaler.transform(raw_values)
                    return pd.Series(normalized, index=raw_values.index, dtype=float)
            elif isinstance(self.scaler, RunningMeanStd) and self.config.reset_on_session_break:
                # Session-aware running normalization: reset at overnight/weekend gaps
                return self._transform_session_aware(raw_values)
            elif isinstance(self.scaler, (RollingWindowScaler, RunningMeanStd)):
                # RollingWindowScaler and non-session-aware RunningMeanStd handle NaNs internally
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

    def _transform_session_aware(self, raw_values: pd.Series) -> pd.Series:
        """Transform with session resets for running normalization.

        Detects session breaks (overnight/weekend gaps) and resets running stats
        at each boundary. This prevents overnight gap prices from distorting
        morning normalization statistics.

        Args:
            raw_values: Raw feature values with DatetimeIndex

        Returns:
            Normalized series with same index as input
        """
        if not isinstance(raw_values.index, pd.DatetimeIndex):
            # Fallback: no session detection for non-datetime index
            normalized = self.scaler.transform(raw_values)
            return pd.Series(normalized, index=raw_values.index, dtype=float)

        from trading_rl.features.utils import detect_session_breaks

        # Detect session breaks
        session_starts = detect_session_breaks(
            raw_values.index,
            threshold_hours=self.config.session_break_threshold_hours,
        )

        # Normalize each session independently
        all_normalized = []

        for i in range(len(session_starts)):
            start_idx = session_starts[i]
            end_idx = session_starts[i + 1] if i + 1 < len(session_starts) else len(raw_values)

            if start_idx >= len(raw_values):
                break

            session_data = raw_values.iloc[start_idx:end_idx]

            # Reset running stats for new session
            self.scaler.reset()

            # Normalize each row from session-local stats observed so far.
            normalized_session = self._transform_running_session_online(session_data)
            all_normalized.append(normalized_session)

        # Concatenate all sessions
        if all_normalized:
            result = pd.concat(all_normalized)
            return result
        else:
            # Fallback: no sessions detected
            normalized = self.scaler.transform(raw_values)
            return pd.Series(normalized, index=raw_values.index, dtype=float)

    def _transform_session_aware_time_weighted(self, raw_values: pd.Series) -> pd.Series:
        """Transform with session resets for time-weighted running normalization.

        Similar to _transform_session_aware but uses time-weighted statistics.

        Args:
            raw_values: Raw feature values with DatetimeIndex

        Returns:
            Normalized series with same index as input
        """
        if not isinstance(raw_values.index, pd.DatetimeIndex):
            # Fallback: no session detection for non-datetime index
            normalized = self.scaler.transform(raw_values)
            return pd.Series(normalized, index=raw_values.index, dtype=float)

        from trading_rl.features.utils import detect_session_breaks

        # Detect session breaks
        session_starts = detect_session_breaks(
            raw_values.index,
            threshold_hours=self.config.session_break_threshold_hours,
        )

        # Normalize each session independently
        all_normalized = []

        for i in range(len(session_starts)):
            start_idx = session_starts[i]
            end_idx = session_starts[i + 1] if i + 1 < len(session_starts) else len(raw_values)

            if start_idx >= len(raw_values):
                break

            session_data = raw_values.iloc[start_idx:end_idx]

            # Reset running stats for new session
            self.scaler.reset()

            # Normalize each row from session-local stats observed so far.
            normalized_session = self._transform_time_weighted_session_online(
                session_data
            )
            all_normalized.append(normalized_session)

        # Concatenate all sessions
        if all_normalized:
            result = pd.concat(all_normalized)
            return result
        else:
            # Fallback: no sessions detected
            normalized = self.scaler.transform(raw_values)
            return pd.Series(normalized, index=raw_values.index, dtype=float)

    def _transform_running_session_online(self, session_data: pd.Series) -> pd.Series:
        """Normalize one session using cumulative stats within that session."""
        normalized_values: list[float] = []

        for value in session_data.astype(float):
            if pd.isna(value):
                normalized_values.append(0.0)
                continue

            self.scaler.update(np.asarray([value], dtype=float))
            count = getattr(self.scaler, "count", 0)
            if count < 2:
                normalized = 0.0
            else:
                normalized = (value - self.scaler.mean) / np.sqrt(
                    self.scaler.var + self.scaler.epsilon
                )
                if not np.isfinite(normalized):
                    normalized = 0.0

            normalized_values.append(float(normalized))

        return pd.Series(normalized_values, index=session_data.index, dtype=float)

    def _transform_time_weighted_session_online(
        self, session_data: pd.Series
    ) -> pd.Series:
        """Normalize one session using cumulative time-weighted stats."""
        normalized_values: list[float] = []
        values = session_data.astype(float)

        if isinstance(values.index, pd.DatetimeIndex):
            weights = values.index.to_series().diff().dt.total_seconds().fillna(1.0)
            weights = weights.clip(lower=0.0).to_numpy(dtype=float)
        else:
            weights = np.ones(len(values), dtype=float)

        for value, weight in zip(values, weights):
            if pd.isna(value):
                normalized_values.append(0.0)
                continue

            self.scaler.update(
                np.asarray([value], dtype=float),
                np.asarray([weight], dtype=float),
            )
            total_weight = getattr(self.scaler, "total_weight", 0.0)
            if total_weight <= weight:
                normalized = 0.0
            else:
                normalized = (value - self.scaler.mean) / np.sqrt(
                    self.scaler.var + self.scaler.epsilon
                )
                if not np.isfinite(normalized):
                    normalized = 0.0

            normalized_values.append(float(normalized))

        return pd.Series(normalized_values, index=session_data.index, dtype=float)

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
