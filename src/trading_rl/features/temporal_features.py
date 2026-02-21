"""Temporal and cyclical features for capturing time-based patterns."""

import numpy as np
import pandas as pd

from trading_rl.features.base import Feature
from trading_rl.features.registry import register_feature


@register_feature("hour_sin")
class HourSinFeature(Feature):
    """Sine component of hour-of-day cyclical encoding.

    Captures intraday patterns using cyclical encoding. Requires datetime index.
    Use together with hour_cos to get full cyclical representation.

    The hour is converted to a fraction of the day [0, 1) and then:
    sin(2π × hour_fraction)

    This ensures hour 23 is close to hour 0 in feature space.
    """

    def __init__(self, config):
        # Temporal features use cyclical encoding, not z-score normalization
        config.normalize = False
        super().__init__(config)

    def required_columns(self) -> list[str]:
        return []  # Uses index, not columns

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Compute sine of hour-of-day."""
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("hour_sin requires DataFrame with DatetimeIndex")

        # Extract hour and convert to fraction of day [0, 1)
        hour = df.index.hour
        hour_fraction = hour / 24.0

        # Apply sine transformation
        return pd.Series(
            np.sin(2 * np.pi * hour_fraction),
            index=df.index,
            name="hour_sin"
        )


@register_feature("hour_cos")
class HourCosFeature(Feature):
    """Cosine component of hour-of-day cyclical encoding.

    Captures intraday patterns using cyclical encoding. Requires datetime index.
    Use together with hour_sin to get full cyclical representation.

    The hour is converted to a fraction of the day [0, 1) and then:
    cos(2π × hour_fraction)
    """

    def __init__(self, config):
        # Temporal features use cyclical encoding, not z-score normalization
        config.normalize = False
        super().__init__(config)

    def required_columns(self) -> list[str]:
        return []  # Uses index, not columns

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Compute cosine of hour-of-day."""
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("hour_cos requires DataFrame with DatetimeIndex")

        # Extract hour and convert to fraction of day [0, 1)
        hour = df.index.hour
        hour_fraction = hour / 24.0

        # Apply cosine transformation
        return pd.Series(
            np.cos(2 * np.pi * hour_fraction),
            index=df.index,
            name="hour_cos"
        )


@register_feature("day_of_week_sin")
class DayOfWeekSinFeature(Feature):
    """Sine component of day-of-week cyclical encoding.

    Captures weekly patterns (e.g., Monday effect, weekend patterns).
    Requires datetime index. Use together with day_of_week_cos.

    Day of week (0=Monday, 6=Sunday) is converted to fraction [0, 1) and then:
    sin(2π × day_fraction)

    This ensures Sunday is close to Monday in feature space.
    """

    def __init__(self, config):
        # Temporal features use cyclical encoding, not z-score normalization
        config.normalize = False
        super().__init__(config)

    def required_columns(self) -> list[str]:
        return []  # Uses index, not columns

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Compute sine of day-of-week."""
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("day_of_week_sin requires DataFrame with DatetimeIndex")

        # Extract day of week (0=Monday, 6=Sunday) and convert to fraction [0, 1)
        day_of_week = df.index.dayofweek
        day_fraction = day_of_week / 7.0

        # Apply sine transformation
        return pd.Series(
            np.sin(2 * np.pi * day_fraction),
            index=df.index,
            name="day_of_week_sin"
        )


@register_feature("day_of_week_cos")
class DayOfWeekCosFeature(Feature):
    """Cosine component of day-of-week cyclical encoding.

    Captures weekly patterns (e.g., Monday effect, weekend patterns).
    Requires datetime index. Use together with day_of_week_sin.

    Day of week (0=Monday, 6=Sunday) is converted to fraction [0, 1) and then:
    cos(2π × day_fraction)
    """

    def __init__(self, config):
        # Temporal features use cyclical encoding, not z-score normalization
        config.normalize = False
        super().__init__(config)

    def required_columns(self) -> list[str]:
        return []  # Uses index, not columns

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Compute cosine of day-of-week."""
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("day_of_week_cos requires DataFrame with DatetimeIndex")

        # Extract day of week (0=Monday, 6=Sunday) and convert to fraction [0, 1)
        day_of_week = df.index.dayofweek
        day_fraction = day_of_week / 7.0

        # Apply cosine transformation
        return pd.Series(
            np.cos(2 * np.pi * day_fraction),
            index=df.index,
            name="day_of_week_cos"
        )


@register_feature("minute_of_hour_sin")
class MinuteOfHourSinFeature(Feature):
    """Sine component of minute-of-hour cyclical encoding.

    Captures sub-hourly patterns for high-frequency data. Requires datetime index.
    Use together with minute_of_hour_cos for full cyclical representation.

    Minute (0-59) is converted to fraction [0, 1) and then:
    sin(2π × minute_fraction)
    """

    def __init__(self, config):
        # Temporal features use cyclical encoding, not z-score normalization
        config.normalize = False
        super().__init__(config)

    def required_columns(self) -> list[str]:
        return []  # Uses index, not columns

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Compute sine of minute-of-hour."""
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("minute_of_hour_sin requires DataFrame with DatetimeIndex")

        # Extract minute and convert to fraction of hour [0, 1)
        minute = df.index.minute
        minute_fraction = minute / 60.0

        # Apply sine transformation
        return pd.Series(
            np.sin(2 * np.pi * minute_fraction),
            index=df.index,
            name="minute_of_hour_sin"
        )


@register_feature("minute_of_hour_cos")
class MinuteOfHourCosFeature(Feature):
    """Cosine component of minute-of-hour cyclical encoding.

    Captures sub-hourly patterns for high-frequency data. Requires datetime index.
    Use together with minute_of_hour_sin for full cyclical representation.

    Minute (0-59) is converted to fraction [0, 1) and then:
    cos(2π × minute_fraction)
    """

    def __init__(self, config):
        # Temporal features use cyclical encoding, not z-score normalization
        config.normalize = False
        super().__init__(config)

    def required_columns(self) -> list[str]:
        return []  # Uses index, not columns

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Compute cosine of minute-of-hour."""
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("minute_of_hour_cos requires DataFrame with DatetimeIndex")

        # Extract minute and convert to fraction of hour [0, 1)
        minute = df.index.minute
        minute_fraction = minute / 60.0

        # Apply cosine transformation
        return pd.Series(
            np.cos(2 * np.pi * minute_fraction),
            index=df.index,
            name="minute_of_hour_cos"
        )
