"""Volume-based features."""

import numpy as np
import pandas as pd

from trading_rl.features.base import Feature
from trading_rl.features.registry import register_feature


@register_feature("log_volume")
class LogVolumeFeature(Feature):
    """Log-transformed volume: log1p(volume).

    Log transform handles the heavy right tail of volume distribution.
    log1p = log(1 + x) handles zero volumes safely.
    """

    def required_columns(self) -> list[str]:
        return ["volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Compute log-transformed volume."""
        return np.log1p(df["volume"])


@register_feature("volume_change")
class VolumeChangeFeature(Feature):
    """Volume change: (volume_t / volume_t-1) - 1.

    Captures changes in trading activity.
    """

    def required_columns(self) -> list[str]:
        return ["volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Compute volume change."""
        return (df["volume"] / df["volume"].shift(1).replace(0, 1) - 1).fillna(0)


@register_feature("volume_ma_ratio")
class VolumeMAFeature(Feature):
    """Volume relative to moving average.

    Detects unusual trading activity relative to recent history.
    """

    def required_columns(self) -> list[str]:
        return ["volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Compute volume relative to its moving average."""
        window = self.config.params.get("window", 20)

        volume_ma = df["volume"].rolling(window=window, min_periods=1).mean()
        return (df["volume"] / (volume_ma + 1e-8) - 1).fillna(0)
