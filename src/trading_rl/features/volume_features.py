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


@register_feature("amihud_illiquidity")
class AmihudIlliquidityFeature(Feature):
    """Amihud Illiquidity: |Return| / Volume.

    Measures price impact per unit of volume (dollar volume proxy).
    High values indicate thin, illiquid markets.

    Parameters:
        window: Rolling window for smoothing (default: 1, i.e., instantaneous)
    """

    def required_columns(self) -> list[str]:
        return ["close", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Compute Amihud illiquidity."""
        window = int(self.config.params.get("window", 1))

        # Absolute log return
        abs_log_return = np.abs(np.log(df["close"] / df["close"].shift(1))).fillna(0.0)

        # Basic illiquidity: |r| / Vol
        # We use a small epsilon to avoid division by zero
        illiquidity = abs_log_return / (df["volume"] + 1e-8)

        if window > 1:
            return illiquidity.rolling(window=window, min_periods=1).mean()

        return illiquidity


@register_feature("relative_volume")
class RelativeVolumeFeature(Feature):
    """Relative Volume (RVOL): Volume / Average Volume.

    Identifies breakout sessions where activity is significantly higher
    than the recent baseline.

    Parameters:
        window: Lookback window for average (default: 20)
    """

    def required_columns(self) -> list[str]:
        return ["volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Compute relative volume."""
        window = int(self.config.params.get("window", 20))

        avg_volume = df["volume"].rolling(window=window, min_periods=1).mean()
        rvol = df["volume"] / (avg_volume + 1e-8)

        return rvol.fillna(1.0)
