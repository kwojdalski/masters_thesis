"""Volatility-based features."""

import numpy as np
import pandas as pd

from trading_rl.features.base import Feature
from trading_rl.features.registry import register_feature


@register_feature("realized_volatility")
class RealizedVolatilityFeature(Feature):
    """Rolling realized volatility from log returns.

    Parameters:
        window: Lookback window (default: 20)
    """

    def required_columns(self) -> list[str]:
        return ["close"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        window = int(self.config.params.get("window", 20))
        log_returns = np.log(df["close"] / df["close"].shift(1)).fillna(0.0)
        # Keep realized volatility in per-period units (not annualized).
        return log_returns.rolling(window=window, min_periods=1).std().fillna(0.0)


@register_feature("volatility_ratio")
class VolatilityRatioFeature(Feature):
    """Ratio of short-horizon to long-horizon realized volatility.

    Parameters:
        short_window: Short lookback window (default: 5)
        long_window: Long lookback window (default: 60)
    """

    def required_columns(self) -> list[str]:
        return ["close"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        short_window = int(self.config.params.get("short_window", 5))
        long_window = int(self.config.params.get("long_window", 60))
        if short_window <= 0 or long_window <= 0:
            raise ValueError("short_window and long_window must be > 0")
        if short_window >= long_window:
            raise ValueError("short_window must be < long_window")

        log_returns = np.log(df["close"] / df["close"].shift(1)).fillna(0.0)
        rv_short = log_returns.rolling(window=short_window, min_periods=1).std()
        rv_long = log_returns.rolling(window=long_window, min_periods=1).std()
        ratio = rv_short / (rv_long + 1e-8)
        return ratio.fillna(0.0)
