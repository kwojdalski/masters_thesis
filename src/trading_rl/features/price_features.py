"""Price-based features."""

import numpy as np
import pandas as pd

from trading_rl.features.base import Feature
from trading_rl.features.registry import register_feature


@register_feature("log_return")
class LogReturnFeature(Feature):
    """Log return feature: log(close_t / close_t-1).

    Captures price momentum and direction.
    """

    def required_columns(self) -> list[str]:
        return ["close"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Compute log returns."""
        return np.log(df["close"] / df["close"].shift(1)).fillna(0)


@register_feature("high")
class HighFeature(Feature):
    """High relative to close: (high / close) - 1.

    Captures intra-candle volatility to the upside.
    """

    def required_columns(self) -> list[str]:
        return ["high", "close"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Compute high relative to close."""
        return (df["high"] / df["close"] - 1).fillna(0)


@register_feature("low")
class LowFeature(Feature):
    """Low relative to close: (low / close) - 1.

    Captures intra-candle volatility to the downside.
    """

    def required_columns(self) -> list[str]:
        return ["low", "close"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Compute low relative to close."""
        return (df["low"] / df["close"] - 1).fillna(0)


@register_feature("trend")
class TrendFeature(Feature):
    """Cumulative price trend: close / initial_close.

    Captures long-term trend direction as price relative to episode start.
    Raw value is 1.0 at episode start, >1.0 for uptrend, <1.0 for downtrend.

    Normalization is controlled by FeatureConfig.normalization_method:
    - "none": raw ratio (e.g. 1.05 = +5% from start) — no look-ahead bias
    - "running": causal z-score via Welford's algorithm — no look-ahead bias
    - "rolling": causal rolling window z-score — no look-ahead bias
    - "global": StandardScaler fit on training data — mild look-ahead (train set only)
    """

    def required_columns(self) -> list[str]:
        return ["close"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Compute cumulative trend as price ratio relative to episode start."""
        return df["close"] / df["close"].iloc[0]


@register_feature("simple_return")
class SimpleReturnFeature(Feature):
    """Simple return: close.pct_change().

    Alternative to log_return for those who prefer percentage changes.
    """

    def required_columns(self) -> list[str]:
        return ["close"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Compute simple returns."""
        return df["close"].pct_change().fillna(0)


@register_feature("rsi")
class RSIFeature(Feature):
    """Relative Strength Index (RSI).

    Momentum oscillator that measures speed and magnitude of price changes.
    """

    def required_columns(self) -> list[str]:
        return ["close"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Compute RSI with configurable period."""
        period = self.config.params.get("period", 14)

        # Calculate price changes
        delta = df["close"].diff()

        # Separate gains and losses
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)

        # Calculate average gain and loss
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()

        # Calculate RS and RSI
        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))

        # Normalize to [-1, 1] range for consistency
        return (rsi - 50) / 50


@register_feature("return_lag")
class ReturnLagFeature(Feature):
    """Lagged return feature.

    Computes simple return for a column and shifts it by a specified lag.
    Useful for building autoregressive signals without leaking future data.
    """

    def required_columns(self) -> list[str]:
        column = self.config.params.get("column", "close")
        return [column]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Compute lagged simple return."""
        column = self.config.params.get("column", "close")
        lag = int(self.config.params.get("lag", 1))
        returns = df[column].pct_change().fillna(0)
        return returns.shift(lag).fillna(0)
