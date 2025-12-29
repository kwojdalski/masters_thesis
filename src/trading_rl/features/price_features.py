"""Price-based features."""

import pandas as pd

from trading_rl.features.base import Feature
from trading_rl.features.registry import register_feature


@register_feature("log_return")
class LogReturnFeature(Feature):
    """Log return feature: (close_t / close_t-1) - 1.

    Captures price momentum and direction.
    """

    def required_columns(self) -> list[str]:
        return ["close"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Compute log returns."""
        return (df["close"] / df["close"].shift(1) - 1).fillna(0)


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

    Captures long-term trend, normalized to [0, 1] range.
    Note: Uses min-max normalization instead of z-score to preserve trend direction.
    """

    def __init__(self, config):
        # Trend uses min-max normalization, not z-score
        config.normalize = False
        super().__init__(config)

    def required_columns(self) -> list[str]:
        return ["close"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Compute cumulative trend with min-max normalization."""
        # Price relative to initial price
        trend = df["close"] / df["close"].iloc[0]

        # Min-max normalization to [0, 1] instead of z-score
        trend_min = trend.min()
        trend_max = trend.max()
        normalized = (trend - trend_min) / (trend_max - trend_min + 1e-8)

        return normalized


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
