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
        ret = np.log(df["close"] / df["close"].shift(1))
        # -inf/inf (close == 0 or a zero-to-nonzero jump) is a data anomaly,
        # not a "no move": left unguarded, the normalization pipeline's
        # inf-replace silently turns the largest possible price move into a
        # z-score of exactly 0.0, indistinguishable from "at the mean."
        non_finite = ret.notna() & ~np.isfinite(ret)
        if non_finite.any():
            raise ValueError(
                f"log_return: non-finite value(s) (close is 0 at the current "
                f"or previous row) at {list(ret.index[non_finite][:5])}"
            )
        return ret.fillna(0)


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

    Resets at session boundaries (see detect_session_breaks), using the same
    session_break_threshold_hours as session-aware normalization. Without
    this, a single compute() call over a concatenated multi-split frame
    (as prepare_data() does for caching) would anchor every row to the very
    first row of the whole frame -- e.g. validation/test rows would be
    "relative to episode start" of the training split instead of their own.

    Normalization is controlled by FeatureConfig.normalization_method:
    - "none": raw ratio (e.g. 1.05 = +5% from start) — no look-ahead bias
    - "running": causal z-score via Welford's algorithm — no look-ahead bias
    - "rolling": causal rolling window z-score — no look-ahead bias
    - "global": StandardScaler fit on training data — mild look-ahead (train set only)
    """

    def required_columns(self) -> list[str]:
        return ["close"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Compute cumulative trend as price ratio relative to each session's start."""
        from trading_rl.features.utils import detect_session_breaks

        close = df["close"]
        session_starts = detect_session_breaks(
            df.index, threshold_hours=self.config.session_break_threshold_hours
        )
        result = pd.Series(index=close.index, dtype=float)
        for i, start_idx in enumerate(session_starts):
            end_idx = (
                session_starts[i + 1] if i + 1 < len(session_starts) else len(close)
            )
            session_close = close.iloc[start_idx:end_idx]
            result.iloc[start_idx:end_idx] = session_close / session_close.iloc[0]
        return result


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
        """Compute RSI with configurable period using Wilder's exponential smoothing."""
        period = self.config.params.get("period", 14)

        # Calculate price changes
        delta = df["close"].diff()

        # Separate gains and losses
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)

        # Wilder's smoothing: alpha = 1/period
        avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

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
