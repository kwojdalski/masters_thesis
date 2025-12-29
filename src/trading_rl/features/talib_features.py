"""TA-Lib technical analysis features.

This module provides 20 popular technical indicators from TA-Lib library.
All features are automatically normalized using z-score unless specified otherwise.

Categories:
- Trend: SMA, EMA, WMA, DEMA, TEMA
- Momentum: RSI, MACD, MOM, ROC, CMO, WILLR, CCI
- Volatility: ATR, NATR, BBANDS
- Volume: OBV, AD, ADOSC
- Trend Strength: ADX, AROON
"""

import numpy as np
import pandas as pd
import talib

from trading_rl.features.base import Feature
from trading_rl.features.registry import register_feature


# =============================================================================
# TREND INDICATORS
# =============================================================================


@register_feature("sma")
class SMAFeature(Feature):
    """Simple Moving Average.

    Parameters:
        period: Lookback period (default: 20)
    """

    def required_columns(self) -> list[str]:
        return ["close"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        period = self.config.params.get("period", 20)
        sma = talib.SMA(df["close"].values, timeperiod=period)
        # Return as ratio to close for stationarity
        return pd.Series(sma / df["close"].values - 1, index=df.index).fillna(0)


@register_feature("ema")
class EMAFeature(Feature):
    """Exponential Moving Average.

    Parameters:
        period: Lookback period (default: 20)
    """

    def required_columns(self) -> list[str]:
        return ["close"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        period = self.config.params.get("period", 20)
        ema = talib.EMA(df["close"].values, timeperiod=period)
        # Return as ratio to close for stationarity
        return pd.Series(ema / df["close"].values - 1, index=df.index).fillna(0)


@register_feature("wma")
class WMAFeature(Feature):
    """Weighted Moving Average.

    Parameters:
        period: Lookback period (default: 20)
    """

    def required_columns(self) -> list[str]:
        return ["close"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        period = self.config.params.get("period", 20)
        wma = talib.WMA(df["close"].values, timeperiod=period)
        return pd.Series(wma / df["close"].values - 1, index=df.index).fillna(0)


@register_feature("dema")
class DEMAFeature(Feature):
    """Double Exponential Moving Average.

    Parameters:
        period: Lookback period (default: 20)
    """

    def required_columns(self) -> list[str]:
        return ["close"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        period = self.config.params.get("period", 20)
        dema = talib.DEMA(df["close"].values, timeperiod=period)
        return pd.Series(dema / df["close"].values - 1, index=df.index).fillna(0)


@register_feature("tema")
class TEMAFeature(Feature):
    """Triple Exponential Moving Average.

    Parameters:
        period: Lookback period (default: 20)
    """

    def required_columns(self) -> list[str]:
        return ["close"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        period = self.config.params.get("period", 20)
        tema = talib.TEMA(df["close"].values, timeperiod=period)
        return pd.Series(tema / df["close"].values - 1, index=df.index).fillna(0)


# =============================================================================
# MOMENTUM INDICATORS
# =============================================================================


@register_feature("talib_rsi")
class TALibRSIFeature(Feature):
    """Relative Strength Index (TA-Lib version).

    Parameters:
        period: Lookback period (default: 14)
    """

    def required_columns(self) -> list[str]:
        return ["close"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        period = self.config.params.get("period", 14)
        rsi = talib.RSI(df["close"].values, timeperiod=period)
        # Normalize to [-1, 1] range
        return pd.Series((rsi - 50) / 50, index=df.index).fillna(0)


@register_feature("macd")
class MACDFeature(Feature):
    """MACD (Moving Average Convergence Divergence).

    Returns the MACD line (not signal or histogram).

    Parameters:
        fastperiod: Fast EMA period (default: 12)
        slowperiod: Slow EMA period (default: 26)
        signalperiod: Signal line period (default: 9)
    """

    def required_columns(self) -> list[str]:
        return ["close"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        fastperiod = self.config.params.get("fastperiod", 12)
        slowperiod = self.config.params.get("slowperiod", 26)
        signalperiod = self.config.params.get("signalperiod", 9)

        macd, signal, hist = talib.MACD(
            df["close"].values,
            fastperiod=fastperiod,
            slowperiod=slowperiod,
            signalperiod=signalperiod,
        )
        # Return MACD line normalized by close price
        return pd.Series(macd / df["close"].values, index=df.index).fillna(0)


@register_feature("mom")
class MOMFeature(Feature):
    """Momentum.

    Parameters:
        period: Lookback period (default: 10)
    """

    def required_columns(self) -> list[str]:
        return ["close"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        period = self.config.params.get("period", 10)
        mom = talib.MOM(df["close"].values, timeperiod=period)
        # Normalize by close price
        return pd.Series(mom / df["close"].values, index=df.index).fillna(0)


@register_feature("roc")
class ROCFeature(Feature):
    """Rate of Change.

    Parameters:
        period: Lookback period (default: 10)
    """

    def required_columns(self) -> list[str]:
        return ["close"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        period = self.config.params.get("period", 10)
        roc = talib.ROC(df["close"].values, timeperiod=period)
        # ROC already returns percentage, just convert to ratio
        return pd.Series(roc / 100, index=df.index).fillna(0)


@register_feature("cmo")
class CMOFeature(Feature):
    """Chandra Momentum Oscillator.

    Parameters:
        period: Lookback period (default: 14)
    """

    def required_columns(self) -> list[str]:
        return ["close"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        period = self.config.params.get("period", 14)
        cmo = talib.CMO(df["close"].values, timeperiod=period)
        # CMO ranges from -100 to +100, normalize to [-1, 1]
        return pd.Series(cmo / 100, index=df.index).fillna(0)


@register_feature("willr")
class WILLRFeature(Feature):
    """Williams %R.

    Parameters:
        period: Lookback period (default: 14)
    """

    def required_columns(self) -> list[str]:
        return ["high", "low", "close"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        period = self.config.params.get("period", 14)
        willr = talib.WILLR(
            df["high"].values, df["low"].values, df["close"].values, timeperiod=period
        )
        # WILLR ranges from -100 to 0, normalize to [-1, 0]
        return pd.Series(willr / 100, index=df.index).fillna(0)


@register_feature("cci")
class CCIFeature(Feature):
    """Commodity Channel Index.

    Parameters:
        period: Lookback period (default: 14)
    """

    def required_columns(self) -> list[str]:
        return ["high", "low", "close"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        period = self.config.params.get("period", 14)
        cci = talib.CCI(
            df["high"].values, df["low"].values, df["close"].values, timeperiod=period
        )
        # CCI typically ranges -200 to +200, clip and normalize
        return pd.Series(np.clip(cci, -200, 200) / 200, index=df.index).fillna(0)


# =============================================================================
# VOLATILITY INDICATORS
# =============================================================================


@register_feature("atr")
class ATRFeature(Feature):
    """Average True Range.

    Parameters:
        period: Lookback period (default: 14)
    """

    def required_columns(self) -> list[str]:
        return ["high", "low", "close"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        period = self.config.params.get("period", 14)
        atr = talib.ATR(
            df["high"].values, df["low"].values, df["close"].values, timeperiod=period
        )
        # Normalize by close price
        return pd.Series(atr / df["close"].values, index=df.index).fillna(0)


@register_feature("natr")
class NATRFeature(Feature):
    """Normalized Average True Range.

    Parameters:
        period: Lookback period (default: 14)
    """

    def required_columns(self) -> list[str]:
        return ["high", "low", "close"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        period = self.config.params.get("period", 14)
        natr = talib.NATR(
            df["high"].values, df["low"].values, df["close"].values, timeperiod=period
        )
        # NATR already normalized as percentage
        return pd.Series(natr / 100, index=df.index).fillna(0)


@register_feature("bbands")
class BBANDSFeature(Feature):
    """Bollinger Bands - returns %B (position within bands).

    Parameters:
        period: Lookback period (default: 20)
        nbdevup: Upper band deviation (default: 2)
        nbdevdn: Lower band deviation (default: 2)
    """

    def required_columns(self) -> list[str]:
        return ["close"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        period = self.config.params.get("period", 20)
        nbdevup = self.config.params.get("nbdevup", 2)
        nbdevdn = self.config.params.get("nbdevdn", 2)

        upper, middle, lower = talib.BBANDS(
            df["close"].values,
            timeperiod=period,
            nbdevup=nbdevup,
            nbdevdn=nbdevdn,
        )

        # Return %B: position within bands (0 = lower, 1 = upper)
        percent_b = (df["close"].values - lower) / (upper - lower + 1e-8)
        # Normalize to [-1, 1] range (0.5 = middle band)
        return pd.Series((percent_b - 0.5) * 2, index=df.index).fillna(0)


# =============================================================================
# VOLUME INDICATORS
# =============================================================================


@register_feature("obv")
class OBVFeature(Feature):
    """On Balance Volume.

    Note: OBV is cumulative, so we return rate of change instead.
    """

    def required_columns(self) -> list[str]:
        return ["close", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        obv = talib.OBV(df["close"].values, df["volume"].values)
        # Return rate of change to make it stationary
        obv_series = pd.Series(obv, index=df.index)
        return obv_series.pct_change().fillna(0)


@register_feature("ad")
class ADFeature(Feature):
    """Accumulation/Distribution Line.

    Note: AD is cumulative, so we return rate of change instead.
    """

    def required_columns(self) -> list[str]:
        return ["high", "low", "close", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        ad = talib.AD(
            df["high"].values,
            df["low"].values,
            df["close"].values,
            df["volume"].values,
        )
        # Return rate of change to make it stationary
        ad_series = pd.Series(ad, index=df.index)
        return ad_series.pct_change().fillna(0)


@register_feature("adosc")
class ADOSCFeature(Feature):
    """Chaikin A/D Oscillator.

    Parameters:
        fastperiod: Fast period (default: 3)
        slowperiod: Slow period (default: 10)
    """

    def required_columns(self) -> list[str]:
        return ["high", "low", "close", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        fastperiod = self.config.params.get("fastperiod", 3)
        slowperiod = self.config.params.get("slowperiod", 10)

        adosc = talib.ADOSC(
            df["high"].values,
            df["low"].values,
            df["close"].values,
            df["volume"].values,
            fastperiod=fastperiod,
            slowperiod=slowperiod,
        )
        # Normalize by volume
        return pd.Series(
            adosc / (df["volume"].values + 1e-8), index=df.index
        ).fillna(0)


# =============================================================================
# TREND STRENGTH INDICATORS
# =============================================================================


@register_feature("adx")
class ADXFeature(Feature):
    """Average Directional Index.

    Parameters:
        period: Lookback period (default: 14)
    """

    def required_columns(self) -> list[str]:
        return ["high", "low", "close"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        period = self.config.params.get("period", 14)
        adx = talib.ADX(
            df["high"].values, df["low"].values, df["close"].values, timeperiod=period
        )
        # ADX ranges 0-100, normalize to [0, 1]
        return pd.Series(adx / 100, index=df.index).fillna(0)


@register_feature("aroon")
class AROONFeature(Feature):
    """Aroon Oscillator (Aroon Up - Aroon Down).

    Parameters:
        period: Lookback period (default: 14)
    """

    def required_columns(self) -> list[str]:
        return ["high", "low"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        period = self.config.params.get("period", 14)
        aroon_down, aroon_up = talib.AROON(
            df["high"].values, df["low"].values, timeperiod=period
        )
        # Return oscillator: difference normalized to [-1, 1]
        oscillator = (aroon_up - aroon_down) / 100
        return pd.Series(oscillator, index=df.index).fillna(0)
