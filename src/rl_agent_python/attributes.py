"""
Attribute generation for financial time series analysis.

This module provides functions for calculating various technical indicators
and attributes from financial time series data.
"""

from typing import Dict, List

import numpy as np
import pandas as pd
import talib


def return_on_investment(quotes: pd.Series, period: int) -> pd.Series:
    """
    Calculate return on investment for a given period.

    Args:
        quotes: Series of price quotes
        period: Investment period

    Returns:
        Series of returns
    """
    differences = quotes.diff(period)
    roi = pd.Series(index=quotes.index, dtype=float)
    roi.iloc[period:] = differences.iloc[period:] / quotes.iloc[:-period] + 1
    roi.iloc[:period] = np.nan
    return roi


def min_max(quotes_high: pd.Series, quotes_low: pd.Series) -> pd.Series:
    """
    Calculate percentage difference between high and low quotes.

    Args:
        quotes_high: Series of high prices
        quotes_low: Series of low prices

    Returns:
        Series of percentage differences
    """
    return 100 * (quotes_high - quotes_low) / quotes_low


def attributes_all(
    df_currencies: pd.DataFrame,
    currencies: List[str],
    delay: int,
    base_series: List[str] = ["price", "return"],
) -> Dict[str, pd.DataFrame]:
    """
    Generate attributes for all currencies.

    Args:
        df_currencies: DataFrame containing currency data
        currencies: List of currency codes
        delay: Time delay for calculations
        base_series: List of base series to use

    Returns:
        Dictionary mapping currencies to their attribute DataFrames
    """
    attributes_dict = {}
    for asset in currencies:
        attributes_dict[asset] = attributes(
            [asset], {asset: df_currencies[asset]}, delay, False
        )
    return attributes_dict


def attributes(
    assets: List[str], quotes: Dict[str, pd.Series], delay: int, na_omit: bool
) -> pd.DataFrame:
    """
    Generate attributes for specified assets.

    Args:
        assets: List of asset codes
        quotes: Dictionary mapping assets to their price series
        delay: Time delay for calculations
        na_omit: Whether to omit NA values

    Returns:
        DataFrame containing calculated attributes
    """
    df_attributes = create_attributes(assets, quotes, delay)
    return df_attributes


def create_attributes(
    assets: List[str], quotes: Dict[str, pd.Series], delay: int
) -> pd.DataFrame:
    """
    Create technical indicators and attributes for the given assets.

    Args:
        assets: List of asset codes
        quotes: Dictionary mapping assets to their price series
        delay: Time delay for calculations

    Returns:
        DataFrame containing all calculated attributes
    """
    df_attributes = pd.DataFrame(index=quotes[assets[0]].index)

    for asset in assets:
        # Return on investment
        roi = return_on_investment(quotes[asset], delay)
        df_attributes[f"return_{asset}"] = roi

        # Exponential moving averages
        df_attributes[f"EMA3_return_{asset}"] = talib.EMA(roi, timeperiod=3)
        df_attributes[f"EMA10_return_{asset}"] = talib.EMA(roi, timeperiod=10)

        # Triple Exponential Moving Average (TEMA)
        price = quotes[asset]
        ema3 = talib.EMA(price, timeperiod=3)
        ema3_2 = talib.EMA(ema3, timeperiod=3)
        ema3_3 = talib.EMA(ema3_2, timeperiod=3)
        tema3 = 3 * ema3 - 3 * ema3_2 + ema3_3
        df_attributes[f"TEMA3_{asset}"] = tema3

        # MACD
        macd, macd_signal, macd_hist = talib.MACD(
            price, fastperiod=12, slowperiod=26, signalperiod=9
        )
        df_attributes[f"MACD_{asset}"] = macd
        df_attributes[f"MACDsigLine_{asset}"] = macd_signal
        df_attributes[f"MACDsignal_{asset}"] = macd_hist

        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(
            price, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
        )
        df_attributes[f"BBands20dn_{asset}"] = lower
        df_attributes[f"BBands20mavg_{asset}"] = middle
        df_attributes[f"BBands20up_{asset}"] = upper
        df_attributes[f"BBands20pctB_{asset}"] = (price - lower) / (upper - lower)
        df_attributes[f"BBands20Bdiv_{asset}"] = upper - lower
        df_attributes[f"BBands20Pdiv_{asset}"] = price - middle

        # Aroon Oscillator
        aroon_up, aroon_down = talib.AROON(price, timeperiod=14)
        df_attributes[f"AROON14up_{asset}"] = aroon_up
        df_attributes[f"AROON14dn_{asset}"] = aroon_down
        df_attributes[f"AROON14osc_{asset}"] = aroon_up - aroon_down

        # Average True Range (ATR)
        atr = talib.ATR(price, price, price, timeperiod=14)
        df_attributes[f"ATR14_{asset}"] = atr

        # Chaikin Money Flow
        cmf = talib.ADOSC(price, price, price, price, fastperiod=3, slowperiod=10)
        df_attributes[f"ChaikinMF_{asset}"] = cmf

        # Volume indicators (if volume data is available)
        if "Volume" in quotes[asset].name:
            volume = quotes[asset]
            df_attributes[f"Volume_{asset}"] = volume
            df_attributes[f"MA8_Volume_{asset}"] = talib.SMA(volume, timeperiod=8)
            df_attributes[f"MA15_Volume_{asset}"] = talib.SMA(volume, timeperiod=15)
            df_attributes[f"MA60_Volume_{asset}"] = talib.SMA(volume, timeperiod=60)

    return df_attributes
