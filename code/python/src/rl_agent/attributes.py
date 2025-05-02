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
    # Create a Series of the same length as quotes, filled with NaN
    roi = pd.Series(np.nan, index=quotes.index)

    # Calculate (current_price - price_n_periods_ago) / price_n_periods_ago
    # in a vectorized way
    shifted = quotes.shift(period)
    roi[period:] = (quotes[period:] - shifted[period:]) / shifted[period:]

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

    return df_attributes
