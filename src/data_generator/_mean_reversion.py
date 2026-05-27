from __future__ import annotations

from logging import Logger
from pathlib import Path

import numpy as np
import pandas as pd

from ._types import DEFAULT_SYNTHETIC_START_DATE
from ._io import log_dataset_summary


def generate_mean_reversion_pattern(
    output_dir: Path,
    output_file: str,
    logger: Logger,
    *,
    n_samples: int = 500,
    mean_price: float = 50000.0,
    reversion_strength: float = 0.1,
    volatility: float = 0.05,
    shock_probability: float = 0.02,
    shock_magnitude: float = 0.15,
    start_date: str = DEFAULT_SYNTHETIC_START_DATE,
) -> pd.DataFrame:
    """Generate synthetic OHLCV data with mean-reverting price dynamics."""
    logger.info(
        "Generating mean reversion pattern -> samples=%s, mean_price=%.2f, reversion_strength=%.2f",
        n_samples, mean_price, reversion_strength,
    )

    dates = pd.date_range(start=pd.to_datetime(start_date), periods=n_samples, freq="h")
    prices = np.zeros(n_samples)
    prices[0] = mean_price * (1 + np.random.normal(0, 0.1))

    for i in range(1, n_samples):
        reversion = -reversion_strength * (prices[i - 1] - mean_price)
        shock = (
            np.random.choice([-1, 1]) * shock_magnitude * mean_price
            if np.random.random() < shock_probability
            else 0.0
        )
        noise = np.random.normal(0, volatility * mean_price)
        prices[i] = max(prices[i - 1] + reversion + shock + noise, mean_price * 0.1)

    high_var = np.abs(np.random.normal(0, volatility * mean_price * 0.3, n_samples))
    low_var = np.abs(np.random.normal(0, volatility * mean_price * 0.3, n_samples))
    highs = prices + high_var
    lows = prices - low_var

    opens = np.roll(prices, 1)
    opens[0] = prices[0]
    opens = opens + np.random.normal(0, 0.5 * volatility * mean_price, n_samples)
    opens = np.clip(opens, lows, highs)
    prices = np.clip(prices, lows, highs)

    base_volume = 1_000_000
    price_deviation = np.abs(prices - mean_price) / mean_price
    volumes = (
        base_volume * (1 + 2 * price_deviation)
        + np.random.normal(0, base_volume * 0.1, n_samples)
    )
    volumes = np.maximum(volumes, base_volume * 0.1)

    df = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": prices, "volume": volumes},
        index=dates,
    ).abs()

    output_path = output_dir / output_file
    df.to_parquet(output_path)
    log_dataset_summary(df, output_path, context="Mean reversion pattern", logger=logger)
    mean_deviation = np.mean(np.abs(prices - mean_price))
    max_deviation = np.max(np.abs(prices - mean_price))
    logger.info(
        "Mean price target %.2f, actual %.2f (mean deviation %.2f, max deviation %.2f)",
        mean_price, df["close"].mean(), mean_deviation, max_deviation,
    )
    logger.info(
        "Strategy hint -> accumulate below %.0f, distribute above %.0f",
        mean_price * 0.98, mean_price * 1.02,
    )
    return df
