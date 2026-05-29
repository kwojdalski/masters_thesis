from __future__ import annotations

from typing import Any
from pathlib import Path

import numpy as np
import pandas as pd

from ._types import DEFAULT_SYNTHETIC_START_DATE
from ._io import log_dataset_summary


def generate_upward_drift_pattern(
    output_dir: Path,
    output_file: str,
    logger: Any,
    *,
    n_samples: int = 1000,
    base_price: float = 50000.0,
    drift_rate: float = 0.015,
    volatility: float = 0.0005,
    pullback_floor: float = 0.995,
    start_date: str = DEFAULT_SYNTHETIC_START_DATE,
) -> pd.DataFrame:
    """Generate synthetic OHLCV data with strong upward drift and minimal volatility."""
    logger.info(
        "Generating upward drift pattern -> samples={}, base_price={}, "
        "drift_rate={}, volatility={}",
        n_samples, base_price, drift_rate, volatility,
    )

    dates = pd.date_range(start=pd.to_datetime(start_date), periods=n_samples, freq="h")
    steps = np.arange(n_samples)
    drift_curve = base_price * np.exp(drift_rate * steps)

    noise = np.random.normal(0, volatility, n_samples)
    close_prices = (
        pd.Series(drift_curve * (1 + noise)).rolling(window=5, min_periods=1).mean().to_numpy()
    )

    # Enforce shallow pullbacks relative to the drift curve
    floor = pullback_floor * np.maximum.accumulate(drift_curve)
    close_prices = np.maximum(close_prices, floor)
    close_prices[0] = base_price

    spread_scale = max(volatility * 10, 0.001)
    highs = close_prices * (1 + spread_scale * (1 + 0.1 * np.random.uniform(-1, 1, n_samples)))
    lows = close_prices * (1 - spread_scale * (1 + 0.1 * np.random.uniform(-1, 1, n_samples)))

    opens = np.roll(close_prices, 1)
    opens[0] = close_prices[0]
    opens = opens + np.random.normal(0, volatility * np.mean(close_prices) * 0.1, n_samples)
    opens = np.clip(opens, lows, highs)
    close_prices = np.clip(close_prices, lows, highs)

    base_volume = 1_000_000
    volumes = (
        base_volume * (1 + 0.8 * (steps / max(steps[-1], 1))) * (1 + np.random.normal(0, 0.05, n_samples))
    )
    volumes = np.maximum(volumes, base_volume * 0.5)

    df = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": close_prices, "volume": volumes},
        index=dates,
    ).abs()

    output_path = output_dir / output_file
    df.to_parquet(output_path)
    log_dataset_summary(df, output_path, context="Upward drift pattern", logger=logger)
    total_return = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100
    logger.info(
        "Upward drift stats -> total return {}%, volatility {}",
        total_return, df["close"].pct_change().std(),
    )
    return df
