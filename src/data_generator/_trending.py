from __future__ import annotations

from typing import Any
from pathlib import Path

import numpy as np
import pandas as pd

from ._types import DEFAULT_SYNTHETIC_START_DATE
from ._io import log_dataset_summary


def generate_trending_pattern(
    output_dir: Path,
    output_file: str,
    logger: Any,
    *,
    n_samples: int = 500,
    base_price: float = 50000.0,
    n_trends: int = 3,
    min_trend_length: int = 50,
    max_trend_length: int = 150,
    trend_strength_range: tuple[float, float] = (0.5, 2.0),
    volatility: float = 0.03,
    consolidation_prob: float = 0.2,
    start_date: str = DEFAULT_SYNTHETIC_START_DATE,
) -> pd.DataFrame:
    """Generate synthetic OHLCV data with sustained trend and consolidation segments."""
    logger.info(
        "Generating trending pattern -> samples={}, base_price={}, n_trends={}",
        n_samples, base_price, n_trends,
    )

    dates = pd.date_range(start=pd.to_datetime(start_date), periods=n_samples, freq="h")
    prices = np.zeros(n_samples)
    prices[0] = base_price

    remaining = n_samples - 1
    segments: list[dict] = []
    for i in range(n_trends):
        if i == n_trends - 1:
            length = remaining
        else:
            max_len = min(max_trend_length, remaining - (n_trends - i - 1) * min_trend_length)
            length = int(np.random.randint(min_trend_length, max_len + 1))
        direction = np.random.choice([-1, 1])
        strength = float(np.random.uniform(*trend_strength_range)) * direction
        segments.append({
            "length": length,
            "strength": strength,
            "is_consolidation": np.random.random() < consolidation_prob and i > 0,
        })
        remaining -= length
        if remaining <= 0:
            break

    current_idx = 1
    for seg in segments:
        end_idx = min(current_idx + seg["length"], n_samples)
        seg_len = end_idx - current_idx
        if seg["is_consolidation"]:
            base_level = prices[current_idx - 1]
            for j in range(seg_len):
                if current_idx + j >= n_samples:
                    break
                prices[current_idx + j] = (
                    prices[current_idx + j - 1]
                    + np.random.normal(0, volatility * base_level)
                    + np.random.normal(0, 0.1 * volatility * base_level)
                )
        else:
            trend_per_step = seg["strength"] * base_price / 100
            for j in range(seg_len):
                if current_idx + j >= n_samples:
                    break
                current_price = prices[current_idx + j - 1]
                noise = np.random.normal(0, volatility * max(abs(current_price), base_price * 0.1))
                prices[current_idx + j] = max(current_price + trend_per_step + noise, base_price * 0.05)
        current_idx = end_idx
        if current_idx >= n_samples:
            break

    prices = np.maximum(prices, base_price * 0.1)

    price_changes = np.abs(np.diff(prices, prepend=prices[0]))
    highs = prices + 0.3 * price_changes + np.abs(np.random.normal(0, volatility * prices * 0.2, n_samples))
    lows = prices - 0.3 * price_changes - np.abs(np.random.normal(0, volatility * prices * 0.2, n_samples))

    opens = np.roll(prices, 1)
    opens[0] = prices[0]
    opens = opens + np.random.normal(0, 0.3 * volatility * prices, n_samples)
    opens = np.clip(opens, lows, highs)
    prices = np.clip(prices, lows, highs)

    base_volume = 1_000_000
    momentum = np.abs(np.diff(prices, prepend=prices[0]))
    volumes = (
        base_volume * (1 + 3 * momentum / (np.mean(momentum) or 1.0))
        + np.random.normal(0, base_volume * 0.15, n_samples)
    )
    volumes = np.maximum(volumes, base_volume * 0.1)

    df = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": prices, "volume": volumes},
        index=dates,
    ).abs()

    output_path = output_dir / output_file
    df.to_parquet(output_path)
    log_dataset_summary(df, output_path, context="Trending pattern", logger=logger)
    total_return = (prices[-1] / prices[0] - 1) * 100
    max_drawdown = np.min(prices / np.maximum.accumulate(prices) - 1) * 100
    logger.info("Momentum stats -> total return {:.2f}%, max drawdown {:.2f}%", total_return, max_drawdown)
    logger.info("Trends identified: {} | Strategy hint -> ride momentum, manage reversals", len(segments))
    return df
