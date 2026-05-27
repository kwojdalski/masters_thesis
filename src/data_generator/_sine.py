from __future__ import annotations

from logging import Logger
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ._types import DEFAULT_SYNTHETIC_START_DATE
from ._io import log_dataset_summary


def generate_sine_wave_pattern(
    output_dir: Path,
    output_file: str,
    logger: Logger,
    *,
    n_periods: int = 5,
    samples_per_period: int = 100,
    base_price: float = 50000.0,
    amplitude: float = 30.0,
    trend_slope: float = 0,
    volatility: float = 0.0,
    start_date: str = DEFAULT_SYNTHETIC_START_DATE,
    freq: str = "h",
) -> pd.DataFrame:
    """Generate synthetic OHLCV data with sine wave pattern and optional upward trend."""
    logger.info(
        "Generating sine wave pattern -> periods=%s, samples_per_period=%s, "
        "amplitude=%.2f, trend_slope=%.2f, freq=%s",
        n_periods, samples_per_period, amplitude, trend_slope, freq,
    )

    total_samples = n_periods * samples_per_period
    t = np.linspace(0, 2 * np.pi * n_periods, total_samples)
    dates = pd.date_range(start=pd.to_datetime(start_date), periods=total_samples, freq=freq)

    trend = trend_slope * np.arange(total_samples)
    base_prices = base_price + trend + amplitude * np.sin(t)

    if volatility > 0:
        noise = (
            volatility * base_price
            * np.sin(2.0 * t + np.pi / 4)
        )
    else:
        noise = np.zeros_like(base_prices)
    close_prices = base_prices + noise

    close_prices = (
        pd.Series(close_prices).rolling(window=5, min_periods=1, center=False).mean().to_numpy()
    )

    variation_scale = max(amplitude * 0.05, base_price * 0.002)
    highs = close_prices + variation_scale * (0.5 + 0.3 * np.random.uniform(-1, 1, total_samples))
    lows = close_prices - variation_scale * (0.5 + 0.3 * np.random.uniform(-1, 1, total_samples))

    opens = np.roll(close_prices, 1)
    opens[0] = close_prices[0]
    gap_amplitude = 0.1 * volatility * base_price
    if gap_amplitude > 0:
        opens = opens + gap_amplitude * np.sin(t + np.pi / 6)

    opens = np.clip(opens, lows, highs)
    close_prices = np.clip(close_prices, lows, highs)

    price_changes = np.abs(np.diff(close_prices, prepend=close_prices[0]))
    price_change_scale = np.max(price_changes) or 1.0
    base_volume = 1_000_000
    volumes = base_volume * (
        1.0 + 0.5 * (price_changes / price_change_scale) * (0.5 + 0.5 * np.sin(t + np.pi / 2))
    )
    volumes = np.maximum(volumes, base_volume * 0.1)

    df = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": close_prices, "volume": volumes},
        index=dates,
    ).abs()

    output_path = output_dir / output_file
    df.to_parquet(output_path)
    log_dataset_summary(df, output_path, context="Sine wave pattern", logger=logger)
    logger.info(
        "Sine wave trading cues -> buy ≈ %.0f, sell ≈ %.0f",
        base_price - amplitude, base_price + amplitude,
    )
    logger.debug("Close price std dev: %.2f", df["close"].std())
    return df


def generate_hft_sine_wave_lob(
    output_dir: Path,
    output_file: str,
    logger: Logger,
    *,
    n_events: int = 20000,
    n_periods: int = 5,
    base_price: float = 270.0,
    amplitude: float = 5.0,
    spread: float = 0.12,
    level_spacing: float = 0.10,
    tick_size: float = 0.01,
    symbol: str = "AAPLUSD",
    start_datetime: str = "2026-02-25 14:30:00",
    session_duration_seconds: float = 23400.0,
    odd_lot_fraction: float = 0.08,
    seed: int = 42,
    price_noise_std: float = 0.01,
) -> pd.DataFrame:
    """Generate synthetic HFT LOB data matching the MBP-10 structure of real stock data."""
    logger.info(
        "Generating HFT sine wave LOB -> n_events=%s, n_periods=%s, "
        "base_price=%.2f, amplitude=%.2f, spread=%.4f",
        n_events, n_periods, base_price, amplitude, spread,
    )

    rng = np.random.default_rng(seed)
    lob_levels = 10
    half_spread = spread / 2.0

    t = np.linspace(0, 2 * np.pi * n_periods, n_events)
    mid_prices = base_price + amplitude * np.sin(t)

    # Uniform spacing with small jitter keeps all time-of-day features non-constant
    # across splits — exponential gaps would cluster events inside a single hour.
    session_duration_ns = int(session_duration_seconds * 1e9)
    step_ns = session_duration_ns // n_events
    base_offsets = np.arange(n_events, dtype=np.int64) * step_ns
    max_jitter_ns = min(step_ns // 10, int(1e9))
    jitter = rng.integers(-max_jitter_ns, max_jitter_ns + 1, n_events, dtype=np.int64)
    offsets = np.clip(base_offsets + jitter, 0, session_duration_ns - 1)
    start_ns = pd.Timestamp(start_datetime, tz="UTC").value
    timestamps_ns = start_ns + offsets

    bid_px = np.empty((n_events, lob_levels))
    ask_px = np.empty((n_events, lob_levels))
    bid_sz = np.empty((n_events, lob_levels), dtype=np.uint32)
    ask_sz = np.empty((n_events, lob_levels), dtype=np.uint32)
    bid_ct = np.empty((n_events, lob_levels), dtype=np.uint32)
    ask_ct = np.empty((n_events, lob_levels), dtype=np.uint32)

    for lvl in range(lob_levels):
        bid_noise = rng.normal(0.0, price_noise_std, n_events)
        ask_noise = rng.normal(0.0, price_noise_std, n_events)
        raw_bid = mid_prices - half_spread - lvl * level_spacing + bid_noise
        raw_ask = mid_prices + half_spread + lvl * level_spacing + ask_noise
        bid_px[:, lvl] = np.round(raw_bid / tick_size) * tick_size
        ask_px[:, lvl] = np.round(raw_ask / tick_size) * tick_size

        base_sz = max(50, 300 - lvl * 25)
        bid_sz[:, lvl] = rng.integers(max(1, base_sz - 50), base_sz + 150, n_events, dtype=np.uint32)
        ask_sz[:, lvl] = rng.integers(max(1, base_sz - 50), base_sz + 150, n_events, dtype=np.uint32)
        bid_ct[:, lvl] = rng.integers(1, 5, n_events, dtype=np.uint32)
        ask_ct[:, lvl] = rng.integers(1, 5, n_events, dtype=np.uint32)

    # Action distribution matches real AAPL: A ~41%, C ~37%, T ~22%
    actions = rng.choice(np.array(["A", "C", "T"], dtype=object), size=n_events, p=[0.41, 0.37, 0.22])
    is_trade = actions == "T"
    sides = rng.choice(np.array(["B", "A"], dtype=object), size=n_events)

    depths = (rng.geometric(p=0.6, size=n_events) - 1).astype(np.uint8)
    depths = np.clip(depths, 0, lob_levels - 1).astype(np.uint8)
    depths[is_trade] = 0

    is_bid_side = sides == "B"
    row_idx = np.arange(n_events)
    event_prices = np.where(is_bid_side, bid_px[row_idx, depths], ask_px[row_idx, depths])

    sizes = rng.integers(100, 1001, n_events, dtype=np.uint32)
    n_odd = int(is_trade.sum() * odd_lot_fraction)
    trade_indices = np.where(is_trade)[0]
    odd_indices = rng.choice(trade_indices, size=n_odd, replace=False)
    sizes[odd_indices] = rng.integers(1, 100, n_odd, dtype=np.uint32)
    volumes = np.where(is_trade, sizes.astype(np.float64), 0.0)

    sequences = (np.arange(1, n_events + 1) * 10).astype(np.uint32)
    ts_in_deltas = rng.integers(1_000, 500_001, n_events, dtype=np.int32)

    utc_index = pd.DatetimeIndex(timestamps_ns, tz="UTC")
    ts_event_naive = pd.DatetimeIndex(timestamps_ns)
    start_date_str = pd.Timestamp(timestamps_ns[0], unit="ns", tz="UTC").date().isoformat()

    data: dict[str, Any] = {
        "symbol": pd.array([symbol] * n_events, dtype="string"),
        "price": event_prices,
        "source": pd.array(["synthetic"] * n_events, dtype="string"),
        "venue": pd.array(["SYNTHETIC"] * n_events, dtype="string"),
        "volume": volumes,
        "ts_event": ts_event_naive,
        "rtype": np.full(n_events, 10, dtype=np.uint8),
        "publisher_id": np.full(n_events, 2, dtype=np.uint16),
        "instrument_id": np.zeros(n_events, dtype=np.uint32),
        "action": actions,
        "side": sides,
        "depth": depths,
        "size": sizes,
        "flags": np.full(n_events, 128, dtype=np.uint8),
        "ts_in_delta": ts_in_deltas,
        "sequence": sequences,
    }
    for lvl in range(lob_levels):
        tag = f"{lvl:02d}"
        data[f"bid_px_{tag}"] = bid_px[:, lvl]
        data[f"ask_px_{tag}"] = ask_px[:, lvl]
        data[f"bid_sz_{tag}"] = bid_sz[:, lvl]
        data[f"ask_sz_{tag}"] = ask_sz[:, lvl]
        data[f"bid_ct_{tag}"] = bid_ct[:, lvl]
        data[f"ask_ct_{tag}"] = ask_ct[:, lvl]
    data["_is_normalized"] = np.ones(n_events, dtype=bool)
    data["date"] = start_date_str

    df = pd.DataFrame(data, index=utc_index)
    output_path = output_dir / output_file
    df.to_parquet(output_path)
    log_dataset_summary(df, output_path, context="HFT sine wave LOB pattern", logger=logger)
    logger.info(
        "Mid-price range: %.4f - %.4f | action mix A/C/T: %d/%d/%d",
        mid_prices.min(), mid_prices.max(),
        (actions == "A").sum(), (actions == "C").sum(), is_trade.sum(),
    )
    return df
