from __future__ import annotations

from typing import Any

import yaml

from ._types import DEFAULT_SYNTHETIC_START_DATE, PatternType


def load_config(config_path: str) -> dict[str, Any]:
    """Load data generator configuration from a YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def dispatch_from_config(
    generator: Any,
    config_path: str,
    output_file: str | None = None,
) -> Any:
    """Parse a YAML config and delegate to the matching generator method."""
    config = load_config(config_path)
    cfg = config.get("data_generator", {})
    pattern_type = PatternType(cfg.get("pattern_type", PatternType.UPWARD_DRIFT))

    if output_file is None:
        output_file = f"{pattern_type}_validation.parquet"

    generator.logger.info("generate pattern type={} config={}", pattern_type, config_path)

    if pattern_type == PatternType.UPWARD_DRIFT:
        return generator.generate_upward_drift_pattern(
            output_file=output_file,
            n_samples=cfg.get("n_samples", 500),
            base_price=cfg.get("base_price", 50000.0),
            drift_rate=cfg.get("drift_rate", 0.015),
            volatility=cfg.get("volatility", 0.0005),
            pullback_floor=cfg.get("pullback_floor", 0.995),
            start_date=cfg.get("start_date", DEFAULT_SYNTHETIC_START_DATE),
        )
    if pattern_type == PatternType.SINE_WAVE:
        return generator.generate_sine_wave_pattern(
            output_file=output_file,
            n_periods=cfg.get("n_periods", 5),
            samples_per_period=cfg.get("samples_per_period", 100),
            base_price=cfg.get("base_price", 50000.0),
            amplitude=cfg.get("amplitude", 30.0),
            trend_slope=cfg.get("trend_slope", 0),
            volatility=cfg.get("volatility", 0.0),
            start_date=cfg.get("start_date", DEFAULT_SYNTHETIC_START_DATE),
            freq=cfg.get("freq", "h"),
        )
    if pattern_type == PatternType.MEAN_REVERSION:
        return generator.generate_mean_reversion_pattern(
            output_file=output_file,
            n_samples=cfg.get("n_samples", 500),
            mean_price=cfg.get("mean_price", 50000.0),
            reversion_strength=cfg.get("reversion_strength", 0.1),
            volatility=cfg.get("volatility", 0.05),
            shock_probability=cfg.get("shock_probability", 0.02),
            shock_magnitude=cfg.get("shock_magnitude", 0.15),
            start_date=cfg.get("start_date", DEFAULT_SYNTHETIC_START_DATE),
        )
    if pattern_type == PatternType.TRENDING:
        return generator.generate_trending_pattern(
            output_file=output_file,
            n_samples=cfg.get("n_samples", 500),
            base_price=cfg.get("base_price", 50000.0),
            n_trends=cfg.get("n_trends", 3),
            min_trend_length=cfg.get("min_trend_length", 50),
            max_trend_length=cfg.get("max_trend_length", 150),
            trend_strength_range=tuple(cfg.get("trend_strength_range", [0.5, 2.0])),
            volatility=cfg.get("volatility", 0.03),
            consolidation_prob=cfg.get("consolidation_prob", 0.2),
            start_date=cfg.get("start_date", DEFAULT_SYNTHETIC_START_DATE),
        )
    if pattern_type == PatternType.RANDOM_WALK:
        return generator.generate_random_walk_pattern(
            output_file=output_file,
            n_samples=cfg.get("n_samples", 500),
            base_price=cfg.get("base_price", 50000.0),
            volatility=cfg.get("volatility", 0.001),
            start_date=cfg.get("start_date", DEFAULT_SYNTHETIC_START_DATE),
        )
    if pattern_type == PatternType.HFT_SINE_WAVE_LOB:
        return generator.generate_hft_sine_wave_lob(
            output_file=output_file,
            n_events=cfg.get("n_events", 20000),
            n_periods=cfg.get("n_periods", 5),
            base_price=cfg.get("base_price", 270.0),
            amplitude=cfg.get("amplitude", 5.0),
            spread=cfg.get("spread", 0.12),
            level_spacing=cfg.get("level_spacing", 0.10),
            tick_size=cfg.get("tick_size", 0.01),
            symbol=cfg.get("symbol", "AAPLUSD"),
            start_datetime=cfg.get("start_datetime", "2026-02-25 14:30:00"),
            session_duration_seconds=cfg.get("session_duration_seconds", 23400.0),
            odd_lot_fraction=cfg.get("odd_lot_fraction", 0.08),
            seed=cfg.get("seed", 42),
            price_noise_std=cfg.get("price_noise_std", 0.01),
        )
    return None
