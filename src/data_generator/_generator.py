from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

from logger import get_logger

from ._config import dispatch_from_config, load_config
from ._drift import generate_random_walk_pattern, generate_upward_drift_pattern
from ._io import copy_data, list_source_files, load_data, log_dataset_summary
from ._mean_reversion import generate_mean_reversion_pattern
from ._sine import generate_hft_sine_wave_lob, generate_sine_wave_pattern
from ._trending import generate_trending_pattern
from ._types import DEFAULT_SYNTHETIC_START_DATE, _parse_log_level


class PriceDataGenerator:
    """Generator for creating synthetic price data from existing parquet files.

    Delegates each pattern type to a focused module-level function.  The class
    itself only owns I/O context (source_dir, output_dir, logger) and acts as a
    thin orchestration layer.
    """

    def __init__(
        self,
        source_dir: str = "data/raw/binance",
        output_dir: str = "data/raw/synthetic",
        log_level: int | str | None = None,
    ) -> None:
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        requested_level = log_level or os.getenv("DATA_GENERATOR_LOG_LEVEL")
        if requested_level is not None:
            self.logger.setLevel(_parse_log_level(requested_level))

        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.debug(
            "Initialized generator with source_dir={}, output_dir={}",
            self.source_dir, self.output_dir,
        )

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def load_data(self, filename: str) -> pd.DataFrame:
        """Load data from a parquet file in source_dir."""
        return load_data(self.source_dir, filename, self.logger)

    def copy_data(self, source_file: str, output_file: str | None = None) -> None:
        """Copy a parquet file from source_dir to output_dir without modification."""
        copy_data(self.source_dir, self.output_dir, source_file, output_file, self.logger)

    def list_source_files(self) -> list[str]:
        """Return names of all parquet files in source_dir."""
        return list_source_files(self.source_dir, self.logger)

    def load_config(self, config_path: str) -> dict[str, Any]:
        """Load data generator configuration from a YAML file."""
        return load_config(config_path)

    # ------------------------------------------------------------------
    # Sampling from real data
    # ------------------------------------------------------------------

    def generate_synthetic_sample(
        self,
        source_file: str,
        output_file: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        sample_size: int | None = None,
    ) -> pd.DataFrame:
        """Generate synthetic data by sampling from an existing parquet file."""
        df = self.load_data(source_file)

        if start_date is not None:
            df = df[df.index >= start_date]
        if end_date is not None:
            df = df[df.index <= end_date]
        if start_date or end_date:
            self.logger.debug(
                "Filtered data to date range {} -> {} (remaining rows: {})",
                start_date or df.index.min(), end_date or df.index.max(), len(df),
            )

        if sample_size is not None and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42).sort_index()
            self.logger.trace("sampled rows={}", len(df))

        if output_file is None:
            output_file = source_file.replace(".parquet", "_synthetic.parquet")
        output_path = self.output_dir / output_file

        if df.empty:
            self.logger.warning(
                "Generated dataset for {} is empty after filtering; writing empty frame",
                output_path,
            )
        df.to_parquet(output_path)
        log_dataset_summary(df, output_path, context="Synthetic sample", logger=self.logger)
        return df

    # ------------------------------------------------------------------
    # YAML config dispatcher
    # ------------------------------------------------------------------

    def generate_from_config(
        self, config_path: str, output_file: str | None = None
    ) -> pd.DataFrame:
        """Generate synthetic data using parameters from a YAML config file."""
        return dispatch_from_config(self, config_path, output_file)

    # ------------------------------------------------------------------
    # Pattern generators
    # ------------------------------------------------------------------

    def generate_sine_wave_pattern(
        self,
        output_file: str,
        n_periods: int = 5,
        samples_per_period: int = 100,
        base_price: float = 50000.0,
        amplitude: float = 30.0,
        trend_slope: float = 0,
        volatility: float = 0.0,
        start_date: str = DEFAULT_SYNTHETIC_START_DATE,
        freq: str = "h",
    ) -> pd.DataFrame:
        return generate_sine_wave_pattern(
            self.output_dir, output_file, self.logger,
            n_periods=n_periods,
            samples_per_period=samples_per_period,
            base_price=base_price,
            amplitude=amplitude,
            trend_slope=trend_slope,
            volatility=volatility,
            start_date=start_date,
            freq=freq,
        )

    def generate_upward_drift_pattern(
        self,
        output_file: str,
        n_samples: int = 1000,
        base_price: float = 50000.0,
        drift_rate: float = 0.015,
        volatility: float = 0.0005,
        pullback_floor: float = 0.995,
        start_date: str = DEFAULT_SYNTHETIC_START_DATE,
    ) -> pd.DataFrame:
        return generate_upward_drift_pattern(
            self.output_dir, output_file, self.logger,
            n_samples=n_samples,
            base_price=base_price,
            drift_rate=drift_rate,
            volatility=volatility,
            pullback_floor=pullback_floor,
            start_date=start_date,
        )

    def generate_random_walk_pattern(
        self,
        output_file: str,
        n_samples: int = 500,
        base_price: float = 50000.0,
        volatility: float = 0.001,
        start_date: str = DEFAULT_SYNTHETIC_START_DATE,
    ) -> pd.DataFrame:
        return generate_random_walk_pattern(
            self.output_dir, output_file, self.logger,
            n_samples=n_samples,
            base_price=base_price,
            volatility=volatility,
            start_date=start_date,
        )

    def generate_mean_reversion_pattern(
        self,
        output_file: str,
        n_samples: int = 500,
        mean_price: float = 50000.0,
        reversion_strength: float = 0.1,
        volatility: float = 0.05,
        shock_probability: float = 0.02,
        shock_magnitude: float = 0.15,
        start_date: str = DEFAULT_SYNTHETIC_START_DATE,
    ) -> pd.DataFrame:
        return generate_mean_reversion_pattern(
            self.output_dir, output_file, self.logger,
            n_samples=n_samples,
            mean_price=mean_price,
            reversion_strength=reversion_strength,
            volatility=volatility,
            shock_probability=shock_probability,
            shock_magnitude=shock_magnitude,
            start_date=start_date,
        )

    def generate_trending_pattern(
        self,
        output_file: str,
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
        return generate_trending_pattern(
            self.output_dir, output_file, self.logger,
            n_samples=n_samples,
            base_price=base_price,
            n_trends=n_trends,
            min_trend_length=min_trend_length,
            max_trend_length=max_trend_length,
            trend_strength_range=trend_strength_range,
            volatility=volatility,
            consolidation_prob=consolidation_prob,
            start_date=start_date,
        )

    def generate_hft_sine_wave_lob(
        self,
        output_file: str,
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
        return generate_hft_sine_wave_lob(
            self.output_dir, output_file, self.logger,
            n_events=n_events,
            n_periods=n_periods,
            base_price=base_price,
            amplitude=amplitude,
            spread=spread,
            level_spacing=level_spacing,
            tick_size=tick_size,
            symbol=symbol,
            start_datetime=start_datetime,
            session_duration_seconds=session_duration_seconds,
            odd_lot_fraction=odd_lot_fraction,
            seed=seed,
            price_noise_std=price_noise_std,
        )
