"""
Simple price data generator that loads existing parquet files and dumps them to synthetic data folder.
"""

import logging
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from logger import get_logger


def _parse_log_level(level: int | str | None) -> int:
    """Convert log level provided as string or int into logging module constant."""
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        return logging._nameToLevel.get(level.upper(), logging.INFO)
    return logging.INFO


class PriceDataGenerator:
    """Generator for creating synthetic price data from existing parquet files."""

    def __init__(
        self,
        source_dir: str = "data/raw/binance",
        output_dir: str = "data/raw/synthetic",
        log_level: int | str | None = None,
    ):
        """
        Initialize the price data generator.

        Parameters
        ----------
        source_dir : str
            Directory containing source parquet files
        output_dir : str
            Directory to dump synthetic data
        """
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        requested_level = log_level or os.getenv("DATA_GENERATOR_LOG_LEVEL")
        if requested_level is not None:
            self.logger.setLevel(_parse_log_level(requested_level))

        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.debug(
            "Initialized generator with source_dir=%s, output_dir=%s",
            self.source_dir,
            self.output_dir,
        )

    def _log_dataset_summary(
        self,
        df: pd.DataFrame,
        output_path: Path,
        *,
        context: str,
    ) -> None:
        """Log common dataset summary information."""
        self.logger.info("%s saved to %s", context, output_path)
        self.logger.debug("Shape=%s", df.shape)
        if not df.empty:
            self.logger.debug(
                "Index range: %s -> %s",
                df.index.min(),
                df.index.max(),
            )
            if "close" in df.columns:
                self.logger.debug(
                    "Close price range: %.2f -> %.2f",
                    df["close"].min(),
                    df["close"].max(),
                )

    def load_data(self, filename: str) -> pd.DataFrame:
        """
        Load data from a parquet file.

        Parameters
        ----------
        filename : str
            Name of the parquet file to load

        Returns
        -------
        pd.DataFrame
            Loaded price data
        """
        filepath = self.source_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        self.logger.debug("Loading dataset from %s", filepath)
        df = pd.read_parquet(filepath)
        self.logger.debug("Loaded %s rows from %s", len(df), filepath)
        return df

    def generate_synthetic_sample(
        self,
        source_file: str,
        output_file: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        sample_size: int | None = None,
    ) -> pd.DataFrame:
        """
        Generate synthetic data by sampling from source file.

        Parameters
        ----------
        source_file : str
            Source parquet file name
        output_file : str, optional
            Output file name (if None, uses source_file name with _synthetic suffix)
        start_date : str, optional
            Start date for filtering (format: 'YYYY-MM-DD')
        end_date : str, optional
            End date for filtering (format: 'YYYY-MM-DD')
        sample_size : int, optional
            Number of rows to sample (if None, uses all data)

        Returns
        -------
        pd.DataFrame
            Generated synthetic data
        """
        # Load source data
        df = self.load_data(source_file)

        # Filter by date range if specified
        if start_date is not None:
            df = df[df.index >= start_date]
        if end_date is not None:
            df = df[df.index <= end_date]
        if start_date or end_date:
            self.logger.debug(
                "Filtered data to date range %s -> %s (remaining rows: %s)",
                start_date or df.index.min(),
                end_date or df.index.max(),
                len(df),
            )

        # Sample if specified
        if sample_size is not None and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42).sort_index()
            self.logger.debug("Sampled %s rows from source data", len(df))

        # Save to output directory
        if output_file is None:
            output_file = source_file.replace(".parquet", "_synthetic.parquet")

        output_path = self.output_dir / output_file
        if df.empty:
            self.logger.warning(
                "Generated dataset for %s is empty after filtering; writing empty frame",
                output_path,
            )
        df.to_parquet(output_path)

        self._log_dataset_summary(
            df,
            output_path,
            context="Synthetic sample",
        )

        return df

    def copy_data(self, source_file: str, output_file: str | None = None) -> None:
        """
        Copy data from source to synthetic folder without modifications.

        Parameters
        ----------
        source_file : str
            Source parquet file name
        output_file : str, optional
            Output file name (if None, uses source_file name)
        """
        source_path = self.source_dir / source_file

        if output_file is None:
            output_file = source_file

        output_path = self.output_dir / output_file

        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        self.logger.debug("Copying file from %s to %s", source_path, output_path)
        shutil.copy2(source_path, output_path)
        self.logger.info("Copied %s to %s", source_path, output_path)

    def list_source_files(self) -> list[str]:
        """
        List all parquet files in source directory.

        Returns
        -------
        list[str]
            List of parquet file names
        """
        files = list(self.source_dir.glob("*.parquet"))
        filenames = [f.name for f in files]
        self.logger.debug(
            "Discovered %s parquet files in %s", len(filenames), self.source_dir
        )
        return filenames

    def generate_sine_wave_pattern(
        self,
        output_file: str,
        n_periods: int = 5,
        samples_per_period: int = 100,
        base_price: float = 50000.0,
        amplitude: float = 30.0,
        trend_slope: float = 0,
        volatility: float = 0.0,
        start_date: str = "2024-01-01",
    ) -> pd.DataFrame:
        """
        Generate synthetic OHLCV data with sine wave pattern and upward trend.

        This creates predictable patterns where:
        - Buy signals optimal at sine wave troughs
        - Sell signals optimal at sine wave peaks
        - Overall upward trend provides directional bias

        Parameters
        ----------
        output_file : str
            Output parquet file name
        n_periods : int
            Number of complete sine wave cycles
        samples_per_period : int
            Number of data points per cycle
        base_price : float
            Base price level (center of sine wave)
        amplitude : float
            Sine wave amplitude (peak-to-trough variation)
        trend_slope : float
            Linear trend increment per time step
        volatility : float
            Random noise factor (fraction of price)
        start_date : str
            Start date for the time index

        Returns
        -------
        pd.DataFrame
            Generated OHLCV data with datetime index
        """
        self.logger.info(
            "Generating sine wave pattern -> periods=%s, samples_per_period=%s, amplitude=%.2f, trend_slope=%.2f",
            n_periods,
            samples_per_period,
            amplitude,
            trend_slope,
        )

        total_samples = n_periods * samples_per_period

        # Create time series
        t = np.linspace(0, 2 * np.pi * n_periods, total_samples)

        # Generate datetime index (hourly frequency)
        start_dt = pd.to_datetime(start_date)
        dates = pd.date_range(start=start_dt, periods=total_samples, freq="h")

        # Generate sine wave with trend
        trend = trend_slope * np.arange(total_samples)
        sine_component = amplitude * np.sin(t)

        # Base price series (close prices)
        base_prices = base_price + trend + sine_component

        # Add deterministic secondary wave instead of random noise
        if volatility > 0:
            harmonic_frequency = 2.0
            harmonic_phase = np.pi / 4
            noise = (
                volatility
                * base_price
                * np.sin(harmonic_frequency * t + harmonic_phase)
            )
        else:
            noise = np.zeros_like(base_prices)
        close_prices = base_prices + noise

        # Light smoothing to reduce short-term volatility
        close_prices = (
            pd.Series(close_prices)
            .rolling(window=5, min_periods=1, center=False)
            .mean()
            .to_numpy()
        )

        # Generate realistic OHLC from close prices
        # High: close + some upward variation
        # Low: close - some downward variation
        # Open: previous close with small gap

        variation_scale = max(amplitude * 0.05, base_price * 0.002)
        # Use small random variations instead of additional sine waves
        high_variation = variation_scale * (
            0.5 + 0.3 * np.random.uniform(-1, 1, total_samples)
        )
        low_variation = variation_scale * (
            0.5 + 0.3 * np.random.uniform(-1, 1, total_samples)
        )

        highs = close_prices + high_variation
        lows = close_prices - low_variation

        # Opens: lag close by 1 period with small deterministic gap
        opens = np.roll(close_prices, 1)
        opens[0] = close_prices[0]  # First open = first close
        gap_amplitude = 0.1 * volatility * base_price
        if gap_amplitude > 0:
            opens = opens + gap_amplitude * np.sin(t + np.pi / 6)

        # Ensure OHLC relationships are valid: low ≤ open, close ≤ high
        opens = np.clip(opens, lows, highs)
        close_prices = np.clip(close_prices, lows, highs)

        # Generate volume with inverse correlation to price changes
        price_changes = np.abs(np.diff(close_prices, prepend=close_prices[0]))
        base_volume = 1000000
        price_change_scale = np.max(price_changes)
        if price_change_scale == 0:
            price_change_scale = 1.0
        normalized_changes = price_changes / price_change_scale
        volume_harmonic = 0.5 + 0.5 * np.sin(t + np.pi / 2)
        volumes = base_volume * (1.0 + 0.5 * normalized_changes * volume_harmonic)
        volumes = np.maximum(volumes, base_volume * 0.1)  # Minimum volume

        # Create DataFrame
        df = pd.DataFrame(
            {
                "open": opens,
                "high": highs,
                "low": lows,
                "close": close_prices,
                "volume": volumes,
            },
            index=dates,
        )

        # Ensure all values are positive
        df = df.abs()

        # Save to output directory
        output_path = self.output_dir / output_file
        df.to_parquet(output_path)

        self._log_dataset_summary(
            df,
            output_path,
            context="Sine wave pattern",
        )
        self.logger.info(
            "Sine wave trading cues -> buy ≈ %.0f, sell ≈ %.0f",
            base_price - amplitude,
            base_price + amplitude,
        )
        self.logger.debug(
            "Close price std dev: %.2f",
            df["close"].std(),
        )

        return df

    def load_config(self, config_path: str) -> dict[str, Any]:
        """
        Load configuration from YAML file.

        Parameters
        ----------
        config_path : str
            Path to YAML configuration file

        Returns
        -------
        dict[str, Any]
            Configuration dictionary
        """
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return config

    def generate_from_config(
        self, config_path: str, output_file: str | None = None
    ) -> pd.DataFrame:
        """
        Generate synthetic data using parameters from YAML config file.

        Parameters
        ----------
        config_path : str
            Path to YAML configuration file
        output_file : str, optional
            Output file name (if None, uses pattern_type from config)

        Returns
        -------
        pd.DataFrame
            Generated synthetic data
        """
        config = self.load_config(config_path)
        data_gen_config = config.get("data_generator", {})

        pattern_type = data_gen_config.get("pattern_type", "upward_drift")

        if output_file is None:
            output_file = f"{pattern_type}_validation.parquet"

        self.logger.info(f"Generating {pattern_type} pattern from config {config_path}")

        if pattern_type == "upward_drift":
            return self.generate_upward_drift_pattern(
                output_file=output_file,
                n_samples=data_gen_config.get("n_samples", 500),
                base_price=data_gen_config.get("base_price", 50000.0),
                drift_rate=data_gen_config.get("drift_rate", 0.015),
                volatility=data_gen_config.get("volatility", 0.0005),
                pullback_floor=data_gen_config.get("pullback_floor", 0.995),
                start_date=data_gen_config.get("start_date", "2024-01-01"),
            )
        elif pattern_type == "sine_wave":
            return self.generate_sine_wave_pattern(
                output_file=output_file,
                n_periods=data_gen_config.get("n_periods", 5),
                samples_per_period=data_gen_config.get("samples_per_period", 100),
                base_price=data_gen_config.get("base_price", 50000.0),
                amplitude=data_gen_config.get("amplitude", 30.0),
                trend_slope=data_gen_config.get("trend_slope", 0),
                volatility=data_gen_config.get("volatility", 0.0),
                start_date=data_gen_config.get("start_date", "2024-01-01"),
            )
        elif pattern_type == "mean_reversion":
            return self.generate_mean_reversion_pattern(
                output_file=output_file,
                n_samples=data_gen_config.get("n_samples", 500),
                mean_price=data_gen_config.get("mean_price", 50000.0),
                reversion_strength=data_gen_config.get("reversion_strength", 0.1),
                volatility=data_gen_config.get("volatility", 0.05),
                shock_probability=data_gen_config.get("shock_probability", 0.02),
                shock_magnitude=data_gen_config.get("shock_magnitude", 0.15),
                start_date=data_gen_config.get("start_date", "2024-01-01"),
            )
        elif pattern_type == "trending":
            return self.generate_trending_pattern(
                output_file=output_file,
                n_samples=data_gen_config.get("n_samples", 500),
                base_price=data_gen_config.get("base_price", 50000.0),
                n_trends=data_gen_config.get("n_trends", 3),
                min_trend_length=data_gen_config.get("min_trend_length", 50),
                max_trend_length=data_gen_config.get("max_trend_length", 150),
                trend_strength_range=tuple(
                    data_gen_config.get("trend_strength_range", [0.5, 2.0])
                ),
                volatility=data_gen_config.get("volatility", 0.03),
                consolidation_prob=data_gen_config.get("consolidation_prob", 0.2),
                start_date=data_gen_config.get("start_date", "2024-01-01"),
            )
        else:
            raise ValueError(f"Unknown pattern_type: {pattern_type}")

    def generate_upward_drift_pattern(
        self,
        output_file: str,
        n_samples: int = 1000,
        base_price: float = 50000.0,
        drift_rate: float = 0.015,
        volatility: float = 0.0005,
        pullback_floor: float = 0.995,
        start_date: str = "2024-01-01",
    ) -> pd.DataFrame:
        """
        Generate synthetic OHLCV data with strong upward drift and minimal volatility.

        The pattern is intended for validating momentum strategies that assume a
        persistent uptrend with very shallow pullbacks. Prices follow an exponential
        drift curve with light, smoothed noise and tight OHLC spreads.

        Parameters
        ----------
        output_file : str
            Output parquet file name.
        n_samples : int
            Number of data points to generate.
        base_price : float
            Starting price level.
        drift_rate : float
            Exponential drift rate per step (e.g., 0.0015 ≈ 0.15%).
        volatility : float
            Multiplicative noise factor applied to the drift curve.
        pullback_floor : float
            Floor applied as a fraction of the running drift curve to cap drawdowns.
        start_date : str
            Start date for the time index.

        Returns
        -------
        pd.DataFrame
            Generated OHLCV data with datetime index.
        """
        self.logger.info(
            "Generating upward drift pattern -> samples=%s, base_price=%.2f, drift_rate=%.4f, volatility=%.4f",
            n_samples,
            base_price,
            drift_rate,
            volatility,
        )

        # Generate datetime index (hourly frequency)
        start_dt = pd.to_datetime(start_date)
        dates = pd.date_range(start=start_dt, periods=n_samples, freq="h")

        # Deterministic exponential drift curve
        steps = np.arange(n_samples)
        drift_curve = base_price * np.exp(drift_rate * steps)

        # Light multiplicative noise to keep volatility minimal
        noise = np.random.normal(0, volatility, n_samples)
        noisy_curve = drift_curve * (1 + noise)

        # Smooth noise to avoid erratic swings
        close_prices = (
            pd.Series(noisy_curve).rolling(window=5, min_periods=1).mean().to_numpy()
        )

        # Enforce shallow pullbacks relative to the drift curve
        floor = pullback_floor * np.maximum.accumulate(drift_curve)
        close_prices = np.maximum(close_prices, floor)
        close_prices[0] = base_price

        # Generate tight OHLC ranges around close prices with consistent upward drift
        spread_scale = max(volatility * 10, 0.001)
        # Use small random variations instead of sine waves to maintain drift consistency
        high_spread = spread_scale * (1 + 0.1 * np.random.uniform(-1, 1, n_samples))
        low_spread = spread_scale * (1 + 0.1 * np.random.uniform(-1, 1, n_samples))

        highs = close_prices * (1 + high_spread)
        lows = close_prices * (1 - low_spread)

        opens = np.roll(close_prices, 1)
        opens[0] = close_prices[0]
        # Use simple random gap noise to avoid patterns
        gap_noise = np.random.normal(
            0, volatility * np.mean(close_prices) * 0.1, n_samples
        )
        opens = opens + gap_noise

        # Ensure OHLC relationships are valid: low ≤ open, close ≤ high
        opens = np.clip(opens, lows, highs)
        close_prices = np.clip(close_prices, lows, highs)

        # Volume trending higher with the drift
        base_volume = 1_000_000
        volume_trend = 1 + 0.8 * (steps / max(steps[-1], 1))
        volume_noise = np.random.normal(0, 0.05, n_samples)
        volumes = base_volume * volume_trend * (1 + volume_noise)
        volumes = np.maximum(volumes, base_volume * 0.5)

        df = pd.DataFrame(
            {
                "open": opens,
                "high": highs,
                "low": lows,
                "close": close_prices,
                "volume": volumes,
            },
            index=dates,
        )

        df = df.abs()

        output_path = self.output_dir / output_file
        df.to_parquet(output_path)

        total_return = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100

        self._log_dataset_summary(
            df,
            output_path,
            context="Upward drift pattern",
        )
        self.logger.info(
            "Upward drift stats -> total return %.2f%%, volatility %.4f",
            total_return,
            df["close"].pct_change().std(),
        )

        return df

    def generate_mean_reversion_pattern(
        self,
        output_file: str,
        n_samples: int = 500,
        mean_price: float = 50000.0,
        reversion_strength: float = 0.1,
        volatility: float = 0.05,
        shock_probability: float = 0.02,
        shock_magnitude: float = 0.15,
        start_date: str = "2024-01-01",
    ) -> pd.DataFrame:
        """
        Generate synthetic OHLCV data with mean reversion pattern.

        This creates patterns where:
        - Price tends to revert to a long-term mean
        - Optimal strategy: buy when price < mean, sell when price > mean
        - Occasional price shocks create larger reversion opportunities

        Parameters
        ----------
        output_file : str
            Output parquet file name
        n_samples : int
            Number of data points
        mean_price : float
            Long-term mean price level
        reversion_strength : float
            Speed of mean reversion (0-1, higher = faster reversion)
        volatility : float
            Random volatility factor
        shock_probability : float
            Probability of price shock per period
        shock_magnitude : float
            Magnitude of price shocks (fraction of mean)
        start_date : str
            Start date for the time index

        Returns
        -------
        pd.DataFrame
            Generated OHLCV data with datetime index
        """
        self.logger.info(
            "Generating mean reversion pattern -> samples=%s, mean_price=%.2f, reversion_strength=%.2f",
            n_samples,
            mean_price,
            reversion_strength,
        )

        # Generate datetime index (hourly frequency)
        start_dt = pd.to_datetime(start_date)
        dates = pd.date_range(start=start_dt, periods=n_samples, freq="h")

        # Initialize price series
        prices = np.zeros(n_samples)
        prices[0] = mean_price * (1 + np.random.normal(0, 0.1))  # Start near mean

        # Generate mean-reverting price series
        for i in range(1, n_samples):
            # Mean reversion component
            deviation = prices[i - 1] - mean_price
            reversion = -reversion_strength * deviation

            # Random shock component
            shock = 0
            if np.random.random() < shock_probability:
                shock = np.random.choice([-1, 1]) * shock_magnitude * mean_price

            # Random noise
            noise = np.random.normal(0, volatility * mean_price)

            # Update price
            prices[i] = prices[i - 1] + reversion + shock + noise

            # Ensure price stays positive
            prices[i] = max(prices[i], mean_price * 0.1)

        # Generate realistic OHLC from close prices
        high_variation = np.abs(
            np.random.normal(0, volatility * mean_price * 0.3, n_samples)
        )
        low_variation = np.abs(
            np.random.normal(0, volatility * mean_price * 0.3, n_samples)
        )

        highs = prices + high_variation
        lows = prices - low_variation

        # Opens: previous close with small gap
        opens = np.roll(prices, 1)
        opens[0] = prices[0]
        gap_noise = np.random.normal(0, 0.5 * volatility * mean_price, n_samples)
        opens = opens + gap_noise

        # Ensure OHLC relationships are valid: low ≤ open, close ≤ high
        opens = np.clip(opens, lows, highs)
        prices = np.clip(prices, lows, highs)

        # Generate volume (higher when price deviates more from mean)
        price_deviation = np.abs(prices - mean_price) / mean_price
        base_volume = 1000000
        volume_multiplier = 1 + 2 * price_deviation  # Higher volume when far from mean
        volumes = base_volume * volume_multiplier + np.random.normal(
            0, base_volume * 0.1, n_samples
        )
        volumes = np.maximum(volumes, base_volume * 0.1)

        # Create DataFrame
        df = pd.DataFrame(
            {
                "open": opens,
                "high": highs,
                "low": lows,
                "close": prices,
                "volume": volumes,
            },
            index=dates,
        )

        # Ensure all values are positive
        df = df.abs()

        # Save to output directory
        output_path = self.output_dir / output_file
        df.to_parquet(output_path)

        mean_deviation = np.mean(np.abs(prices - mean_price))
        max_deviation = np.max(np.abs(prices - mean_price))

        self._log_dataset_summary(
            df,
            output_path,
            context="Mean reversion pattern",
        )
        self.logger.info(
            "Mean price target %.2f, actual %.2f (mean deviation %.2f, max deviation %.2f)",
            mean_price,
            df["close"].mean(),
            mean_deviation,
            max_deviation,
        )
        lower_bound = mean_price * 0.98
        upper_bound = mean_price * 1.02
        self.logger.info(
            "Strategy hint -> accumulate below %.0f, distribute above %.0f",
            lower_bound,
            upper_bound,
        )

        return df

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
        start_date: str = "2024-01-01",
    ) -> pd.DataFrame:
        """
        Generate synthetic OHLCV data with trending market patterns.

        This creates patterns with:
        - Sustained upward and downward trends
        - Brief consolidation periods between trends
        - Optimal strategy: momentum following (buy uptrends, sell downtrends)

        Parameters
        ----------
        output_file : str
            Output parquet file name
        n_samples : int
            Number of data points
        base_price : float
            Starting price level
        n_trends : int
            Number of distinct trend periods
        min_trend_length : int
            Minimum samples per trend
        max_trend_length : int
            Maximum samples per trend
        trend_strength_range : tuple[float, float]
            Range of trend strength (price change per period)
        volatility : float
            Random volatility factor
        consolidation_prob : float
            Probability of consolidation periods between trends
        start_date : str
            Start date for the time index

        Returns
        -------
        pd.DataFrame
            Generated OHLCV data with datetime index
        """
        self.logger.info(
            "Generating trending pattern -> samples=%s, base_price=%.2f, n_trends=%s",
            n_samples,
            base_price,
            n_trends,
        )

        # Generate datetime index (hourly frequency)
        start_dt = pd.to_datetime(start_date)
        dates = pd.date_range(start=start_dt, periods=n_samples, freq="h")

        # Initialize price series
        prices = np.zeros(n_samples)
        prices[0] = base_price

        # Plan trend segments
        remaining_samples = n_samples - 1
        trend_segments = []

        for i in range(n_trends):
            if i == n_trends - 1:  # Last trend uses remaining samples
                trend_length = remaining_samples
            else:
                max_len = min(
                    max_trend_length,
                    remaining_samples - (n_trends - i - 1) * min_trend_length,
                )
                trend_length = np.random.randint(min_trend_length, max_len + 1)

            # Random trend direction and strength
            direction = np.random.choice([-1, 1])  # -1 for down, +1 for up
            strength = np.random.uniform(*trend_strength_range) * direction

            trend_segments.append(
                {
                    "length": trend_length,
                    "strength": strength,
                    "is_consolidation": np.random.random() < consolidation_prob
                    and i > 0,
                }
            )

            remaining_samples -= trend_length
            if remaining_samples <= 0:
                break

        # Generate price series based on trend segments
        current_idx = 1

        for segment in trend_segments:
            end_idx = min(current_idx + segment["length"], n_samples)
            segment_length = end_idx - current_idx

            if segment["is_consolidation"]:
                # Consolidation: price oscillates around current level
                base_level = prices[current_idx - 1]
                for j in range(segment_length):
                    if current_idx + j >= n_samples:
                        break
                    noise = np.random.normal(0, volatility * base_level)
                    small_drift = np.random.normal(0, 0.1 * volatility * base_level)
                    prices[current_idx + j] = (
                        prices[current_idx + j - 1] + noise + small_drift
                    )
            else:
                # Trending: sustained directional movement
                trend_per_step = segment["strength"]
                for j in range(segment_length):
                    if current_idx + j >= n_samples:
                        break
                    trend_component = (
                        trend_per_step * base_price / 100
                    )  # Convert to price units
                    current_price = prices[current_idx + j - 1]
                    noise = np.random.normal(
                        0, volatility * max(abs(current_price), base_price * 0.1)
                    )
                    prices[current_idx + j] = current_price + trend_component + noise
                    # Ensure price stays positive during generation
                    prices[current_idx + j] = max(
                        prices[current_idx + j], base_price * 0.05
                    )

            current_idx = end_idx
            if current_idx >= n_samples:
                break

        # Ensure all prices are positive
        prices = np.maximum(prices, base_price * 0.1)

        # Generate realistic OHLC from close prices
        price_changes = np.abs(np.diff(prices, prepend=prices[0]))
        high_variation = 0.3 * price_changes + np.abs(
            np.random.normal(0, volatility * prices * 0.2, n_samples)
        )
        low_variation = 0.3 * price_changes + np.abs(
            np.random.normal(0, volatility * prices * 0.2, n_samples)
        )

        highs = prices + high_variation
        lows = prices - low_variation

        # Opens: previous close with small gap
        opens = np.roll(prices, 1)
        opens[0] = prices[0]
        gap_noise = np.random.normal(0, 0.3 * volatility * prices, n_samples)
        opens = opens + gap_noise

        # Ensure OHLC relationships are valid: low ≤ open, close ≤ high
        opens = np.clip(opens, lows, highs)
        prices = np.clip(prices, lows, highs)

        # Generate volume (higher during trend changes)
        momentum = np.abs(np.diff(prices, prepend=prices[0]))
        base_volume = 1000000
        volume_multiplier = 1 + 3 * momentum / np.mean(
            momentum
        )  # Higher volume during strong moves
        volumes = base_volume * volume_multiplier + np.random.normal(
            0, base_volume * 0.15, n_samples
        )
        volumes = np.maximum(volumes, base_volume * 0.1)

        # Create DataFrame
        df = pd.DataFrame(
            {
                "open": opens,
                "high": highs,
                "low": lows,
                "close": prices,
                "volume": volumes,
            },
            index=dates,
        )

        # Ensure all values are positive
        df = df.abs()

        # Save to output directory
        output_path = self.output_dir / output_file
        df.to_parquet(output_path)

        total_return = (prices[-1] / prices[0] - 1) * 100
        max_drawdown = np.min(prices / np.maximum.accumulate(prices) - 1) * 100

        self._log_dataset_summary(
            df,
            output_path,
            context="Trending pattern",
        )
        self.logger.info(
            "Momentum stats -> total return %.2f%%, max drawdown %.2f%%",
            total_return,
            max_drawdown,
        )
        self.logger.info(
            "Trends identified: %s | Strategy hint -> ride momentum, manage reversals",
            len(trend_segments),
        )

        return df


def main():
    """Example usage of PriceDataGenerator."""
    # Initialize generator
    generator = PriceDataGenerator()
    logger = generator.logger

    # List available source files
    logger.info("Available source files:")
    source_files = generator.list_source_files()
    for f in source_files:
        logger.info("  - %s", f)

    # Generate synthetic patterns for DDPG validation
    logger.info("Generating synthetic patterns for algorithm validation...")

    # Generate sine wave pattern with trend (good for DDPG validation)
    generator.generate_sine_wave_pattern(
        output_file="sine_wave_validation.parquet",
        n_periods=5,
        samples_per_period=100,
        base_price=50000.0,
        amplitude=100.0,
        trend_slope=0.01,
        volatility=0.05,
    )

    # Generate low-volatility upward drift pattern
    generator.generate_upward_drift_pattern(
        output_file="upward_drift_validation.parquet",
        n_samples=1000,
        base_price=50000.0,
        drift_rate=0.0012,
        volatility=0.0004,
        pullback_floor=0.997,
    )

    # Generate mean reversion pattern
    generator.generate_mean_reversion_pattern(
        output_file="mean_reversion_validation.parquet",
        n_samples=500,
        mean_price=50000.0,
        reversion_strength=0.12,
        volatility=0.04,
        shock_probability=0.03,
        shock_magnitude=0.15,
    )

    # Generate trending pattern
    generator.generate_trending_pattern(
        output_file="trending_validation.parquet",
        n_samples=400,
        base_price=50000.0,
        n_trends=3,
        min_trend_length=60,
        max_trend_length=140,
        trend_strength_range=(1.0, 2.5),
        volatility=0.03,
        consolidation_prob=0.25,
    )

    # Generate synthetic data from BTCUSDT if available
    if "binance-BTCUSDT-1h.parquet" in source_files:
        logger.info("Generating synthetic data from BTCUSDT...")

        # Full dataset copy
        generator.copy_data("binance-BTCUSDT-1h.parquet", "BTCUSDT_full.parquet")

        # Sample for 2023
        generator.generate_synthetic_sample(
            source_file="binance-BTCUSDT-1h.parquet",
            output_file="BTCUSDT_2023.parquet",
            start_date="2023-01-01",
            end_date="2023-12-31",
        )

        # Sample for 2024
        generator.generate_synthetic_sample(
            source_file="binance-BTCUSDT-1h.parquet",
            output_file="BTCUSDT_2024.parquet",
            start_date="2024-01-01",
            end_date="2024-12-31",
        )

        # Small sample for testing (1000 rows)
        generator.generate_synthetic_sample(
            source_file="binance-BTCUSDT-1h.parquet",
            output_file="BTCUSDT_sample_1000.parquet",
            sample_size=1000,
        )


if __name__ == "__main__":
    main()
