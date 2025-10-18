"""
Simple price data generator that loads existing parquet files and dumps them to synthetic data folder.
"""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd


class PriceDataGenerator:
    """Generator for creating synthetic price data from existing parquet files."""

    def __init__(
        self,
        source_dir: str = "data/raw/binance",
        output_dir: str = "data/raw/synthetic",
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
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

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

        df = pd.read_parquet(filepath)
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

        # Sample if specified
        if sample_size is not None and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42).sort_index()

        # Save to output directory
        if output_file is None:
            output_file = source_file.replace(".parquet", "_synthetic.parquet")

        output_path = self.output_dir / output_file
        df.to_parquet(output_path)

        print(f"Generated synthetic data: {output_path}")
        print(f"Shape: {df.shape}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")

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

        shutil.copy2(source_path, output_path)
        print(f"Copied {source_path} to {output_path}")

    def list_source_files(self) -> list[str]:
        """
        List all parquet files in source directory.

        Returns
        -------
        list[str]
            List of parquet file names
        """
        files = list(self.source_dir.glob("*.parquet"))
        return [f.name for f in files]

    def generate_sine_wave_pattern(
        self,
        output_file: str,
        n_periods: int = 5,
        samples_per_period: int = 100,
        base_price: float = 50000.0,
        amplitude: float = 5000.0,
        trend_slope: float = 50.0,
        volatility: float = 0.02,
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

        # Add controlled noise
        noise = np.random.normal(0, volatility * base_prices, total_samples)
        close_prices = base_prices + noise

        # Generate realistic OHLC from close prices
        # High: close + some upward variation
        # Low: close - some downward variation
        # Open: previous close with small gap

        high_variation = np.abs(
            np.random.normal(0, 0.5 * amplitude / 10, total_samples)
        )
        low_variation = np.abs(np.random.normal(0, 0.5 * amplitude / 10, total_samples))

        highs = close_prices + high_variation
        lows = close_prices - low_variation

        # Opens: lag close by 1 period with small random gap
        opens = np.roll(close_prices, 1)
        opens[0] = close_prices[0]  # First open = first close
        gap_noise = np.random.normal(0, 0.1 * volatility * close_prices, total_samples)
        opens = opens + gap_noise

        # Ensure OHLC relationships are valid
        highs = np.maximum(highs, np.maximum(opens, close_prices))
        lows = np.minimum(lows, np.minimum(opens, close_prices))

        # Generate volume with inverse correlation to price changes
        price_changes = np.abs(np.diff(close_prices, prepend=close_prices[0]))
        base_volume = 1000000
        volume_variation = base_volume * (
            0.5 + 0.5 * price_changes / np.max(price_changes)
        )
        volumes = (
            base_volume
            + volume_variation
            + np.random.normal(0, base_volume * 0.1, total_samples)
        )
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

        print(f"Generated sine wave pattern: {output_path}")
        print(f"Shape: {df.shape}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print(f"Price range: {df['close'].min():.2f} to {df['close'].max():.2f}")
        print(f"Periods: {n_periods}, Samples per period: {samples_per_period}")
        print(
            f"Expected strategy: Buy at troughs (~{(base_price - amplitude):.0f}), Sell at peaks (~{(base_price + amplitude):.0f})"
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

        # Ensure OHLC relationships are valid
        highs = np.maximum(highs, np.maximum(opens, prices))
        lows = np.minimum(lows, np.minimum(opens, prices))

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

        print(f"Generated mean reversion pattern: {output_path}")
        print(f"Shape: {df.shape}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print(f"Price range: {df['close'].min():.2f} to {df['close'].max():.2f}")
        print(f"Mean price: {mean_price:.2f}, Actual mean: {df['close'].mean():.2f}")
        print(
            f"Mean deviation: {mean_deviation:.2f}, Max deviation: {max_deviation:.2f}"
        )
        print(
            f"Strategy: Buy when price < {mean_price:.0f}, Sell when price > {mean_price:.0f}"
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

        # Ensure OHLC relationships are valid
        highs = np.maximum(highs, np.maximum(opens, prices))
        lows = np.minimum(lows, np.minimum(opens, prices))

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

        print(f"Generated trending pattern: {output_path}")
        print(f"Shape: {df.shape}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print(f"Price range: {df['close'].min():.2f} to {df['close'].max():.2f}")
        print(f"Total return: {total_return:.2f}%, Max drawdown: {max_drawdown:.2f}%")
        print(
            f"Trends: {len(trend_segments)}, Strategy: Follow momentum (buy uptrends, sell downtrends)"
        )

        return df


def main():
    """Example usage of PriceDataGenerator."""
    # Initialize generator
    generator = PriceDataGenerator()

    # List available source files
    print("Available source files:")
    source_files = generator.list_source_files()
    for f in source_files:
        print(f"  - {f}")

    # Generate synthetic patterns for DDPG validation
    print("\nGenerating synthetic patterns for algorithm validation...")

    # Generate sine wave pattern with trend (good for DDPG validation)
    generator.generate_sine_wave_pattern(
        output_file="sine_wave_validation.parquet",
        n_periods=5,
        samples_per_period=100,
        base_price=50000.0,
        amplitude=8000.0,
        trend_slope=50.0,
        volatility=0.02,
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
        print("\nGenerating synthetic data from BTCUSDT...")

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
