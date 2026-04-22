"""
Base classes for data fetchers and generators.

This module provides abstract base classes to ensure consistency across
all data sources (synthetic generators, stock fetchers, crypto fetchers, etc.)
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from logger import get_logger
from trading_rl.constants import DataFormat


class BaseDataSource(ABC):
    """
    Abstract base class for all data sources.

    All data sources (synthetic generators, stock fetchers, etc.) should
    inherit from this class to ensure consistent interface.

    Attributes:
        output_dir (Path): Directory where data files are saved
        logger: Logger instance for this data source
    """

    def __init__(
        self,
        output_dir: str = "data/raw",
        log_level: int | str | None = None,
    ):
        """
        Initialize the data source.

        Parameters
        ----------
        output_dir : str
            Directory to save output data files
        log_level : int | str | None
            Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        if log_level:
            level = self._parse_log_level(log_level)
            self.logger.setLevel(level)

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.debug(
            f"Initialized {self.__class__.__name__} with output_dir={self.output_dir}"
        )

    @staticmethod
    def _parse_log_level(level: int | str | None) -> int:
        """Convert log level string to logging constant."""
        level_map = {
            "CRITICAL": 50,
            "ERROR": 40,
            "WARNING": 30,
            "INFO": 20,
            "DEBUG": 10,
            "NOTSET": 0,
        }
        if isinstance(level, int):
            return level
        if isinstance(level, str):
            return level_map.get(level.upper(), level_map["INFO"])
        return level_map["INFO"]

    @abstractmethod
    def generate_data(self, **kwargs) -> pd.DataFrame:
        """
        Generate or fetch data.

        This is the main method that all data sources must implement.
        For generators, this creates synthetic data.
        For fetchers, this downloads real market data.

        Parameters
        ----------
        **kwargs
            Source-specific parameters

        Returns
        -------
        pd.DataFrame
            Generated or fetched data in OHLCV format
        """
        pass

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
        self.logger.debug(f"Loading config from {config_path}")
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return config

    def from_config(self, config_path: str, **kwargs) -> pd.DataFrame:
        """
        Generate data using parameters from configuration file.

        This method loads config and calls generate_data with appropriate
        parameters. Subclasses can override this for custom config handling.

        Parameters
        ----------
        config_path : str
            Path to YAML configuration file
        **kwargs
            Additional parameters to override config values

        Returns
        -------
        pd.DataFrame
            Generated or fetched data
        """
        config = self.load_config(config_path)
        self.logger.info(
            f"Generating data from config: {config_path} using {self.__class__.__name__}"
        )
        # Subclasses should implement their own config parsing
        return self.generate_data(**{**config, **kwargs})

    def save_data(
        self,
        df: pd.DataFrame,
        filename: str,
        format: DataFormat = DataFormat.PARQUET,
    ) -> Path:
        """
        Save DataFrame to file.

        Parameters
        ----------
        df : pd.DataFrame
            Data to save
        filename : str
            Output filename (without path)
        format : DataFormat
            File format: DataFormat.PARQUET, DataFormat.CSV, or DataFormat.PICKLE

        Returns
        -------
        Path
            Path to saved file
        """
        output_path = self.output_dir / filename

        if format == DataFormat.PARQUET:
            df.to_parquet(output_path)
        elif format == DataFormat.CSV:
            df.to_csv(output_path)
        elif format == DataFormat.PICKLE:
            df.to_pickle(output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

        self.logger.info(f"Saved data to {output_path} ({format} format)")
        return output_path

    def _log_dataset_summary(
        self,
        df: pd.DataFrame,
        output_path: Path,
        *,
        context: str,
    ) -> None:
        """
        Log common dataset summary information.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset to summarize
        output_path : Path
            Where the dataset was saved
        context : str
            Description of the dataset (e.g., "Synthetic sample", "Stock data")
        """
        self.logger.info(f"{context} saved to {output_path}")
        self.logger.debug(f"Shape={df.shape}")

        if not df.empty:
            self.logger.debug(
                f"Index range: {df.index.min()} -> {df.index.max()}",
            )
            if "close" in df.columns:
                self.logger.debug(
                    f"Close price range: {df['close'].min():.2f} -> {df['close'].max():.2f}",
                )

    def validate_ohlcv(self, df: pd.DataFrame) -> bool:
        """
        Validate that DataFrame has required OHLCV columns.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate

        Returns
        -------
        bool
            True if valid OHLCV data, False otherwise
        """
        required_columns = {"open", "high", "low", "close", "volume"}
        has_required = required_columns.issubset(df.columns)

        if not has_required:
            missing = required_columns - set(df.columns)
            self.logger.warning(f"Missing required OHLCV columns: {missing}")
            return False

        # Validate OHLC relationships
        invalid_ohlc = (
            (df["high"] < df["low"])
            | (df["high"] < df["open"])
            | (df["high"] < df["close"])
            | (df["low"] > df["open"])
            | (df["low"] > df["close"])
        )

        if invalid_ohlc.any():
            self.logger.warning(
                f"Found {invalid_ohlc.sum()} rows with invalid OHLC relationships"
            )
            return False

        return True


class BaseSyntheticGenerator(BaseDataSource):
    """
    Base class for synthetic data generators.

    Provides common functionality for generating various synthetic price patterns.
    """

    def generate_data(self, pattern_type: str, **kwargs) -> pd.DataFrame:
        """
        Generate synthetic data based on pattern type.

        Parameters
        ----------
        pattern_type : str
            Type of pattern to generate (e.g., "sine_wave", "upward_drift")
        **kwargs
            Pattern-specific parameters

        Returns
        -------
        pd.DataFrame
            Generated synthetic data
        """
        # This will be implemented by specific generators
        raise NotImplementedError(
            f"Pattern type '{pattern_type}' not implemented in {self.__class__.__name__}"
        )


class BaseMarketDataFetcher(BaseDataSource):
    """
    Base class for market data fetchers.

    Provides common functionality for fetching real market data from various sources.
    """

    def generate_data(self, **kwargs) -> pd.DataFrame:
        """
        Fetch market data.

        This wraps the abstract fetch_market_data method to provide
        consistent interface with generators.

        Parameters
        ----------
        **kwargs
            Fetcher-specific parameters

        Returns
        -------
        pd.DataFrame
            Fetched market data
        """
        return self.fetch_market_data(**kwargs)

    @abstractmethod
    def fetch_market_data(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Fetch market data from external source.

        Parameters
        ----------
        symbols : list[str]
            List of symbols to fetch
        start_date : str
            Start date
        end_date : str | None
            End date (None for current date)
        **kwargs
            Source-specific parameters

        Returns
        -------
        pd.DataFrame
            Market data in OHLCV format
        """
        pass
