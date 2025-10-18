"""
Simple price data generator that loads existing parquet files and dumps them to synthetic data folder.
"""

import shutil
from pathlib import Path

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


def main():
    """Example usage of PriceDataGenerator."""
    # Initialize generator
    generator = PriceDataGenerator()

    # List available source files
    print("Available source files:")
    source_files = generator.list_source_files()
    for f in source_files:
        print(f"  - {f}")

    # Generate synthetic data from BTCUSDT
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
