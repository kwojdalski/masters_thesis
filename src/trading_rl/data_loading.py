"""Lazy-loading DataFrame for on-demand data access.

Implements DataFrame-like interface but loads from parquet files
lazily. This keeps memory usage low during training by only
loading data when accessed.
"""

from __future__ import annotations

import logging
import pandas as pd
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class LazyDataFrame:
    """DataFrame-like wrapper that loads data lazily from parquet files.

    Implements a subset of the pandas DataFrame interface to be compatible
    with gym_trading_env environments. Only loads data when columns or
    rows are actually accessed, keeping memory usage minimal.

    Attributes are computed on first access and cached. File-based methods
    (to_csv, to_parquet) operate on the in-memory cache.

    Example:
        lazy_df = LazyDataFrame("data/prepared/train.parquet")
        print(len(lazy_df))  # Loads file to get shape
        print(lazy_df["close"].mean())  # Loads and accesses column
        lazy_df.to_parquet("output.parquet")  # Operates on cached DataFrame
    """

    def __init__(
        self,
        file_path: str | Path,
        cache_after_load: bool = True,
    ) -> None:
        """Initialize lazy DataFrame.

        Args:
            file_path: Path to parquet file containing the prepared DataFrame.
            cache_after_load: If True, keep loaded DataFrame in memory after
                first access. If False, reload from disk each time (slower but
                less memory).
        """
        self.file_path = Path(file_path)
        self.cache_after_load = cache_after_load
        self._df: pd.DataFrame | None = None
        self._loaded = False

    def _load_if_needed(self) -> pd.DataFrame:
        """Load DataFrame from file if not already cached."""
        if self._df is None or (not self.cache_after_load and self._loaded):
            logger.debug(f"Lazy loading {self.file_path}")
            self._df = pd.read_parquet(self.file_path)
            self._loaded = True
        return self._df

    @property
    def columns(self) -> pd.Index:
        """Return DataFrame columns (loads file)."""
        df = self._load_if_needed()
        return df.columns

    @property
    def shape(self) -> tuple[int, ...]:
        """Return DataFrame shape (loads file)."""
        df = self._load_if_needed()
        return df.shape

    def __len__(self) -> int:
        """Return number of rows (loads file)."""
        df = self._load_if_needed()
        return len(df)

    def __getitem__(self, key):
        """Get column by name or slice rows (loads file if needed)."""
        df = self._load_if_needed()
        return df[key]

    def iterrows(self) -> pd.DataFrame.iterrows:
        """Iterate over DataFrame rows (loads file)."""
        df = self._load_if_needed()
        return df.iterrows()

    def head(self, n: int = 5) -> pd.DataFrame:
        """Return first n rows (loads file)."""
        df = self._load_if_needed()
        return df.head(n)

    def to_csv(self, *args, **kwargs) -> None:
        """Write to CSV (operates on cached DataFrame)."""
        df = self._load_if_needed()
        df.to_csv(*args, **kwargs)

    def to_parquet(self, *args, **kwargs) -> None:
        """Write to parquet (operates on cached DataFrame)."""
        df = self._load_if_needed()
        df.to_parquet(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attribute access to underlying DataFrame."""
        if name.startswith("_"):
            raise AttributeError(f"LazyDataFrame has no attribute '{name}'")
        df = self._load_if_needed()
        return getattr(df, name)

    def __repr__(self) -> str:
        """String representation showing file path and cached status."""
        status = "cached" if self._df is not None else "not loaded"
        return f"LazyDataFrame(file_path='{self.file_path}', {status})"


def save_prepared_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, str]:
    """Save prepared train/val/test splits to parquet files.

    Args:
        train_df: Training DataFrame with engineered features.
        val_df: Validation DataFrame with engineered features.
        test_df: Test DataFrame with engineered features.
        output_dir: Directory to save files.

    Returns:
        Dictionary mapping split name to file path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        output_path = output_dir / f"{split_name}_prepared.parquet"
        df.to_parquet(output_path)
        paths[split_name] = output_path
        logger.info(f"Saved {split_name} split to {output_path} ({len(df)} rows)")

    return paths


def load_prepared_splits(
    output_dir: str | Path,
) -> dict[str, LazyDataFrame]:
    """Load prepared splits as LazyDataFrame instances.

    Args:
        output_dir: Directory containing prepared parquet files.

    Returns:
        Dictionary mapping split name to LazyDataFrame.
    """
    output_dir = Path(output_dir)
    paths = {}

    for split_name in ["train", "val", "test"]:
        file_path = output_dir / f"{split_name}_prepared.parquet"
        if not file_path.exists():
            raise FileNotFoundError(f"Prepared split file not found: {file_path}")
        paths[split_name] = LazyDataFrame(file_path)

    logger.info(f"Loaded lazy references for {len(paths)} splits from {output_dir}")
    return paths
