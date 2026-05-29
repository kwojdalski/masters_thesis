from __future__ import annotations

import shutil
from typing import Any
from pathlib import Path

import pandas as pd


def log_dataset_summary(
    df: pd.DataFrame,
    output_path: Path,
    *,
    context: str,
    logger: Any,
) -> None:
    """Log common dataset summary information."""
    logger.info("{} saved to {}", context, output_path)
    logger.trace("n_rows={} n_cols={}", *df.shape)
    if not df.empty:
        logger.trace("Index range: {} -> {}", df.index.min(), df.index.max())
        if "close" in df.columns:
            logger.trace(
                "Close price range: %.2f -> %.2f",
                df["close"].min(),
                df["close"].max(),
            )


def load_data(source_dir: Path, filename: str, logger: Any) -> pd.DataFrame:
    """Load data from a parquet file in source_dir."""
    filepath = source_dir / filename
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    logger.debug("load dataset path={}", filepath)
    df = pd.read_parquet(filepath)
    logger.trace("loaded rows={} path={}", len(df), filepath)
    return df


def copy_data(
    source_dir: Path,
    output_dir: Path,
    source_file: str,
    output_file: str | None,
    logger: Any,
) -> None:
    """Copy a parquet file from source_dir to output_dir without modification."""
    source_path = source_dir / source_file
    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")
    dest = output_dir / (output_file or source_file)
    logger.trace("copy file src={} dst={}", source_path, dest)
    shutil.copy2(source_path, dest)
    logger.info("copied src={} dst={}", source_path, dest)


def list_source_files(source_dir: Path, logger: Any) -> list[str]:
    """Return names of all parquet files in source_dir."""
    filenames = [f.name for f in source_dir.glob("*.parquet")]
    logger.trace("Discovered {} parquet files in {}", len(filenames), source_dir)
    return filenames
