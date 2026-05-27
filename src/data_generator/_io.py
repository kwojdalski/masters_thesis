from __future__ import annotations

import shutil
from logging import Logger
from pathlib import Path

import pandas as pd


def log_dataset_summary(
    df: pd.DataFrame,
    output_path: Path,
    *,
    context: str,
    logger: Logger,
) -> None:
    """Log common dataset summary information."""
    logger.info("%s saved to %s", context, output_path)
    logger.debug("n_rows=%d n_cols=%d", *df.shape)
    if not df.empty:
        logger.debug("Index range: %s -> %s", df.index.min(), df.index.max())
        if "close" in df.columns:
            logger.debug(
                "Close price range: %.2f -> %.2f",
                df["close"].min(),
                df["close"].max(),
            )


def load_data(source_dir: Path, filename: str, logger: Logger) -> pd.DataFrame:
    """Load data from a parquet file in source_dir."""
    filepath = source_dir / filename
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    logger.debug("load dataset path=%s", filepath)
    df = pd.read_parquet(filepath)
    logger.debug("loaded rows=%s path=%s", len(df), filepath)
    return df


def copy_data(
    source_dir: Path,
    output_dir: Path,
    source_file: str,
    output_file: str | None,
    logger: Logger,
) -> None:
    """Copy a parquet file from source_dir to output_dir without modification."""
    source_path = source_dir / source_file
    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")
    dest = output_dir / (output_file or source_file)
    logger.debug("copy file src=%s dst=%s", source_path, dest)
    shutil.copy2(source_path, dest)
    logger.info("copied src=%s dst=%s", source_path, dest)


def list_source_files(source_dir: Path, logger: Logger) -> list[str]:
    """Return names of all parquet files in source_dir."""
    filenames = [f.name for f in source_dir.glob("*.parquet")]
    logger.debug("Discovered %s parquet files in %s", len(filenames), source_dir)
    return filenames
