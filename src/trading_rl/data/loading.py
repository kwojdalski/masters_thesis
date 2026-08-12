"""Data loading utilities for trading RL."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from gym_trading_env.downloader import download

from logger import get_logger
from trading_rl.data_loading import MemmapPaths

logger = get_logger(__name__)


@dataclass(frozen=True)
class PreparedDataset:
    """Prepared RL dataset with split dataframes and derived metadata."""

    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    feature_columns: list[str]
    price_column: str
    raw_columns: list[str]
    # Per-symbol memmap paths for StreamingTradingEnv. None when memmap_dir is
    # not configured; set by _build_pooled_dataset / build_prepared_dataset.
    memmap_train_paths: list[MemmapPaths] | None = None
    # Fitted scaler states keyed by feature output name, saved with checkpoints
    # so evaluation can use training-time normalization statistics.
    feature_pipeline_state: dict[str, dict[str, float]] | None = None


@dataclass(frozen=True)
class FeaturePipelineRestoreResult:
    """Feature pipeline plus metadata about restored training-time state."""

    pipeline: Any
    restored: bool
    state_size: int = 0
    source: str | None = None


def restore_pipeline_state(pipeline: Any, state: dict[str, dict[str, float]]) -> None:
    """Restore training-time scaler statistics into a FeaturePipeline.

    Delegates to ``pipeline.load_state()`` so that the fitted-state flag is
    managed inside the class rather than poked from outside.

    Args:
        pipeline: A FeaturePipeline instance (fitted or not).
        state: Mapping from feature output names to scaler state dicts, as
            saved by dump_pipeline_state / save_checkpoint.
    """
    restored = pipeline.load_state(state)
    logger.debug(
        "restore pipeline state restored={} total={}",
        restored,
        len(pipeline.features),
    )


def load_feature_pipeline_state_from_checkpoint(
    checkpoint_path: str | Path | None,
) -> dict[str, dict[str, float]] | None:
    """Load saved feature pipeline scaler state from a training checkpoint."""
    if checkpoint_path is None:
        return None

    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        return None

    try:
        import torch

        checkpoint = torch.load(checkpoint_file, weights_only=True)
    except Exception as exc:
        logger.warning(
            "failed to load feature pipeline state checkpoint={} error={}",
            checkpoint_file,
            exc,
        )
        return None

    state = checkpoint.get("feature_pipeline_state")
    return state if state else None


def build_feature_pipeline_with_state(
    feature_config: str | Path,
    *,
    feature_pipeline_state: dict[str, dict[str, float]] | None = None,
    checkpoint_path: str | Path | None = None,
) -> FeaturePipelineRestoreResult:
    """Build a FeaturePipeline and restore training-time scaler state when available.

    The checkpoint serialization detail lives here so evaluation callers do not
    each need to know how feature scaler state is stored on disk.
    """
    from trading_rl.features import FeaturePipeline

    pipeline = FeaturePipeline.from_yaml(str(feature_config))

    state_source = "provided"
    state = feature_pipeline_state
    if not state:
        state_source = "checkpoint"
        state = load_feature_pipeline_state_from_checkpoint(checkpoint_path)

    if state:
        restore_pipeline_state(pipeline, state)
        return FeaturePipelineRestoreResult(
            pipeline=pipeline,
            restored=True,
            state_size=len(state),
            source=state_source,
        )

    return FeaturePipelineRestoreResult(pipeline=pipeline, restored=False)


def dump_pipeline_state(pipeline: Any) -> dict[str, dict] | None:
    """Extract scaler state from a fitted FeaturePipeline for checkpointing.

    Symmetric counterpart to restore_pipeline_state.  Returns None when the
    pipeline has no features with serialisable scaler state.

    Args:
        pipeline: A fitted FeaturePipeline instance.

    Returns:
        Mapping from feature output names to scaler state dicts, or None.
    """
    if pipeline is None:
        return None
    state: dict[str, dict] = {}
    for feature in pipeline.features:
        name = feature.get_output_name()
        scaler = getattr(feature, "scaler", None)
        if scaler is not None and hasattr(scaler, "state_dict"):
            state[name] = scaler.state_dict()
    return state if state else None


def download_trading_data(
    exchange_names: list[str],
    symbols: list[str],
    timeframe: str,
    data_dir: str,
    since: Any | None = None,
) -> None:
    """Download historical trading data from exchanges.

    Args:
        exchange_names: List of exchange names (e.g., ["binance"])
        symbols: List of trading pairs (e.g., ["BTC/USDT"])
        timeframe: Timeframe for candles (e.g., "1h", "1d")
        data_dir: Directory to save downloaded data
        since: Start date for data download
    """
    if download is None:
        raise ImportError(
            "gym_trading_env package is required for data downloading. "
            "Install it with: pip install gym-trading-env"
        )

    logger.info("download data symbols={} exchanges={}", symbols, exchange_names)
    download(
        exchange_names=exchange_names,
        symbols=symbols,
        timeframe=timeframe,
        dir=data_dir,
        since=since,
    )
    logger.info("download data complete")


def load_trading_data(data_path: str) -> pd.DataFrame:
    """Load trading data from parquet or pickle file.

    Args:
        data_path: Path to parquet or pickle file

    Returns:
        DataFrame with OHLCV data
    """
    data_file = Path(data_path)
    logger.info("load data path={}", data_file)
    suffix = data_file.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        df = pd.read_pickle(data_file)  # noqa: S301 -- local project data cache, not untrusted input
    elif suffix in {".parquet"}:
        df = pd.read_parquet(data_file)
    else:
        raise ValueError(
            f"Unsupported data format '{suffix}' for file {data_file}. "
            "Supported formats: .pkl, .pickle, .parquet"
        )
    logger.info("load data n_rows={}", len(df))
    return df
