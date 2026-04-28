"""Checkpoint resumption logic for RL training experiments."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

from trading_rl.callbacks import MLflowTrainingCallback
from trading_rl.config import ExperimentConfig


@dataclass
class CheckpointResumptionResult:
    """Result from checkpoint resumption setup."""

    mlflow_callback: MLflowTrainingCallback
    effective_experiment_name: str
    original_steps: int


def _setup_mlflow_tracking_from_checkpoint(trainer: Any) -> str | None:
    """Extract and setup MLflow tracking URI from checkpoint."""
    tracking_uri = getattr(trainer, "mlflow_tracking_uri", None)
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    return tracking_uri


def _resolve_experiment_name_from_checkpoint(
    trainer: Any,
    config: ExperimentConfig,
    effective_experiment_name: str,
    logger: logging.Logger,
) -> str:
    """Resolve experiment name from checkpoint metadata, updating config if needed."""
    resume_experiment_name = getattr(trainer, "mlflow_experiment_name", None)

    if not resume_experiment_name:
        experiment_id = getattr(trainer, "mlflow_experiment_id", None)
        if experiment_id:
            experiment = mlflow.get_experiment(experiment_id)
            resume_experiment_name = experiment.name if experiment else None

    if resume_experiment_name:
        if resume_experiment_name != config.experiment_name:
            logger.info("resume experiment name=%s", resume_experiment_name)
        config.experiment_name = resume_experiment_name
        effective_experiment_name = resume_experiment_name
        if hasattr(trainer, "checkpoint_prefix"):
            trainer.checkpoint_prefix = resume_experiment_name

    return effective_experiment_name


def _start_mlflow_run_for_resumption(
    trainer: Any,
    original_steps: int,
    logger: logging.Logger,
) -> None:
    """Start or resume MLflow run for checkpoint resumption."""
    resume_run_id = getattr(trainer, "mlflow_run_id", None)
    if resume_run_id:
        logger.info("resume mlflow run run_id=%s", resume_run_id)
        mlflow.start_run(run_id=resume_run_id)
    else:
        logger.info("create mlflow run from_step=%d", original_steps)
        mlflow.start_run(run_name=f"resumed_step_{original_steps}")


def _get_episode_count_from_trainer(trainer: Any) -> int:
    """Extract current episode count from trainer state."""
    logged = (
        trainer.logs.get("episode_log_count") if hasattr(trainer, "logs") else None
    )
    if logged:
        return int(logged[-1])
    total = trainer.total_episodes
    if hasattr(total, "item"):
        return int(total.item())
    return int(total)


def _create_resumption_callback(
    trainer: Any,
    config: ExperimentConfig,
    train_df: pd.DataFrame,
    effective_experiment_name: str,
    tracking_uri: str | None,
) -> MLflowTrainingCallback:
    """Create MLflow callback for resumed training with proper state."""
    configured_price_column = getattr(config.env, "price_column", None)
    if (
        isinstance(configured_price_column, str)
        and configured_price_column in train_df.columns
    ):
        price_series = train_df[configured_price_column]
    elif "close" in train_df.columns:
        price_series = train_df["close"]
    elif "price" in train_df.columns:
        price_series = train_df["price"]
    else:
        price_series = None

    fallback_uri = getattr(getattr(config, "tracking", None), "tracking_uri", None)
    mlflow_callback = MLflowTrainingCallback(
        effective_experiment_name,
        tracking_uri=tracking_uri or fallback_uri,
        price_series=price_series,
        start_run=False,
    )
    mlflow_callback._episode_count = _get_episode_count_from_trainer(trainer)
    return mlflow_callback


def setup_checkpoint_resumption(
    checkpoint_path: str,
    trainer: Any,
    config: ExperimentConfig,
    train_df: pd.DataFrame,
    effective_experiment_name: str,
    additional_steps: int | None,
    logger: logging.Logger,
    setup_mlflow_experiment_fn: Any,
) -> CheckpointResumptionResult:
    """Load checkpoint, setup MLflow tracking, and create callback for resumed training.

    Args:
        checkpoint_path: Path to checkpoint file.
        trainer: Trainer instance to load checkpoint into.
        config: Experiment configuration (modified in place).
        train_df: Training dataframe for price series.
        effective_experiment_name: Initial experiment name (may be updated).
        additional_steps: Optional additional training steps.
        logger: Logger instance.
        setup_mlflow_experiment_fn: Callable matching setup_mlflow_experiment's signature.

    Returns:
        CheckpointResumptionResult with callback, experiment name, and original steps.
    """
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info("resume training from checkpoint path=%s", checkpoint_path)

    trainer.load_checkpoint(str(checkpoint_path))
    original_steps = trainer.total_count
    logger.info("load checkpoint step=%d", original_steps)

    if additional_steps:
        trainer.config.max_steps = original_steps + additional_steps
        logger.info("extend training additional_steps=%d target_steps=%d", additional_steps, trainer.config.max_steps)

    tracking_uri = _setup_mlflow_tracking_from_checkpoint(trainer)
    effective_experiment_name = _resolve_experiment_name_from_checkpoint(
        trainer, config, effective_experiment_name, logger
    )
    setup_mlflow_experiment_fn(config, effective_experiment_name)
    _start_mlflow_run_for_resumption(trainer, original_steps, logger)

    mlflow_callback = _create_resumption_callback(
        trainer, config, train_df, effective_experiment_name, tracking_uri
    )

    if mlflow.active_run():
        MLflowTrainingCallback.log_parameter_faq_artifact()
        MLflowTrainingCallback.log_training_parameters(config)
        MLflowTrainingCallback.log_config_artifact(config)
        MLflowTrainingCallback.log_data_overview(train_df, config)

    return CheckpointResumptionResult(
        mlflow_callback=mlflow_callback,
        effective_experiment_name=effective_experiment_name,
        original_steps=original_steps,
    )
