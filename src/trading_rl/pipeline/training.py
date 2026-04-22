"""Training/runtime orchestration helpers for the main RL pipeline."""

from __future__ import annotations

import datetime
import logging
import os
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import torch
from rich.console import Console
from rich.table import Table

from logger import get_logger as get_project_logger
from logger import setup_logging as configure_root_logging
from trading_rl.callbacks import MLflowTrainingCallback
from trading_rl.config import ExperimentConfig
from trading_rl.constants import Algorithm
from trading_rl.data_utils import PreparedDataset, build_prepared_dataset
from trading_rl.envs import AlgorithmicEnvironmentBuilder
from trading_rl.envs.trading_envs import EnvBackend
from trading_rl.trainers.ppo import PPOTrainerContinuous
from trading_rl.training import DDPGTrainer, PPOTrainer, TD3Trainer


@dataclass(frozen=True)
class TrainingBundle:
    """Constructed training runtime objects derived from config and dataset."""

    train_env: Any
    trainer: Any
    mlflow_callback: MLflowTrainingCallback | None
    algorithm: str
    n_obs: int
    n_act: int


@dataclass(frozen=True)
class ExperimentRuntime:
    """Top-level runtime bundle for one experiment execution."""

    logger: logging.Logger
    effective_experiment_name: str
    prepared_dataset: PreparedDataset
    training_bundle: TrainingBundle


def _format_head_value(value: Any) -> str:
    if isinstance(value, float):
        abs_value = abs(value)
        if (abs_value >= 1e5) or (0 < abs_value < 1e-3):
            return f"{value:.4e}"
        return f"{value:.6f}"
    if isinstance(value, (int, np.integer)):
        return str(value)
    return str(value)


def _print_training_data_head_table(
    train_df: pd.DataFrame, n_rows: int = 5, max_columns: int = 12
) -> None:
    if train_df.empty:
        return

    head_df = train_df.head(n_rows)
    visible_columns = list(head_df.columns[:max_columns])
    hidden_count = max(0, len(head_df.columns) - len(visible_columns))

    table = Table(title="Training Data Head")
    table.add_column("index", style="cyan")
    for col in visible_columns:
        table.add_column(str(col), justify="right")

    for idx, row in head_df.iterrows():
        row_cells = [str(idx)]
        row_cells.extend(_format_head_value(row[col]) for col in visible_columns)
        table.add_row(*row_cells)

    console = Console()
    console.print(table)
    if hidden_count > 0:
        console.print(
            f"[dim]Showing {len(visible_columns)} of {len(head_df.columns)} columns "
            f"({hidden_count} hidden).[/dim]"
        )


def setup_logging(config: ExperimentConfig) -> logging.Logger:
    """Setup logging configuration."""
    log_file_path = Path(config.logging.log_dir) / config.logging.log_file
    log_level = os.getenv("LOG_LEVEL") or config.logging.log_level

    configure_root_logging(
        level=log_level,
        log_file=str(log_file_path),
        console_output=True,
        colored_output=True,
    )

    import warnings

    from plotnine.exceptions import PlotnineWarning

    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", category=PlotnineWarning)

    logger = get_project_logger(__name__)
    logger.info("Starting experiment: %s", config.experiment_name)
    return logger


def set_seed(seed: int | None) -> int:
    """Set random seeds for reproducibility."""
    import random

    if seed is None:
        seed = random.randint(1, 100000)  # noqa: S311
        logging.getLogger(__name__).info("Generated random seed: %s", seed)

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    return seed


def _select_trainer_class(algorithm: str, backend: str):
    is_continuous_env = backend in {EnvBackend.TRADINGENV, EnvBackend.GYM_TRADING_CONTINUOUS}
    algorithm_upper = algorithm.upper()

    if algorithm_upper == Algorithm.PPO:
        return PPOTrainerContinuous if is_continuous_env else PPOTrainer
    if algorithm_upper == Algorithm.TD3:
        return TD3Trainer
    if algorithm_upper == Algorithm.DDPG:
        return DDPGTrainer
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def _build_train_env(
    dataset: PreparedDataset,
    config: ExperimentConfig,
    logger: logging.Logger,
) -> Any:
    logger.info("Creating environment...")
    env = AlgorithmicEnvironmentBuilder().create(dataset.train_df, config)
    logger.debug("Environment specs:")
    logger.debug("  Observation spec: %s", env.observation_spec)
    logger.debug("  Action spec: %s", env.action_spec)
    logger.debug("  Reward spec: %s", env.reward_spec)
    return env


def _build_trainer(
    env: Any,
    config: ExperimentConfig,
    algorithm: str,
    effective_experiment_name: str,
    logger: logging.Logger,
) -> tuple[Any, int, int]:
    n_obs = env.observation_spec["observation"].shape[-1]
    n_act = env.action_spec.shape[-1]
    logger.info("Environment: %s observations, %s actions", n_obs, n_act)

    backend = getattr(config.env, "backend", "")
    logger.info(
        "Creating models for %s algorithm (Backend: %s)...",
        algorithm,
        backend,
    )
    trainer_cls = _select_trainer_class(algorithm, backend)

    if trainer_cls == PPOTrainerContinuous:
        logger.info("Selected PPOTrainerContinuous for continuous environment")
    elif trainer_cls == PPOTrainer:
        logger.info("Selected PPOTrainer for discrete environment")
    else:
        logger.info("Selected %s", trainer_cls.__name__)

    if algorithm == Algorithm.TD3:
        actor, qvalue_net = trainer_cls.build_models(n_obs, n_act, config, env)
        trainer = trainer_cls(
            actor=actor,
            qvalue_net=qvalue_net,
            env=env,
            config=config.training,
            checkpoint_dir=config.logging.log_dir,
            checkpoint_prefix=effective_experiment_name,
        )
    else:
        actor, value_net = trainer_cls.build_models(n_obs, n_act, config, env)
        trainer = trainer_cls(
            actor=actor,
            value_net=value_net,
            env=env,
            config=config.training,
            checkpoint_dir=config.logging.log_dir,
            checkpoint_prefix=effective_experiment_name,
        )

    return trainer, n_obs, n_act


def _build_mlflow_callback(
    *,
    config: ExperimentConfig,
    dataset: PreparedDataset,
    effective_experiment_name: str,
    progress_bar: Any,
) -> MLflowTrainingCallback:
    tracking_uri = getattr(getattr(config, "tracking", None), "tracking_uri", None)
    estimated_episodes = max(1, config.training.max_steps // config.data.train_size)
    price_series = dataset.train_df[dataset.price_column]

    return MLflowTrainingCallback(
        effective_experiment_name,
        tracking_uri=tracking_uri,
        progress_bar=progress_bar,
        total_episodes=estimated_episodes if progress_bar else None,
        price_series=price_series,
        initial_portfolio_value=config.env.initial_portfolio_value,
        reward_type=config.env.reward_type,
        config_for_run_name=config,
    )


def _build_training_bundle(
    *,
    config: ExperimentConfig,
    dataset: PreparedDataset,
    effective_experiment_name: str,
    logger: logging.Logger,
    progress_bar: Any,
    create_mlflow_callback: bool,
) -> TrainingBundle:
    algorithm = getattr(config.training, "algorithm", "PPO").upper()
    train_env = _build_train_env(dataset, config, logger)
    trainer, n_obs, n_act = _build_trainer(
        train_env,
        config,
        algorithm,
        effective_experiment_name,
        logger,
    )

    mlflow_callback = None
    if create_mlflow_callback:
        mlflow_callback = _build_mlflow_callback(
            config=config,
            dataset=dataset,
            effective_experiment_name=effective_experiment_name,
            progress_bar=progress_bar,
        )

    return TrainingBundle(
        train_env=train_env,
        trainer=trainer,
        mlflow_callback=mlflow_callback,
        algorithm=algorithm,
        n_obs=n_obs,
        n_act=n_act,
    )


def _print_config_debug(config: ExperimentConfig, logger: logging.Logger) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return

    def format_key(key: str) -> str:
        return key.replace("_", " ").title()

    def format_value(value: Any) -> str:
        if isinstance(value, datetime.datetime):
            return value.isoformat()
        if isinstance(value, list):
            return str(value)
        return str(value)

    def print_dataclass(obj: Any, indent: int = 0) -> None:
        if not is_dataclass(obj):
            return

        prefix = "  " * indent
        for field in fields(obj):
            key = field.name
            value = getattr(obj, key)
            formatted_key = format_key(key)

            if is_dataclass(value):
                if indent == 0:
                    logger.debug("%s\033[93m%s:\033[0m", prefix, formatted_key)
                else:
                    logger.debug("%s%s:", prefix, formatted_key)
                print_dataclass(value, indent + 1)
            else:
                logger.debug("%s%s: %s", prefix, formatted_key, format_value(value))

    logger.debug("=" * 60)
    logger.debug("CONFIGURATION VALUES")
    logger.debug("=" * 60)
    print_dataclass(config)
    logger.debug("=" * 60)


def setup_mlflow_experiment(
    config: ExperimentConfig, experiment_name: str | None = None
) -> str:
    """Setup MLflow experiment for tracking.

    Args:
        config: Experiment configuration.
        experiment_name: Optional override for the MLflow experiment name.

    Returns:
        The effective experiment name that was set.
    """
    tracking_uri = getattr(getattr(config, "tracking", None), "tracking_uri", None)
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    exp_name = experiment_name or config.experiment_name
    mlflow.set_experiment(exp_name)
    return exp_name


def build_experiment_runtime(
    config: ExperimentConfig,
    experiment_name: str | None = None,
    progress_bar: Any = None,
    create_mlflow_callback: bool = True,
) -> ExperimentRuntime:
    """Build typed runtime state used by fresh and resumed runs."""
    effective_experiment_name = experiment_name or config.experiment_name

    logger = setup_logging(config)
    config.seed = set_seed(config.seed)
    _print_config_debug(config, logger)

    logger.info("Preparing data...")
    logger.debug("  Data path: %s", config.data.data_path)
    logger.debug("  Train size: %s", config.data.train_size)
    logger.debug(
        "  Feature config: %s",
        getattr(config.data, "feature_config", None),
    )

    prepared_dataset = build_prepared_dataset(config, logger)
    train_df = prepared_dataset.train_df
    val_df = prepared_dataset.val_df
    test_df = prepared_dataset.test_df

    if logger.isEnabledFor(logging.INFO):
        _print_training_data_head_table(train_df)

    logger.debug(
        "Data loaded - train: %s, val: %s, test: %s, columns: %s",
        train_df.shape,
        val_df.shape,
        test_df.shape,
        list(train_df.columns),
    )

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Training data statistics:")
        if "close" in train_df.columns:
            logger.debug(
                "  Close price - min: %.2f, max: %.2f, mean: %.2f",
                train_df["close"].min(),
                train_df["close"].max(),
                train_df["close"].mean(),
            )
            logger.debug("  Close price std: %.2f", train_df["close"].std())

        feature_cols = [col for col in train_df.columns if "feature" in col.lower()]
        if feature_cols:
            logger.debug("  Features found: %s", feature_cols)
        else:
            logger.debug("  No feature_* columns found in prepared data")

    training_bundle = _build_training_bundle(
        config=config,
        dataset=prepared_dataset,
        effective_experiment_name=effective_experiment_name,
        logger=logger,
        progress_bar=progress_bar,
        create_mlflow_callback=create_mlflow_callback,
    )

    if mlflow.active_run():
        MLflowTrainingCallback.log_parameter_faq_artifact()
        MLflowTrainingCallback.log_training_parameters(config)
        MLflowTrainingCallback.log_config_artifact(config)
        MLflowTrainingCallback.log_data_overview(train_df, config)

    return ExperimentRuntime(
        logger=logger,
        effective_experiment_name=effective_experiment_name,
        prepared_dataset=prepared_dataset,
        training_bundle=training_bundle,
    )

