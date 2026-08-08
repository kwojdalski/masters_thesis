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
import torch
from torchrl.envs import TransformedEnv
from loguru import logger
from logger import get_logger as get_project_logger
from logger import is_level_enabled
from logger import log_banner
from logger import print_df_head
from logger import setup_logging as configure_root_logging
from trading_rl.profiler import get_profiler
from trading_rl.callbacks import MLflowTrainingCallback
from trading_rl.config import ExperimentConfig, LoggingParams, MLflowCallbackParams, TrainerConstructionParams
from trading_rl.data_utils import PreparedDataset, build_prepared_dataset
from trading_rl.envs import AlgorithmicEnvironmentBuilder, EnvBuildParams
from trading_rl.envs.trading_envs import EnvBackend
from trading_rl.trainers.base import BaseTrainer
import trading_rl.trainers.ddpg  # noqa: F401 — registers DDPGTrainer
import trading_rl.trainers.ppo  # noqa: F401 — registers PPOTrainer, PPOTrainerContinuous
import trading_rl.trainers.random_trainer  # noqa: F401 — registers RandomTrainer
import trading_rl.trainers.recurrent_ppo  # noqa: F401 — registers RecurrentPPOTrainer
import trading_rl.trainers.sac  # noqa: F401 — registers SACTrainer
import trading_rl.trainers.td3  # noqa: F401 — registers TD3Trainer
from trading_rl.trainers.registry import TrainerRegistry


@dataclass(frozen=True)
class TrainingBundle:
    """Constructed training runtime objects derived from config and dataset."""

    train_env: TransformedEnv
    trainer: BaseTrainer
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




def setup_logging(params: LoggingParams, experiment_name: str | None = None) -> logging.Logger:
    """Setup logging configuration."""
    log_level = os.getenv("LOG_LEVEL") or params.log_level
    Path(params.log_dir).mkdir(parents=True, exist_ok=True)
    Path(params.tensorboard_dir).mkdir(parents=True, exist_ok=True)
    log_file_path = (
        str(Path(params.log_dir) / params.log_file)
        if params.log_to_file
        else None
    )

    configure_root_logging(
        level=log_level,
        log_file=log_file_path,
        console_output=True,
        colored_output=True,
    )

    import warnings

    from plotnine.exceptions import PlotnineWarning

    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", category=PlotnineWarning)

    logger = get_project_logger(__name__)
    if experiment_name:
        logger.info("start experiment name={}", experiment_name)
    return logger


def set_seed(seed: int | None) -> int:
    """Set random seeds for reproducibility."""
    import os
    import random

    if seed is None:
        seed = random.randint(1, 100000)  # noqa: S311
        logger.info("Generated random seed: {}", seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


def _select_trainer_class(algorithm: str, backend: str):
    is_continuous_env = backend in {EnvBackend.TRADINGENV, EnvBackend.GYM_TRADING_CONTINUOUS}
    return TrainerRegistry.get(algorithm, is_continuous=is_continuous_env)


def _build_train_env(
    dataset: PreparedDataset,
    config: ExperimentConfig,
    logger: logging.Logger,
) -> Any:
    logger.info("build environment")
    env = AlgorithmicEnvironmentBuilder().create(dataset.train_df, EnvBuildParams.from_config(config))
    logger.trace("environment obs_spec={} action_spec={} reward_spec={}", env.observation_spec, env.action_spec, env.reward_spec)
    return env


def _build_trainer(
    env: Any,
    config: ExperimentConfig,
    algorithm: str,
    effective_experiment_name: str,
    logger: logging.Logger,
    *,
    eval_env: Any | None = None,
    eval_data_len: int | None = None,
) -> Any:
    import math
    n_obs = math.prod(env.observation_spec["observation"].shape)
    n_act = env.action_spec.shape[-1]
    logger.info("build environment n_obs={} n_act={}", n_obs, n_act)

    backend = getattr(config.env, "backend", "")
    logger.info("build models algorithm={} backend={}", algorithm, backend)
    trainer_cls = _select_trainer_class(algorithm, backend)
    logger.info("select trainer cls={}", trainer_cls.__name__)

    trainer_params = TrainerConstructionParams.from_config(
        config, n_obs, n_act, effective_experiment_name
    )
    actor, value_net = trainer_cls.build_models(n_obs, n_act, config, env)
    return trainer_cls(
        actor=actor,
        value_net=value_net,
        env=env,
        config=trainer_params.config,
        n_obs=trainer_params.n_obs,
        n_act=trainer_params.n_act,
        actor_hidden_dims=trainer_params.actor_hidden_dims,
        value_hidden_dims=trainer_params.value_hidden_dims,
        eval_config=trainer_params.eval_config,
        eval_env=eval_env,
        eval_data_len=eval_data_len,
        checkpoint_dir=trainer_params.checkpoint_dir,
        checkpoint_prefix=trainer_params.checkpoint_prefix,
    )


def _build_mlflow_callback(
    *,
    params: MLflowCallbackParams,
    effective_experiment_name: str,
    progress_bar: Any,
    config_for_run_name: ExperimentConfig,
) -> MLflowTrainingCallback:
    return MLflowTrainingCallback(
        effective_experiment_name,
        tracking_uri=params.tracking_uri,
        progress_bar=progress_bar,
        total_episodes=params.total_episodes if progress_bar else None,
        price_series=params.price_series,
        initial_portfolio_value=params.initial_portfolio_value,
        reward_type=params.reward_type,
        reward_scale=params.reward_scale,
        action_positions=params.action_positions,
        config_for_run_name=config_for_run_name,
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
    eval_env = None
    if getattr(config.training, "eval_interval", 0) > 0:
        eval_env = AlgorithmicEnvironmentBuilder().create(
            dataset.val_df, EnvBuildParams.from_config(config), use_memmap=False
        )
    trainer = _build_trainer(
        train_env,
        config,
        algorithm,
        effective_experiment_name,
        logger,
        eval_env=eval_env,
        eval_data_len=len(dataset.val_df),
    )

    mlflow_callback = None
    if create_mlflow_callback:
        mlflow_params = MLflowCallbackParams.from_config(config, dataset)
        mlflow_callback = _build_mlflow_callback(
            params=mlflow_params,
            effective_experiment_name=effective_experiment_name,
            progress_bar=progress_bar,
            config_for_run_name=config,
        )

    return TrainingBundle(
        train_env=train_env,
        trainer=trainer,
        mlflow_callback=mlflow_callback,
        algorithm=algorithm,
        n_obs=trainer.n_obs,
        n_act=trainer.n_act,
    )


def _print_config_debug(config: ExperimentConfig, logger: logging.Logger) -> None:
    if not is_level_enabled("TRACE"):
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
                logger.trace("{}{}:", prefix, formatted_key)
                print_dataclass(value, indent + 1)
            else:
                logger.trace("{}{}: {}", prefix, formatted_key, format_value(value))

    logger.trace("=" * 60)
    logger.trace("configuration values")
    logger.trace("=" * 60)
    print_dataclass(config)
    logger.trace("=" * 60)


def setup_mlflow_experiment(
    experiment_name: str,
    tracking_uri: str | None = None,
) -> str:
    """Configure the MLflow tracking URI and set the active experiment.

    Args:
        experiment_name: The MLflow experiment name to activate.
        tracking_uri: Optional tracking server URI; uses MLflow default when None.

    Returns:
        experiment_name unchanged.
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return experiment_name


def _log_data_diagnostics(
    prepared_dataset: "PreparedDataset",
    logger: logging.Logger,
) -> None:
    """Log data shape and feature diagnostics after preparation."""
    train_df = prepared_dataset.train_df
    val_df = prepared_dataset.val_df
    test_df = prepared_dataset.test_df

    if is_level_enabled("INFO"):
        feature_cols = [c for c in train_df.columns if str(c).startswith("feature_")]
        other_cols = [c for c in train_df.columns if not str(c).startswith("feature_")]
        print_df_head(
            train_df[feature_cols + other_cols],
            title=f"Prepared Training Split  ({len(feature_cols)} feature_* cols used as observations, {len(train_df.columns)} total)",
            max_columns=7,
            paginate=True,
        )

    logger.debug(
        "Data loaded - train: {}, val: {}, test: {}, columns: {}",
        train_df.shape, val_df.shape, test_df.shape, list(train_df.columns),
    )

    if is_level_enabled("TRACE"):
        logger.trace("training data statistics")
        if "close" in train_df.columns:
            logger.trace(
                "  Close price - min: {}, max: {}, mean: {}",
                train_df["close"].min(), train_df["close"].max(), train_df["close"].mean(),
            )
            logger.trace("  Close price std: {:.2f}", train_df["close"].std())
        feature_cols = [col for col in train_df.columns if "feature" in col.lower()]
        logger.trace("  Features found: {}" if feature_cols else "  No feature_* columns found in prepared data", feature_cols or "")

    n_feat = len([c for c in train_df.columns if str(c).startswith("feature_")])
    log_banner(
        logger,
        f"DATA PREPARATION END  train={train_df.shape[0]:,}  val={val_df.shape[0]:,}  test={test_df.shape[0]:,}  features={n_feat}",
    )


def _log_mlflow_artifacts(
    config: ExperimentConfig,
    prepared_dataset: "PreparedDataset",
    create_mlflow_callback: bool,
) -> None:
    """Log static MLflow artifacts (config, FAQs, optional data overviews)."""
    if not (create_mlflow_callback and mlflow.active_run()):
        return
    MLflowTrainingCallback.log_parameter_faq_artifact()
    MLflowTrainingCallback.log_training_parameters(config)
    MLflowTrainingCallback.log_config_artifact(config)
    if getattr(getattr(config, "logging", None), "log_data_overviews", False):
        train_df = prepared_dataset.train_df
        MLflowTrainingCallback.log_raw_data_overview(train_df, config)
        MLflowTrainingCallback.log_transformed_data_overview(train_df, config)


def _configure_experiment_environment(
    config: ExperimentConfig,
    experiment_name: str | None,
) -> tuple[Any, str]:
    """Configure logging and seed for an experiment; return (logger, effective_name).

    Extracted from build_experiment_runtime so this phase can be called
    independently (e.g. in resumed runs or tests that need a configured logger
    without building the full training bundle).
    """
    effective_experiment_name = experiment_name or config.experiment_name
    logging_params = LoggingParams.from_config(config)
    logger = setup_logging(logging_params, effective_experiment_name)
    config.seed = set_seed(config.seed)
    _print_config_debug(config, logger)
    return logger, effective_experiment_name


def build_experiment_runtime(
    config: ExperimentConfig,
    experiment_name: str | None = None,
    progress_bar: Any = None,
    create_mlflow_callback: bool = True,
) -> ExperimentRuntime:
    """Build typed runtime state used by fresh and resumed runs."""
    logger, effective_experiment_name = _configure_experiment_environment(
        config, experiment_name
    )

    profiler = get_profiler()

    logger.info("prepare data")
    logger.debug("data path={} train_size={} feature_config={}", config.data.data_path, config.data.train_size, getattr(config.data, "feature_config", None))

    log_banner(logger, "DATA PREPARATION START")
    with profiler.stage("data_preparation", 2):
        prepared_dataset = build_prepared_dataset(config, logger)
    _log_data_diagnostics(prepared_dataset, logger)

    with profiler.stage("train_env_build", 2):
        training_bundle = _build_training_bundle(
            config=config,
            dataset=prepared_dataset,
            effective_experiment_name=effective_experiment_name,
            logger=logger,
            progress_bar=progress_bar,
            create_mlflow_callback=create_mlflow_callback,
        )

    _log_mlflow_artifacts(config, prepared_dataset, create_mlflow_callback)

    return ExperimentRuntime(
        logger=logger,
        effective_experiment_name=effective_experiment_name,
        prepared_dataset=prepared_dataset,
        training_bundle=training_bundle,
    )

