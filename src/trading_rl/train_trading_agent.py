"""Main training script for trading agent - refactored version.

This is a clean, modular version of the trading RL training script.
All configuration, data processing, models, and training logic have been
separated into reusable modules.
"""

# %%
import contextlib
import logging
import os
import warnings
from pathlib import Path

import gym_trading_env  # noqa: F401
import mlflow
import numpy as np
import torch
import torch.multiprocessing as mp
from joblib import Memory

# No matplotlib configuration needed since we use plotnine exclusively
from logger import get_logger as get_project_logger
from logger import setup_logging as configure_root_logging
from logger import trace_calls
from trading_rl.callbacks import MLflowTrainingCallback
from trading_rl.config import ExperimentConfig
from trading_rl.data_utils import prepare_data
from trading_rl.envs import AlgorithmicEnvironmentBuilder
from trading_rl.plotting import visualize_training
from trading_rl.training import DDPGTrainer, PPOTrainer, TD3Trainer
from trading_rl.trainers.ppo import PPOTrainerContinuous

# Avoid torch_shm_manager requirement in restricted environments
mp.set_sharing_strategy("file_system")
# gym_trading_env sets warnings to errors; reset to defaults for TorchRL
warnings.filterwarnings("default")

# Setup joblib memory for caching expensive operations
memory = Memory(location=".cache/joblib", verbose=1)


def clear_cache():
    """Clear all joblib caches."""
    memory.clear(warn=True)


# %%
def setup_logging(config: ExperimentConfig):
    """Setup logging configuration."""
    # No matplotlib logging to disable since we use plotnine exclusively

    log_file_path = Path(config.logging.log_dir) / config.logging.log_file

    # Check for LOG_LEVEL environment variable override
    log_level = os.getenv("LOG_LEVEL") or config.logging.log_level

    configure_root_logging(
        level=log_level,
        log_file=str(log_file_path),
        console_output=True,
        # sys.stdout.isatty(),
        colored_output=True,
    )
    # Suppress noisy external library loggers
    import logging

    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    logger = get_project_logger(__name__)
    logger.info(f"Starting experiment: {config.experiment_name}")
    return logger


# %%
def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    import random

    # Seed all random number generators
    random.seed(seed)  # Python's built-in random module
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


# Environment builder used throughout training
env_builder = AlgorithmicEnvironmentBuilder()


# %%
# %%
@trace_calls(show_return=True)
def setup_mlflow_experiment(
    config: ExperimentConfig, experiment_name: str | None = None
) -> str:
    """Setup MLflow experiment for tracking.

    Args:
        experiment_name: Name of the MLflow experiment
    """
    tracking_uri = getattr(getattr(config, "tracking", None), "tracking_uri", None)
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    exp_name = experiment_name or config.experiment_name
    mlflow.set_experiment(exp_name)
    return exp_name


# %%


def _print_config_debug(config: ExperimentConfig, logger) -> None:
    """Print configuration values in debug mode using automatic traversal."""
    import datetime
    from dataclasses import fields, is_dataclass

    if not logger.isEnabledFor(logging.DEBUG):
        return

    def format_key(key: str) -> str:
        """Format key: remove underscores and title case."""
        return key.replace("_", " ").title()

    def format_value(value) -> str:
        """Format value for display."""
        if isinstance(value, datetime.datetime):
            return value.isoformat()
        elif isinstance(value, list):
            return str(value)
        else:
            return str(value)

    def print_dataclass(obj, indent: int = 0, logger=logger):
        """Recursively print dataclass fields."""
        if not is_dataclass(obj):
            return

        prefix = "  " * indent
        for field in fields(obj):
            key = field.name
            value = getattr(obj, key)
            formatted_key = format_key(key)

            if is_dataclass(value):
                # Print section header for nested dataclass with yellow highlighting
                if indent == 0:
                    # Top-level sections get yellow highlighting
                    logger.debug(f"{prefix}\033[93m{formatted_key}:\033[0m")
                else:
                    logger.debug(f"{prefix}{formatted_key}:")
                print_dataclass(value, indent + 1, logger)
            else:
                # Print key-value pair
                formatted_value = format_value(value)
                logger.debug(f"{prefix}{formatted_key}: {formatted_value}")

    logger.debug("=" * 60)
    logger.debug("CONFIGURATION VALUES")
    logger.debug("=" * 60)
    print_dataclass(config)
    logger.debug("=" * 60)


def build_training_context(
    config: ExperimentConfig,
    experiment_name: str | None = None,
    progress_bar=None,
) -> dict:
    """Build common training context used by fresh and resumed runs."""
    effective_experiment_name = experiment_name or config.experiment_name

    logger = setup_logging(config)
    set_seed(config.seed)

    _print_config_debug(config, logger)

    logger.info("Preparing data...")
    logger.debug(f"  Data path: {config.data.data_path}")
    logger.debug(f"  Train size: {config.data.train_size}")
    logger.debug(f"  No features: {getattr(config.data, 'no_features', False)}")

    df = prepare_data(
        data_path=config.data.data_path,
        download_if_missing=config.data.download_data,
        exchange_names=config.data.exchange_names,
        symbols=config.data.symbols,
        timeframe=config.data.timeframe,
        data_dir=config.data.data_dir,
        since=config.data.download_since,
        no_features=getattr(config.data, "no_features", False),
    )

    logger.debug(f"Data loaded - shape: {df.shape}, columns: {list(df.columns)}")

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Data statistics:")
        logger.debug(
            "  Close price - min: %.2f, max: %.2f, mean: %.2f",
            df["close"].min(),
            df["close"].max(),
            df["close"].mean(),
        )
        logger.debug(f"  Close price std: {df['close'].std():.2f}")

        feature_cols = [col for col in df.columns if "feature" in col.lower()]
        if feature_cols:
            logger.debug(f"  Features found: {feature_cols}")
        else:
            logger.debug("  No features found in data (using raw OHLCV only)")

    if mlflow.active_run():
        MLflowTrainingCallback.log_parameter_faq_artifact()
        MLflowTrainingCallback.log_training_parameters(config)
        MLflowTrainingCallback.log_config_artifact(config)
        MLflowTrainingCallback.log_data_overview(df, config)

    logger.info("Creating environment...")
    env = env_builder.create(df, config)

    n_obs = env.observation_spec["observation"].shape[-1]
    n_act = env.action_spec.shape[-1]
    logger.info(f"Environment: {n_obs} observations, {n_act} actions")

    logger.debug("Environment specs:")
    logger.debug(f"  Observation spec: {env.observation_spec}")
    logger.debug(f"  Action spec: {env.action_spec}")
    logger.debug(f"  Reward spec: {env.reward_spec}")

    backend = getattr(config.env, "backend", "")
    is_continuous_env = (
        backend == "tradingenv"
        or backend == "gym_trading_env.continuous"
    )

    algorithm = getattr(config.training, "algorithm", "PPO").upper()
    logger.info(f"Creating models for {algorithm} algorithm (Backend: {backend})...")

    if algorithm == "PPO":
        if is_continuous_env:
            logger.info("Selected PPOTrainerContinuous for continuous environment")
            trainer_cls = PPOTrainerContinuous
        else:
            logger.info("Selected PPOTrainer for discrete environment")
            trainer_cls = PPOTrainer
    elif algorithm == "TD3":
        trainer_cls = TD3Trainer
    elif algorithm == "DDPG":
        trainer_cls = DDPGTrainer
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    if algorithm == "TD3":
        actor, qvalue_net = trainer_cls.build_models(n_obs, n_act, config, env)
        trainer = trainer_cls(
            actor=actor,
            qvalue_net=qvalue_net,
            env=env,
            config=config.training,
            checkpoint_dir=config.logging.log_dir,
            checkpoint_prefix=config.experiment_name,
        )
    else:
        actor, value_net = trainer_cls.build_models(n_obs, n_act, config, env)
        trainer = trainer_cls(
            actor=actor,
            value_net=value_net,
            env=env,
            config=config.training,
            checkpoint_dir=config.logging.log_dir,
            checkpoint_prefix=config.experiment_name,
        )

    tracking_uri = getattr(getattr(config, "tracking", None), "tracking_uri", None)
    estimated_episodes = max(1, config.training.max_steps // config.data.train_size)
    mlflow_callback = MLflowTrainingCallback(
        effective_experiment_name,
        tracking_uri=tracking_uri,
        progress_bar=progress_bar,
        total_episodes=estimated_episodes if progress_bar else None,
        price_series=df["close"][: config.data.train_size],
    )

    return {
        "logger": logger,
        "df": df,
        "env": env,
        "trainer": trainer,
        "mlflow_callback": mlflow_callback,
        "algorithm": algorithm,
        "n_obs": n_obs,
        "n_act": n_act,
        "effective_experiment_name": effective_experiment_name,
    }


@trace_calls(show_return=False)
def run_single_experiment(
    custom_config: ExperimentConfig | None = None,
    experiment_name: str | None = None,
    progress_bar=None,
) -> dict:
    """Run a single training experiment with MLflow tracking.

    This function tracks both losses and position statistics in MLflow:
    - Multiple metrics logged simultaneously (losses, positions, rewards)
    - Parameters logged for configuration tracking
    - Artifacts logged for plots and models

    Args:
        custom_config: Optional custom configuration
        experiment_name: Optional override for MLflow experiment name (uses config.experiment_name by default)
        progress_bar: Optional Rich progress bar for episode tracking

    Returns:
        Dictionary with results
    """
    config = custom_config or ExperimentConfig()
    context = build_training_context(
        config=config,
        experiment_name=experiment_name,
        progress_bar=progress_bar,
    )
    logger = context["logger"]
    df = context["df"]
    trainer = context["trainer"]
    mlflow_callback = context["mlflow_callback"]
    algorithm = context["algorithm"]
    n_obs = context["n_obs"]
    n_act = context["n_act"]
    effective_experiment_name = context["effective_experiment_name"]
    # Train
    logger.info("Starting training...")
    logs = trainer.train(callback=mlflow_callback)

    # Save checkpoint
    checkpoint_path = (
        Path(config.logging.log_dir) / f"{config.experiment_name}_checkpoint.pt"
    )
    trainer.save_checkpoint(str(checkpoint_path))

    # Evaluate agent
    logger.info("Evaluating agent...")
    # Ensure max_steps doesn't exceed available data size
    eval_max_steps = min(
        config.training.eval_steps, len(df) - 1, config.data.train_size - 1
    )  # Use the smallest of: eval_steps, actual data size, or train_size
    reward_plot, action_plot, action_probs_plot, final_reward, last_positions = (
        trainer.evaluate(
            df[: config.data.train_size],  # Pass only the training portion
            max_steps=eval_max_steps,
            config=config,
            algorithm=algorithm,
        )
    )

    # Save evaluation plots as MLflow artifacts
    MLflowTrainingCallback.log_evaluation_plots(
        reward_plot=reward_plot,
        action_plot=action_plot,
        action_probs_plot=action_probs_plot,
        logs=logs,
    )

    # Detect backend type for proper metric naming
    is_portfolio_backend = config.env.backend == "tradingenv"

    # Prepare comprehensive metrics
    final_metrics = {
        # Performance metrics
        "final_reward": final_reward,
        "training_steps": len(logs.get("loss_value", [])),
        "evaluation_steps": eval_max_steps,  # actual max_steps used in evaluation
        # Use backend-aware naming for positions/weights
        ("portfolio_weights" if is_portfolio_backend else "last_position_per_episode"): last_positions,
        # Dataset metadata
        "data_start_date": str(df.index[0]) if not df.empty else "unknown",
        "data_end_date": str(df.index[-1]) if not df.empty else "unknown",
        "data_size": len(df),
        "train_size": config.data.train_size,
        # Environment configuration
        "trading_fees": config.env.trading_fees,
        "borrow_interest_rate": config.env.borrow_interest_rate,
        "positions": str(config.env.positions),
        # Network architecture
        "actor_hidden_dims": config.network.actor_hidden_dims,
        "value_hidden_dims": config.network.value_hidden_dims,
        "n_observations": n_obs,
        "n_actions": n_act,
        # Training configuration
        "experiment_name": config.experiment_name,
        "seed": config.seed,
        "actor_lr": config.training.actor_lr,
        "value_lr": config.training.value_lr,
        "buffer_size": config.training.buffer_size,
    }

    # Log final metrics to MLflow
    MLflowTrainingCallback.log_final_metrics(logs, final_metrics, mlflow_callback)

    logger.info("Training complete!")
    logger.info(f"Final reward: {final_reward:.4f}")
    logger.info(f"Checkpoint saved to: {checkpoint_path}")

    return {
        "trainer": trainer,
        "logs": logs,
        "final_metrics": final_metrics,
        "plots": {
            "loss": visualize_training(logs)
            if logs.get("loss_value") or logs.get("loss_actor")
            else None,
            "reward": reward_plot,
            "action": action_plot,
        },
    }


# %%
@trace_calls(show_return=True)
def run_multiple_experiments(
    n_trials: int = 5,
    base_seed: int | None = None,
    custom_config: ExperimentConfig | None = None,
    experiment_name: str | None = None,
    show_progress: bool = True,
) -> str:
    """Run multiple experiments and track with MLflow.

    Each experiment tracks:
    - Multiple metrics simultaneously (losses, positions, rewards)
    - Parameters for configuration comparison
    - Artifacts for plots and models

    Args:
        n_trials: Number of experiments to run
        base_seed: Base seed for reproducible experiments (each trial uses base_seed + trial_number)
        custom_config: Optional custom configuration
        experiment_name: Optional override for MLflow experiment name (uses config.experiment_name by default)
        show_progress: Whether to show progress bar for episodes

    Returns:
        MLflow experiment name with all results
    """
    # Load configuration to get experiment name
    from rich.progress import Progress

    from trading_rl.config import ExperimentConfig

    config = custom_config or ExperimentConfig()
    effective_experiment_name = experiment_name or config.experiment_name

    # Setup MLflow experiment
    setup_mlflow_experiment(config, effective_experiment_name)

    results = []

    logger = get_project_logger(__name__)

    # Create progress bar context if requested
    progress_context = Progress() if show_progress else None

    with progress_context if progress_context else contextlib.nullcontext() as progress:
        for trial_number in range(n_trials):
            logger.info(f"Running trial {trial_number + 1}/{n_trials}")

            # Create config with deterministic seed based on trial number
            if custom_config is not None:
                # Use custom config as base and copy it for this trial
                import copy

                trial_config = copy.deepcopy(custom_config)
            else:
                trial_config = ExperimentConfig()

            if base_seed is not None:
                trial_config.seed = base_seed + trial_number
            else:
                import random

                trial_config.seed = random.randint(1, 100000)  # noqa: S311

            # Keep the same experiment name for all trials
            trial_config.experiment_name = effective_experiment_name

            # Start a new MLflow run for this trial
            with mlflow.start_run(run_name=f"trial_{trial_number}"):
                result = run_single_experiment(
                    custom_config=trial_config,
                    progress_bar=progress if show_progress else None,
                )
                results.append(result)

    # Note: Comparison plots removed to avoid plotting issues

    return effective_experiment_name


@trace_calls(show_return=True)
def run_experiment_from_config(config_path: str, n_trials: int = 1) -> str:
    """Load experiment config from YAML file and run experiment(s).

    This is a convenient wrapper that:
    1. Loads configuration from a YAML file
    2. Uses the experiment_name from the config for MLflow
    3. Runs the specified number of trials

    Args:
        config_path: Path to YAML configuration file
        n_trials: Number of trials to run (defaults to 1)

    Returns:
        MLflow experiment name
    """
    from trading_rl.config import ExperimentConfig

    # Load config from file
    config = ExperimentConfig.from_yaml(config_path)

    # Ensure MLflow experiment is set up before starting run
    setup_mlflow_experiment(config)

    if n_trials == 1:
        # Run single experiment
        with mlflow.start_run():
            run_single_experiment(custom_config=config)
        return config.experiment_name
    else:
        # Run multiple experiments
        return run_multiple_experiments(n_trials=n_trials, custom_config=config)
