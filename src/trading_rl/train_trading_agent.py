"""Main training script for DDPG trading agent - refactored version.

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
from typing import Any

import gym_trading_env  # noqa: F401
import gymnasium as gym
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from joblib import Memory

# No matplotlib configuration needed since we use plotnine exclusively
from plotnine import aes, geom_line
from plotnine.exceptions import PlotnineWarning
from tensordict.nn import InteractionType
from torchrl.envs import GymWrapper, TransformedEnv
from torchrl.envs.transforms import StepCounter
from torchrl.envs.utils import set_exploration_type

from logger import get_logger as get_project_logger
from logger import setup_logging as configure_root_logging
from trading_rl.config import ExperimentConfig
from trading_rl.continuous_action_wrapper import ContinuousToDiscreteAction
from trading_rl.data_utils import prepare_data, reward_function
from trading_rl.models import (
    create_actor,
    create_ppo_actor,
    create_ppo_value_network,
    create_td3_actor,
    create_td3_qvalue_network,
    create_value_network,
)
from trading_rl.plotting import visualize_training
from trading_rl.training import DDPGTrainer, PPOTrainer, TD3Trainer
from trading_rl.utils import compare_rollouts
from trading_rl.callbacks import MLflowTrainingCallback, log_final_metrics_to_mlflow

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


# %%
def create_continuous_trading_environment(
    df: pd.DataFrame, config: ExperimentConfig
) -> TransformedEnv:
    """Create continuous action space trading environment for TD3/DDPG.

    This creates a discrete trading environment and wraps it with a transform
    that converts continuous actions [-1, 1] to discrete trading actions.

    Args:
        df: DataFrame with features
        config: Experiment configuration

    Returns:
        Trading environment with continuous action space
    """
    logger = get_project_logger(__name__)

    # Use standard discrete positions for the base environment
    positions = config.env.positions

    logger.debug(f"Creating continuous environment for {config.training.algorithm}")
    logger.debug(f"  Data shape for training: {df[: config.data.train_size].shape}")
    logger.debug(f"  Positions: {positions}")
    logger.debug(f"  Trading fees: {config.env.trading_fees}")
    logger.debug(f"  Borrow interest rate: {config.env.borrow_interest_rate}")

    # Create base discrete trading environment
    base_env = gym.make(
        "TradingEnv",
        name=config.env.name,
        df=df[: config.data.train_size],
        positions=positions,
        trading_fees=config.env.trading_fees,
        borrow_interest_rate=config.env.borrow_interest_rate,
        reward_function=reward_function,
        verbose=0,  # Disable gym_trading_env print statements
    )

    logger.info(f"Created discrete base environment with positions: {positions}")

    # Wrap for TorchRL
    env = GymWrapper(base_env)

    # Add continuous action wrapper BEFORE step counter
    continuous_wrapper = ContinuousToDiscreteAction(
        discrete_actions=positions,
        thresholds=[-0.33, 0.33],  # Standard thresholds for 3-action mapping
        device=getattr(config.training, "device", "cpu"),
    )

    env = TransformedEnv(env, continuous_wrapper)
    logger.info("Added continuous-to-discrete action wrapper")
    logger.debug(f"  Continuous action spec: {env.action_spec}")

    # Add step counter last
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*auto_unwrap_transformed_env.*")
        env = TransformedEnv(env, StepCounter())

    logger.info("Continuous trading environment created successfully")
    logger.debug(f"  Observation spec: {env.observation_spec}")
    return env


def create_environment(df: pd.DataFrame, config: ExperimentConfig) -> TransformedEnv:
    """Create and configure trading environment.

    Args:
        df: DataFrame with features
        config: Experiment configuration

    Returns:
        Wrapped and transformed environment
    """
    logger = get_project_logger(__name__)

    # TD3 and DDPG require continuous action space
    algorithm = getattr(config.training, "algorithm", "PPO").upper()

    if algorithm in ["TD3", "DDPG"]:
        logger.info(
            f"{algorithm} algorithm detected - creating continuous action environment"
        )
        return create_continuous_trading_environment(df, config)

    # For other algorithms (PPO, DQN), use standard discrete environment
    logger.info(
        f"{algorithm} algorithm detected - creating discrete action environment"
    )

    positions = config.env.positions

    # Create base trading environment
    base_env = gym.make(
        "TradingEnv",
        name=config.env.name,
        df=df[: config.data.train_size],
        positions=positions,
        trading_fees=config.env.trading_fees,
        borrow_interest_rate=config.env.borrow_interest_rate,
        reward_function=reward_function,
        verbose=0,  # Disable gym_trading_env print statements
    )

    # Wrap for TorchRL
    env = GymWrapper(base_env)

    # Add step counter
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*auto_unwrap_transformed_env.*")
        env = TransformedEnv(env, StepCounter())

    return env


# %%
# %%
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
def _log_parameter_faq_artifact():
    """Log parameter FAQ as both markdown and HTML artifacts."""
    logger = get_project_logger(__name__)

    try:
        # Check MLflow run status
        active_run = mlflow.active_run()
        if not active_run:
            logger.warning("No active MLflow run - skipping FAQ artifacts")
            return

        faq_path = Path(__file__).parent / "docs" / "parameter_faq.md"
        if not faq_path.exists():
            logger.warning(f"FAQ file not found: {faq_path}")
            return

        # Log original markdown
        try:
            mlflow.log_artifact(str(faq_path), "documentation")
            logger.info("Successfully logged FAQ markdown artifact")
        except Exception as md_error:
            logger.error(f"Failed to log markdown FAQ: {md_error}")
            return  # Don't continue if markdown fails

        # Convert to HTML and log
        try:
            import markdown

            # Read markdown content
            with open(faq_path, encoding="utf-8") as f:
                md_content = f.read()

            # Convert to HTML with basic extensions (avoiding potential missing extensions)
            try:
                html_content = markdown.markdown(
                    md_content, extensions=["tables", "fenced_code", "toc"]
                )
            except Exception as ext_error:
                logger.warning(
                    f"Failed with extensions, trying basic conversion: {ext_error}"
                )
                # Fallback to basic conversion without extensions
                html_content = markdown.markdown(md_content)

            # Add basic CSS styling
            styled_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Parameter FAQ - Trading RL Experiments</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; line-height: 1.6; }}
        h1, h2, h3 {{ color: #333; }}
        h1 {{ border-bottom: 2px solid #eee; padding-bottom: 10px; }}
        h2 {{ border-bottom: 1px solid #eee; padding-bottom: 5px; margin-top: 30px; }}
        code {{ background: #f5f5f5; padding: 2px 4px; border-radius: 3px; font-family: 'Monaco', 'Consolas', monospace; }}
        pre {{ background: #f5f5f5; padding: 10px; border-radius: 5px; overflow-x: auto; }}
        ul, ol {{ padding-left: 20px; }}
        li {{ margin: 5px 0; }}
        strong {{ color: #2c3e50; }}
        blockquote {{ border-left: 4px solid #ddd; margin-left: 0; padding-left: 20px; color: #666; }}
        table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f5f5f5; }}
    </style>
</head>
<body>
{html_content}
</body>
</html>"""

            # Save HTML to temporary file with proper name and log
            import os
            import tempfile

            # Create temp file with proper name
            temp_dir = tempfile.gettempdir()
            html_temp_path = os.path.join(temp_dir, "parameter_faq.html")

            logger.info(f"Creating HTML file at: {html_temp_path}")
            logger.info(f"HTML content length: {len(styled_html)} characters")

            with open(html_temp_path, "w", encoding="utf-8") as f:
                f.write(styled_html)

            # Verify file was created
            if os.path.exists(html_temp_path):
                file_size = os.path.getsize(html_temp_path)
                logger.info(f"HTML file created successfully, size: {file_size} bytes")
            else:
                logger.error("HTML file was not created!")
                return

            mlflow.log_artifact(html_temp_path, "documentation")
            logger.info("Successfully logged FAQ HTML artifact")

            # Verify cleanup
            if os.path.exists(html_temp_path):
                os.unlink(html_temp_path)
                logger.info("Cleaned up temporary HTML file")
            else:
                logger.warning("Temporary HTML file already missing")

        except ImportError:
            logger.warning("Markdown library not available - skipping HTML conversion")
        except Exception as html_error:
            logger.error(f"Failed to create/log HTML FAQ: {html_error}")

    except Exception as e:
        logger.error(f"Error in FAQ artifact logging: {e}")


def evaluate_agent(
    env: TransformedEnv,
    actor: Any,
    df: pd.DataFrame,
    max_steps: int = 1000,
    save_path: str | None = None,
    config: Any = None,
    trainer: Any | None = None,
    algorithm: str | None = None,
) -> tuple:
    """Evaluate trained agent and create visualizations.

    Args:
        env: Trading environment
        actor: Trained actor network
        df: Original dataframe for benchmarks
        max_steps: Number of steps for evaluation
        save_path: Optional path to save plots
        trainer: Optional trainer instance (used for PPO-specific plots)
        algorithm: Algorithm name to gate algorithm-specific visuals

    Returns:
        Tuple of (reward_plot, action_plot, action_probs_plot, final_reward, last_positions)
    """
    logger = get_project_logger(__name__)

    # Run evaluation rollouts
    logger.debug(f"Running deterministic evaluation for {max_steps} steps")
    with set_exploration_type(InteractionType.MODE):
        rollout_deterministic = env.rollout(max_steps=max_steps, policy=actor)

    logger.debug(f"Running random evaluation for {max_steps} steps")
    with set_exploration_type(InteractionType.RANDOM):
        rollout_random = env.rollout(max_steps=max_steps, policy=actor)

    # DEBUG: Log rollout statistics
    if logger.isEnabledFor(logging.DEBUG):
        det_actions = rollout_deterministic["action"]
        det_rewards = rollout_deterministic["next", "reward"]

        logger.debug("=" * 60)
        logger.debug("DETERMINISTIC ROLLOUT STATISTICS")
        logger.debug("=" * 60)
        logger.debug(f"  Actions shape: {det_actions.shape}")
        logger.debug(
            f"  Actions - mean: {det_actions.mean():.4f}, std: {det_actions.std():.4f}"
        )
        logger.debug(
            f"  Actions - min: {det_actions.min():.4f}, max: {det_actions.max():.4f}"
        )
        logger.debug(
            f"  Rewards - mean: {det_rewards.mean():.6f}, std: {det_rewards.std():.6f}"
        )
        logger.debug(f"  Rewards - sum: {det_rewards.sum():.6f}")

        # Show first 20 actions
        actions_flat = det_actions.flatten()[:20].cpu().detach().numpy()
        logger.debug(f"  First 20 actions: {actions_flat}")

        # Check if stuck
        if det_actions.std() < 0.01:
            logger.warning("⚠️  AGENT IS STUCK - Taking nearly identical actions!")
            logger.warning(f"    Action std={det_actions.std():.6f} is very low")
        logger.debug("=" * 60)

    # Create benchmark comparisons
    benchmark_df = pd.DataFrame(
        {
            "x": range(max_steps),
            "buy_and_hold": np.log(df["close"] / df["close"].shift(1))
            .fillna(0)
            .cumsum()[:max_steps],
            "max_profit": np.log(abs(df["close"] / df["close"].shift(1) - 1) + 1)
            .fillna(0)
            .cumsum()[:max_steps],
        }
    )

    # Compare rollouts
    reward_plot, action_plot = compare_rollouts(
        [rollout_deterministic, rollout_random],
        n_obs=max_steps,
    )

    # Add benchmarks to reward plot with proper legend labels
    from plotnine import ggplot, labs, scale_color_manual

    # Create benchmark data for legend
    benchmark_data = []
    for step, (bh_val, mp_val) in enumerate(
        zip(benchmark_df["buy_and_hold"], benchmark_df["max_profit"], strict=False)
    ):
        benchmark_data.extend(
            [
                {"Steps": step, "Cumulative_Reward": bh_val, "Run": "Buy-and-Hold"},
                {"Steps": step, "Cumulative_Reward": mp_val, "Run": "Max Profit"},
            ]
        )

    # Get original data from the plot
    existing_data = reward_plot.data
    combined_data = pd.concat(
        [existing_data, pd.DataFrame(benchmark_data)], ignore_index=True
    )

    # Recreate plot with all data including benchmarks
    reward_plot = (
        ggplot(combined_data, aes(x="Steps", y="Cumulative_Reward", color="Run"))
        + geom_line()
        + labs(title="Cumulative Rewards Comparison", x="Steps", y="Cumulative Reward")
        + scale_color_manual(
            values={
                "Deterministic": "#F8766D",  # Default ggplot red
                "Random": "#00BFC4",  # Default ggplot blue
                "Buy-and-Hold": "violet",
                "Max Profit": "green",
            }
        )
    )

    if save_path:
        reward_plot.save(f"{save_path}_rewards.png", width=8, height=5, dpi=150)
        action_plot.save(f"{save_path}_actions.png", width=8, height=5, dpi=150)

    # Calculate final reward for metrics
    final_reward = float(rollout_deterministic["next"]["reward"].sum().item())

    # Calculate and log final returns (replaces gym_trading_env verbose output)
    initial_portfolio = 10000.0
    final_portfolio = initial_portfolio * np.exp(final_reward)
    portfolio_return = 100 * (final_portfolio / initial_portfolio - 1)
    market_return = 100 * (np.exp(final_reward) - 1)

    logger.info("=" * 60)
    logger.info("FINAL EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Market Return      : {market_return:5.2f}%")
    logger.info(f"Portfolio Return   : {portfolio_return:5.2f}%")
    logger.info(f"Initial Portfolio  : ${initial_portfolio:,.2f}")
    logger.info(f"Final Portfolio    : ${final_portfolio:,.2f}")
    logger.info(f"Total Reward (log) : {final_reward:.6f}")
    logger.info("=" * 60)

    # Extract positions from the deterministic rollout for tracking
    # Convert actions to positions (-1, 0, 1) based on action mapping
    action_tensor = rollout_deterministic["action"].squeeze()

    if action_tensor.ndim > 1 and action_tensor.shape[-1] > 1:
        # One-hot encoded (discrete) -> argmax
        actions = action_tensor.argmax(dim=-1)
    else:
        # Continuous or scalar discrete -> use values directly
        # If continuous in [-1, 1], we might want to map to nearest discrete for "position" logging
        # But simply converting to int/list is fine for now to avoid crash
        actions = action_tensor

    # Handle different tensor shapes
    if actions.dim() == 0:  # Single action case
        actions = [actions.item()]
    else:
        # Flatten the tensor and convert to list
        actions = actions.flatten().tolist()

    # Map actions to positions: action 0 -> position -1, action 1 -> position 0, action 2 -> position 1
    # Ensure each action is a scalar number before subtraction
    last_positions = []
    for action in actions:
        if isinstance(action, (list, tuple)):
            # If action is still a list/tuple, take the first element
            action_val = action[0] if len(action) > 0 else 0
        else:
            action_val = action

        # If we have continuous actions (floats), we assume they represent the position directly
        # or we interpret them. If discrete indices (0,1,2), we map to -1,0,1.
        if isinstance(action_val, float):
            # Continuous action [-1, 1] roughly maps to position directly
            last_positions.append(action_val)
        else:
            # Discrete index 0, 1, 2 -> -1, 0, 1
            last_positions.append(int(action_val) - 1)

    action_probs_plot = None
    if (
        algorithm
        and algorithm.upper() == "PPO"
        and trainer
        and hasattr(trainer, "create_action_probabilities_plot")
    ):
        action_probs_plot = trainer.create_action_probabilities_plot(
            max_steps, df, config
        )

    return (
        reward_plot,
        action_plot,
        action_probs_plot,
        final_reward,
        last_positions,
    )


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


def _log_training_parameters(config: ExperimentConfig) -> None:
    """Log all training parameters to MLflow."""
    import json

    try:
        # Basic experiment parameters
        mlflow.log_param("experiment_name", str(config.experiment_name))
        mlflow.log_param("seed", int(config.seed))

        # Data parameters
        mlflow.log_param("data_train_size", int(config.data.train_size))
        mlflow.log_param("data_timeframe", str(config.data.timeframe))
        mlflow.log_param("data_exchange_names", json.dumps(config.data.exchange_names))
        mlflow.log_param("data_symbols", json.dumps(config.data.symbols))
        mlflow.log_param("data_download_data", bool(config.data.download_data))
        mlflow.log_param(
            "data_no_features", bool(getattr(config.data, "no_features", False))
        )

        # Environment parameters
        mlflow.log_param("env_name", str(config.env.name))
        mlflow.log_param("env_positions", json.dumps(config.env.positions))
        mlflow.log_param("env_trading_fees", float(config.env.trading_fees))
        mlflow.log_param(
            "env_borrow_interest_rate", float(config.env.borrow_interest_rate)
        )
        mlflow.log_param(
            "env_backend", str(getattr(config.env, "backend", "gym_anytrading.forex"))
        )

        # Network architecture
        mlflow.log_param(
            "network_actor_hidden_dims", json.dumps(config.network.actor_hidden_dims)
        )
        mlflow.log_param(
            "network_value_hidden_dims", json.dumps(config.network.value_hidden_dims)
        )

        # Training parameters
        mlflow.log_param("training_algorithm", str(config.training.algorithm))
        mlflow.log_param("training_actor_lr", float(config.training.actor_lr))
        mlflow.log_param("training_value_lr", float(config.training.value_lr))
        mlflow.log_param(
            "training_value_weight_decay", float(config.training.value_weight_decay)
        )
        mlflow.log_param("training_max_steps", int(config.training.max_steps))
        mlflow.log_param(
            "training_init_rand_steps", int(config.training.init_rand_steps)
        )
        mlflow.log_param(
            "training_frames_per_batch", int(config.training.frames_per_batch)
        )
        mlflow.log_param(
            "training_optim_steps_per_batch", int(config.training.optim_steps_per_batch)
        )
        mlflow.log_param("training_sample_size", int(config.training.sample_size))
        mlflow.log_param("training_buffer_size", int(config.training.buffer_size))
        mlflow.log_param("training_loss_function", str(config.training.loss_function))
        mlflow.log_param("training_eval_steps", int(config.training.eval_steps))
        mlflow.log_param("training_eval_interval", int(config.training.eval_interval))
        mlflow.log_param("training_log_interval", int(config.training.log_interval))

        # Algorithm-specific parameters
        if hasattr(config.training, "tau"):
            mlflow.log_param("training_tau", float(config.training.tau))
        if hasattr(config.training, "clip_epsilon"):
            mlflow.log_param(
                "training_clip_epsilon", float(config.training.clip_epsilon)
            )
        if hasattr(config.training, "entropy_bonus"):
            mlflow.log_param(
                "training_entropy_bonus", float(config.training.entropy_bonus)
            )
        if hasattr(config.training, "vf_coef"):
            mlflow.log_param("training_vf_coef", float(config.training.vf_coef))
        if hasattr(config.training, "ppo_epochs"):
            mlflow.log_param("training_ppo_epochs", int(config.training.ppo_epochs))

        # Logging parameters
        mlflow.log_param("logging_log_dir", str(config.logging.log_dir))
        mlflow.log_param("logging_log_level", str(config.logging.log_level))

    except Exception as e:
        get_project_logger(__name__).warning(
            f"Failed to log some training parameters: {e}"
        )


def _log_config_artifact(config: ExperimentConfig) -> None:
    """Log YAML config file as MLflow artifact if available."""
    import tempfile
    from pathlib import Path

    import yaml

    # Try to find the config file based on experiment name
    config_dir = Path("src/configs")
    possible_configs = [
        config_dir / f"{config.experiment_name}.yaml",
        config_dir / f"{config.experiment_name}_ppo.yaml",
        config_dir / f"{config.experiment_name}_ddpg.yaml",
    ]

    config_file = None
    for path in possible_configs:
        if path.exists():
            config_file = path
            break

    if config_file and config_file.exists():
        # Log the original config file
        mlflow.log_artifact(str(config_file), "config")
    else:
        # Create a config file from the current config object
        config_dict = {
            "experiment_name": config.experiment_name,
            "seed": config.seed,
            "data": {
                "data_path": config.data.data_path,
                "download_data": config.data.download_data,
                "exchange_names": config.data.exchange_names,
                "symbols": config.data.symbols,
                "timeframe": config.data.timeframe,
                "data_dir": config.data.data_dir,
                "download_since": config.data.download_since,
                "train_size": config.data.train_size,
                "no_features": getattr(config.data, "no_features", False),
            },
            "env": {
                "name": config.env.name,
                "positions": config.env.positions,
                "trading_fees": config.env.trading_fees,
                "borrow_interest_rate": config.env.borrow_interest_rate,
            },
            "network": {
                "actor_hidden_dims": config.network.actor_hidden_dims,
                "value_hidden_dims": config.network.value_hidden_dims,
            },
            "training": {
                "algorithm": config.training.algorithm,
                "actor_lr": config.training.actor_lr,
                "value_lr": config.training.value_lr,
                "value_weight_decay": config.training.value_weight_decay,
                "max_steps": config.training.max_steps,
                "init_rand_steps": config.training.init_rand_steps,
                "frames_per_batch": config.training.frames_per_batch,
                "optim_steps_per_batch": config.training.optim_steps_per_batch,
                "sample_size": config.training.sample_size,
                "buffer_size": config.training.buffer_size,
                "loss_function": config.training.loss_function,
                "eval_steps": config.training.eval_steps,
                "eval_interval": config.training.eval_interval,
                "log_interval": config.training.log_interval,
            },
            "logging": {
                "log_dir": config.logging.log_dir,
                "log_level": config.logging.log_level,
            },
        }

        # Add algorithm-specific parameters
        if hasattr(config.training, "tau"):
            config_dict["training"]["tau"] = config.training.tau
        if hasattr(config.training, "clip_epsilon"):
            config_dict["training"]["clip_epsilon"] = config.training.clip_epsilon
        if hasattr(config.training, "entropy_bonus"):
            config_dict["training"]["entropy_bonus"] = config.training.entropy_bonus
        if hasattr(config.training, "vf_coef"):
            config_dict["training"]["vf_coef"] = config.training.vf_coef
        if hasattr(config.training, "ppo_epochs"):
            config_dict["training"]["ppo_epochs"] = config.training.ppo_epochs

        # Create temporary YAML file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)
            mlflow.log_artifact(f.name, "config")


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
    # Load configuration
    config = custom_config or ExperimentConfig()

    # Use experiment name from config, with optional override
    effective_experiment_name = experiment_name or config.experiment_name

    # Setup
    logger = setup_logging(config)
    set_seed(config.seed)

    # Print config values in debug mode
    _print_config_debug(config, logger)

    # Prepare data
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

    # DEBUG: Show data statistics
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Data statistics:")
        logger.debug(
            f"  Close price - min: {df['close'].min():.2f}, max: {df['close'].max():.2f}, mean: {df['close'].mean():.2f}"
        )
        logger.debug(f"  Close price std: {df['close'].std():.2f}")

        # Check for features
        feature_cols = [col for col in df.columns if "feature" in col.lower()]
        if feature_cols:
            logger.debug(f"  Features found: {feature_cols}")
        else:
            logger.debug("  No features found in data (using raw OHLCV only)")

    # Log data overview to MLflow
    if mlflow.active_run():
        # Log parameter FAQ as artifacts (early in trial for immediate availability)
        _log_parameter_faq_artifact()

        # Log all training parameters
        _log_training_parameters(config)

        # Log YAML config file as artifact if available
        _log_config_artifact(config)

        # Log dataset metadata
        mlflow.log_param("dataset_shape", f"{df.shape[0]}x{df.shape[1]}")
        mlflow.log_param("dataset_columns", list(df.columns))
        mlflow.log_param("date_range", f"{df.index.min()} to {df.index.max()}")
        mlflow.log_param("data_source", config.data.data_path)

        # Price statistics removed as they are not relevant for model evaluation

        # Log data sample as CSV artifact and generate plots
        try:
            import os
            import tempfile

            from plotnine import (
                aes,
                element_text,
                geom_line,
                ggplot,
                labs,
                theme,
                theme_minimal,
            )

            warnings.filterwarnings("ignore")  # Suppress plotnine warnings

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False
            ) as f:
                # Get first 50 rows for overview
                sample_df = df.head(50)
                sample_df.to_csv(f.name)
                mlflow.log_artifact(f.name, "data_overview")
                os.unlink(f.name)  # Clean up temp file

            # Also log data statistics summary
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            ) as f:
                f.write("Dataset Overview\n")
                f.write("================\n\n")
                f.write(f"Shape: {df.shape}\n")
                f.write(f"Columns: {list(df.columns)}\n")
                f.write(f"Date Range: {df.index.min()} to {df.index.max()}\n\n")
                f.write("Data Types:\n")
                f.write(str(df.dtypes))
                f.write("\n\nStatistical Summary:\n")
                f.write(str(df.describe()))
                f.flush()
                mlflow.log_artifact(f.name, "data_overview")
                os.unlink(f.name)  # Clean up temp file

            # Generate plotnine plots for each variable
            plot_df = df.head(200).reset_index()
            plot_df["time_index"] = range(len(plot_df))  # Add explicit time index

            # OHLCV columns to plot
            ohlcv_columns = ["open", "high", "low", "close", "volume"]
            available_columns = [col for col in ohlcv_columns if col in plot_df.columns]

            # Also include any feature columns (non-OHLCV)
            feature_columns = [
                col
                for col in plot_df.columns
                if col not in [*ohlcv_columns, plot_df.columns[0], "time_index"]
            ]

            all_plot_columns = (
                available_columns + feature_columns[:5]
            )  # Limit features to avoid too many plots

            for column in all_plot_columns:
                try:
                    # Create plot based on column type
                    if column == "volume":
                        # Line plot for volume
                        p = (
                            ggplot(plot_df, aes(x="time_index", y=column))
                            + geom_line(color="blue", size=0.8)
                            + theme_minimal()
                            + labs(
                                title=f"{column.title()} Over Time",
                                x="Time Index",
                                y=column.title(),
                            )
                            + theme(plot_title=element_text(size=14, face="bold"))
                        )
                    else:
                        # Line plot for price/feature data
                        color = "green" if column == "close" else "steelblue"
                        p = (
                            ggplot(plot_df, aes(x="time_index", y=column))
                            + geom_line(color=color, size=0.8)
                            + theme_minimal()
                            + labs(
                                title=f"{column.title()} Over Time",
                                x="Time Index",
                                y=column.title(),
                            )
                            + theme(plot_title=element_text(size=14, face="bold"))
                        )

                    # Save plot
                    with tempfile.NamedTemporaryFile(
                        suffix=".png", delete=False
                    ) as plot_file:
                        p.save(plot_file.name, width=8, height=5, dpi=150)
                        mlflow.log_artifact(plot_file.name, "data_overview/plots")
                        os.unlink(plot_file.name)

                except Exception as plot_error:
                    logger.warning(f"Failed to create plot for {column}: {plot_error}")

            # Create combined OHLC plot if all price columns are available
            if all(col in plot_df.columns for col in ["open", "high", "low", "close"]):
                try:
                    # Ensure consistent data before melting
                    ohlc_subset = plot_df[
                        ["time_index", "open", "high", "low", "close"]
                    ].copy()
                    ohlc_subset = ohlc_subset.dropna()  # Remove any NaN values

                    # Reshape data for multi-line plot
                    import pandas as pd

                    ohlc_melted = pd.melt(
                        ohlc_subset,
                        id_vars=["time_index"],
                        value_vars=["open", "high", "low", "close"],
                        var_name="price_type",
                        value_name="price",
                    )

                    p_combined = (
                        ggplot(
                            ohlc_melted,
                            aes(x="time_index", y="price", color="price_type"),
                        )
                        + geom_line(size=0.8)
                        + theme_minimal()
                        + labs(
                            title="OHLC Prices Over Time",
                            x="Time Index",
                            y="Price",
                            color="Price Type",
                        )
                        + theme(plot_title=element_text(size=14, face="bold"))
                    )

                    with tempfile.NamedTemporaryFile(
                        suffix=".png", delete=False
                    ) as plot_file:
                        p_combined.save(plot_file.name, width=10, height=5, dpi=150)
                        mlflow.log_artifact(plot_file.name, "data_overview/plots")
                        os.unlink(plot_file.name)

                except Exception as combined_error:
                    logger.warning(
                        f"Failed to create combined OHLC plot: {combined_error}"
                    )

        except Exception as e:
            logger.warning(f"Failed to log data overview: {e}")

    # Create environment
    logger.info("Creating environment...")
    env = create_environment(df, config)

    # Get environment specs
    n_obs = env.observation_spec["observation"].shape[-1]
    n_act = env.action_spec.shape[-1]
    logger.info(f"Environment: {n_obs} observations, {n_act} actions")

    logger.debug("Environment specs:")
    logger.debug(f"  Observation spec: {env.observation_spec}")
    logger.debug(f"  Action spec: {env.action_spec}")
    logger.debug(f"  Reward spec: {env.reward_spec}")

    # Create models based on algorithm choice
    algorithm = getattr(config.training, "algorithm", "PPO")
    logger.info(f"Creating models for {algorithm} algorithm...")

    qvalue_nets: list[Any] | None = None
    # Actor and value network depend on algorithm
    if algorithm.upper() == "PPO":
        # PPO needs actor with log prob outputs
        actor = create_ppo_actor(
            n_obs,
            n_act,
            hidden_dims=config.network.actor_hidden_dims,
            spec=env.action_spec,
        )
        value_net = create_ppo_value_network(
            n_obs,
            hidden_dims=config.network.value_hidden_dims,
        )
    elif algorithm.upper() == "TD3":
        actor = create_td3_actor(
            n_obs,
            n_act,
            hidden_dims=config.network.actor_hidden_dims,
            spec=env.action_spec,
        )
        qvalue_nets = [
            create_td3_qvalue_network(
                n_obs,
                n_act,
                hidden_dims=config.network.value_hidden_dims,
            ),
            create_td3_qvalue_network(
                n_obs,
                n_act,
                hidden_dims=config.network.value_hidden_dims,
            ),
        ]
        value_net = qvalue_nets

    else:  # DDPG
        # DDPG uses original actor
        actor = create_actor(
            n_obs,
            n_act,
            hidden_dims=config.network.actor_hidden_dims,
            spec=env.action_spec,
        )
        value_net = create_value_network(
            n_obs,
            n_act,
            hidden_dims=config.network.value_hidden_dims,
        )

    # Create trainer based on algorithm
    logger.info(f"Initializing {algorithm} trainer...")
    if algorithm.upper() == "PPO":
        trainer = PPOTrainer(
            actor=actor,
            value_net=value_net,
            env=env,
            config=config.training,
        )
    elif algorithm.upper() == "TD3":
        # value_net holds the StackedQValueNetwork created earlier
        trainer = TD3Trainer(
            actor=actor,
            qvalue_nets=qvalue_nets,
            env=env,
            config=config.training,
        )
    else:  # DDPG
        trainer = DDPGTrainer(
            actor=actor,
            value_net=value_net,
            env=env,
            config=config.training,
        )

    # Create MLflow callback with progress bar
    tracking_uri = getattr(getattr(config, "tracking", None), "tracking_uri", None)

    # Estimate total episodes (rough approximation based on steps and episode length)
    # Average episode length is approximately train_size for trading environments
    estimated_episodes = max(1, config.training.max_steps // config.data.train_size)

    mlflow_callback = MLflowTrainingCallback(
        effective_experiment_name,
        tracking_uri=tracking_uri,
        progress_bar=progress_bar,
        total_episodes=estimated_episodes if progress_bar else None,
    )

    # Re-configure logging because MLflow might have hijacked the root logger handlers
    logger = setup_logging(config)

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
        evaluate_agent(
            env,
            actor,
            df[: config.data.train_size],  # Pass only the training portion
            max_steps=eval_max_steps,
            config=config,
            trainer=trainer,
            algorithm=algorithm,
        )
    )

    # Save evaluation plots as MLflow artifacts
    if mlflow.active_run():
        import contextlib
        import io
        import os
        import tempfile

        # Context manager to suppress all plotnine output
        @contextlib.contextmanager
        def suppress_plotnine_output():
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                yield

        try:
            # Save reward plot using plotnine
            with tempfile.NamedTemporaryFile(
                suffix="_rewards.png", delete=False
            ) as tmp_reward:
                try:
                    with warnings.catch_warnings(), suppress_plotnine_output():
                        warnings.simplefilter("ignore", PlotnineWarning)
                        reward_plot.save(tmp_reward.name, width=8, height=5, dpi=150)
                    mlflow.log_artifact(tmp_reward.name, "evaluation_plots")
                except Exception:
                    logger.exception("Failed to save reward plot")
                finally:
                    if os.path.exists(tmp_reward.name):
                        os.unlink(tmp_reward.name)

            # Save action/position plot using plotnine
            with tempfile.NamedTemporaryFile(
                suffix="_positions.png", delete=False
            ) as tmp_action:
                try:
                    with warnings.catch_warnings(), suppress_plotnine_output():
                        warnings.simplefilter("ignore", PlotnineWarning)
                        action_plot.save(tmp_action.name, width=8, height=5, dpi=150)
                    mlflow.log_artifact(tmp_action.name, "evaluation_plots")
                except Exception:
                    logger.exception("Failed to save action plot")
                finally:
                    if os.path.exists(tmp_action.name):
                        os.unlink(tmp_action.name)

            # Save action probabilities plot (if present)
            if action_probs_plot is not None:
                with tempfile.NamedTemporaryFile(
                    suffix="_action_probabilities.png", delete=False
                ) as tmp_probs:
                    try:
                        with warnings.catch_warnings(), suppress_plotnine_output():
                            warnings.simplefilter("ignore", PlotnineWarning)
                            if hasattr(action_probs_plot, "save"):
                                action_probs_plot.save(
                                    tmp_probs.name, width=8, height=5, dpi=150
                                )  # type: ignore[arg-type]
                            elif hasattr(action_probs_plot, "savefig"):
                                action_probs_plot.savefig(tmp_probs.name, dpi=150)
                            else:
                                raise RuntimeError("Unsupported plot object for saving")
                        mlflow.log_artifact(tmp_probs.name, "evaluation_plots")
                    except Exception:
                        logger.exception("Failed to save action probabilities plot")
                    finally:
                        if os.path.exists(tmp_probs.name):
                            os.unlink(tmp_probs.name)

            # Create and save training loss plots
            if logs.get("loss_value") or logs.get("loss_actor"):
                import pandas as pd
                from plotnine import aes, geom_line, ggplot, labs, theme_minimal

                # Create loss plot data
                loss_data = []
                if logs.get("loss_value"):
                    loss_data.extend(
                        [
                            {"step": i, "loss": loss, "type": "Value Loss"}
                            for i, loss in enumerate(logs["loss_value"])
                        ]
                    )
                if logs.get("loss_actor"):
                    loss_data.extend(
                        [
                            {"step": i, "loss": loss, "type": "Actor Loss"}
                            for i, loss in enumerate(logs["loss_actor"])
                        ]
                    )

                if loss_data:
                    loss_df = pd.DataFrame(loss_data)

                    # Create training loss plot
                    loss_plot = (
                        ggplot(loss_df, aes(x="step", y="loss", color="type"))
                        + geom_line(size=1.2)
                        + labs(
                            title="Training Losses",
                            x="Training Step",
                            y="Loss Value",
                            color="Loss Type",
                        )
                        + theme_minimal()
                    )

                    # Save training loss plot using plotnine
                    with tempfile.NamedTemporaryFile(
                        suffix="_training_losses.png", delete=False
                    ) as tmp_loss:
                        try:
                            with warnings.catch_warnings(), suppress_plotnine_output():
                                warnings.simplefilter("ignore", PlotnineWarning)
                                loss_plot.save(
                                    tmp_loss.name, width=8, height=5, dpi=150
                                )
                            mlflow.log_artifact(tmp_loss.name, "training_plots")
                        except Exception:
                            logger.exception("Failed to save training loss plot")
                        finally:
                            if os.path.exists(tmp_loss.name):
                                os.unlink(tmp_loss.name)

            logger.info("Saved evaluation and training plots as MLflow artifacts")
        except Exception as e:
            logger.warning(f"Failed to save plots as artifacts: {e}")

    # Prepare comprehensive metrics
    final_metrics = {
        # Performance metrics
        "final_reward": final_reward,
        "training_steps": len(logs.get("loss_value", [])),
        "evaluation_steps": eval_max_steps,  # actual max_steps used in evaluation
        "last_position_per_episode": last_positions,  # Agent positions during final episode
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
    log_final_metrics_to_mlflow(logs, final_metrics, mlflow_callback)

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

    if n_trials == 1:
        # Run single experiment
        with mlflow.start_run():
            run_single_experiment(custom_config=config)
        return config.experiment_name
    else:
        # Run multiple experiments
        return run_multiple_experiments(n_trials=n_trials, custom_config=config)
