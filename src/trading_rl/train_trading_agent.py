"""Main training script for DDPG trading agent - refactored version.

This is a clean, modular version of the trading RL training script.
All configuration, data processing, models, and training logic have been
separated into reusable modules.
"""

# %%
import logging
import sys
from pathlib import Path
from typing import Any

import gym_trading_env  # noqa: F401
import gymnasium as gym
import numpy as np
import optuna
import pandas as pd
import torch
from joblib import Memory
from plotnine import aes, facet_wrap, geom_line, ggplot
from tensordict.nn import InteractionType
from torchrl.envs import GymWrapper, TransformedEnv
from torchrl.envs.transforms import StepCounter
from torchrl.envs.utils import set_exploration_type

from logger import get_logger as get_project_logger
from logger import setup_logging as configure_root_logging
from trading_rl.config import ExperimentConfig
from trading_rl.data_utils import prepare_data, reward_function
from trading_rl.models import create_actor, create_value_network
from trading_rl.training import DDPGTrainer
from trading_rl.utils import compare_rollouts

# Setup joblib memory for caching expensive operations
memory = Memory(location=".cache/joblib", verbose=1)


def clear_cache():
    """Clear all joblib caches."""
    memory.clear(warn=True)


# %%
def setup_logging(config: ExperimentConfig) -> logging.Logger:
    """Setup logging configuration.

    Args:
        config: Experiment configuration

    Returns:
        Logger instance
    """
    # Disable matplotlib logging
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    log_file_path = Path(config.logging.log_dir) / config.logging.log_file

    configure_root_logging(
        level=config.logging.log_level,
        log_file=str(log_file_path),
        console_output=True,
        colored_output=sys.stdout.isatty(),
    )

    logger = get_project_logger(__name__)
    logger.info(f"Starting experiment: {config.experiment_name}")
    return logger


# %%
def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


# %%
def create_environment(df: pd.DataFrame, config: ExperimentConfig) -> TransformedEnv:
    """Create and configure trading environment.

    Args:
        df: DataFrame with features
        config: Experiment configuration

    Returns:
        Wrapped and transformed environment
    """
    # Create base trading environment
    base_env = gym.make(
        "TradingEnv",
        name=config.env.name,
        df=df[: config.data.train_size],
        positions=config.env.positions,
        trading_fees=config.env.trading_fees,
        borrow_interest_rate=config.env.borrow_interest_rate,
        reward_function=reward_function,
    )

    # Wrap for TorchRL
    env = GymWrapper(base_env)
    env = TransformedEnv(env, StepCounter())

    return env


# %%
def visualize_training(logs: dict, save_path: str | None = None):
    """Visualize training progress.

    Args:
        logs: Dictionary of training logs
        save_path: Optional path to save plot
    """
    # Create loss dataframe
    loss_df = pd.DataFrame(
        {
            "step": range(len(logs["loss_value"])),
            "Value Loss": logs["loss_value"],
            "Actor Loss": logs["loss_actor"],
        }
    )

    # Create plot
    plot = (
        ggplot(loss_df.melt(id_vars=["step"], var_name="Loss Type", value_name="Loss"))
        + geom_line(aes(x="step", y="Loss", color="Loss Type"))
        + facet_wrap("Loss Type", ncol=1, scales="free")
    )

    if save_path:
        plot.save(save_path)

    return plot


# %%
def create_optuna_study(
    study_name: str, storage_url: str | None = None
) -> optuna.Study:
    """Create or load Optuna study for experiment tracking.

    Args:
        study_name: Name of the study
        storage_url: Optional storage URL (defaults to SQLite)

    Returns:
        Optuna study object
    """
    if storage_url is None:
        storage_url = f"sqlite:///{study_name}.db"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction="maximize",  # We'll track final reward
        load_if_exists=True,
    )
    return study


# %%
class OptunaTrainingCallback:
    """Callback for logging training progress to Optuna during training."""

    def __init__(self, trial: optuna.Trial):
        self.trial = trial
        self.step_count = 0
        self.intermediate_rewards = []
        self.intermediate_losses = {"actor": [], "value": []}
        self.training_stats = {
            "episode_rewards": [],
            "portfolio_values": [],
            "actions_taken": [],
            "exploration_ratio": [],
        }

    def log_training_step(self, step: int, actor_loss: float, value_loss: float):
        """Log losses from a training step."""
        self.intermediate_losses["actor"].append(actor_loss)
        self.intermediate_losses["value"].append(value_loss)
        self.step_count = step

        # Report intermediate values every 100 steps for Optuna visualization
        if step % 100 == 0:
            avg_actor_loss = np.mean(self.intermediate_losses["actor"][-100:])
            avg_value_loss = np.mean(self.intermediate_losses["value"][-100:])

            # Report average loss as intermediate value
            self.trial.report(avg_value_loss, step=step)

    def log_episode_stats(
        self,
        episode_reward: float,
        portfolio_value: float,
        actions: list,
        exploration_ratio: float,
    ):
        """Log statistics from an episode."""
        self.training_stats["episode_rewards"].append(episode_reward)
        self.training_stats["portfolio_values"].append(portfolio_value)
        self.training_stats["actions_taken"].extend(actions)
        self.training_stats["exploration_ratio"].append(exploration_ratio)

    def get_training_curves(self) -> dict:
        """Get training curves for storage."""
        return {
            "actor_losses": self.intermediate_losses["actor"],
            "value_losses": self.intermediate_losses["value"],
            "episode_rewards": self.training_stats["episode_rewards"],
            "portfolio_values": self.training_stats["portfolio_values"],
            "exploration_ratios": self.training_stats["exploration_ratio"],
        }


def log_metrics_to_optuna(
    trial: optuna.Trial,
    logs: dict,
    final_metrics: dict,
    training_callback: OptunaTrainingCallback = None,
):
    """Log training metrics to Optuna trial.

    Args:
        trial: Optuna trial object
        logs: Training logs dictionary
        final_metrics: Final evaluation metrics
        training_callback: Optional callback with intermediate training data
    """
    # Log final metrics as trial value (for tracking purposes)
    trial.report(
        final_metrics["final_reward"], step=final_metrics.get("training_steps", 0)
    )

    # Log performance metrics
    trial.set_user_attr("final_reward", final_metrics["final_reward"])
    trial.set_user_attr("training_steps", final_metrics["training_steps"])
    trial.set_user_attr("evaluation_steps", final_metrics["evaluation_steps"])
    trial.set_user_attr(
        "final_value_loss", logs["loss_value"][-1] if logs["loss_value"] else 0.0
    )
    trial.set_user_attr(
        "final_actor_loss", logs["loss_actor"][-1] if logs["loss_actor"] else 0.0
    )
    trial.set_user_attr(
        "avg_value_loss", np.mean(logs["loss_value"]) if logs["loss_value"] else 0.0
    )
    trial.set_user_attr(
        "avg_actor_loss", np.mean(logs["loss_actor"]) if logs["loss_actor"] else 0.0
    )

    # Log training curves if callback provided
    if training_callback:
        training_curves = training_callback.get_training_curves()

        # Store training curves as JSON strings
        trial.set_user_attr("training_curves", str(training_curves))

        # Store summary statistics
        if training_curves["episode_rewards"]:
            trial.set_user_attr(
                "avg_episode_reward", np.mean(training_curves["episode_rewards"])
            )
            trial.set_user_attr(
                "max_episode_reward", np.max(training_curves["episode_rewards"])
            )
            trial.set_user_attr(
                "min_episode_reward", np.min(training_curves["episode_rewards"])
            )

        if training_curves["portfolio_values"]:
            trial.set_user_attr(
                "final_portfolio_value", training_curves["portfolio_values"][-1]
            )
            trial.set_user_attr(
                "max_portfolio_value", np.max(training_curves["portfolio_values"])
            )

        if training_curves["exploration_ratios"]:
            trial.set_user_attr(
                "avg_exploration_ratio", np.mean(training_curves["exploration_ratios"])
            )

    # Log dataset metadata
    trial.set_user_attr(
        "data_start_date", final_metrics.get("data_start_date", "unknown")
    )
    trial.set_user_attr("data_end_date", final_metrics.get("data_end_date", "unknown"))
    trial.set_user_attr("data_size", final_metrics.get("data_size", 0))
    trial.set_user_attr("train_size", final_metrics.get("train_size", 0))

    # Log environment configuration
    trial.set_user_attr("trading_fees", final_metrics.get("trading_fees", 0.0))
    trial.set_user_attr(
        "borrow_interest_rate", final_metrics.get("borrow_interest_rate", 0.0)
    )
    trial.set_user_attr("positions", final_metrics.get("positions", "unknown"))

    # Log network architecture
    trial.set_user_attr(
        "actor_hidden_dims", str(final_metrics.get("actor_hidden_dims", []))
    )
    trial.set_user_attr(
        "value_hidden_dims", str(final_metrics.get("value_hidden_dims", []))
    )
    trial.set_user_attr("n_observations", final_metrics.get("n_observations", 0))
    trial.set_user_attr("n_actions", final_metrics.get("n_actions", 0))

    # Log experiment configuration
    trial.set_user_attr(
        "experiment_name", final_metrics.get("experiment_name", "unknown")
    )
    trial.set_user_attr("seed", final_metrics.get("seed", 0))

    trial.set_user_attr("actor_lr", final_metrics.get("actor_lr", 0.0))
    trial.set_user_attr("value_lr", final_metrics.get("value_lr", 0.0))
    trial.set_user_attr("buffer_size", final_metrics.get("buffer_size", 0))


def evaluate_agent(
    env: TransformedEnv,
    actor: Any,
    df: pd.DataFrame,
    max_steps: int = 1000,
    save_path: str | None = None,
) -> tuple:
    """Evaluate trained agent and create visualizations.

    Args:
        env: Trading environment
        actor: Trained actor network
        df: Original dataframe for benchmarks
        max_steps: Number of steps for evaluation
        save_path: Optional path to save plots

    Returns:
        Tuple of (reward_plot, action_plot, final_reward)
    """
    # Run evaluation rollouts
    with set_exploration_type(InteractionType.MODE):
        rollout_deterministic = env.rollout(max_steps=max_steps, policy=actor)

    with set_exploration_type(InteractionType.RANDOM):
        rollout_random = env.rollout(max_steps=max_steps, policy=actor)

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

    # Add benchmarks to reward plot
    reward_plot = (
        reward_plot
        + geom_line(
            data=benchmark_df,
            mapping=aes(x="x", y="buy_and_hold"),
            color="violet",
        )
        + geom_line(
            data=benchmark_df,
            mapping=aes(x="x", y="max_profit"),
            color="green",
            linetype="dashed",
        )
    )

    if save_path:
        reward_plot.save(f"{save_path}_rewards.png")
        action_plot.save(f"{save_path}_actions.png")

    # Calculate final reward for metrics
    final_reward = float(rollout_deterministic["next"]["reward"].sum().item())

    return reward_plot, action_plot, final_reward


def run_single_experiment(
    trial: optuna.Trial | None = None, custom_config: ExperimentConfig | None = None
) -> dict:
    """Run a single training experiment.

    Args:
        trial: Optional Optuna trial for logging
        custom_config: Optional custom configuration

    Returns:
        Dictionary with results
    """
    # Load configuration
    config = custom_config or ExperimentConfig()

    # Setup
    logger = setup_logging(config)
    set_seed(config.seed)

    # Prepare data
    logger.info("Preparing data...")
    df = prepare_data(
        data_path=config.data.data_path,
        download_if_missing=config.data.download_data,
        exchange_names=config.data.exchange_names,
        symbols=config.data.symbols,
        timeframe=config.data.timeframe,
        data_dir=config.data.data_dir,
        since=config.data.download_since,
    )

    # Create environment
    logger.info("Creating environment...")
    env = create_environment(df, config)

    # Get environment specs
    n_obs = env.observation_spec["observation"].shape[-1]
    n_act = env.action_spec.shape[-1]
    logger.info(f"Environment: {n_obs} observations, {n_act} actions")

    # Create models
    logger.info("Creating models...")
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

    # Create trainer
    logger.info("Initializing trainer...")
    trainer = DDPGTrainer(
        actor=actor,
        value_net=value_net,
        env=env,
        config=config.training,
    )

    # Create Optuna callback if trial is provided
    optuna_callback = None
    if trial is not None:
        optuna_callback = OptunaTrainingCallback(trial)

    # Train
    logger.info("Starting training...")
    logs = trainer.train(callback=optuna_callback)

    # Save checkpoint
    checkpoint_path = (
        Path(config.logging.log_dir) / f"{config.experiment_name}_checkpoint.pt"
    )
    trainer.save_checkpoint(str(checkpoint_path))

    # Evaluate agent
    logger.info("Evaluating agent...")
    reward_plot, action_plot, final_reward = evaluate_agent(
        env,
        actor,
        df,
        max_steps=1000,
    )

    # Prepare comprehensive metrics
    final_metrics = {
        # Performance metrics
        "final_reward": final_reward,
        "training_steps": len(logs.get("loss_value", [])),
        "evaluation_steps": 1000,  # max_steps from evaluation
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

    # Log to Optuna if trial provided
    if trial is not None:
        log_metrics_to_optuna(trial, logs, final_metrics, optuna_callback)

    logger.info("Training complete!")
    logger.info(f"Final reward: {final_reward:.4f}")
    logger.info(f"Checkpoint saved to: {checkpoint_path}")

    return {
        "trainer": trainer,
        "logs": logs,
        "final_metrics": final_metrics,
        "plots": {
            "loss": visualize_training(logs),
            "reward": reward_plot,
            "action": action_plot,
        },
    }


# %%
def run_multiple_experiments(study_name: str, n_trials: int = 5) -> optuna.Study:
    """Run multiple experiments and track with Optuna.

    Args:
        study_name: Name for the Optuna study
        n_trials: Number of experiments to run

    Returns:
        Optuna study with all results
    """
    study = create_optuna_study(study_name)

    def objective(trial):
        # You can vary parameters here if desired
        # For now, we just run the same experiment multiple times
        result = run_single_experiment(trial)
        return result["final_metrics"]["final_reward"]

    study.optimize(objective, n_trials=n_trials)
    return study
