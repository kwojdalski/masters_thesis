"""Main training script for DDPG trading agent - refactored version.

This is a clean, modular version of the trading RL training script.
All configuration, data processing, models, and training logic have been
separated into reusable modules.
"""

# %%
import logging
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from plotnine import aes, facet_wrap, geom_line, ggplot
from tensordict.nn import InteractionType
from torchrl.envs import GymWrapper, TransformedEnv
from torchrl.envs.transforms import StepCounter
from torchrl.envs.utils import set_exploration_type

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import gym_trading_env  # noqa: F401, E402
from scripts.utils import compare_rollouts  # noqa: E402
from trading_rl import (  # noqa: E402
    DDPGTrainer,
    ExperimentConfig,
    create_actor,
    create_value_network,
    prepare_data,
    reward_function,
)


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

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, config.logging.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(Path(config.logging.log_dir) / config.logging.log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logger = logging.getLogger(__name__)
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
def visualize_training(logs: dict, save_path: str | None = None) -> None:
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
def evaluate_agent(
    env: TransformedEnv,
    actor: any,
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
        Tuple of (reward_plot, action_plot)
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

    return reward_plot, action_plot


# %%
def main():
    """Main training pipeline."""
    # Load configuration
    config = ExperimentConfig()

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

    # Train
    logger.info("Starting training...")
    logs = trainer.train()

    # Save checkpoint
    checkpoint_path = (
        Path(config.logging.log_dir) / f"{config.experiment_name}_checkpoint.pt"
    )
    trainer.save_checkpoint(str(checkpoint_path))

    # Visualize results
    logger.info("Creating visualizations...")
    loss_plot = visualize_training(
        logs,
        save_path=str(
            Path(config.logging.log_dir) / f"{config.experiment_name}_losses.png"
        ),
    )

    reward_plot, action_plot = evaluate_agent(
        env,
        actor,
        df,
        max_steps=1000,
        save_path=str(Path(config.logging.log_dir) / f"{config.experiment_name}_eval"),
    )

    logger.info("Training complete!")
    logger.info(f"Checkpoint saved to: {checkpoint_path}")

    return {
        "trainer": trainer,
        "logs": logs,
        "plots": {
            "loss": loss_plot,
            "reward": reward_plot,
            "action": action_plot,
        },
    }


# %%
if __name__ == "__main__":
    results = main()
