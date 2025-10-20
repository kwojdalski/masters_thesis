"""Main training script for DDPG trading agent - refactored version.

This is a clean, modular version of the trading RL training script.
All configuration, data processing, models, and training logic have been
separated into reusable modules.
"""

# %%
import logging
from pathlib import Path
from typing import Any

import gym_trading_env  # noqa: F401
import gymnasium as gym
import mlflow
import numpy as np
import pandas as pd
import torch
from joblib import Memory
from plotnine import aes, geom_line
from tensordict.nn import InteractionType
from torchrl.envs import GymWrapper, TransformedEnv
from torchrl.envs.transforms import StepCounter
from torchrl.envs.utils import set_exploration_type

from logger import get_logger as get_project_logger
from logger import setup_logging as configure_root_logging
from trading_rl.config import ExperimentConfig
from trading_rl.data_utils import prepare_data, reward_function
from trading_rl.models import create_actor, create_value_network
from trading_rl.plotting import create_mlflow_comparison_plots, visualize_training
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
        # sys.stdout.isatty(),
        colored_output=True,
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
# %%
def setup_mlflow_experiment(experiment_name: str) -> None:
    """Setup MLflow experiment for tracking.

    Args:
        experiment_name: Name of the MLflow experiment
    """
    mlflow.set_experiment(experiment_name)
    return experiment_name


# %%
class MLflowTrainingCallback:
    """Callback for logging training progress to MLflow.

    Logs multiple metrics simultaneously:
    - Actor and value losses
    - Position changes and trading activity
    - Portfolio values and rewards
    - Custom plots and artifacts
    """

    def __init__(self, experiment_name: str = "trading_rl"):
        self.step_count = 0
        self.intermediate_losses = {"actor": [], "value": []}
        self.position_change_counts: list[int] = []
        self.training_stats = {
            "episode_rewards": [],
            "portfolio_values": [],
            "actions_taken": [],
            "exploration_ratio": [],
            "position_change_counts": [],
        }

        # Set MLflow experiment
        mlflow.set_experiment(experiment_name)

        # Start run if not already active
        if not mlflow.active_run():
            mlflow.start_run()

    def log_training_step(self, step: int, actor_loss: float, value_loss: float):
        """Log losses from a training step to MLflow.

        Logs multiple metrics simultaneously for rich tracking.
        """
        self.intermediate_losses["actor"].append(actor_loss)
        self.intermediate_losses["value"].append(value_loss)
        self.step_count = step

        # Log losses to MLflow every step
        mlflow.log_metric("actor_loss", actor_loss, step=step)
        mlflow.log_metric("value_loss", value_loss, step=step)

        # Log position changes if available
        if self.position_change_counts:
            window = min(len(self.position_change_counts), 100)
            avg_position_changes = float(np.mean(self.position_change_counts[-window:]))
            mlflow.log_metric("avg_position_changes", avg_position_changes, step=step)

    def log_episode_stats(
        self,
        episode_reward: float,
        portfolio_value: float,
        actions: list,
        exploration_ratio: float,
    ):
        """Log statistics from an episode to MLflow."""
        self.training_stats["episode_rewards"].append(episode_reward)
        self.training_stats["portfolio_values"].append(portfolio_value)
        self.training_stats["actions_taken"].extend(actions)
        self.training_stats["exploration_ratio"].append(exploration_ratio)

        position_changes = self._count_position_changes(actions)
        self.position_change_counts.append(position_changes)
        self.training_stats["position_change_counts"].append(position_changes)

        # Log episode metrics to MLflow
        episode_num = len(self.training_stats["episode_rewards"])
        mlflow.log_metric("episode_reward", episode_reward, step=episode_num)
        mlflow.log_metric("portfolio_value", portfolio_value, step=episode_num)
        mlflow.log_metric(
            "position_changes_per_episode", position_changes, step=episode_num
        )
        mlflow.log_metric("exploration_ratio", exploration_ratio, step=episode_num)

    def get_training_curves(self) -> dict:
        """Get training curves for storage."""
        return {
            "actor_losses": self.intermediate_losses["actor"],
            "value_losses": self.intermediate_losses["value"],
            "episode_rewards": self.training_stats["episode_rewards"],
            "portfolio_values": self.training_stats["portfolio_values"],
            "exploration_ratios": self.training_stats["exploration_ratio"],
            "position_change_counts": self.training_stats["position_change_counts"],
        }

    @staticmethod
    def _count_position_changes(actions: list, tolerance: float = 1e-6) -> int:
        """Count how often the agent changes positions within an episode."""
        if len(actions) < 2:
            return 0

        changes = 0
        prev_action = actions[0]
        for action in actions[1:]:
            if abs(action - prev_action) > tolerance:
                changes += 1
                prev_action = action
        return changes


def log_final_metrics_to_mlflow(
    logs: dict,
    final_metrics: dict,
    training_callback: MLflowTrainingCallback = None,
):
    """Log final training metrics to MLflow.

    Args:
        logs: Training logs dictionary
        final_metrics: Final evaluation metrics
        training_callback: Optional callback with training data
    """
    # Log final performance metrics
    mlflow.log_metric("final_reward", final_metrics["final_reward"])
    mlflow.log_metric("training_steps", final_metrics["training_steps"])
    mlflow.log_metric("evaluation_steps", final_metrics["evaluation_steps"])

    if logs["loss_value"]:
        mlflow.log_metric("final_value_loss", logs["loss_value"][-1])
        mlflow.log_metric("avg_value_loss", np.mean(logs["loss_value"]))

    if logs["loss_actor"]:
        mlflow.log_metric("final_actor_loss", logs["loss_actor"][-1])
        mlflow.log_metric("avg_actor_loss", np.mean(logs["loss_actor"]))

    # Log training curves if callback provided
    if training_callback:
        training_curves = training_callback.get_training_curves()

        # Log summary statistics
        if training_curves["episode_rewards"]:
            mlflow.log_metric(
                "avg_episode_reward", np.mean(training_curves["episode_rewards"])
            )
            mlflow.log_metric(
                "max_episode_reward", np.max(training_curves["episode_rewards"])
            )
            mlflow.log_metric(
                "min_episode_reward", np.min(training_curves["episode_rewards"])
            )

        if training_curves["portfolio_values"]:
            mlflow.log_metric(
                "final_portfolio_value", training_curves["portfolio_values"][-1]
            )
            mlflow.log_metric(
                "max_portfolio_value", np.max(training_curves["portfolio_values"])
            )

        if training_curves["exploration_ratios"]:
            mlflow.log_metric(
                "avg_exploration_ratio", np.mean(training_curves["exploration_ratios"])
            )

        if training_curves["position_change_counts"]:
            position_changes = training_curves["position_change_counts"]
            mlflow.log_metric(
                "avg_position_change_per_episode", float(np.mean(position_changes))
            )
            mlflow.log_metric(
                "max_position_changes_per_episode", int(np.max(position_changes))
            )
            mlflow.log_metric("total_position_changes", int(np.sum(position_changes)))

    # Log dataset metadata as parameters
    mlflow.log_param("data_start_date", final_metrics.get("data_start_date", "unknown"))
    mlflow.log_param("data_end_date", final_metrics.get("data_end_date", "unknown"))
    mlflow.log_param("data_size", final_metrics.get("data_size", 0))
    mlflow.log_param("train_size", final_metrics.get("train_size", 0))

    # Log environment configuration
    mlflow.log_param("trading_fees", final_metrics.get("trading_fees", 0.0))
    mlflow.log_param(
        "borrow_interest_rate", final_metrics.get("borrow_interest_rate", 0.0)
    )
    mlflow.log_param("positions", final_metrics.get("positions", "unknown"))

    # Log network architecture
    mlflow.log_param(
        "actor_hidden_dims", str(final_metrics.get("actor_hidden_dims", []))
    )
    mlflow.log_param(
        "value_hidden_dims", str(final_metrics.get("value_hidden_dims", []))
    )
    mlflow.log_param("n_observations", final_metrics.get("n_observations", 0))
    mlflow.log_param("n_actions", final_metrics.get("n_actions", 0))

    # Log experiment configuration
    mlflow.log_param("experiment_name", final_metrics.get("experiment_name", "unknown"))
    mlflow.log_param("seed", final_metrics.get("seed", 0))
    mlflow.log_param("actor_lr", final_metrics.get("actor_lr", 0.0))
    mlflow.log_param("value_lr", final_metrics.get("value_lr", 0.0))
    mlflow.log_param("buffer_size", final_metrics.get("buffer_size", 0))


def _log_parameter_faq_artifact():
    """Log parameter FAQ as both markdown and HTML artifacts."""
    import logging

    logger = logging.getLogger(__name__)

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
    experiment_name: str = "trading_rl", custom_config: ExperimentConfig | None = None
) -> dict:
    """Run a single training experiment with MLflow tracking.

    This function tracks both losses and position statistics in MLflow:
    - Multiple metrics logged simultaneously (losses, positions, rewards)
    - Parameters logged for configuration tracking
    - Artifacts logged for plots and models

    Args:
        experiment_name: MLflow experiment name
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

    # Log data overview to MLflow
    if mlflow.active_run():
        # Log parameter FAQ as artifacts (early in trial for immediate availability)
        _log_parameter_faq_artifact()

        # Log dataset metadata
        mlflow.log_param("dataset_shape", f"{df.shape[0]}x{df.shape[1]}")
        mlflow.log_param("dataset_columns", list(df.columns))
        mlflow.log_param("date_range", f"{df.index.min()} to {df.index.max()}")
        mlflow.log_param("data_source", config.data.data_path)

        # Log price statistics
        if "close" in df.columns:
            mlflow.log_metric("price_min", df["close"].min())
            mlflow.log_metric("price_max", df["close"].max())
            mlflow.log_metric("price_mean", df["close"].mean())
            mlflow.log_metric("price_std", df["close"].std())

        # Log data sample as CSV artifact and generate plots
        try:
            import os
            import tempfile
            import warnings

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
                        p.save(plot_file.name, width=10, height=6, dpi=150)
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
                        p_combined.save(plot_file.name, width=12, height=6, dpi=150)
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

    # Create MLflow callback
    mlflow_callback = MLflowTrainingCallback(experiment_name)

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
    # Ensure max_steps doesn't exceed available data
    eval_max_steps = min(1000, len(df) - 1)  # -1 to account for shift operation
    reward_plot, action_plot, final_reward = evaluate_agent(
        env,
        actor,
        df,
        max_steps=eval_max_steps,
    )

    # Prepare comprehensive metrics
    final_metrics = {
        # Performance metrics
        "final_reward": final_reward,
        "training_steps": len(logs.get("loss_value", [])),
        "evaluation_steps": eval_max_steps,  # actual max_steps used in evaluation
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
            "loss": visualize_training(logs),
            "reward": reward_plot,
            "action": action_plot,
        },
    }


# %%
def run_multiple_experiments(
    experiment_name: str,
    n_trials: int = 5,
    base_seed: int | None = None,
    custom_config: ExperimentConfig | None = None,
) -> str:
    """Run multiple experiments and track with MLflow.

    Each experiment tracks:
    - Multiple metrics simultaneously (losses, positions, rewards)
    - Parameters for configuration comparison
    - Artifacts for plots and models

    Args:
        experiment_name: Name for the MLflow experiment
        n_trials: Number of experiments to run
        base_seed: Base seed for reproducible experiments (each trial uses base_seed + trial_number)

    Returns:
        MLflow experiment name with all results
    """
    # Setup MLflow experiment
    setup_mlflow_experiment(experiment_name)

    results = []

    logger = logging.getLogger(__name__)

    for trial_number in range(n_trials):
        logger.info(f"Running trial {trial_number + 1}/{n_trials}")

        # Create config with deterministic seed based on trial number
        from trading_rl.config import ExperimentConfig

        if custom_config is not None:
            # Use custom config as base and copy it for this trial
            import copy

            config = copy.deepcopy(custom_config)
        else:
            config = ExperimentConfig()

        if base_seed is not None:
            config.seed = base_seed + trial_number
        else:
            import random

            config.seed = random.randint(1, 100000)  # noqa: S311

        # Update experiment name to include trial number
        config.experiment_name = f"{experiment_name}_trial_{trial_number}"

        # Start a new MLflow run for this trial
        with mlflow.start_run(run_name=f"trial_{trial_number}"):
            result = run_single_experiment(experiment_name, custom_config=config)
            results.append(result)

    # Generate comparison plots after all trials complete
    create_mlflow_comparison_plots(experiment_name, results)

    return experiment_name
