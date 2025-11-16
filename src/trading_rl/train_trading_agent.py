"""Main training script for DDPG trading agent - refactored version.

This is a clean, modular version of the trading RL training script.
All configuration, data processing, models, and training logic have been
separated into reusable modules.
"""

# %%
import logging
import os
import tempfile
import warnings
from pathlib import Path
from typing import Any

import gym_trading_env  # noqa: F401
import gymnasium as gym
import mlflow
import numpy as np
import pandas as pd
import torch
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
from trading_rl.data_utils import prepare_data, reward_function
from trading_rl.models import (
    create_actor,
    create_ppo_actor,
    create_ppo_value_network,
    create_value_network,
)
from trading_rl.plotting import visualize_training
from trading_rl.training import DDPGTrainer, PPOTrainer
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
    # No matplotlib logging to disable since we use plotnine exclusively

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
class MLflowTrainingCallback:
    """Callback for logging training progress to MLflow.

    Logs multiple metrics simultaneously:
    - Actor and value losses
    - Position changes and trading activity
    - Portfolio values and rewards
    - Custom plots and artifacts
    """

    def __init__(
        self,
        experiment_name: str = "trading_rl",
        tracking_uri: str | None = None,
    ):
        self.step_count = 0
        self.intermediate_losses = {"actor": [], "value": []}
        self.position_change_counts: list[int] = []
        self.training_stats = {
            "episode_rewards": [],
            "portfolio_valuations": [],
            "actions_taken": [],
            "exploration_ratio": [],
            "position_change_counts": [],
        }

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
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
        portfolio_valuation: float,
        actions: list,
        exploration_ratio: float,
    ):
        """Log statistics from an episode to MLflow."""
        self.training_stats["episode_rewards"].append(episode_reward)
        self.training_stats["portfolio_valuations"].append(portfolio_valuation)
        self.training_stats["actions_taken"].extend(actions)
        self.training_stats["exploration_ratio"].append(exploration_ratio)

        position_changes = self._count_position_changes(actions)
        self.position_change_counts.append(position_changes)
        self.training_stats["position_change_counts"].append(position_changes)

        # Calculate position change ratio (position changes / episode length)
        episode_length = len(actions)
        position_change_ratio = (
            position_changes / episode_length if episode_length > 0 else 0.0
        )

        # Log episode metrics to MLflow
        episode_num = len(self.training_stats["episode_rewards"])
        mlflow.log_metric("episode_reward", episode_reward, step=episode_num)
        mlflow.log_metric(
            "episode_portfolio_valuation", portfolio_valuation, step=episode_num
        )
        mlflow.log_metric(
            "episode_position_changes", position_changes, step=episode_num
        )
        mlflow.log_metric(
            "episode_position_change_ratio", position_change_ratio, step=episode_num
        )
        mlflow.log_metric(
            "episode_exploration_ratio", exploration_ratio, step=episode_num
        )

    def get_training_curves(self) -> dict:
        """Get training curves for storage."""
        return {
            "actor_losses": self.intermediate_losses["actor"],
            "value_losses": self.intermediate_losses["value"],
            "episode_rewards": self.training_stats["episode_rewards"],
            "portfolio_valuations": self.training_stats["portfolio_valuations"],
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

    # Log position statistics from final episode
    if "last_position_per_episode" in final_metrics:
        positions = final_metrics["last_position_per_episode"]
        if positions:
            import json

            # Log position sequence length as metric
            mlflow.log_metric("last_position_sequence_length", len(positions))
            # Store full position sequence as artifact
            position_str = json.dumps(positions[:100])  # Limit to first 100
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                f.write(position_str)
                f.flush()
                mlflow.log_artifact(f.name, "position_data")
                os.unlink(f.name)

    if logs["loss_value"]:
        mlflow.log_metric("final_value_loss", logs["loss_value"][-1])
        # mlflow.log_metric("avg_value_loss", np.mean(logs["loss_value"]))

    if logs["loss_actor"]:
        mlflow.log_metric("final_actor_loss", logs["loss_actor"][-1])
        mlflow.log_metric("avg_actor_loss", np.mean(logs["loss_actor"]))

    # Log training curves if callback provided
    if training_callback:
        training_curves = training_callback.get_training_curves()

        # Log summary statistics
        if training_curves["episode_rewards"]:
            mlflow.log_metric(
                "episode_avg_reward", np.mean(training_curves["episode_rewards"])
            )
            mlflow.log_metric(
                "episode_max_reward", np.max(training_curves["episode_rewards"])
            )
            mlflow.log_metric(
                "episode_min_reward", np.min(training_curves["episode_rewards"])
            )

        if training_curves["portfolio_valuations"]:
            mlflow.log_metric(
                "episode_portfolio_valuation",
                training_curves["portfolio_valuations"][-1],
            )

        # Removed avg_exploration_ratio metric

        if training_curves["position_change_counts"]:
            position_changes = training_curves["position_change_counts"]
            mlflow.log_metric(
                "episode_avg_position_change", float(np.mean(position_changes))
            )
            # Removed max_position_changes_per_episode metric
            mlflow.log_metric("total_position_changes", int(np.sum(position_changes)))

            # Calculate and log average position change ratio
            # Note: We need episode lengths to calculate this properly
            # For now, we'll estimate based on total actions vs episodes
            total_episodes = len(training_curves["episode_rewards"])
            total_actions = (
                len(training_callback.training_stats["actions_taken"])
                if training_callback
                else 0
            )
            avg_episode_length = (
                total_actions / total_episodes
                if total_episodes > 0 and total_actions > 0
                else 1
            )
            avg_position_change_ratio = np.mean(position_changes) / avg_episode_length
            mlflow.log_metric(
                "episode_avg_position_change_ratio", float(avg_position_change_ratio)
            )

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
    config: Any = None,
) -> tuple:
    """Evaluate trained agent and create visualizations.

    Args:
        env: Trading environment
        actor: Trained actor network
        df: Original dataframe for benchmarks
        max_steps: Number of steps for evaluation
        save_path: Optional path to save plots

    Returns:
        Tuple of (reward_plot, action_plot, action_probs_plot, final_reward, last_positions)
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

    # Extract positions from the deterministic rollout for tracking
    # Convert actions to positions (-1, 0, 1) based on action mapping
    # Use argmax to handle one-hot encoded actions (same as action plot)
    actions = rollout_deterministic["action"].squeeze().argmax(dim=-1)

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
        last_positions.append(int(action_val) - 1)

    # Create action probabilities plot
    action_probs_plot = _create_action_probabilities_plot(
        env, actor, max_steps, df, config
    )

    return reward_plot, action_plot, action_probs_plot, final_reward, last_positions


def _create_action_probabilities_plot(env, actor, max_steps, df=None, config=None):
    """Create a plot showing action probability distributions over time steps."""
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from plotnine import (
        element_text,
        geom_area,
        ggplot,
        labs,
        scale_fill_manual,
        scale_x_continuous,
        theme,
        theme_minimal,
    )

    try:
        max_viz_steps = min(max_steps, 200)
        max_episode_length = 20

        with torch.no_grad():
            # Use the original environment but force it to give us varying observations
            # by resetting and running multiple short episodes
            env_to_use = env

            # Reset environment and collect probabilities for each step
            obs = env_to_use.reset()
            action_names = {0: "Short", 1: "Hold", 2: "Long"}
            action_probs_data = []

            # Limit steps for visualization (too many steps make plot unreadable)
            current_episode_steps = 0

            for step in range(max_viz_steps):
                # Reset environment periodically to get different observations
                if current_episode_steps >= max_episode_length:
                    obs = env_to_use.reset()
                    current_episode_steps = 0
                # Extract observation from TensorDict if needed
                if hasattr(obs, "get") and "observation" in obs:
                    current_obs = obs["observation"]
                else:
                    current_obs = obs

                # Get action probabilities from actor
                actor_output = actor(current_obs)

                # Handle different actor output formats
                if hasattr(actor_output, "logits"):
                    # Standard distribution with logits
                    logits = actor_output.logits
                    probs = torch.softmax(logits, dim=-1)

                elif isinstance(actor_output, tuple) and len(actor_output) >= 1:
                    # PPO case - returns tuple with probabilities as first element
                    probs = actor_output[0]  # First element contains probabilities

                elif hasattr(actor_output, "loc"):
                    # DDPG case - continuous action, need to discretize
                    action_val = torch.clamp(actor_output.loc, -1, 1)
                    # Convert continuous [-1, 1] to discrete probabilities
                    if action_val < -0.33:
                        probs = torch.tensor([0.7, 0.2, 0.1])  # Mostly short
                    elif action_val > 0.33:
                        probs = torch.tensor([0.1, 0.2, 0.7])  # Mostly long
                    else:
                        probs = torch.tensor([0.2, 0.6, 0.2])  # Mostly hold

                else:
                    # Fallback - uniform distribution
                    probs = torch.tensor([0.33, 0.34, 0.33])

                # Ensure probs is the right shape and sum to 1
                probs = probs.squeeze()
                if probs.dim() == 0 or len(probs) != 3:
                    probs = torch.tensor([0.33, 0.34, 0.33])
                probs = probs / probs.sum()  # Normalize

                # Add data for each action at this step
                for action_idx, action_name in action_names.items():
                    action_probs_data.append(
                        {
                            "Step": step,
                            "Action": action_name,
                            "Probability": float(probs[action_idx]),
                        }
                    )

                # Take a step in the environment
                if hasattr(actor_output, "sample"):
                    action = actor_output.sample()
                elif isinstance(actor_output, tuple) and len(actor_output) >= 2:
                    # PPO actor returns (probs, action, log_prob)
                    action = actor_output[1]
                else:
                    # Fallback - sample from probabilities manually (categorical draw)
                    action_idx = torch.multinomial(probs, 1).item()
                    action = F.one_hot(
                        torch.tensor(action_idx), num_classes=len(action_names)
                    ).to(torch.float32)

                # Ensure action matches env spec (one-hot vector)
                if isinstance(action, torch.Tensor):
                    action = action.to(torch.float32)
                else:
                    action = torch.tensor(action, dtype=torch.float32)

                # Convert scalar/discrete actions to one-hot vectors
                if action.dim() == 0:
                    action = F.one_hot(
                        action.long(), num_classes=len(action_names)
                    ).to(torch.float32)
                elif action.dim() == 1 and action.shape[0] != len(action_names):
                    action = F.one_hot(
                        action.long(), num_classes=len(action_names)
                    ).to(torch.float32)

                # Ensure batch dimension matches observation batch size
                if hasattr(obs, "batch_size") and obs.batch_size:
                    expected_batch = obs.batch_size[0]
                else:
                    expected_batch = 1

                if action.dim() == 1:
                    action = action.unsqueeze(0)
                if action.shape[0] != expected_batch:
                    action = action.expand(expected_batch, action.shape[1])

                # Step the environment using a cloned tensordict
                if hasattr(obs, "clone") and hasattr(obs, "set"):
                    action_td = obs.clone()
                    action_td.set("action", action)
                    step_result = env_to_use.step(action_td)
                else:
                    # If obs is not a TensorDict, we can't safely step the env.
                    # Break out so fallback plot is used.
                    raise RuntimeError("Environment observation is not a TensorDict")

                # Extract post-step observation for next iteration
                if "next" in step_result.keys():
                    next_obs = step_result.get("next").clone()
                else:
                    next_obs = step_result.clone()
                obs = next_obs

                # Increment episode step counter
                current_episode_steps += 1

                # Break if episode is done
                done_tensor = None
                if hasattr(obs, "get"):
                    done_tensor = obs.get("done", torch.tensor([False]))
                if done_tensor is not None and torch.as_tensor(done_tensor).any():
                    break

        df_probs = pd.DataFrame(action_probs_data)
        action_order = ["Short", "Hold", "Long"]
        df_probs["Action"] = pd.Categorical(
            df_probs["Action"], categories=action_order, ordered=True
        )

        # Create stacked area plot showing probability distributions over time
        plot = (
            ggplot(df_probs, aes(x="Step", y="Probability", fill="Action"))
            + geom_area(position="stack", alpha=0.8)
            + labs(
                title="Action Probability Distributions Over Time",
                x="Time Step",
                y="Probability",
                fill="Action",
            )
            + theme_minimal()
            + scale_fill_manual(
                name="Action",
                values={
                    "Short": "#F8766D",  # red-ish at bottom
                    "Hold": "#C0C0C0",  # neutral middle
                    "Long": "#00BFC4",  # teal top
                },
            )
            + scale_x_continuous(expand=(0, 0))
            + theme(
                figure_size=(12, 6),
                axis_title=element_text(size=11),
                legend_position="right",
            )
        )

        return plot

    except Exception:
        # Fallback plot in case of errors - show uniform distribution over steps
        fallback_steps = min(max_steps, 500)
        fallback_data = []
        for step in range(fallback_steps):
            for action in ["Short", "Hold", "Long"]:
                fallback_data.append(
                    {"Step": step, "Action": action, "Probability": 0.33}
                )

        df_fallback = pd.DataFrame(fallback_data)
        df_fallback["Action"] = pd.Categorical(
            df_fallback["Action"], categories=["Short", "Hold", "Long"], ordered=True
        )
        plot = (
            ggplot(df_fallback, aes(x="Step", y="Probability", fill="Action"))
            + geom_area(position="stack", alpha=0.8)
            + labs(
                title="Action Probability Distribution (Fallback)",
                x="Time Step",
                y="Probability",
            )
            + theme_minimal()
            + scale_fill_manual(
                name="Action",
                values={
                    "Short": "#F8766D",
                    "Hold": "#C0C0C0",
                    "Long": "#00BFC4",
                },
            )
        )

        return plot


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
        logging.getLogger(__name__).warning(
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
    custom_config: ExperimentConfig | None = None, experiment_name: str | None = None
) -> dict:
    """Run a single training experiment with MLflow tracking.

    This function tracks both losses and position statistics in MLflow:
    - Multiple metrics logged simultaneously (losses, positions, rewards)
    - Parameters logged for configuration tracking
    - Artifacts logged for plots and models

    Args:
        custom_config: Optional custom configuration
        experiment_name: Optional override for MLflow experiment name (uses config.experiment_name by default)

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
        no_features=getattr(config.data, "no_features", False),
    )

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

    # Create models based on algorithm choice
    algorithm = getattr(config.training, "algorithm", "PPO")
    logger.info(f"Creating models for {algorithm} algorithm...")

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
    else:  # DDPG
        trainer = DDPGTrainer(
            actor=actor,
            value_net=value_net,
            env=env,
            config=config.training,
        )

    # Create MLflow callback
    tracking_uri = getattr(getattr(config, "tracking", None), "tracking_uri", None)
    mlflow_callback = MLflowTrainingCallback(
        effective_experiment_name, tracking_uri=tracking_uri
    )

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

            # Save action probabilities plot using plotnine
            with tempfile.NamedTemporaryFile(
                suffix="_action_probabilities.png", delete=False
            ) as tmp_probs:
                try:
                    with warnings.catch_warnings(), suppress_plotnine_output():
                        warnings.simplefilter("ignore", PlotnineWarning)
                        action_probs_plot.save(
                            tmp_probs.name, width=8, height=5, dpi=150
                        )
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
            "loss": visualize_training(logs),
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

    Returns:
        MLflow experiment name with all results
    """
    # Load configuration to get experiment name
    from trading_rl.config import ExperimentConfig

    config = custom_config or ExperimentConfig()
    effective_experiment_name = experiment_name or config.experiment_name

    # Setup MLflow experiment
    setup_mlflow_experiment(config, effective_experiment_name)

    results = []

    logger = logging.getLogger(__name__)

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
            result = run_single_experiment(custom_config=trial_config)
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
