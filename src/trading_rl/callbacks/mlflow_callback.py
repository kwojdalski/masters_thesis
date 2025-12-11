"""MLflow callback and utilities for training logging."""

import json
import os
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import yaml

from logger import get_logger as get_project_logger


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
        progress_bar=None,
        total_episodes: int | None = None,
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

        # Progress bar tracking
        self.progress_bar = progress_bar
        self.progress_task = None
        self.total_episodes = total_episodes

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        # Start run if not already active
        if not mlflow.active_run():
            mlflow.start_run()

        # Initialize progress bar task if provided
        if self.progress_bar and self.total_episodes:
            self.progress_task = self.progress_bar.add_task(
                "[cyan]Training Episodes", total=self.total_episodes
            )

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

        # Update progress bar if available
        if self.progress_bar and self.progress_task is not None:
            self.progress_bar.update(
                self.progress_task,
                advance=1,
                description=f"[cyan]Episode {episode_num} | Reward: {episode_reward:.4f} | Portfolio: ${portfolio_valuation:,.0f}",
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

    @staticmethod
    def log_config_artifact(config) -> None:
        """Log YAML config file as an MLflow artifact."""
        # Try to find an existing config file by experiment name
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
            mlflow.log_artifact(str(config_file), "config")
            return

        # Otherwise, serialize the in-memory config
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

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)
            mlflow.log_artifact(f.name, "config")
            os.unlink(f.name)

    @staticmethod
    def log_training_parameters(config) -> None:
        """Log core training parameters to MLflow."""
        try:
            mlflow.log_param("experiment_name", str(config.experiment_name))
            mlflow.log_param("seed", int(config.seed))

            # Data parameters
            mlflow.log_param("data_train_size", int(config.data.train_size))
            mlflow.log_param("data_timeframe", str(config.data.timeframe))
            mlflow.log_param(
                "data_exchange_names", json.dumps(config.data.exchange_names)
            )
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

            # Network parameters
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
                "training_optim_steps_per_batch",
                int(config.training.optim_steps_per_batch),
            )
            mlflow.log_param("training_sample_size", int(config.training.sample_size))
            mlflow.log_param("training_buffer_size", int(config.training.buffer_size))
            mlflow.log_param("training_loss_function", str(config.training.loss_function))
            mlflow.log_param("training_eval_steps", int(config.training.eval_steps))
            mlflow.log_param("training_eval_interval", int(config.training.eval_interval))
            mlflow.log_param("training_log_interval", int(config.training.log_interval))

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

        except Exception as e:  # pragma: no cover - defensive
            get_project_logger(__name__).warning(
                f"Failed to log some training parameters: {e}"
            )

    @staticmethod
    def log_parameter_faq_artifact() -> None:
        """Log parameter FAQ as both markdown and HTML artifacts."""
        logger = get_project_logger(__name__)

        try:
            # Check MLflow run status
            active_run = mlflow.active_run()
            if not active_run:
                logger.warning("No active MLflow run - skipping FAQ artifacts")
                return

            faq_path = Path(__file__).resolve().parent.parent / "docs" / "parameter_faq.md"
            if not faq_path.exists():
                logger.warning(f"FAQ file not found: {faq_path}")
                return

            # Log original markdown
            try:
                mlflow.log_artifact(str(faq_path), "documentation")
                logger.info("Successfully logged FAQ markdown artifact")
            except Exception as md_error:
                logger.error(f"Failed to log markdown FAQ: {md_error}")
                return

            # Convert to HTML and log
            try:
                import markdown

                with open(faq_path, encoding="utf-8") as f:
                    md_content = f.read()

                try:
                    html_content = markdown.markdown(
                        md_content, extensions=["tables", "fenced_code", "toc"]
                    )
                except Exception as ext_error:
                    logger.warning(
                        f"Failed with extensions, trying basic conversion: {ext_error}"
                    )
                    html_content = markdown.markdown(md_content)

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

                temp_dir = tempfile.gettempdir()
                html_temp_path = os.path.join(temp_dir, "parameter_faq.html")

                logger.info(f"Creating HTML file at: {html_temp_path}")
                logger.info(f"HTML content length: {len(styled_html)} characters")

                with open(html_temp_path, "w", encoding="utf-8") as f:
                    f.write(styled_html)

                if os.path.exists(html_temp_path):
                    mlflow.log_artifact(html_temp_path, "documentation")
                    logger.info("Successfully logged FAQ HTML artifact")
                    os.unlink(html_temp_path)
                else:
                    logger.error("HTML file was not created!")

            except ImportError:
                logger.warning("Markdown library not available - skipping HTML conversion")
            except Exception as html_error:
                logger.error(f"Failed to create/log HTML FAQ: {html_error}")

        except Exception as e:
            logger.error(f"Error in FAQ artifact logging: {e}")


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
    logger = get_project_logger(__name__)
    # Log final performance metrics
    mlflow.log_metric("final_reward", final_metrics["final_reward"])
    mlflow.log_metric("training_steps", final_metrics["training_steps"])
    mlflow.log_metric("evaluation_steps", final_metrics["evaluation_steps"])

    # Log position statistics from final episode
    if "last_position_per_episode" in final_metrics:
        positions = final_metrics["last_position_per_episode"]
        if positions:
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

    if logs.get("loss_value") and len(logs["loss_value"]) > 0:
        mlflow.log_metric("final_value_loss", logs["loss_value"][-1])
        # mlflow.log_metric("avg_value_loss", np.mean(logs["loss_value"]))
    else:
        logger.warning(
            "No value loss data available for logging - training may have been skipped due to tensor shape issues"
        )

    if logs.get("loss_actor") and len(logs["loss_actor"]) > 0:
        mlflow.log_metric("final_actor_loss", logs["loss_actor"][-1])
        mlflow.log_metric("avg_actor_loss", np.mean(logs["loss_actor"]))
    else:
        logger.warning(
            "No actor loss data available for logging - training may have been skipped due to tensor shape issues"
        )

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
                else 1.0
            )

            # Avoid division by zero
            if avg_episode_length > 0:
                avg_position_change_ratio = (
                    np.mean(position_changes) / avg_episode_length
                )
                mlflow.log_metric(
                    "episode_avg_position_change_ratio", avg_position_change_ratio
                )
