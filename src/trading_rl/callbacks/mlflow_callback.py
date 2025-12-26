"""MLflow callback and utilities for training logging."""

import json
import logging
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
        price_series=None,
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
        self.price_series = price_series

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

    def log_training_step(
        self,
        step: int,
        actor_loss: float,
        value_loss: float,
        extra_metrics: dict[str, float] | None = None,
    ):
        """Log losses from a training step to MLflow.

        Logs multiple metrics simultaneously for rich tracking.
        """
        self.intermediate_losses["actor"].append(actor_loss)
        self.intermediate_losses["value"].append(value_loss)
        self.step_count = step

        # Log losses to MLflow every step
        mlflow.log_metric("actor_loss", actor_loss, step=step)
        mlflow.log_metric("value_loss", value_loss, step=step)

        if extra_metrics:
            for key, val in extra_metrics.items():
                mlflow.log_metric(key, val, step=step)

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
                "network_actor_hidden_dims",
                json.dumps(config.network.actor_hidden_dims),
            )
            mlflow.log_param(
                "network_value_hidden_dims",
                json.dumps(config.network.value_hidden_dims),
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
            mlflow.log_param(
                "training_checkpoint_interval",
                int(getattr(config.training, "checkpoint_interval", 0)),
            )
            mlflow.log_param(
                "training_loss_function", str(config.training.loss_function)
            )
            mlflow.log_param("training_eval_steps", int(config.training.eval_steps))
            mlflow.log_param(
                "training_eval_interval", int(config.training.eval_interval)
            )
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

            faq_path = (
                Path(__file__).resolve().parent.parent / "docs" / "parameter_faq.md"
            )
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
                logger.warning(
                    "Markdown library not available - skipping HTML conversion"
                )
            except Exception as html_error:
                logger.error(f"Failed to create/log HTML FAQ: {html_error}")

        except Exception as e:
            logger.error(f"Error in FAQ artifact logging: {e}")

    @staticmethod
    def log_data_overview(df, config) -> None:
        """Log dataset overview, sample, and quick visuals to MLflow."""
        logger = get_project_logger(__name__)

        if not mlflow.active_run():
            logger.warning("No active MLflow run - skipping data overview logging")
            return

        try:
            import tempfile

            import pandas as pd
            from plotnine import aes, element_text, geom_line, ggplot, labs, theme

            # Log dataset metadata
            mlflow.log_param("dataset_shape", f"{df.shape[0]}x{df.shape[1]}")
            mlflow.log_param("dataset_columns", list(df.columns))
            mlflow.log_param("date_range", f"{df.index.min()} to {df.index.max()}")
            mlflow.log_param("data_source", config.data.data_path)

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False
            ) as f:
                sample_df = df.head(50)
                sample_df.to_csv(f.name)
                mlflow.log_artifact(f.name, "data_overview")
                os.unlink(f.name)

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
                os.unlink(f.name)

            plot_df = df.head(200).reset_index()
            plot_df["time_index"] = range(len(plot_df))

            ohlcv_columns = ["open", "high", "low", "close", "volume"]
            available_columns = [col for col in ohlcv_columns if col in plot_df.columns]
            feature_columns = [
                col
                for col in plot_df.columns
                if col not in [*ohlcv_columns, plot_df.columns[0], "time_index"]
            ]
            all_plot_columns = available_columns + feature_columns[:5]

            for column in all_plot_columns:
                try:
                    p = (
                        ggplot(plot_df, aes(x="time_index", y=column))
                        + geom_line(color="steelblue", size=0.8)
                        + labs(
                            title=f"{column.title()} Over Time",
                            x="Time Index",
                            y=column.title(),
                        )
                        + theme(plot_title=element_text(size=14, face="bold"))
                    )

                    with tempfile.NamedTemporaryFile(
                        suffix=".png", delete=False
                    ) as plot_file:
                        p.save(plot_file.name, width=8, height=5, dpi=150)
                        mlflow.log_artifact(plot_file.name, "data_overview/plots")
                        os.unlink(plot_file.name)

                except Exception as plot_error:  # pragma: no cover - logging only
                    logger.warning(f"Failed to create plot for {column}: {plot_error}")

            if all(col in plot_df.columns for col in ["open", "high", "low", "close"]):
                try:
                    ohlc_subset = plot_df[
                        ["time_index", "open", "high", "low", "close"]
                    ].copy()
                    ohlc_subset = ohlc_subset.dropna()

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

                except Exception as combined_error:  # pragma: no cover - logging only
                    logger.warning(
                        f"Failed to create combined OHLC plot: {combined_error}"
                    )

        except Exception as e:  # pragma: no cover - logging only
            logger.warning(f"Failed to log data overview: {e}")

    @staticmethod
    def log_final_metrics(
        logs: dict,
        final_metrics: dict,
        training_callback: "MLflowTrainingCallback | None" = None,
    ) -> None:
        """Log final training metrics to MLflow."""
        logger = get_project_logger(__name__)
        mlflow.log_metric("final_reward", final_metrics["final_reward"])
        mlflow.log_metric("training_steps", final_metrics["training_steps"])
        mlflow.log_metric("evaluation_steps", final_metrics["evaluation_steps"])

        # Handle both discrete positions and portfolio weights
        if "last_position_per_episode" in final_metrics:
            positions = final_metrics["last_position_per_episode"]
            if positions:
                mlflow.log_metric("last_position_sequence_length", len(positions))
                position_str = json.dumps(positions[:100])
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False
                ) as f:
                    f.write(position_str)
                    f.flush()
                    mlflow.log_artifact(f.name, "position_data")
                    os.unlink(f.name)
        elif "portfolio_weights" in final_metrics:
            weights = final_metrics["portfolio_weights"]
            if weights:
                mlflow.log_metric("portfolio_weights_sequence_length", len(weights))
                weights_str = json.dumps(weights[:100])
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False
                ) as f:
                    f.write(weights_str)
                    f.flush()
                    mlflow.log_artifact(f.name, "portfolio_weights_data")
                    os.unlink(f.name)

        if logs.get("loss_value"):
            mlflow.log_metric("final_value_loss", logs["loss_value"][-1])
        else:
            logger.warning(
                "No value loss data available for logging - training may have been skipped due to tensor shape issues"
            )

        if logs.get("loss_actor"):
            mlflow.log_metric("final_actor_loss", logs["loss_actor"][-1])
            mlflow.log_metric("avg_actor_loss", np.mean(logs["loss_actor"]))
        else:
            logger.warning(
                "No actor loss data available for logging - training may have been skipped due to tensor shape issues"
            )

        if training_callback:
            training_curves = training_callback.get_training_curves()

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

            if training_curves["position_change_counts"]:
                position_changes = training_curves["position_change_counts"]
                mlflow.log_metric(
                    "episode_avg_position_change", float(np.mean(position_changes))
                )
                mlflow.log_metric(
                    "total_position_changes", int(np.sum(position_changes))
                )

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

                if avg_episode_length > 0:
                    avg_position_change_ratio = (
                        np.mean(position_changes) / avg_episode_length
                    )
                mlflow.log_metric(
                    "episode_avg_position_change_ratio",
                    avg_position_change_ratio,
                )

    @staticmethod
    def log_evaluation_plots(
        reward_plot,
        action_plot,
        action_probs_plot=None,
        logs=None,
    ) -> None:
        """Save evaluation/training plots as MLflow artifacts."""
        logger = get_project_logger(__name__)

        if not mlflow.active_run():
            logger.warning("No active MLflow run - skipping plot artifact logging")
            return

        import contextlib
        import io
        import tempfile
        import warnings

        from plotnine.exceptions import PlotnineWarning

        # Context manager to suppress all plotnine output
        @contextlib.contextmanager
        def suppress_plotnine_output():
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                yield

        saved_paths: dict[str, str] = {}

        try:

            def _save_plot_as_artifact(plot_obj, suffix, artifact_dir, logger):
                """Helper to save plot to temp file and log as MLflow artifact."""
                with tempfile.NamedTemporaryFile(
                    suffix=suffix, delete=False
                ) as tmp_file:
                    tmp_path = tmp_file.name
                try:
                    with warnings.catch_warnings(), suppress_plotnine_output():
                        warnings.simplefilter("ignore", PlotnineWarning)
                        # Handle plotnine plots ("save") and matplotlib figures ("savefig")
                        if hasattr(plot_obj, "save"):
                            plot_obj.save(tmp_path, width=8, height=5, dpi=150)
                        elif hasattr(plot_obj, "savefig"):
                            plot_obj.savefig(tmp_path, dpi=150)
                        else:
                            raise RuntimeError("Unsupported plot object for saving")
                    # Temporarily silence noisy third-party loggers during save

                    pil_logger = logging.getLogger("PIL.PngImagePlugin")
                    previous_level = pil_logger.level
                    pil_logger.setLevel(logging.INFO)
                    try:
                        mlflow.log_artifact(tmp_path, artifact_dir)
                    finally:
                        pil_logger.setLevel(previous_level)
                    saved_paths[suffix] = tmp_path
                except Exception:
                    logger.exception(f"Failed to save {suffix} plot")
                # cleanup deferred for optional combination

            # Save reward plot using plotnine
            _save_plot_as_artifact(
                reward_plot, "_rewards.png", "evaluation_plots", logger
            )

            # Save action/position plot using plotnine
            _save_plot_as_artifact(
                action_plot, "_positions.png", "evaluation_plots", logger
            )

            # Save action probabilities plot (if present)
            if action_probs_plot is not None:
                _save_plot_as_artifact(
                    action_probs_plot,
                    "_action_probabilities.png",
                    "evaluation_plots",
                    logger,
                )
            else:
                logger.info("Action probability plot missing; skipping that artifact")

            # Create and save training loss plots
            if logs and (logs.get("loss_value") or logs.get("loss_actor")):
                import pandas as pd
                from plotnine import aes, geom_line, ggplot, labs

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

                    from plotnine import facet_wrap

                    loss_plot = (
                        ggplot(loss_df, aes(x="step", y="loss", color="type"))
                        + geom_line(size=1.2)
                        + facet_wrap("type", ncol=1, scales="free")
                        + labs(
                            title="Training Losses",
                            x="Training Step",
                            y="Loss Value",
                            color="Loss Type",
                        )
                    )

                    _save_plot_as_artifact(
                        loss_plot, "_training_losses.png", "training_plots", logger
                    )

            # Combine existing evaluation plots using plotnine patchwork when available
            combined_path = None
            try:
                from plotnine import patchwork  # type: ignore

                combined_plot = None
                if reward_plot is not None and action_plot is not None:
                    combined_plot = reward_plot | action_plot
                    if action_probs_plot is not None:
                        combined_plot = combined_plot / action_probs_plot
                elif reward_plot is not None:
                    combined_plot = reward_plot
                elif action_plot is not None:
                    combined_plot = action_plot
                elif action_probs_plot is not None:
                    combined_plot = action_probs_plot

                if combined_plot is not None:
                    _save_plot_as_artifact(
                        combined_plot,
                        "_combined_evaluation.png",
                        "evaluation_plots",
                        logger,
                    )
            except ImportError:
                # Fallback: combine PNGs with Pillow in a (p1 | p2) / p3 layout
                if {
                    "_rewards.png",
                    "_positions.png",
                    "_action_probabilities.png",
                } <= set(saved_paths.keys()):
                    try:
                        from PIL import Image

                        reward_img = Image.open(saved_paths["_rewards.png"])
                        action_img = Image.open(saved_paths["_positions.png"])
                        probs_img = Image.open(saved_paths["_action_probabilities.png"])

                        top_width = reward_img.width + action_img.width
                        top_height = max(reward_img.height, action_img.height)
                        bottom_width = probs_img.width
                        bottom_height = probs_img.height

                        combined_width = max(top_width, bottom_width)
                        combined_height = top_height + bottom_height

                        combined = Image.new(
                            "RGB", (combined_width, combined_height), "white"
                        )
                        combined.paste(reward_img, (0, 0))
                        combined.paste(action_img, (reward_img.width, 0))
                        combined.paste(probs_img, (0, top_height))

                        with tempfile.NamedTemporaryFile(
                            suffix="_combined_evaluation.png", delete=False
                        ) as tmp_combined:
                            combined.save(tmp_combined.name, format="PNG")
                            mlflow.log_artifact(tmp_combined.name, "evaluation_plots")
                            combined_path = tmp_combined.name
                    except (
                        Exception
                    ) as combine_error:  # pragma: no cover - logging only
                        logger.warning(
                            f"Failed to create combined evaluation plot: {combine_error}"
                        )

            logger.info("Saved evaluation and training plots as MLflow artifacts")
        except Exception as e:  # pragma: no cover - defensive logging only
            logger.warning(f"Failed to save plots as artifacts: {e}")
        finally:
            # Clean up temporary files we kept for optional combination
            for path in saved_paths.values():
                if os.path.exists(path):
                    os.unlink(path)
