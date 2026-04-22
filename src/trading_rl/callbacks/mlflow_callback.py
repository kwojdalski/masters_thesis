"""MLflow callback for training progress logging."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch

from trading_rl.config import DEFAULT_INITIAL_PORTFOLIO_VALUE

from trading_rl.callbacks.artifacts import (
    log_config_artifact,
    log_data_overview,
    log_evaluation_plots,
    log_evaluation_report,
    log_explainability_results,
    log_final_metrics,
    log_parameter_faq_artifact,
    log_statistical_tests,
    log_training_parameters,
)

_POSITION_CHANGE_TOLERANCE = 0.1
_POSITION_CHANGE_WINDOW = 100


class MLflowTrainingCallback:
    """Callback for logging training progress to MLflow.

    Logs multiple metrics per step and episode:
    - Actor and value losses
    - Position changes and trading activity
    - Portfolio values and rewards

    Artifact and metric logging helpers (config, data overview, evaluation
    reports, statistical tests, explainability, plots) live in
    ``trading_rl.callbacks.artifacts`` and are re-exported here as class
    attributes for backward compatibility.
    """

    # Re-export artifact helpers so callers using
    # ``MLflowTrainingCallback.log_*(…)`` continue to work unchanged.
    log_config_artifact = staticmethod(log_config_artifact)
    log_data_overview = staticmethod(log_data_overview)
    log_evaluation_plots = staticmethod(log_evaluation_plots)
    log_evaluation_report = staticmethod(log_evaluation_report)
    log_explainability_results = staticmethod(log_explainability_results)
    log_final_metrics = staticmethod(log_final_metrics)
    log_parameter_faq_artifact = staticmethod(log_parameter_faq_artifact)
    log_statistical_tests = staticmethod(log_statistical_tests)
    log_training_parameters = staticmethod(log_training_parameters)

    def __init__(
        self,
        experiment_name: str = "trading_rl",
        tracking_uri: str | None = None,
        progress_bar=None,
        total_episodes: int | None = None,
        price_series=None,
        start_run: bool = True,
        initial_portfolio_value: float = DEFAULT_INITIAL_PORTFOLIO_VALUE,
        reward_type: str = "log_return",
        config_for_run_name: Any | None = None,
    ):
        self.step_count = 0
        self._episode_count = 0
        self.initial_portfolio_value = initial_portfolio_value
        self.reward_type = reward_type
        self._portfolio_value = initial_portfolio_value
        self.intermediate_losses = {"actor": [], "value": []}
        self.position_change_counts: list[int] = []
        self.training_stats = {
            "episode_rewards": [],
            "portfolio_valuations": [],
            "actions_taken": [],
            "exploration_ratio": [],
            "position_change_counts": [],
            "sum_positions": [],
        }
        self.price_series = price_series

        self.progress_bar = progress_bar
        self.progress_task = None
        self.total_episodes = total_episodes

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        if start_run and not mlflow.active_run():
            mlflow.start_run(
                run_name=self._default_run_name(
                    experiment_name=experiment_name,
                    config=config_for_run_name,
                )
            )

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
        """Log losses from a training step to MLflow."""
        self.intermediate_losses["actor"].append(actor_loss)
        self.intermediate_losses["value"].append(value_loss)
        self.step_count = step

        mlflow.log_metric("actor_loss", actor_loss, step=step)
        mlflow.log_metric("value_loss", value_loss, step=step)

        if extra_metrics:
            for key, val in extra_metrics.items():
                mlflow.log_metric(key, val, step=step)

        if self.position_change_counts:
            window = min(len(self.position_change_counts), _POSITION_CHANGE_WINDOW)
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
        # TODO: Make tolerance configurable
        position_changes = self._count_position_changes(actions, tolerance=_POSITION_CHANGE_TOLERANCE)
        self.position_change_counts.append(position_changes)
        self.training_stats["position_change_counts"].append(position_changes)
        episode_sum_position = float(np.sum(actions)) if actions else 0.0
        self.training_stats["sum_positions"].append(episode_sum_position)

        episode_length = len(actions)
        position_change_ratio = position_changes / episode_length if episode_length > 0 else 0.0

        episode_base = self._episode_count
        if isinstance(episode_base, torch.Tensor):
            episode_base = int(episode_base.item())
        episode_num = int(episode_base) + len(self.training_stats["episode_rewards"])

        mlflow.log_metric("episode_reward", episode_reward, step=episode_num)
        mlflow.log_metric("episode_portfolio_valuation", portfolio_valuation, step=episode_num)
        mlflow.log_metric("episode_position_changes", position_changes, step=episode_num)
        mlflow.log_metric("episode_position_change_ratio", position_change_ratio, step=episode_num)
        mlflow.log_metric("episode_sum_position", episode_sum_position, step=episode_num)
        mlflow.log_metric("episode_exploration_ratio", exploration_ratio, step=episode_num)

        if self.progress_bar and self.progress_task is not None:
            self.progress_bar.update(
                self.progress_task,
                advance=1,
                description=(
                    f"[cyan]Episode {episode_num} | Reward: {episode_reward:.4f} "
                    f"| Portfolio: ${portfolio_valuation:,.0f}"
                ),
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
            "sum_positions": self.training_stats["sum_positions"],
        }

    # ------------------------------------------------------------------
    # Private helpers used only by instance methods above
    # ------------------------------------------------------------------

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
    def _default_run_name(experiment_name: str, config: Any | None = None) -> str:
        """Build deterministic human-readable run name for fresh runs."""
        safe_experiment = experiment_name.replace("/", "_").replace(" ", "_")
        if config is None:
            return safe_experiment
        return MLflowTrainingCallback._config_signature(config)

    @staticmethod
    def _normalize_for_hash(value: Any) -> Any:
        """Normalize nested values so hashing is deterministic."""
        if is_dataclass(value):
            return MLflowTrainingCallback._normalize_for_hash(asdict(value))
        if isinstance(value, dict):
            return {
                str(k): MLflowTrainingCallback._normalize_for_hash(v)
                for k, v in sorted(value.items(), key=lambda item: str(item[0]))
            }
        if isinstance(value, (list, tuple)):
            return [MLflowTrainingCallback._normalize_for_hash(v) for v in value]
        if isinstance(value, datetime):
            return value.astimezone(UTC).isoformat()
        if isinstance(value, Path):
            return str(value)
        return value

    @staticmethod
    def _config_signature(config: Any) -> str:
        """Encode config into a stable 3-word signature."""
        adjectives = (
            "amber", "brisk", "calm", "daring", "eager", "frozen",
            "golden", "hidden", "ivory", "jade", "keen", "lunar",
            "mellow", "noble", "opal", "proud",
        )
        nouns = (
            "atlas", "beacon", "cipher", "delta", "ember", "falcon",
            "grove", "harbor", "island", "jungle", "kernel", "lagoon",
            "matrix", "nebula", "orbit", "pulse",
        )
        suffixes = (
            "alpha", "bravo", "charlie", "delta", "echo", "foxtrot",
            "golf", "hotel", "india", "juliet", "kilo", "lima",
            "mike", "november", "oscar", "papa",
        )

        normalized = MLflowTrainingCallback._normalize_for_hash(config)
        payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
        digest = sha256(payload.encode("utf-8")).hexdigest()
        b0 = int(digest[0:2], 16)
        b1 = int(digest[2:4], 16)
        b2 = int(digest[4:6], 16)
        return f"{adjectives[b0 % len(adjectives)]}-{nouns[b1 % len(nouns)]}-{suffixes[b2 % len(suffixes)]}"
