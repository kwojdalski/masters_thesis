"""Post-training finalization helpers for experiment execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow

from trading_rl.callbacks import MLflowTrainingCallback
from trading_rl.plotting import visualize_training


def build_final_checkpoint_path(
    *,
    config: Any,
    effective_experiment_name: str,
    trainer: Any,
    checkpoint_path: str | None,
) -> Path:
    """Build the final checkpoint path for a fresh or resumed run."""
    run = mlflow.active_run()
    if run and run.info.run_name:
        base_name = (
            run.info.run_name.replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
        )
    else:
        base_name = effective_experiment_name

    if checkpoint_path:
        return (
            Path(config.logging.log_dir)
            / f"{base_name}_checkpoint_step_{trainer.total_count}.pt"
        )
    return Path(config.logging.log_dir) / f"{base_name}_checkpoint.pt"


def save_final_checkpoint(
    *,
    config: Any,
    effective_experiment_name: str,
    trainer: Any,
    checkpoint_path: str | None,
) -> Path:
    """Persist the final checkpoint and return its path."""
    final_checkpoint_path = build_final_checkpoint_path(
        config=config,
        effective_experiment_name=effective_experiment_name,
        trainer=trainer,
        checkpoint_path=checkpoint_path,
    )
    trainer.save_checkpoint(str(final_checkpoint_path))
    return final_checkpoint_path


def log_final_metrics(
    *,
    logs: dict[str, Any],
    final_metrics: dict[str, Any],
    mlflow_callback: Any,
) -> None:
    """Emit final aggregate metrics through the MLflow callback helpers."""
    MLflowTrainingCallback.log_final_metrics(logs, final_metrics, mlflow_callback)


def build_experiment_result(
    *,
    trainer: Any,
    logs: dict[str, Any],
    interrupted: bool,
    final_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Build the public result payload returned by run_single_experiment."""
    return {
        "trainer": trainer,
        "logs": logs,
        "interrupted": interrupted,
        "final_metrics": final_metrics,
        "plots": {
            "loss": visualize_training(logs)
            if logs.get("loss_value") or logs.get("loss_actor")
            else None,
        },
    }
