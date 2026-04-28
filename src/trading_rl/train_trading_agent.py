"""Main training entry point for trading RL experiments.

Public API
----------
run_single_experiment   — run one fresh or resumed experiment
run_multiple_experiments — run N seeded trials under one MLflow experiment
run_experiment_from_config — load a YAML config and run experiment(s)

Everything else (data prep, environment construction, evaluation,
checkpoint management, explainability) lives in trading_rl/pipeline/.
"""

# %%
import contextlib
import warnings
from typing import Any

import gym_trading_env  # noqa: F401

# gym_trading_env sets warnings.filterwarnings("error") at import time.
# Reset to defaults now so that TorchRL's own DeprecationWarnings (e.g.
# VanillaWeightUpdater) don't become fatal errors during the next imports.
warnings.filterwarnings("default")
# Suppress TorchRL internal deprecation warnings about WeightUpdaterBase subclasses;
# these are library-internal and not actionable from user code.
warnings.filterwarnings(
    "ignore",
    message=".*inherits from WeightUpdaterBase is deprecated.*",
    category=DeprecationWarning,
)

import mlflow
import torch.multiprocessing as mp

from logger import get_logger as get_project_logger
from logger import trace_calls
from trading_rl.config import ExperimentConfig
from trading_rl.data_utils import (
    ensure_close_column_for_hft,
    ensure_unique_index_for_hft_tradingenv,
)
from trading_rl.pipeline.experiment_runner import execute_single_experiment
from trading_rl.pipeline.training import (
    ExperimentRuntime,
    TrainingBundle,
    build_experiment_runtime,
    set_seed,
    setup_logging,
    setup_mlflow_experiment,
)
from trading_rl.callbacks import MLflowTrainingCallback
from trading_rl.plotting import visualize_training

# Avoid torch_shm_manager requirement in restricted environments
mp.set_sharing_strategy("file_system")
# tradingenv TrackRecord converts pandas.Timestamp -> datetime (microseconds),
# which emits this noisy warning when nanoseconds are present.
warnings.filterwarnings(
    "ignore",
    message=r"Discarding nonzero nanoseconds in conversion\.",
    category=UserWarning,
)

# ---------------------------------------------------------------------------
# Thin re-exports kept for test backwards-compatibility
# ---------------------------------------------------------------------------

# Tests in test_hft_close_derivation.py import these by name from this module.
_ensure_close_column_for_hft = ensure_close_column_for_hft
_ensure_unique_index_for_hft_tradingenv = ensure_unique_index_for_hft_tradingenv


def build_training_context(
    config: ExperimentConfig,
    experiment_name: str | None = None,
    progress_bar: Any = None,
    create_mlflow_callback: bool = True,
) -> dict[str, Any]:
    """Build backward-compatible dict context used by older callers/tests."""
    runtime = build_experiment_runtime(
        config=config,
        experiment_name=experiment_name,
        progress_bar=progress_bar,
        create_mlflow_callback=create_mlflow_callback,
    )
    prepared_dataset = runtime.prepared_dataset
    training_bundle = runtime.training_bundle
    return {
        "logger": runtime.logger,
        "prepared_dataset": prepared_dataset,
        "train_df": prepared_dataset.train_df,
        "val_df": prepared_dataset.val_df,
        "test_df": prepared_dataset.test_df,
        "env": training_bundle.train_env,
        "training_bundle": training_bundle,
        "trainer": training_bundle.trainer,
        "mlflow_callback": training_bundle.mlflow_callback,
        "algorithm": training_bundle.algorithm,
        "n_obs": training_bundle.n_obs,
        "n_act": training_bundle.n_act,
        "effective_experiment_name": runtime.effective_experiment_name,
    }


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

@trace_calls(show_return=False)
def run_single_experiment(
    custom_config: ExperimentConfig | None = None,
    experiment_name: str | None = None,
    progress_bar: Any = None,
    checkpoint_path: str | None = None,
    additional_steps: int | None = None,
) -> dict[str, Any]:
    """Run a single training experiment with MLflow tracking (fresh or resumed).

    Args:
        custom_config: Optional custom configuration.
        experiment_name: Optional override for MLflow experiment name.
        progress_bar: Optional Rich progress bar for episode tracking.
        checkpoint_path: Optional path to checkpoint to resume from.
        additional_steps: Optional additional steps when resuming.

    Returns:
        Dictionary with experiment results.
    """
    config = custom_config or ExperimentConfig()
    return execute_single_experiment(
        config=config,
        experiment_name=experiment_name,
        progress_bar=progress_bar,
        checkpoint_path=checkpoint_path,
        additional_steps=additional_steps,
        build_experiment_runtime_fn=build_experiment_runtime,
    )


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

    Args:
        n_trials: Number of experiments to run.
        base_seed: Base seed for reproducible experiments.
        custom_config: Optional custom configuration.
        experiment_name: Optional override for MLflow experiment name.
        show_progress: Whether to show progress bar for episodes.

    Returns:
        MLflow experiment name with all results.
    """
    import copy
    import random

    from rich.progress import Progress

    config = custom_config or ExperimentConfig()
    effective_experiment_name = experiment_name or config.experiment_name
    setup_mlflow_experiment(config, effective_experiment_name)

    logger = get_project_logger(__name__)
    progress_context = Progress() if show_progress else None

    with progress_context if progress_context else contextlib.nullcontext() as progress:
        for trial_number in range(n_trials):
            logger.info("run trial trial=%d n_trials=%d", trial_number + 1, n_trials)

            if custom_config is not None:
                trial_config = copy.deepcopy(custom_config)
            else:
                trial_config = ExperimentConfig()

            if base_seed is not None:
                trial_config.seed = base_seed + trial_number
            else:
                trial_config.seed = random.randint(1, 100000)  # noqa: S311

            trial_config.experiment_name = effective_experiment_name

            with mlflow.start_run(run_name=f"trial_{trial_number}"):
                run_single_experiment(
                    custom_config=trial_config,
                    progress_bar=progress if show_progress else None,
                )

    return effective_experiment_name


@trace_calls(show_return=True)
def run_experiment_from_config(config_path: str, n_trials: int = 1) -> str:
    """Load experiment config from YAML file and run experiment(s).

    Args:
        config_path: Path to YAML configuration file.
        n_trials: Number of trials to run (defaults to 1).

    Returns:
        MLflow experiment name.
    """
    config = ExperimentConfig.from_yaml(config_path)
    setup_mlflow_experiment(config)

    if n_trials == 1:
        with mlflow.start_run():
            run_single_experiment(custom_config=config)
        return config.experiment_name
    else:
        return run_multiple_experiments(n_trials=n_trials, custom_config=config)
