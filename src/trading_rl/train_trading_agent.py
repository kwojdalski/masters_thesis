"""Main training script for trading agent - refactored version.

This is a clean, modular version of the trading RL training script.
All configuration, data processing, models, and training logic have been
separated into reusable modules.
"""

# %%
import contextlib
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gym_trading_env  # noqa: F401
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from joblib import Memory

from logger import trace_calls
from trading_rl.callbacks import MLflowTrainingCallback
from trading_rl.config import ExperimentConfig
from trading_rl.data_utils import (
    PreparedDataset,
    build_prepared_dataset,
    ensure_close_column_for_hft,
    ensure_unique_index_for_hft_tradingenv,
    validate_prepared_data,
)
from trading_rl.envs import AlgorithmicEnvironmentBuilder
from trading_rl.evaluation import (
    EvaluationContext,
    build_evaluation_report_for_trainer,
    run_all_statistical_tests,
)
from trading_rl.evaluation.explainability import RLInterpretabilityAnalyzer
from trading_rl.pipeline.evaluation import (
    SplitEvaluationResult,
    build_evaluation_context_for_split as _build_evaluation_context_for_split_impl,
    build_final_metrics as _build_final_metrics_impl,
    compute_strategy_simple_returns_for_split as _compute_strategy_simple_returns_for_split_impl,
    evaluate_all_splits as _evaluate_all_splits_impl,
    evaluate_split as _evaluate_split_impl,
    resolve_primary_split_result as _resolve_primary_split_result_impl,
    resolve_price_series_for_split as _resolve_price_series_for_split_impl,
    run_primary_split_explainability as _run_primary_split_explainability_impl,
    run_statistical_tests_for_split as _run_statistical_tests_for_split_impl,
)
from trading_rl.pipeline.training import (
    ExperimentRuntime,
    TrainingBundle,
    build_experiment_runtime,
)
from trading_rl.plotting import visualize_training

# Avoid torch_shm_manager requirement in restricted environments
mp.set_sharing_strategy("file_system")

# gym_trading_env sets warnings to errors; reset to defaults for TorchRL
warnings.filterwarnings("default")
# tradingenv TrackRecord converts pandas.Timestamp -> datetime (microseconds),
# which emits this noisy warning when nanoseconds are present.
warnings.filterwarnings(
    "ignore",
    message=r"Discarding nonzero nanoseconds in conversion\.",
    category=UserWarning,
)

# Setup joblib memory for caching expensive operations
memory = Memory(location=".cache/joblib", verbose=1)


def clear_cache():
    """Clear all joblib caches."""
    memory.clear(warn=True)


@dataclass
class CheckpointResumptionResult:
    """Result from checkpoint resumption setup."""

    mlflow_callback: MLflowTrainingCallback
    effective_experiment_name: str
    original_steps: int


@trace_calls(show_return=True)
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


def _validate_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: ExperimentConfig,
) -> None:
    """Validate loaded data for training.

    Args:
        train_df: Training dataframe
        val_df: Validation dataframe
        test_df: Test dataframe
        config: Experiment configuration

    Raises:
        ValueError: If data is invalid (empty, missing columns, NaN values)
    """
    validate_prepared_data(train_df, val_df, test_df, config)


def _ensure_close_column_for_hft(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: ExperimentConfig,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Ensure raw `close` exists in HFT mode by deriving mid-price from L1 book.

    For HFT/LOB datasets we may not have OHLC `close`, but training validation
    still expects a raw close column for environment pricing checks.
    """
    return ensure_close_column_for_hft(train_df, val_df, test_df, config, logger)


def _ensure_unique_index_for_hft_tradingenv(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: ExperimentConfig,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Ensure unique, monotonic timestamps for HFT data used with TradingEnv.

    TradingEnv rejects duplicate indices. For event-level LOB feeds, multiple
    updates can share the same timestamp. We preserve every row by applying the
    minimal offsets needed to make the index strictly increasing with at least
    1-second spacing, which is required by tradingenv latency partitioning.
    """
    return ensure_unique_index_for_hft_tradingenv(
        train_df, val_df, test_df, config, logger
    )


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


def _setup_mlflow_tracking_from_checkpoint(trainer: Any) -> str | None:
    """Extract and setup MLflow tracking URI from checkpoint."""
    tracking_uri = getattr(trainer, "mlflow_tracking_uri", None)
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    return tracking_uri


def _run_explainability_analysis(
    *,
    config: ExperimentConfig,
    trainer: Any,
    eval_ctx: EvaluationContext,
    train_df: pd.DataFrame,
    logger: logging.Logger,
    artifact_path_prefix: str | None = None,
) -> None:
    """Run explainability analysis and log results to MLflow.

    Args:
        config: Experiment configuration with explainability settings
        trainer: Trained agent
        eval_ctx: Evaluation context with environment for rollout
        train_df: Training dataframe to extract feature names
        logger: Logger instance
        artifact_path_prefix: Optional path prefix for MLflow artifacts
            (default: "explainability", for temp: "explainability_temp/step_00005000")
    """
    if not config.explainability.enabled:
        return

    logger.info("Running explainability analysis...")

    try:
        # Extract feature names from environment config (matches observation dimensions)
        if config.env.feature_columns:
            feature_names = config.env.feature_columns
        else:
            # Fallback: extract all feature columns from train_df
            feature_names = [col for col in train_df.columns if str(col).startswith("feature_")]

        logger.debug(f"Using {len(feature_names)} features for explainability: {feature_names}")

        # Get rollout for analysis
        rollout = eval_ctx.env.rollout(max_steps=config.explainability.n_steps)
        obs_batch = rollout["observation"]
        logger.debug(f"Observation batch shape: {obs_batch.shape}")

        # Create analyzer
        analyzer = RLInterpretabilityAnalyzer(trainer, feature_names)

        # Collect results for merged plot
        results = {}

        # Run each requested method
        for method in config.explainability.methods:
            if method == "permutation":
                logger.info("Computing permutation importance...")
                df = analyzer.compute_global_importance(obs_batch)
                plot = analyzer.plot_importance(df, title="Global Feature Importance (Permutation)", color="steelblue")
                metrics = analyzer.quantify_interpretability(df)
                MLflowTrainingCallback.log_explainability_results(
                    df, plot, method="permutation", metrics=metrics, artifact_path_prefix=artifact_path_prefix
                )
                results["permutation"] = df
                logger.info("Permutation importance analysis complete.")

            elif method == "integrated_gradients":
                logger.info("Computing integrated gradients importance...")
                df = analyzer.compute_global_ig(obs_batch)
                plot = analyzer.plot_importance(df, title="Global Feature Importance (Integrated Gradients)", color="coral")
                metrics = analyzer.quantify_interpretability(df)
                MLflowTrainingCallback.log_explainability_results(
                    df, plot, method="integrated_gradients", metrics=metrics, artifact_path_prefix=artifact_path_prefix
                )
                results["integrated_gradients"] = df
                logger.info("Integrated gradients importance analysis complete.")

        # Create merged plot if both methods were run
        if "permutation" in results and "integrated_gradients" in results:
            logger.info("Creating merged explainability plot...")
            merged_plot = analyzer.plot_importance_merged(
                results["permutation"],
                results["integrated_gradients"]
            )
            MLflowTrainingCallback.log_explainability_results(
                None, merged_plot, method="merged", metrics=None, artifact_path_prefix=artifact_path_prefix
            )
            logger.info("Merged explainability plot saved.")

    except Exception as e:
        logger.error(f"Failed to run explainability analysis: {e}")


def _build_evaluation_context_for_split(
    *,
    split: str,
    df: pd.DataFrame,
    config: ExperimentConfig,
) -> EvaluationContext:
    """Build an evaluation context for a given data split.

    The caller is responsible for selecting the correct DataFrame for the split.
    This function only builds the environment and computes max_steps, keeping
    the dataframe and environment coupled so they cannot silently diverge.

    Args:
        split: Split name ("train", "val", or "test") — stored on the context
            for logging and artifact namespacing.
        df: DataFrame for this split.
        config: Experiment configuration.
    """
    return _build_evaluation_context_for_split_impl(
        split=split,
        df=df,
        config=config,
        build_environment=lambda split_df, split_config: AlgorithmicEnvironmentBuilder().create(
            split_df, split_config
        ),
    )


def _resolve_experiment_name_from_checkpoint(
    trainer: Any,
    config: ExperimentConfig,
    effective_experiment_name: str,
    logger: logging.Logger,
) -> str:
    """Resolve experiment name from checkpoint metadata, updating config if needed."""
    resume_experiment_name = getattr(trainer, "mlflow_experiment_name", None)

    # Fallback to experiment ID lookup
    if not resume_experiment_name:
        experiment_id = getattr(trainer, "mlflow_experiment_id", None)
        if experiment_id:
            experiment = mlflow.get_experiment(experiment_id)
            resume_experiment_name = experiment.name if experiment else None

    # Update config and trainer if we found a name
    if resume_experiment_name:
        if resume_experiment_name != config.experiment_name:
            logger.info(
                "Resuming MLflow experiment name from checkpoint: %s",
                resume_experiment_name,
            )
        config.experiment_name = resume_experiment_name
        effective_experiment_name = resume_experiment_name
        if hasattr(trainer, "checkpoint_prefix"):
            trainer.checkpoint_prefix = resume_experiment_name

    return effective_experiment_name


def _start_mlflow_run_for_resumption(
    trainer: Any,
    original_steps: int,
    logger: logging.Logger,
) -> None:
    """Start or resume MLflow run for checkpoint resumption."""
    resume_run_id = getattr(trainer, "mlflow_run_id", None)
    if resume_run_id:
        logger.info(f"Resuming MLflow run: {resume_run_id}")
        mlflow.start_run(run_id=resume_run_id)
    else:
        logger.info(
            f"Creating new MLflow run for resumed training from step {original_steps}"
        )
        mlflow.start_run(run_name=f"resumed_step_{original_steps}")


def _get_episode_count_from_trainer(trainer: Any) -> int:
    """Extract current episode count from trainer state."""
    logged = (
        trainer.logs.get("episode_log_count") if hasattr(trainer, "logs") else None
    )
    if logged:
        return int(logged[-1])
    total = trainer.total_episodes
    if hasattr(total, "item"):
        return int(total.item())
    return int(total)


def _create_resumption_callback(
    trainer: Any,
    config: ExperimentConfig,
    train_df: pd.DataFrame,
    effective_experiment_name: str,
    tracking_uri: str | None,
) -> MLflowTrainingCallback:
    """Create MLflow callback for resumed training with proper state."""
    configured_price_column = getattr(config.env, "price_column", None)
    if (
        isinstance(configured_price_column, str)
        and configured_price_column in train_df.columns
    ):
        price_series = train_df[configured_price_column]
    elif "close" in train_df.columns:
        price_series = train_df["close"]
    elif "price" in train_df.columns:
        price_series = train_df["price"]
    else:
        price_series = None
    fallback_uri = getattr(getattr(config, "tracking", None), "tracking_uri", None)

    mlflow_callback = MLflowTrainingCallback(
        effective_experiment_name,
        tracking_uri=tracking_uri or fallback_uri,
        price_series=price_series,
        start_run=False,  # We already started it
    )
    mlflow_callback._episode_count = _get_episode_count_from_trainer(trainer)

    return mlflow_callback


def _setup_checkpoint_resumption(
    checkpoint_path: str,
    trainer: Any,
    config: ExperimentConfig,
    train_df: pd.DataFrame,
    effective_experiment_name: str,
    additional_steps: int | None,
    logger: logging.Logger,
) -> CheckpointResumptionResult:
    """Load checkpoint, setup MLflow tracking, and create callback for resumed training.

    This handles:
    1. Checkpoint validation and loading
    2. MLflow experiment/run resumption
    3. Callback creation with proper state

    Modifies trainer and config in place.

    Args:
        checkpoint_path: Path to checkpoint file
        trainer: Trainer instance to load checkpoint into
        config: Experiment configuration (modified in place)
        train_df: Training dataframe for price series
        effective_experiment_name: Initial experiment name (may be updated)
        additional_steps: Optional additional training steps
        logger: Logger instance

    Returns:
        CheckpointResumptionResult with callback, experiment name, and original steps
    """
    # Validate checkpoint exists
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info("Resuming training from checkpoint")
    logger.info(f"Checkpoint: {checkpoint_path}")

    # Load checkpoint
    trainer.load_checkpoint(str(checkpoint_path))
    original_steps = trainer.total_count
    logger.info(f"Checkpoint loaded! Resuming from step {original_steps}")

    # Update max_steps for additional training
    if additional_steps:
        trainer.config.max_steps = original_steps + additional_steps
        logger.info(f"Training for {additional_steps} additional steps")
        logger.info(f"Target: {trainer.config.max_steps} total steps")

    # Setup MLflow tracking
    tracking_uri = _setup_mlflow_tracking_from_checkpoint(trainer)

    # Resolve experiment name from checkpoint
    effective_experiment_name = _resolve_experiment_name_from_checkpoint(
        trainer, config, effective_experiment_name, logger
    )

    setup_mlflow_experiment(config, effective_experiment_name)

    # Start MLflow run (resume or create new)
    _start_mlflow_run_for_resumption(trainer, original_steps, logger)

    # Create callback
    mlflow_callback = _create_resumption_callback(
        trainer, config, train_df, effective_experiment_name, tracking_uri
    )

    # Log artifacts
    if mlflow.active_run():
        MLflowTrainingCallback.log_parameter_faq_artifact()
        MLflowTrainingCallback.log_training_parameters(config)
        MLflowTrainingCallback.log_config_artifact(config)
        MLflowTrainingCallback.log_data_overview(train_df, config)

    return CheckpointResumptionResult(
        mlflow_callback=mlflow_callback,
        effective_experiment_name=effective_experiment_name,
        original_steps=original_steps,
    )


def _compute_strategy_simple_returns_for_split(
    *,
    rollout: Any,
    split_ctx: EvaluationContext,
    config: ExperimentConfig,
) -> np.ndarray:
    """Compute simple returns for statistical testing from a rollout."""
    return _compute_strategy_simple_returns_for_split_impl(
        rollout=rollout,
        split_ctx=split_ctx,
        config=config,
    )


def _resolve_price_series_for_split(
    split_ctx: EvaluationContext,
    config: ExperimentConfig,
    logger: logging.Logger,
) -> pd.Series | None:
    """Resolve benchmark price series for one split."""
    return _resolve_price_series_for_split_impl(
        split_ctx=split_ctx,
        config=config,
        logger=logger,
    )


def _run_statistical_tests_for_split(
    *,
    trainer: Any,
    split_ctx: EvaluationContext,
    config: ExperimentConfig,
    logger: logging.Logger,
) -> None:
    """Run and log statistical significance tests for one split."""
    def _run_rollout(rollout_trainer: Any, rollout_split_ctx: EvaluationContext) -> Any:
        from tensordict.nn import InteractionType
        from torchrl.envs.utils import set_exploration_type

        with torch.no_grad():
            try:
                with set_exploration_type(InteractionType.MODE):
                    return rollout_split_ctx.env.rollout(
                        max_steps=rollout_split_ctx.max_steps,
                        policy=rollout_trainer.actor,
                    )
            except RuntimeError:
                with set_exploration_type(InteractionType.DETERMINISTIC):
                    return rollout_split_ctx.env.rollout(
                        max_steps=rollout_split_ctx.max_steps,
                        policy=rollout_trainer.actor,
                    )

    _run_statistical_tests_for_split_impl(
        trainer=trainer,
        split_ctx=split_ctx,
        config=config,
        logger=logger,
        run_rollout=_run_rollout,
        run_all_statistical_tests_fn=run_all_statistical_tests,
        log_statistical_tests_fn=MLflowTrainingCallback.log_statistical_tests,
    )


def _evaluate_split(
    *,
    split: str,
    split_df: pd.DataFrame,
    trainer: Any,
    config: ExperimentConfig,
    algorithm: str,
    logs: dict[str, Any],
    logger: logging.Logger,
) -> SplitEvaluationResult | None:
    """Evaluate one split and log associated artifacts."""
    return _evaluate_split_impl(
        split=split,
        split_df=split_df,
        trainer=trainer,
        config=config,
        algorithm=algorithm,
        logs=logs,
        logger=logger,
        build_evaluation_context_fn=_build_evaluation_context_for_split,
        log_evaluation_plots_fn=MLflowTrainingCallback.log_evaluation_plots,
        build_evaluation_report_for_trainer_fn=build_evaluation_report_for_trainer,
        log_evaluation_report_fn=MLflowTrainingCallback.log_evaluation_report,
        run_statistical_tests_for_split_fn=_run_statistical_tests_for_split,
    )


def _evaluate_all_splits(
    *,
    trainer: Any,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: ExperimentConfig,
    algorithm: str,
    logs: dict[str, Any],
    logger: logging.Logger,
) -> dict[str, dict[str, Any]]:
    """Evaluate train/val/test splits and return serializable results."""
    return _evaluate_all_splits_impl(
        trainer=trainer,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        config=config,
        algorithm=algorithm,
        logs=logs,
        logger=logger,
        evaluate_split_fn=_evaluate_split,
    )


def _resolve_primary_split_result(
    split_results: dict[str, dict[str, Any]],
) -> tuple[str | None, float, list[Any], dict[str, float]]:
    """Resolve primary split result using preference test -> val -> train."""
    return _resolve_primary_split_result_impl(split_results)


def _run_primary_split_explainability(
    *,
    primary_split: str | None,
    trainer: Any,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: ExperimentConfig,
    logger: logging.Logger,
) -> None:
    """Run explainability only on the selected primary split."""
    _run_primary_split_explainability_impl(
        primary_split=primary_split,
        trainer=trainer,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        config=config,
        logger=logger,
        build_evaluation_context_fn=_build_evaluation_context_for_split,
        run_explainability_analysis_fn=_run_explainability_analysis,
    )


def _build_final_metrics(
    *,
    config: ExperimentConfig,
    effective_experiment_name: str,
    interrupted: bool,
    logs: dict[str, Any],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_obs: int,
    n_act: int,
    primary_split: str | None,
    final_reward: float,
    last_positions: list[Any],
    evaluation_report: dict[str, float],
    split_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build final experiment metrics payload."""
    return _build_final_metrics_impl(
        config=config,
        effective_experiment_name=effective_experiment_name,
        interrupted=interrupted,
        logs=logs,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        n_obs=n_obs,
        n_act=n_act,
        primary_split=primary_split,
        final_reward=final_reward,
        last_positions=last_positions,
        evaluation_report=evaluation_report,
        split_results=split_results,
    )


@trace_calls(show_return=False)
def run_single_experiment(
    custom_config: ExperimentConfig | None = None,
    experiment_name: str | None = None,
    progress_bar: Any = None,
    checkpoint_path: str | None = None,
    additional_steps: int | None = None,
) -> dict[str, Any]:
    """Run a single training experiment with MLflow tracking (fresh or resumed).

    This function tracks both losses and position statistics in MLflow:
    - Multiple metrics logged simultaneously (losses, positions, rewards)
    - Parameters logged for configuration tracking
    - Artifacts logged for plots and models
    - Supports resuming from checkpoint with optional additional steps

    Args:
        custom_config: Optional custom configuration
        experiment_name: Optional override for MLflow experiment name (uses config.experiment_name by default)
        progress_bar: Optional Rich progress bar for episode tracking
        checkpoint_path: Optional path to checkpoint to resume from
        additional_steps: Optional additional steps when resuming (extends max_steps)

    Returns:
        Dictionary with results
    """
    config = custom_config or ExperimentConfig()

    # For checkpoint resume, don't create MLflow callback yet
    # (we'll create it after loading checkpoint metadata)
    create_callback = not checkpoint_path

    runtime = build_experiment_runtime(
        config=config,
        experiment_name=experiment_name,
        progress_bar=progress_bar,
        create_mlflow_callback=create_callback,
    )
    logger = runtime.logger
    prepared_dataset = runtime.prepared_dataset
    train_df = prepared_dataset.train_df
    val_df = prepared_dataset.val_df
    test_df = prepared_dataset.test_df
    training_bundle = runtime.training_bundle
    trainer = training_bundle.trainer
    mlflow_callback = training_bundle.mlflow_callback  # May be None for resume
    algorithm = training_bundle.algorithm
    n_obs = training_bundle.n_obs
    n_act = training_bundle.n_act
    effective_experiment_name = runtime.effective_experiment_name

    # Handle checkpoint resumption
    if checkpoint_path:
        result = _setup_checkpoint_resumption(
            checkpoint_path=checkpoint_path,
            trainer=trainer,
            config=config,
            train_df=train_df,
            effective_experiment_name=effective_experiment_name,
            additional_steps=additional_steps,
            logger=logger,
        )
        mlflow_callback = result.mlflow_callback
        effective_experiment_name = result.effective_experiment_name

    # Build train-split context for periodic mid-training evaluation.
    # This must always use train_df — using val/test here would leak
    # out-of-sample signal into training decisions.
    periodic_eval_ctx = _build_evaluation_context_for_split(
        split="train",
        df=train_df,
        config=config,
    )
    trainer.setup_periodic_evaluation(
        df=periodic_eval_ctx.df,
        max_steps=periodic_eval_ctx.max_steps,
        config=config,
        algorithm=algorithm,
        eval_env=periodic_eval_ctx.env,
    )

    # Setup periodic explainability (if enabled in config)
    trainer.setup_periodic_explainability(
        df=periodic_eval_ctx.df,
        max_steps=config.explainability.n_steps,
        config=config,
        eval_env=periodic_eval_ctx.env,
    )

    # Train (Ctrl-C should still trigger final evaluation on the current model state)
    logger.info("Starting training...")
    interrupted = False
    try:
        logs = trainer.train(callback=mlflow_callback)
    except KeyboardInterrupt:
        interrupted = True
        logger.warning(
            "Training interrupted by user (Ctrl-C). Running final evaluation on current model state..."
        )
        logs = dict(trainer.logs)

    # Save checkpoint using MLflow run name if available
    run = mlflow.active_run()
    if run and run.info.run_name:
        # Use MLflow run name, sanitized for filesystem safety
        base_name = (
            run.info.run_name.replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
        )
    else:
        # Fallback to experiment name if no run is active
        base_name = effective_experiment_name

    if checkpoint_path:
        # For resumed training, save with step count
        final_checkpoint_path = (
            Path(config.logging.log_dir)
            / f"{base_name}_checkpoint_step_{trainer.total_count}.pt"
        )
    else:
        # For fresh training, simple name
        final_checkpoint_path = (
            Path(config.logging.log_dir) / f"{base_name}_checkpoint.pt"
        )
    trainer.save_checkpoint(str(final_checkpoint_path))

    split_results = _evaluate_all_splits(
        trainer=trainer,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        config=config,
        algorithm=algorithm,
        logs=logs,
        logger=logger,
    )

    primary_split, final_reward, last_positions, evaluation_report = (
        _resolve_primary_split_result(split_results)
    )
    _run_primary_split_explainability(
        primary_split=primary_split,
        trainer=trainer,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        config=config,
        logger=logger,
    )

    final_metrics = _build_final_metrics(
        config=config,
        effective_experiment_name=effective_experiment_name,
        interrupted=interrupted,
        logs=logs,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        n_obs=n_obs,
        n_act=n_act,
        primary_split=primary_split,
        final_reward=final_reward,
        last_positions=last_positions,
        evaluation_report=evaluation_report,
        split_results=split_results,
    )

    # Log final metrics to MLflow
    MLflowTrainingCallback.log_final_metrics(logs, final_metrics, mlflow_callback)

    if interrupted:
        logger.info("Training interrupted; final evaluation complete!")
    else:
        logger.info("Training complete!")
    logger.info(f"Final reward: {final_reward:.4f}")
    logger.info(f"Checkpoint saved to: {final_checkpoint_path}")

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


@trace_calls(show_return=True)
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

    # Ensure MLflow experiment is set up before starting run
    setup_mlflow_experiment(config)

    if n_trials == 1:
        # Run single experiment
        with mlflow.start_run():
            run_single_experiment(custom_config=config)
        return config.experiment_name
    else:
        # Run multiple experiments
        return run_multiple_experiments(n_trials=n_trials, custom_config=config)
