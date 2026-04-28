"""High-level experiment execution orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from trading_rl.pipeline.checkpoint import setup_checkpoint_resumption
from trading_rl.pipeline.evaluation import (
    build_evaluation_context_for_split,
    build_final_metrics,
    evaluate_all_splits,
    resolve_primary_split_result,
    run_primary_split_explainability,
)
from trading_rl.pipeline.finalization import (
    build_experiment_result,
    log_final_metrics,
    save_final_checkpoint,
)
from trading_rl.pipeline.training import (
    ExperimentRuntime,
    build_experiment_runtime,
    setup_mlflow_experiment,
)


@dataclass(frozen=True)
class TrainingExecutionResult:
    """Outcome of the training phase before final evaluation/finalization."""

    logs: dict[str, Any]
    interrupted: bool


def _resolve_runtime(
    *,
    config: Any,
    experiment_name: str | None,
    progress_bar: Any,
    checkpoint_path: str | None,
    additional_steps: int | None,
    build_experiment_runtime_fn: Any = build_experiment_runtime,
) -> ExperimentRuntime:
    """Build runtime and optionally resume it from checkpoint metadata."""
    runtime = build_experiment_runtime_fn(
        config=config,
        experiment_name=experiment_name,
        progress_bar=progress_bar,
        create_mlflow_callback=not checkpoint_path,
    )
    if not checkpoint_path:
        return runtime

    train_df = runtime.prepared_dataset.train_df
    result = setup_checkpoint_resumption(
        checkpoint_path=checkpoint_path,
        trainer=runtime.training_bundle.trainer,
        config=config,
        train_df=train_df,
        effective_experiment_name=runtime.effective_experiment_name,
        additional_steps=additional_steps,
        logger=runtime.logger,
        setup_mlflow_experiment_fn=setup_mlflow_experiment,
    )
    return ExperimentRuntime(
        logger=runtime.logger,
        effective_experiment_name=result.effective_experiment_name,
        prepared_dataset=runtime.prepared_dataset,
        training_bundle=runtime.training_bundle.__class__(
            train_env=runtime.training_bundle.train_env,
            trainer=runtime.training_bundle.trainer,
            mlflow_callback=result.mlflow_callback,
            algorithm=runtime.training_bundle.algorithm,
            n_obs=runtime.training_bundle.n_obs,
            n_act=runtime.training_bundle.n_act,
        ),
    )


def _configure_periodic_hooks(
    *,
    runtime: ExperimentRuntime,
    config: Any,
) -> None:
    """Wire periodic mid-training evaluation and explainability hooks."""
    train_df = runtime.prepared_dataset.train_df
    periodic_eval_ctx = build_evaluation_context_for_split(
        split="train",
        df=train_df,
        config=config,
    )
    trainer = runtime.training_bundle.trainer
    trainer.setup_periodic_evaluation(
        df=periodic_eval_ctx.df,
        max_steps=periodic_eval_ctx.max_steps,
        config=config,
        algorithm=runtime.training_bundle.algorithm,
        eval_env=periodic_eval_ctx.env,
    )
    trainer.setup_periodic_explainability(
        df=periodic_eval_ctx.df,
        max_steps=config.explainability.n_steps,
        config=config,
        eval_env=periodic_eval_ctx.env,
    )


def _run_training_phase(
    *,
    runtime: ExperimentRuntime,
) -> TrainingExecutionResult:
    """Run training and normalize interruption handling."""
    logger = runtime.logger
    trainer = runtime.training_bundle.trainer
    mlflow_callback = runtime.training_bundle.mlflow_callback

    logger.info("start training")
    interrupted = False
    try:
        logs = trainer.train(callback=mlflow_callback)
    except KeyboardInterrupt:
        interrupted = True
        logger.warning(
            "Training interrupted by user (Ctrl-C). "
            "Running final evaluation on current model state..."
        )
        logs = dict(trainer.logs)
    return TrainingExecutionResult(logs=logs, interrupted=interrupted)


def execute_single_experiment(
    *,
    config: Any,
    experiment_name: str | None = None,
    progress_bar: Any = None,
    checkpoint_path: str | None = None,
    additional_steps: int | None = None,
    build_experiment_runtime_fn: Any = build_experiment_runtime,
) -> dict[str, Any]:
    """Run the end-to-end experiment flow and return the public result payload."""
    runtime = _resolve_runtime(
        config=config,
        experiment_name=experiment_name,
        progress_bar=progress_bar,
        checkpoint_path=checkpoint_path,
        additional_steps=additional_steps,
        build_experiment_runtime_fn=build_experiment_runtime_fn,
    )
    _configure_periodic_hooks(runtime=runtime, config=config)
    training_result = _run_training_phase(runtime=runtime)

    trainer = runtime.training_bundle.trainer
    prepared_dataset = runtime.prepared_dataset
    logger = runtime.logger
    final_checkpoint_path = save_final_checkpoint(
        config=config,
        effective_experiment_name=runtime.effective_experiment_name,
        trainer=trainer,
        checkpoint_path=checkpoint_path,
    )

    split_results = evaluate_all_splits(
        trainer=trainer,
        train_df=prepared_dataset.train_df,
        val_df=prepared_dataset.val_df,
        test_df=prepared_dataset.test_df,
        config=config,
        algorithm=runtime.training_bundle.algorithm,
        logs=training_result.logs,
        logger=logger,
    )
    primary_split, final_reward, last_positions, evaluation_report = (
        resolve_primary_split_result(split_results)
    )
    run_primary_split_explainability(
        primary_split=primary_split,
        trainer=trainer,
        train_df=prepared_dataset.train_df,
        val_df=prepared_dataset.val_df,
        test_df=prepared_dataset.test_df,
        config=config,
        logger=logger,
    )

    final_metrics = build_final_metrics(
        config=config,
        effective_experiment_name=runtime.effective_experiment_name,
        interrupted=training_result.interrupted,
        logs=training_result.logs,
        train_df=prepared_dataset.train_df,
        val_df=prepared_dataset.val_df,
        test_df=prepared_dataset.test_df,
        n_obs=runtime.training_bundle.n_obs,
        n_act=runtime.training_bundle.n_act,
        primary_split=primary_split,
        final_reward=final_reward,
        last_positions=last_positions,
        evaluation_report=evaluation_report,
        split_results=split_results,
    )
    log_final_metrics(
        logs=training_result.logs,
        final_metrics=final_metrics,
        mlflow_callback=runtime.training_bundle.mlflow_callback,
    )

    if training_result.interrupted:
        logger.info("training interrupted final_eval=complete")
    else:
        logger.info("training complete")
    logger.info("final reward=%.4f", final_reward)
    logger.info("save checkpoint path=%s", final_checkpoint_path)

    return build_experiment_result(
        trainer=trainer,
        logs=training_result.logs,
        interrupted=training_result.interrupted,
        final_metrics=final_metrics,
    )
