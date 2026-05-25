"""High-level experiment execution orchestration."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from logger import get_logger as _get_logger
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
from trading_rl.profiler import init_profiler

_logger = _get_logger(__name__)


def _json_default(obj: Any) -> Any:
    import numpy as np
    from trading_rl.evaluation.metrics import MetricReport
    if isinstance(obj, MetricReport):
        return obj.to_dict()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def _sanitise(obj: Any) -> Any:
    """Replace NaN/Inf with None so json.dumps produces valid JSON."""
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    if isinstance(obj, dict):
        return {k: _sanitise(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitise(v) for v in obj]
    return obj


def save_training_results_json(config: Any, final_metrics: dict[str, Any]) -> Path | None:
    """Write results.json to log_dir in the same format the evaluate command produces.

    This lets export_eval_to_thesis.py work after a training run without
    needing a separate evaluate step.
    """
    try:
        from trading_rl.evaluation.metrics import MetricReport

        split_results: dict[str, Any] = final_metrics.get("split_results", {})
        if not split_results:
            _logger.debug("save_training_results_json: no split_results, skipping")
            return None

        payload: dict[str, Any] = {}
        for split_key, entry in split_results.items():
            evaluation_report = entry.get("evaluation_report")
            last_positions = entry.get("last_positions") or []
            metrics_dict: dict[str, Any] | None = None
            if isinstance(evaluation_report, MetricReport):
                metrics_dict = {
                    k: v for k, v in evaluation_report.to_dict().items()
                    if isinstance(v, (int, float)) and math.isfinite(float(v))
                }
            elif isinstance(evaluation_report, dict):
                metrics_dict = evaluation_report

            payload[split_key] = {
                "split": entry.get("split", split_key),
                "final_reward": entry.get("final_reward"),
                "n_steps": len(last_positions),
                "metrics": metrics_dict or {},
            }

        out_path = Path(config.logging.log_dir) / "results.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(_sanitise(payload), indent=2, default=_json_default),
            encoding="utf-8",
        )
        _logger.info("results.json written path=%s", out_path)
        return out_path
    except Exception:
        _logger.warning("save_training_results_json failed", exc_info=True)
        return None


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
    from trading_rl.trainers.runtime_hooks import SplitEvalContext

    dataset = runtime.prepared_dataset
    split_map = {
        "train": dataset.train_df,
        "val": dataset.val_df,
        "test": dataset.test_df,
    }

    configured_splits: list[str] = list(
        getattr(config.training, "temp_eval_splits", ["train"])
    )
    temp_eval_max_steps: int = getattr(config.training, "temp_eval_max_steps", 200000)

    split_contexts: list[SplitEvalContext] = []
    for split_name in configured_splits:
        df = split_map.get(split_name)
        if df is None or len(df) < 2:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "periodic eval: split '%s' not available or too small, skipping", split_name
            )
            continue
        ctx = build_evaluation_context_for_split(split=split_name, df=df, config=config)
        split_contexts.append(
            SplitEvalContext(
                split=split_name,
                df=ctx.df,
                max_steps=min(ctx.max_steps, temp_eval_max_steps),
                eval_env=ctx.env,
            )
        )

    trainer = runtime.training_bundle.trainer
    trainer.setup_periodic_evaluation(
        splits=split_contexts,
        config=config,
        algorithm=runtime.training_bundle.algorithm,
    )

    # Explainability hook continues to use train data only
    train_ctx = build_evaluation_context_for_split(
        split="train", df=dataset.train_df, config=config
    )
    trainer.setup_periodic_explainability(
        df=train_ctx.df,
        max_steps=config.explainability.n_steps,
        config=config,
        eval_env=train_ctx.env,
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
    profiler = init_profiler(level=config.profiling.level if getattr(config, "profiling", None) else 0)

    with profiler.stage("runtime_build"):
        runtime = _resolve_runtime(
            config=config,
            experiment_name=experiment_name,
            progress_bar=progress_bar,
            checkpoint_path=checkpoint_path,
            additional_steps=additional_steps,
            build_experiment_runtime_fn=build_experiment_runtime_fn,
        )
        _configure_periodic_hooks(runtime=runtime, config=config)

    with profiler.stage("training"):
        training_result = _run_training_phase(runtime=runtime)

    trainer = runtime.training_bundle.trainer
    prepared_dataset = runtime.prepared_dataset
    logger = runtime.logger

    with profiler.stage("checkpoint_save"):
        final_checkpoint_path = save_final_checkpoint(
            config=config,
            effective_experiment_name=runtime.effective_experiment_name,
            trainer=trainer,
            checkpoint_path=checkpoint_path,
            feature_pipeline_state=prepared_dataset.feature_pipeline_state,
        )

    split_results = {}
    primary_split = None
    final_reward = float("nan")
    last_positions = []
    evaluation_report = {}

    try:
        with profiler.stage("eval_all_splits"):
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
        with profiler.stage("explainability"):
            run_primary_split_explainability(
                primary_split=primary_split,
                trainer=trainer,
                train_df=prepared_dataset.train_df,
                val_df=prepared_dataset.val_df,
                test_df=prepared_dataset.test_df,
                config=config,
                logger=logger,
            )

        with profiler.stage("finalization"):
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
                total_env_steps=int(trainer.total_count),
                total_episodes=int(trainer.total_episodes),
            )
            log_final_metrics(
                logs=training_result.logs,
                final_metrics=final_metrics,
                mlflow_callback=runtime.training_bundle.mlflow_callback,
            )
    except KeyboardInterrupt:
        logger.warning(
            "Evaluation interrupted by user. Using partial evaluation results..."
        )
        if split_results:
            primary_split, final_reward, last_positions, evaluation_report = (
                resolve_primary_split_result(split_results)
            )
        final_metrics = build_final_metrics(
            config=config,
            effective_experiment_name=runtime.effective_experiment_name,
            interrupted=True,
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
            total_env_steps=int(trainer.total_count),
            total_episodes=int(trainer.total_episodes),
        )
        log_final_metrics(
            logs=training_result.logs,
            final_metrics=final_metrics,
            mlflow_callback=runtime.training_bundle.mlflow_callback,
        )
        logger.info("evaluation interrupted results_saved")

    if training_result.interrupted:
        logger.info("training interrupted final_eval=complete")
    else:
        logger.info("training complete")
    logger.info("final reward=%.4f", final_reward)
    logger.info("save checkpoint path=%s", final_checkpoint_path)

    save_training_results_json(config, final_metrics)

    profiler.print_table()

    return build_experiment_result(
        trainer=trainer,
        logs=training_result.logs,
        interrupted=training_result.interrupted,
        final_metrics=final_metrics,
    )
