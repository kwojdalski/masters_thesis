"""Pipeline building blocks for the trading RL orchestration layer."""

from trading_rl.pipeline.checkpoint import (
    CheckpointResumptionResult,
    setup_checkpoint_resumption,
)
from trading_rl.pipeline.evaluation import (
    SplitEvaluationResult,
    build_evaluation_context_for_split,
    build_final_metrics,
    evaluate_all_splits,
    evaluate_split,
    resolve_primary_split_result,
    run_primary_split_explainability,
    run_statistical_tests_for_split,
)
from trading_rl.pipeline.experiment_runner import execute_single_experiment
from trading_rl.pipeline.finalization import (
    build_experiment_result,
    build_final_checkpoint_path,
    log_final_metrics,
    save_final_checkpoint,
)
from trading_rl.pipeline.explainability import run_explainability_analysis
from trading_rl.pipeline.training import (
    ExperimentRuntime,
    TrainingBundle,
    build_experiment_runtime,
    set_seed,
    setup_logging,
    setup_mlflow_experiment,
)

__all__ = [
    # checkpoint
    "CheckpointResumptionResult",
    "setup_checkpoint_resumption",
    # evaluation
    "SplitEvaluationResult",
    "build_evaluation_context_for_split",
    "build_final_metrics",
    "evaluate_all_splits",
    "evaluate_split",
    "resolve_primary_split_result",
    "run_primary_split_explainability",
    "run_statistical_tests_for_split",
    # experiment_runner
    "execute_single_experiment",
    # finalization
    "build_experiment_result",
    "build_final_checkpoint_path",
    "log_final_metrics",
    "save_final_checkpoint",
    # explainability
    "run_explainability_analysis",
    # training
    "ExperimentRuntime",
    "TrainingBundle",
    "build_experiment_runtime",
    "set_seed",
    "setup_logging",
    "setup_mlflow_experiment",
]
