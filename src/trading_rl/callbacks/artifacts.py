"""Backwards-compatible re-export shim for trading_rl.callbacks.artifacts.

The artifact logging helpers have been split into focused modules:
  - artifacts_config.py     — config / parameter logging
  - artifacts_data.py       — raw / transformed data overview
  - artifacts_evaluation.py — evaluation result logging

Import from those modules directly for new code.
"""

from __future__ import annotations

from trading_rl.callbacks.artifacts_config import (
    _to_yaml_serializable,
    log_config_artifact,
    log_parameter_faq_artifact,
    log_training_parameters,
)
from trading_rl.callbacks.artifacts_data import (
    _log_feature_vs_return_scatter,
    _log_oracle_vs_reward_alignment,
    _log_overview_impl,
    log_feature_descriptive_stats,
    log_raw_data_overview,
    log_transformed_data_overview,
)
from trading_rl.callbacks.artifacts_evaluation import (
    ArtifactPaths,
    log_evaluation_plots,
    log_evaluation_report,
    log_explainability_results,
    log_final_metrics,
    log_statistical_tests,
    save_eval_rollout_artifact,
    save_observation_sample_artifact,
)

__all__ = [
    "ArtifactPaths",
    "_to_yaml_serializable",
    "log_config_artifact",
    "log_training_parameters",
    "log_parameter_faq_artifact",
    "_log_overview_impl",
    "log_raw_data_overview",
    "log_transformed_data_overview",
    "log_feature_descriptive_stats",
    "_log_feature_vs_return_scatter",
    "_log_oracle_vs_reward_alignment",
    "save_observation_sample_artifact",
    "save_eval_rollout_artifact",
    "log_final_metrics",
    "log_evaluation_report",
    "log_statistical_tests",
    "log_explainability_results",
    "log_evaluation_plots",
]
