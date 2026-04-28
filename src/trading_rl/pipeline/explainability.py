"""Post-training explainability analysis for trained RL agents."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from trading_rl.callbacks import MLflowTrainingCallback
from trading_rl.config import ExperimentConfig
from trading_rl.constants import ExplainabilityMethod
from trading_rl.evaluation import EvaluationContext
from trading_rl.evaluation.explainability import RLInterpretabilityAnalyzer


def run_explainability_analysis(
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
        config: Experiment configuration with explainability settings.
        trainer: Trained agent.
        eval_ctx: Evaluation context with environment for rollout.
        train_df: Training dataframe to extract feature names.
        logger: Logger instance.
        artifact_path_prefix: Optional path prefix for MLflow artifacts.
    """
    if not config.explainability.enabled:
        return

    logger.info("run explainability analysis")

    try:
        if config.env.feature_columns:
            feature_names = config.env.feature_columns
        else:
            feature_names = [
                col for col in train_df.columns if str(col).startswith("feature_")
            ]

        logger.debug("explainability n_features=%d features=%s", len(feature_names), feature_names)

        rollout = eval_ctx.env.rollout(max_steps=config.explainability.n_steps)
        obs_batch = rollout["observation"]
        logger.debug("explainability obs_batch n_rows=%d n_cols=%d", *obs_batch.shape)

        analyzer = RLInterpretabilityAnalyzer(trainer, feature_names)
        results = {}

        for method in config.explainability.methods:
            if method == ExplainabilityMethod.PERMUTATION:
                logger.info("compute permutation importance")
                df = analyzer.compute_global_importance(obs_batch)
                plot = analyzer.plot_importance(
                    df,
                    title="Global Feature Importance (Permutation)",
                    color="steelblue",
                )
                metrics = analyzer.quantify_interpretability(df)
                MLflowTrainingCallback.log_explainability_results(
                    df,
                    plot,
                    method=ExplainabilityMethod.PERMUTATION,
                    metrics=metrics,
                    artifact_path_prefix=artifact_path_prefix,
                )
                results[ExplainabilityMethod.PERMUTATION] = df
                logger.info("compute permutation importance complete")

            elif method == ExplainabilityMethod.INTEGRATED_GRADIENTS:
                logger.info("compute integrated gradients importance")
                df = analyzer.compute_global_ig(obs_batch)
                plot = analyzer.plot_importance(
                    df,
                    title="Global Feature Importance (Integrated Gradients)",
                    color="coral",
                )
                metrics = analyzer.quantify_interpretability(df)
                MLflowTrainingCallback.log_explainability_results(
                    df,
                    plot,
                    method=ExplainabilityMethod.INTEGRATED_GRADIENTS,
                    metrics=metrics,
                    artifact_path_prefix=artifact_path_prefix,
                )
                results[ExplainabilityMethod.INTEGRATED_GRADIENTS] = df
                logger.info("compute integrated gradients importance complete")

        if (
            ExplainabilityMethod.PERMUTATION in results
            and ExplainabilityMethod.INTEGRATED_GRADIENTS in results
        ):
            logger.info("create merged explainability plot")
            merged_plot = analyzer.plot_importance_merged(
                results[ExplainabilityMethod.PERMUTATION],
                results[ExplainabilityMethod.INTEGRATED_GRADIENTS],
            )
            MLflowTrainingCallback.log_explainability_results(
                None,
                merged_plot,
                method=ExplainabilityMethod.MERGED,
                metrics=None,
                artifact_path_prefix=artifact_path_prefix,
            )
            logger.info("save merged explainability plot")

    except Exception as e:
        logger.error("explainability analysis failed err=%s", e)
