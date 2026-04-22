"""Evaluation helpers extracted from the main training entrypoint."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from trading_rl.config import ExperimentConfig
from trading_rl.evaluation import EvaluationContext, periods_per_year_from_timeframe


@dataclass(frozen=True)
class SplitEvaluationResult:
    """Evaluation outputs for one data split."""

    final_reward: float
    last_positions: list[Any]
    evaluation_report: dict[str, float]


def build_evaluation_context_for_split(
    *,
    split: str,
    df: pd.DataFrame,
    config: ExperimentConfig,
    build_environment: Callable[[pd.DataFrame, ExperimentConfig], Any],
) -> EvaluationContext:
    eval_env = build_environment(df, config)
    eval_max_steps = min(config.training.eval_steps, len(df) - 1)
    return EvaluationContext(
        split=split,
        df=df,
        env=eval_env,
        max_steps=eval_max_steps,
    )


def compute_strategy_simple_returns_for_split(
    *,
    rollout: Any,
    split_ctx: EvaluationContext,
    config: ExperimentConfig,
) -> np.ndarray:
    from trading_rl.utils import _extract_tradingenv_returns

    reward_type = config.env.reward_type
    backend = config.env.backend

    if str(reward_type).lower() == "log_return":
        strategy_log_returns = (
            rollout["next", "reward"].detach().cpu().reshape(-1).numpy()[
                : split_ctx.max_steps
            ]
        )
        return np.exp(strategy_log_returns) - 1.0

    strategy_simple_returns = np.array([], dtype=float)
    if str(backend).lower() == "tradingenv":
        cumulative_log_returns = _extract_tradingenv_returns(
            split_ctx.env,
            split_ctx.max_steps,
        )
        if cumulative_log_returns is not None and len(cumulative_log_returns) > 0:
            cumulative_log_returns = np.asarray(cumulative_log_returns, dtype=float)
            step_log_returns = np.diff(cumulative_log_returns, prepend=0.0)
            strategy_simple_returns = np.exp(step_log_returns) - 1.0

    if strategy_simple_returns.size == 0:
        proxy_log_returns = (
            rollout["next", "reward"].detach().cpu().reshape(-1).numpy()[
                : split_ctx.max_steps
            ]
        )
        strategy_simple_returns = np.exp(proxy_log_returns) - 1.0

    return strategy_simple_returns


def resolve_price_series_for_split(
    split_ctx: EvaluationContext,
    config: ExperimentConfig,
    logger: logging.Logger,
) -> pd.Series | None:
    benchmark_price_column = config.env.price_column or "close"
    if benchmark_price_column in split_ctx.df.columns:
        return split_ctx.df[benchmark_price_column]
    if "close" in split_ctx.df.columns:
        return split_ctx.df["close"]

    logger.warning(
        "No price column found for %s buy-and-hold comparison",
        split_ctx.split,
    )
    return None


def run_statistical_tests_for_split(
    *,
    trainer: Any,
    split_ctx: EvaluationContext,
    config: ExperimentConfig,
    logger: logging.Logger,
    run_rollout: Callable[[Any, EvaluationContext], Any],
    run_all_statistical_tests_fn: Callable[..., Any],
    log_statistical_tests_fn: Callable[..., None],
) -> None:
    logger.info(
        "Running statistical significance tests for %s split...",
        split_ctx.split,
    )
    try:
        rollout = run_rollout(trainer, split_ctx)
        strategy_simple_returns = compute_strategy_simple_returns_for_split(
            rollout=rollout,
            split_ctx=split_ctx,
            config=config,
        )
        price_series = resolve_price_series_for_split(split_ctx, config, logger)
        periods_per_year = periods_per_year_from_timeframe(
            getattr(config.data, "timeframe", "1d")
        )
        statistical_test_results = run_all_statistical_tests_fn(
            strategy_returns=strategy_simple_returns,
            prices=price_series,
            env=split_ctx.env,
            max_steps=split_ctx.max_steps,
            config=config.statistical_testing,
            market_data=split_ctx.df,
            periods_per_year=periods_per_year,
        )

        log_statistical_tests_fn(
            statistical_test_results,
            split_prefix=split_ctx.split,
            log_to_research_artifacts=config.statistical_testing.log_to_research_artifacts,
            research_artifact_subdir=config.statistical_testing.research_artifact_subdir,
        )
        logger.info(
            "Statistical significance tests complete for %s split",
            split_ctx.split,
        )
    except Exception as error:
        logger.error(
            "Failed to run statistical tests for %s split: %s",
            split_ctx.split,
            error,
        )


def evaluate_split(
    *,
    split: str,
    split_df: pd.DataFrame,
    trainer: Any,
    config: ExperimentConfig,
    algorithm: str,
    logs: dict[str, Any],
    logger: logging.Logger,
    build_evaluation_context_fn: Callable[..., EvaluationContext],
    log_evaluation_plots_fn: Callable[..., None],
    build_evaluation_report_for_trainer_fn: Callable[..., dict[str, float]],
    log_evaluation_report_fn: Callable[..., None],
    run_statistical_tests_for_split_fn: Callable[..., None],
) -> SplitEvaluationResult | None:
    if len(split_df) < 2:
        logger.warning(
            "Skipping %s split evaluation: insufficient data (%d rows)",
            split,
            len(split_df),
        )
        return None

    logger.info("Evaluating agent on %s split (%d rows)...", split, len(split_df))
    split_ctx = build_evaluation_context_fn(split=split, df=split_df, config=config)

    (
        reward_plot,
        action_plot,
        action_probs_plot,
        split_final_reward,
        split_last_positions,
        actual_returns_plot,
        merged_plot,
    ) = trainer.evaluate(
        split_ctx.df,
        max_steps=split_ctx.max_steps,
        config=config,
        algorithm=algorithm,
        eval_env=split_ctx.env,
    )

    log_evaluation_plots_fn(
        reward_plot=reward_plot,
        action_plot=action_plot,
        action_probs_plot=action_probs_plot,
        actual_returns_plot=actual_returns_plot,
        logs=logs,
        merged_plot=merged_plot,
        artifact_path_prefix=f"evaluation_plots/{split}",
    )

    split_evaluation_report = build_evaluation_report_for_trainer_fn(
        trainer=trainer,
        df_prices=split_ctx.df,
        max_steps=split_ctx.max_steps,
        config=config,
        eval_env=split_ctx.env,
    )
    log_evaluation_report_fn(split_evaluation_report, split_prefix=split)

    if config.statistical_testing.enabled:
        run_statistical_tests_for_split_fn(
            trainer=trainer,
            split_ctx=split_ctx,
            config=config,
            logger=logger,
        )

    return SplitEvaluationResult(
        final_reward=split_final_reward,
        last_positions=split_last_positions,
        evaluation_report=split_evaluation_report,
    )


def evaluate_all_splits(
    *,
    trainer: Any,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: ExperimentConfig,
    algorithm: str,
    logs: dict[str, Any],
    logger: logging.Logger,
    evaluate_split_fn: Callable[..., SplitEvaluationResult | None],
) -> dict[str, dict[str, Any]]:
    split_frames = {"train": train_df, "val": val_df, "test": test_df}
    split_results: dict[str, dict[str, Any]] = {}

    for split, split_df in split_frames.items():
        result = evaluate_split_fn(
            split=split,
            split_df=split_df,
            trainer=trainer,
            config=config,
            algorithm=algorithm,
            logs=logs,
            logger=logger,
        )
        if result is None:
            continue
        split_results[split] = {
            "final_reward": result.final_reward,
            "last_positions": result.last_positions,
            "evaluation_report": result.evaluation_report,
        }

    return split_results


def resolve_primary_split_result(
    split_results: dict[str, dict[str, Any]],
) -> tuple[str | None, float, list[Any], dict[str, float]]:
    primary_split = next(
        (split for split in ("test", "val", "train") if split in split_results),
        None,
    )
    final_reward = (
        split_results[primary_split]["final_reward"]
        if primary_split
        else float("nan")
    )
    last_positions = (
        split_results[primary_split]["last_positions"] if primary_split else []
    )
    evaluation_report = (
        split_results[primary_split]["evaluation_report"] if primary_split else {}
    )
    return primary_split, final_reward, last_positions, evaluation_report


def run_primary_split_explainability(
    *,
    primary_split: str | None,
    trainer: Any,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: ExperimentConfig,
    logger: logging.Logger,
    build_evaluation_context_fn: Callable[..., EvaluationContext],
    run_explainability_analysis_fn: Callable[..., None],
) -> None:
    if not primary_split:
        return

    split_frames = {"train": train_df, "val": val_df, "test": test_df}
    explainability_ctx = build_evaluation_context_fn(
        split=primary_split,
        df=split_frames[primary_split],
        config=config,
    )
    run_explainability_analysis_fn(
        config=config,
        trainer=trainer,
        eval_ctx=explainability_ctx,
        train_df=train_df,
        logger=logger,
        artifact_path_prefix=f"explainability/{primary_split}",
    )


def build_final_metrics(
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
    is_portfolio_backend = config.env.backend == "tradingenv"
    return {
        "final_reward": final_reward,
        "training_steps": len(logs.get("loss_value", [])),
        "interrupted": interrupted,
        (
            "portfolio_weights" if is_portfolio_backend else "last_position_per_episode"
        ): last_positions,
        "data_start_date": str(train_df.index[0]) if not train_df.empty else "unknown",
        "data_end_date": (
            str(test_df.index[-1])
            if not test_df.empty
            else (str(train_df.index[-1]) if not train_df.empty else "unknown")
        ),
        "data_size_total": len(train_df) + len(val_df) + len(test_df),
        "train_size": len(train_df),
        "validation_size": len(val_df),
        "test_size": len(test_df),
        "trading_fees": config.env.trading_fees,
        "borrow_interest_rate": config.env.borrow_interest_rate,
        "positions": str(config.env.positions),
        "actor_hidden_dims": config.network.actor_hidden_dims,
        "value_hidden_dims": config.network.value_hidden_dims,
        "n_observations": n_obs,
        "n_actions": n_act,
        "experiment_name": effective_experiment_name,
        "evaluation_split": primary_split or "none",
        "seed": config.seed,
        "actor_lr": config.training.actor_lr,
        "value_lr": config.training.value_lr,
        "buffer_size": config.training.buffer_size,
        "evaluation_report": evaluation_report,
        "split_results": split_results,
    }
