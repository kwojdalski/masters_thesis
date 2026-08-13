"""Evaluation helpers for the main RL training pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from logger import log_banner
from trading_rl.callbacks import ArtifactPaths, MLflowTrainingCallback
from trading_rl.config import ExperimentConfig
from trading_rl.constants import EnvBackend, EvalSymbolSelection, RewardType, SplitName
from trading_rl.envs import AlgorithmicEnvironmentBuilder, EnvBuildParams
from trading_rl.evaluation import (
    EvaluationContext,
    MetricReport,
    build_evaluation_report_for_trainer,
    compute_random_baseline_returns,
    compute_random_returns_from_prices,
    periods_per_year_from_timeframe,
    run_all_statistical_tests,
)
from trading_rl.evaluation.benchmark_table import (
    build_bench_out,
    save_benchmark_table_artifact,
)
from trading_rl.evaluation.benchmarks import BenchmarkEngine
from trading_rl.evaluation.evaluator import StrategyEvaluatorConfig
from trading_rl.evaluation.report import _periods_per_year_from_index
from trading_rl.evaluation.returns import extract_tradingenv_return_series
from trading_rl.pipeline.explainability import run_explainability_analysis
from trading_rl.profiler import get_profiler


@dataclass(frozen=True)
class PipelineSplitResult:
    """Evaluation outputs for one data split."""

    final_reward: float
    last_positions: list[Any]
    evaluation_report: MetricReport


def build_evaluation_context_for_split(
    *,
    split: str,
    df: pd.DataFrame,
    config: ExperimentConfig,
) -> EvaluationContext:
    """Build an evaluation context for a given data split.

    Args:
        split: Split name ("train", "val", or "test").
        df: DataFrame for this split.
        config: Experiment configuration.
    """
    eval_env = AlgorithmicEnvironmentBuilder().create(
        df, EnvBuildParams.from_config(config), use_memmap=False
    )
    eval_max_steps = min(config.evaluation.resolve_eval_steps(len(df)), len(df) - 1)
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
    from trading_rl.evaluation.returns import (
        RewardSeries,
        extract_tradingenv_return_series,
    )

    reward_type = config.env.reward_type
    backend = config.env.backend

    if reward_type == RewardType.LOG_RETURN:
        strategy_log_returns = (
            rollout["next", "reward"]
            .detach()
            .cpu()
            .reshape(-1)
            .numpy()[: split_ctx.max_steps]
        )
        return (
            RewardSeries(
                strategy_log_returns,
                RewardType.LOG_RETURN,
            )
            .to_return_series()
            .to_simple()
            .values
        )

    strategy_simple_returns = np.array([], dtype=float)
    if str(backend).lower() == EnvBackend.TRADINGENV:
        return_series = extract_tradingenv_return_series(
            split_ctx.env,
            split_ctx.max_steps,
        )
        if return_series is not None:
            strategy_simple_returns = return_series.to_simple().values

    return strategy_simple_returns


def _make_rollout_progress_callback(max_steps: int) -> Any:
    """Return a tqdm-backed callback for env.rollout(), or None if tqdm is absent."""
    try:
        from tqdm import tqdm
    except ImportError:
        return None, None

    bar = tqdm(
        total=max_steps,
        desc="rollout",
        unit="step",
        dynamic_ncols=True,
        leave=False,
    )

    def _callback(_td: Any) -> None:
        bar.update(1)

    return bar, _callback


def _run_rollout(trainer: Any, split_ctx: EvaluationContext) -> Any:
    """Execute a deterministic rollout, falling back from MODE to DETERMINISTIC."""
    from tensordict.nn import InteractionType
    from torchrl.envs.utils import set_exploration_type

    bar, callback = _make_rollout_progress_callback(split_ctx.max_steps)
    try:
        with torch.no_grad():
            try:
                with set_exploration_type(InteractionType.MODE):
                    return split_ctx.env.rollout(
                        max_steps=split_ctx.max_steps,
                        policy=trainer.actor,
                        callback=callback,
                    )
            except (NotImplementedError, RuntimeError) as exc:
                if not (
                    isinstance(exc, NotImplementedError)
                    or "does not have a mode" in str(exc)
                    or "analytical mode" in str(exc).lower()
                ):
                    raise
                with set_exploration_type(InteractionType.DETERMINISTIC):
                    return split_ctx.env.rollout(
                        max_steps=split_ctx.max_steps,
                        policy=trainer.actor,
                        callback=callback,
                    )
    finally:
        if bar is not None:
            bar.close()


def _evaluation_strategy_returns(result: Any | None) -> np.ndarray | None:
    simple_returns = getattr(result, "simple_returns", None)
    if simple_returns is None:
        return None
    arr = np.asarray(simple_returns, dtype=float)
    # Return None for empty arrays so build_evaluation_report_for_trainer falls
    # through to TradingEnv broker extraction instead of using an empty array
    # that would produce all-NaN metrics.
    return arr if arr.size > 0 else None


def _evaluation_rollout(result: Any | None) -> Any | None:
    return getattr(result, "rollout", None)


def run_statistical_tests_for_split(
    *,
    trainer: Any,
    split_ctx: EvaluationContext,
    config: ExperimentConfig,
    logger: Any,
    strategy_simple_returns: np.ndarray | None = None,
) -> None:
    logger.info("run statistical significance tests split={}", split_ctx.split)
    try:
        if strategy_simple_returns is None:
            rollout = _run_rollout(trainer, split_ctx)
            strategy_simple_returns = compute_strategy_simple_returns_for_split(
                rollout=rollout,
                split_ctx=split_ctx,
                config=config,
            )
        else:
            strategy_simple_returns = np.asarray(strategy_simple_returns, dtype=float)

        price_column = config.env.price_column or "close"
        benchmarks, _bench_meta = BenchmarkEngine.build(
            split_ctx.df, config.benchmarks, price_column
        )

        random_trials = None
        if config.benchmarks.is_random:
            logger.info(
                "compute random baseline n_trials={}", config.benchmarks.n_random_trials
            )
            random_trials = compute_random_baseline_returns(
                split_ctx.env,
                split_ctx.max_steps,
                n_trials=config.benchmarks.n_random_trials,
                seed=config.benchmarks.random_seed,
                reward_type=config.env.reward_type,
            )

        periods_per_year = _periods_per_year_from_index(
            split_ctx.df
        ) or periods_per_year_from_timeframe(getattr(config.data, "timeframe", "1d"))
        statistical_test_results = run_all_statistical_tests(
            strategy_returns=strategy_simple_returns,
            benchmarks=benchmarks,
            max_steps=split_ctx.max_steps,
            config=config.statistical_testing,
            random_baseline_trials=random_trials,
            periods_per_year=periods_per_year,
        )
        MLflowTrainingCallback.log_statistical_tests(
            statistical_test_results,
            split_prefix=split_ctx.split,
            log_to_research_artifacts=config.statistical_testing.log_to_research_artifacts,
            research_artifact_subdir=config.statistical_testing.research_artifact_subdir,
        )
        logger.info("statistical significance tests complete split={}", split_ctx.split)
    except Exception:
        logger.opt(exception=True).error(
            "statistical tests failed split={}", split_ctx.split
        )
        try:
            import mlflow

            mlflow.log_metric(f"{split_ctx.split}_statistical_tests_failed", 1)
        except Exception:
            logger.debug(
                "mlflow.log_metric for statistical_tests_failed also failed split={}",
                split_ctx.split,
            )


def _save_benchmark_table_for_split(
    *,
    split: str,
    split_ctx: EvaluationContext,
    config: ExperimentConfig,
    evaluation_report: MetricReport,
    logger: Any,
    strategy_simple_returns: np.ndarray | None = None,
) -> None:
    """Build benchmarks and save the benchmark comparison table artifact."""
    price_column = getattr(config.env, "price_column", None) or "close"
    periods_per_year = _periods_per_year_from_index(
        split_ctx.df
    ) or periods_per_year_from_timeframe(getattr(config.data, "timeframe", "1d"))
    benchmarks, _ = BenchmarkEngine.build(split_ctx.df, config.benchmarks, price_column)

    if config.benchmarks.is_random:
        try:
            logger.info(
                "benchmark table: computing random baseline n_trials={} split={}",
                config.benchmarks.n_random_trials,
                split,
            )
            prices = split_ctx.df[price_column]
            random_trials = compute_random_returns_from_prices(
                prices,
                split_ctx.max_steps,
                n_trials=config.benchmarks.n_random_trials,
                seed=config.benchmarks.random_seed,
            )
            if random_trials:
                min_len = min(len(t) for t in random_trials)
                mean_returns = np.mean([t[:min_len] for t in random_trials], axis=0)
                benchmarks.append(BenchmarkEngine.random_actions(mean_returns))
        except Exception as _rand_err:
            logger.warning(
                "benchmark table: random baseline failed split={} err={}",
                split,
                _rand_err,
            )

    if not benchmarks:
        return

    if strategy_simple_returns is None:
        return_series = extract_tradingenv_return_series(
            split_ctx.env, split_ctx.max_steps
        )
        if return_series is None:
            logger.warning(
                "benchmark table: no strategy returns available for split={}", split
            )
            return
        strategy_simple_returns = return_series.to_simple().values
    else:
        strategy_simple_returns = np.asarray(strategy_simple_returns, dtype=float)

    if strategy_simple_returns.size == 0:
        logger.warning("benchmark table: empty strategy returns for split={}", split)
        return

    bench_out = build_bench_out(
        strategy_simple_returns=strategy_simple_returns,
        benchmarks=benchmarks,
        max_steps=split_ctx.max_steps,
        periods_per_year=periods_per_year,
    )

    output_dir = Path(config.logging.log_dir) / "benchmark_tables"
    save_benchmark_table_artifact(
        split=split,
        split_df=split_ctx.df,
        bench_out=bench_out,
        strategy_metrics=evaluation_report.to_dict(),
        output_dir=output_dir,
    )
    logger.info("benchmark table saved split={} dir={}", split, output_dir)


def evaluate_split(
    *,
    split: str,
    split_df: pd.DataFrame,
    trainer: Any,
    config: ExperimentConfig,
    algorithm: str,
    logs: dict[str, Any],
    logger: Any,
) -> PipelineSplitResult | None:
    if len(split_df) < 2:
        logger.warning(
            "Skipping {} split evaluation: insufficient data ({} rows)",
            split,
            len(split_df),
        )
        return None

    profiler = get_profiler()
    log_banner(
        logger, f"EVALUATION START  split={split.upper()}  rows={len(split_df):,}"
    )

    with profiler.stage(f"eval_env_build_{split}"):
        split_ctx = build_evaluation_context_for_split(
            split=split, df=split_df, config=config
        )

    try:
        with profiler.stage(f"eval_observation_sample_{split}"):
            sample_path = MLflowTrainingCallback.save_observation_sample_artifact(
                split=split,
                df=split_ctx.df,
                output_dir=Path(config.logging.log_dir) / ArtifactPaths.EVAL_DATA,
            )
            logger.info("observation sample saved split={} path={}", split, sample_path)
    except Exception as sample_error:
        logger.warning(
            "observation sample artifact failed split={} err={}", split, sample_error
        )

    with profiler.stage(f"eval_rollout_{split}"):
        eval_output = trainer.evaluate(
            split_ctx.df,
            max_steps=split_ctx.max_steps,
            config=config,
            algorithm=algorithm,
            eval_env=split_ctx.env,
        )
        (
            reward_plot,
            action_plot,
            action_probs_plot,
            split_final_reward,
            split_last_positions,
            equity_curve_plot,
            merged_plot,
        ) = eval_output
    last_eval_result = getattr(eval_output, "result", None)
    strategy_simple_returns = _evaluation_strategy_returns(last_eval_result)
    rollout = _evaluation_rollout(last_eval_result)

    if config.evaluation.log_data and last_eval_result is not None:
        try:
            from trading_rl.callbacks.artifacts import save_eval_rollout_artifact

            save_eval_rollout_artifact(
                split=split,
                last_positions=getattr(last_eval_result, "last_positions", []),
                simple_returns=getattr(
                    last_eval_result, "simple_returns", np.array([])
                ),
                cumulative_returns=getattr(
                    last_eval_result, "cumulative_returns", None
                ),
                df_index=split_ctx.df.index,
                output_dir=Path(config.logging.log_dir) / ArtifactPaths.EVAL_DATA,
                artifact_path_prefix=ArtifactPaths.eval_data(split),
            )
        except Exception as _rd_err:
            logger.warning(
                "rollout data artifact failed split={} err={}", split, _rd_err
            )

    # Merge rollout and equity plot DataFrames so the QMD can re-render at any
    # figure size without re-running the rollout.
    _eval_plots_dict = getattr(last_eval_result, "plots", None) or {}
    _plot_data: dict = {
        **((_eval_plots_dict.get("_rollout_plot_data")) or {}),
        **((_eval_plots_dict.get("_equity_plot_data")) or {}),
    }

    with profiler.stage(f"eval_mlflow_plots_{split}"):
        MLflowTrainingCallback.log_evaluation_plots(
            reward_plot=reward_plot,
            action_plot=action_plot,
            action_probs_plot=action_probs_plot,
            equity_curve_plot=equity_curve_plot,
            logs=logs,
            merged_plot=merged_plot,
            artifact_path_prefix=ArtifactPaths.eval_plots(split),
            debug=getattr(config.logging, "debug_plots", False),
            plot_data=_plot_data or None,
        )

    with profiler.stage(f"eval_report_{split}"):
        split_evaluation_report = build_evaluation_report_for_trainer(
            trainer=trainer,
            df_prices=split_ctx.df,
            max_steps=split_ctx.max_steps,
            config=config,
            eval_env=split_ctx.env,
            rollout=rollout,
            strategy_simple_returns=strategy_simple_returns,
        )
        MLflowTrainingCallback.log_evaluation_report(
            split_evaluation_report, split_prefix=split
        )

    try:
        import os
        import tempfile

        import mlflow as _mlflow

        from trading_rl.evaluation.plots import create_metrics_table_figure

        if _mlflow.active_run():
            fig = create_metrics_table_figure(
                split_evaluation_report,
                split=split,
                df=split_ctx.df,
                max_steps=split_ctx.max_steps,
            )
            with tempfile.TemporaryDirectory() as _td:
                _tbl_path = os.path.join(_td, "metrics_table.png")
                fig.savefig(_tbl_path, dpi=150, bbox_inches="tight")
                import matplotlib.pyplot as plt

                plt.close(fig)
                _mlflow.log_artifact(_tbl_path, ArtifactPaths.eval_plots(split))
    except Exception as _mt_err:
        logger.warning("metrics table artifact failed split={} err={}", split, _mt_err)

    try:
        with profiler.stage(f"eval_benchmark_table_{split}"):
            _save_benchmark_table_for_split(
                split=split,
                split_ctx=split_ctx,
                config=config,
                evaluation_report=split_evaluation_report,
                logger=logger,
                strategy_simple_returns=strategy_simple_returns,
            )
    except Exception as _bt_err:
        logger.warning(
            "benchmark table artifact failed split={} err={}", split, _bt_err
        )

    if config.statistical_testing.enabled:
        try:
            with profiler.stage(f"eval_statistical_tests_{split}"):
                run_statistical_tests_for_split(
                    trainer=trainer,
                    split_ctx=split_ctx,
                    config=config,
                    logger=logger,
                    strategy_simple_returns=strategy_simple_returns,
                )
        except KeyboardInterrupt:
            logger.warning(
                "Statistical tests interrupted for {} split. "
                "Returning main evaluation results without statistical tests...",
                split,
            )

    log_banner(
        logger,
        f"EVALUATION END  split={split.upper()}  reward={split_final_reward:.4f}",
    )
    try:
        return PipelineSplitResult(
            final_reward=split_final_reward,
            last_positions=split_last_positions,
            evaluation_report=split_evaluation_report,
        )
    finally:
        try:
            split_ctx.env.close()
        except Exception:
            logger.opt(exception=True).debug("eval env close failed split={}", split)


def evaluate_per_symbol(
    *,
    trainer: Any,
    config: ExperimentConfig,
    feature_pipeline_state: dict[str, Any] | None,
    algorithm: str,
    logs: dict[str, Any],
    logger: Any,
) -> dict[str, dict[str, Any]]:
    """Evaluate the pooled-trained model on each symbol independently.

    Loads each file in ``config.data.val_data_paths``, applies the already-fitted
    pooled pipeline (state restored from ``feature_pipeline_state``), and calls
    ``evaluate_split`` with a ``val__<symbol>`` split key per symbol.

    Args:
        trainer: Trained model.
        config: Experiment configuration.
        feature_pipeline_state: Scaler state saved during training (pooled fit).
        algorithm: Algorithm label string.
        logs: Mutable logs dict passed through to evaluate_split.
        logger: Logger instance.

    Returns:
        Dict mapping ``"val__<symbol>"`` to the per-symbol split result dicts.
    """
    val_data_paths = getattr(config.data, "val_data_paths", None) or []
    if not val_data_paths:
        logger.warning("per_symbol_eval=True but val_data_paths is empty — skipping")
        return {}

    feature_config = getattr(config.data, "feature_config", None)
    filter_lob_levels = getattr(config.data, "filter_lob_levels", None)
    mode = str(getattr(config.env, "mode", "")).lower().strip()

    pipeline = None
    pipeline_state_restored = False
    if feature_config:
        try:
            from trading_rl.data.loading import build_feature_pipeline_with_state

            restore_result = build_feature_pipeline_with_state(
                feature_config,
                feature_pipeline_state=feature_pipeline_state,
            )
            pipeline = restore_result.pipeline
            pipeline_state_restored = restore_result.restored
            if restore_result.restored:
                logger.info(
                    "per-symbol eval: restored pooled pipeline state n_features={}",
                    restore_result.state_size,
                )
            else:
                logger.warning(
                    "per-symbol eval: no pipeline state available — all symbols will be skipped "
                    "to avoid fitting normalization stats on validation data"
                )
        except Exception:
            logger.opt(exception=True).error(
                "per-symbol eval: failed to build/restore feature pipeline — skipping"
            )
            return {}

    results: dict[str, dict[str, Any]] = {}
    for raw_path in val_data_paths:
        data_path = Path(raw_path)
        symbol = data_path.parent.name
        split_key = f"val__{symbol}"
        logger.info("per-symbol eval symbol={} path={}", symbol, data_path)
        try:
            df = pd.read_parquet(data_path).dropna()

            if filter_lob_levels is not None:
                from trading_rl.data.lob_filters import filter_unchanged_lob

                df = filter_unchanged_lob(df, levels=filter_lob_levels)

            if pipeline is not None:
                if not pipeline_state_restored:
                    raise RuntimeError(
                        "pooled feature pipeline state unavailable — refusing to "
                        "fit a fresh pipeline on validation data (would leak "
                        "eval-set statistics into GLOBAL-normalized features)"
                    )
                features = pipeline.transform(df)
                df = pd.concat([df, features], axis=1)

            if mode == "hft":
                from trading_rl.data.hft import _derive_close_hft_single

                df = _derive_close_hft_single(df, data_path.stem, logger)

            result = evaluate_split(
                split=split_key,
                split_df=df,
                trainer=trainer,
                config=config,
                algorithm=algorithm,
                logs=logs,
                logger=logger,
            )
            if result is not None:
                results[split_key] = {
                    "final_reward": result.final_reward,
                    "last_positions": result.last_positions,
                    "evaluation_report": result.evaluation_report,
                }
        except Exception:
            logger.opt(exception=True).error("per-symbol eval failed symbol={}", symbol)

    return results


def evaluate_all_splits(
    *,
    trainer: Any,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: ExperimentConfig,
    algorithm: str,
    logs: dict[str, Any],
    logger: Any,
) -> dict[str, dict[str, Any]]:
    # Project ExperimentConfig to the narrow evaluation config at this boundary.
    # Deeper call chains still receive the full config until those layers are similarly narrowed.
    _eval_config = StrategyEvaluatorConfig.from_experiment_config(config)
    logger.debug(
        "evaluate_all_splits eval_config reward_type={} backend={} periods_per_year={}",
        _eval_config.reward_type,
        _eval_config.backend,
        _eval_config.periods_per_year,
    )

    split_frames = {
        SplitName.TRAIN: train_df,
        SplitName.VAL: val_df,
        SplitName.TEST: test_df,
    }
    split_results: dict[str, dict[str, Any]] = {}

    for split, split_df in split_frames.items():
        result = evaluate_split(
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
) -> tuple[str | None, float, list[Any], MetricReport]:
    primary_split = next(
        (
            split
            for split in (SplitName.TEST, SplitName.VAL, SplitName.TRAIN)
            if split in split_results
        ),
        None,
    )
    final_reward = (
        split_results[primary_split]["final_reward"] if primary_split else float("nan")
    )
    last_positions = (
        split_results[primary_split]["last_positions"] if primary_split else []
    )
    evaluation_report = (
        split_results[primary_split]["evaluation_report"]
        if primary_split
        else MetricReport.all_nan()
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
    logger: Any,
) -> None:
    if not primary_split:
        return

    split_frames = {
        SplitName.TRAIN: train_df,
        SplitName.VAL: val_df,
        SplitName.TEST: test_df,
    }
    explainability_ctx = build_evaluation_context_for_split(
        split=primary_split,
        df=split_frames[primary_split],
        config=config,
    )
    try:
        run_explainability_analysis(
            config=config,
            trainer=trainer,
            eval_ctx=explainability_ctx,
            train_df=train_df,
            logger=logger,
            artifact_path_prefix=ArtifactPaths.explainability(primary_split),
        )
    finally:
        try:
            explainability_ctx.env.close()
        except Exception:
            logger.opt(exception=True).debug(
                "explainability env close failed split={}", primary_split
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
    evaluation_report: MetricReport,
    split_results: dict[str, dict[str, Any]],
    total_env_steps: int = 0,
    total_episodes: int = 0,
) -> dict[str, Any]:
    is_portfolio_backend = config.env.backend == EnvBackend.TRADINGENV
    duration_entries = logs.get("training_duration_s", [])
    training_duration_s = duration_entries[-1] if duration_entries else None

    if config.data.data_paths:
        unique_symbols = sorted({Path(p).parent.name for p in config.data.data_paths})
    else:
        unique_symbols = list(config.data.symbols) if config.data.symbols else []

    val_data_paths = getattr(config.data, "val_data_paths", None) or []
    memmap_dir_cfg = getattr(config.data, "memmap_dir", None)
    if val_data_paths and memmap_dir_cfg:
        symbol_file = Path(memmap_dir_cfg) / ".eval_symbol_used"
        if symbol_file.exists():
            val_test_symbols = [symbol_file.read_text().strip()]
        else:
            strategy = getattr(
                config.data, "eval_symbol_selection", EvalSymbolSelection.FIRST
            )
            if strategy == EvalSymbolSelection.FIRST:
                val_test_symbols = [Path(val_data_paths[0]).parent.name]
            elif strategy == EvalSymbolSelection.ROTATED:
                counter_path = Path(memmap_dir_cfg) / ".eval_symbol_counter"
                try:
                    count = (
                        int(counter_path.read_text().strip())
                        if counter_path.exists()
                        else 1
                    )
                    idx = (count - 1) % len(val_data_paths)
                    val_test_symbols = [Path(val_data_paths[idx]).parent.name]
                except (ValueError, ZeroDivisionError):
                    val_test_symbols = sorted(
                        {Path(p).parent.name for p in val_data_paths}
                    )
            else:
                val_test_symbols = sorted({Path(p).parent.name for p in val_data_paths})
    elif val_data_paths:
        val_test_symbols = sorted({Path(p).parent.name for p in val_data_paths})
    else:
        val_test_symbols = unique_symbols

    def _fmt_ts(ts: Any) -> str:
        if hasattr(ts, "strftime"):
            return ts.strftime("%Y-%m-%d %H:%M:%S UTC")
        return str(ts)[:19]

    split_frames = {
        SplitName.TRAIN: train_df,
        SplitName.VAL: val_df,
        SplitName.TEST: test_df,
    }
    split_symbols = {
        SplitName.TRAIN: unique_symbols,
        SplitName.VAL: val_test_symbols,
        SplitName.TEST: val_test_symbols,
    }
    enriched_split_results: dict[str, dict[str, Any]] = {}
    for split, result in split_results.items():
        df = split_frames.get(split)
        entry = dict(result)
        if df is not None and not df.empty:
            entry["date_start"] = _fmt_ts(df.index[0])
            entry["date_end"] = _fmt_ts(df.index[-1])
        entry["symbols"] = split_symbols.get(split, unique_symbols)
        enriched_split_results[split] = entry

    return {
        "final_reward": final_reward,
        "optimizer_steps": len(logs.get("loss_value", [])),
        "total_env_steps": total_env_steps,
        "total_episodes": int(total_episodes),
        "training_duration_s": training_duration_s,
        "eval_steps": config.evaluation.resolve_eval_steps(len(val_df)),
        "episode_length": getattr(
            config.env,
            "streaming_episode_length",
            len(train_df) if not train_df.empty else 0,
        ),
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
        "unique_symbols": unique_symbols,
        "experiment_name": effective_experiment_name,
        "evaluation_split": primary_split or "none",
        "seed": config.seed,
        "actor_lr": config.training.actor_lr,
        "value_lr": config.training.value_lr,
        "buffer_size": config.training.buffer_size,
        "evaluation_report": evaluation_report,
        "split_results": enriched_split_results,
    }
