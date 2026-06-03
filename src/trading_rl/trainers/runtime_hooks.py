"""Runtime hook orchestration for periodic trainer side effects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mlflow

from logger import get_logger
from trading_rl.callbacks.artifacts import ArtifactPaths

logger = get_logger(__name__)


@dataclass
class SplitEvalContext:
    """One split's data and environment for periodic evaluation."""

    split: str
    df: Any
    max_steps: int
    eval_env: Any = None


@dataclass
class PeriodicEvaluationHook:
    """Periodic evaluation configuration."""

    splits: list[SplitEvalContext]
    config: Any
    algorithm: str
    last_step: int = 0
    effective_interval: int = 0             # current eval cadence (may be shortened in dense mode)
    dense_reward_history: dict = field(default_factory=dict)  # split → [(step, reward)]


@dataclass
class PeriodicExplainabilityHook:
    """Periodic explainability configuration."""

    df: Any
    max_steps: int
    config: Any
    eval_env: Any = None
    last_step: int = 0


class TrainerRuntimeHooks:
    """Manage optional periodic work triggered from the training loop."""

    _MAX_HOOK_FAILURES = 3

    def __init__(self, trainer: Any):
        self.trainer = trainer
        self._evaluation: PeriodicEvaluationHook | None = None
        self._explainability: PeriodicExplainabilityHook | None = None
        self._eval_consecutive_failures = 0
        self._explainability_consecutive_failures = 0
        self._progression_history: dict[str, list[tuple[int, Any]]] = {}
        self._price_plots_logged: set[str] = set()

    def configure_periodic_evaluation(
        self,
        *,
        splits: list[SplitEvalContext],
        config: Any,
        algorithm: str,
    ) -> None:
        """Enable or disable temporary periodic evaluation.

        Args:
            splits: One SplitEvalContext per split to evaluate (e.g. train, val).
            config: Full experiment config.
            algorithm: Algorithm label used in plot titles.
        """
        temp_eval_cfg = config.training.temp_eval
        eval_interval = temp_eval_cfg.interval
        if eval_interval is None or eval_interval <= 0:
            logger.info("periodic evaluation disabled (temp_eval.interval not set)")
            self._evaluation = None
            return

        self._evaluation = PeriodicEvaluationHook(
            splits=splits,
            config=config,
            algorithm=algorithm,
            effective_interval=eval_interval,
        )
        split_names = [s.split for s in splits]
        logger.info(
            "periodic evaluation enabled interval={} splits={} dense_eval={}",
            eval_interval, split_names, temp_eval_cfg.dense_enabled,
        )

    def configure_periodic_explainability(
        self,
        *,
        df: Any,
        max_steps: int,
        config: Any,
        eval_env: Any = None,
    ) -> None:
        """Enable or disable temporary periodic explainability."""
        explainability_interval = getattr(
            config.explainability, "temp_explainability_interval", None
        )
        if explainability_interval is None or explainability_interval <= 0:
            logger.info("periodic explainability disabled (temp_explainability_interval not set)")
            self._explainability = None
            return
        if not config.explainability.enabled:
            logger.info("periodic explainability disabled (explainability.enabled is False)")
            self._explainability = None
            return

        self._explainability = PeriodicExplainabilityHook(
            df=df,
            max_steps=max_steps,
            config=config,
            eval_env=eval_env,
        )
        logger.info(
            "Periodic explainability enabled: will analyze every {} training steps",
            explainability_interval,
        )

    def maybe_run(self, step_number: int) -> None:
        """Run any configured hooks whose cadence has been reached."""
        self._maybe_run_evaluation(step_number)
        self._maybe_run_explainability(step_number)

    def _maybe_run_evaluation(self, step_number: int) -> None:
        hook = self._evaluation
        if hook is None:
            return

        if step_number - hook.last_step < hook.effective_interval:
            return

        self._run_temporary_evaluation(step_number, hook)
        hook.last_step = step_number

    def _maybe_run_explainability(self, step_number: int) -> None:
        hook = self._explainability
        if hook is None:
            return

        explainability_interval = (
            hook.config.explainability.temp_explainability_interval
        )
        if step_number - hook.last_step < explainability_interval:
            return

        self._run_temporary_explainability(step_number, hook)
        hook.last_step = step_number

    def _log_price_plot_once(self, split_ctx: "SplitEvalContext", hook: "PeriodicEvaluationHook") -> None:
        """Save the underlying close price plot to evaluation_plots_temp/{split}/ once."""
        import os
        import tempfile

        import mlflow

        if not mlflow.active_run():
            return
        try:
            from trading_rl.callbacks.artifacts import ArtifactPaths
            from trading_rl.evaluation.plots import create_price_plot
            from trading_rl.evaluation.thesis_theme import (
                FIGURE_HEIGHT,
                FIGURE_WIDTH,
                PLOT_DPI,
                save_plot as _save_plot,
            )

            price_column = getattr(getattr(hook.config, "env", None), "price_column", None) or "close"
            plot = create_price_plot(split_ctx.df, price_column=price_column)
            if plot is None:
                logger.warning(
                    "price plot: column {} not found in split={} df (columns={})",
                    price_column, split_ctx.split, list(split_ctx.df.columns)[:10],
                )
                return
            artifact_dir = f"{ArtifactPaths.EVAL_PLOTS_TEMP}/{split_ctx.split}"
            with tempfile.TemporaryDirectory() as tmpdir:
                path = os.path.join(tmpdir, "close_price.png")
                _save_plot(plot, path, width=FIGURE_WIDTH * 2, height=FIGURE_HEIGHT * 1.5, dpi=PLOT_DPI)
                mlflow.log_artifact(path, artifact_dir)
            logger.info("price plot logged split={} column={}", split_ctx.split, price_column)
        except Exception:
            logger.opt(exception=True).warning("price plot failed split={}", split_ctx.split)

    def _run_temporary_evaluation(
        self,
        step_number: int,
        hook: PeriodicEvaluationHook,
    ) -> None:
        """Run evaluation during training and log artifacts without affecting control flow."""
        import math
        import time
        _t_total = time.monotonic()
        split_names = [s.split for s in hook.splits]
        logger.info(
            "run temporary evaluation step={} splits={}",
            step_number, split_names,
        )

        for split_ctx in hook.splits:
            _t_split = time.monotonic()
            logger.info(
                "temp eval split={} max_steps={} df_rows={}",
                split_ctx.split, split_ctx.max_steps, len(split_ctx.df),
            )
            try:
                _t = time.monotonic()
                eval_output = self.trainer.evaluate(
                    df=split_ctx.df,
                    max_steps=split_ctx.max_steps,
                    config=hook.config,
                    algorithm=hook.algorithm,
                    eval_env=split_ctx.eval_env,
                )
                (
                    reward_plot,
                    action_plot,
                    action_probs_plot,
                    final_reward,
                    _last_positions,
                    equity_curve_plot,
                    merged_plot,
                ) = eval_output
                last_eval_result = getattr(eval_output, "result", None)
                logger.info(
                    "temp eval: trainer.evaluate split={} elapsed_s={}",
                    split_ctx.split, time.monotonic() - _t,
                )

                # Compute full MetricReport from the completed rollout
                try:
                    from trading_rl.evaluation.report import build_evaluation_report_for_trainer
                    from trading_rl.pipeline.evaluation import (
                        _evaluation_rollout,
                        _evaluation_strategy_returns,
                    )
                    metric_report = build_evaluation_report_for_trainer(
                        trainer=self.trainer,
                        df_prices=split_ctx.df,
                        max_steps=split_ctx.max_steps,
                        config=hook.config,
                        eval_env=split_ctx.eval_env,
                        rollout=_evaluation_rollout(last_eval_result),
                        strategy_simple_returns=_evaluation_strategy_returns(
                            last_eval_result
                        ),
                    )
                except Exception:
                    logger.opt(exception=True).warning("temp eval: MetricReport failed split={}", split_ctx.split)
                    metric_report = None

                if metric_report is not None:
                    _sharpe = metric_report.sharpe_ratio
                    _sortino = metric_report.sortino_ratio
                    _tot_ret = metric_report.total_return
                    logger.info(
                        "temp eval metrics split={} sharpe={:.3f} sortino={:.3f} total_return={:.3f}",
                        split_ctx.split,
                        _sharpe if math.isfinite(_sharpe) else float("nan"),
                        _sortino if math.isfinite(_sortino) else float("nan"),
                        _tot_ret if math.isfinite(_tot_ret) else float("nan"),
                    )
                else:
                    logger.warning(
                        "temp eval: metric report unavailable split={} (Sharpe/Sortino will not be logged)",
                        split_ctx.split,
                    )

                # Collect a ReturnSeries for the equity progression plot.
                # Stage in a local variable — only committed to shared
                # _progression_history in the else-clause below so that a
                # failure anywhere in the outer try leaves history unmodified.
                _staged_prog_entry: tuple | None = None
                try:
                    from trading_rl.evaluation.returns import ReturnKind, ReturnSeries
                    _eval_result = last_eval_result
                    if _eval_result is not None:
                        _rs = getattr(_eval_result, "return_series", None)
                        if _rs is None:
                            _simple = getattr(_eval_result, "simple_returns", None)
                            if _simple is not None and len(_simple) > 0:
                                _rs = ReturnSeries(_simple, ReturnKind.SIMPLE)
                        if _rs is not None:
                            _staged_prog_entry = (step_number, _rs)
                            logger.debug(
                                "temp eval: progression entry staged split={} step={}",
                                split_ctx.split, step_number,
                            )
                except Exception:
                    logger.warning(
                        "temp eval: progression history collect failed split={}",
                        split_ctx.split, exc_info=True,
                    )

                if mlflow.active_run():
                    from trading_rl.callbacks import MLflowTrainingCallback

                    # Log the underlying close price plot once per split (not per step).
                    if split_ctx.split not in self._price_plots_logged:
                        self._log_price_plot_once(split_ctx, hook)
                        self._price_plots_logged.add(split_ctx.split)

                    artifact_prefix = ArtifactPaths.eval_plots_temp(split_ctx.split, step_number)
                    _t = time.monotonic()
                    _plot_data = None
                    _last_result = last_eval_result
                    if _last_result is not None and _last_result.plots:
                        _rollout = _last_result.plots.get("_rollout_plot_data")
                        _equity = _last_result.plots.get("_equity_plot_data")
                        if _rollout or _equity:
                            _plot_data = {**(_rollout or {}), **(_equity or {})}
                    MLflowTrainingCallback.log_evaluation_plots(
                        reward_plot=reward_plot,
                        action_plot=action_plot,
                        action_probs_plot=action_probs_plot,
                        equity_curve_plot=equity_curve_plot,
                        logs=None,
                        merged_plot=merged_plot,
                        artifact_path_prefix=artifact_prefix,
                        debug=getattr(hook.config.logging, "debug_plots", False),
                        plot_data=_plot_data,
                    )
                    logger.trace(
                        "temp eval: mlflow upload split={} elapsed_s={}",
                        split_ctx.split, time.monotonic() - _t,
                    )
                    # Upload equity progression plot (all committed checkpoints
                    # plus the staged entry for this step, without mutating history).
                    _prog_history = self._progression_history.get(split_ctx.split, []) + (
                        [_staged_prog_entry] if _staged_prog_entry is not None else []
                    )
                    if len(_prog_history) >= 2:
                        try:
                            import os
                            import tempfile
                            from trading_rl.evaluation.plots import create_equity_progression_plot
                            from trading_rl.evaluation.thesis_theme import (
                                FIGURE_HEIGHT,
                                FIGURE_WIDTH,
                                PLOT_DPI,
                                save_plot as _save_plot,
                            )
                            _prog_plot = create_equity_progression_plot(_prog_history)
                            if _prog_plot is not None:
                                with tempfile.TemporaryDirectory() as _tmpdir:
                                    _tmp_path = os.path.join(_tmpdir, "progression.png")
                                    _save_plot(
                                        _prog_plot, _tmp_path,
                                        width=FIGURE_WIDTH * 1.5,
                                        height=FIGURE_HEIGHT * 1.5,
                                        dpi=PLOT_DPI,
                                    )
                                    mlflow.log_artifact(
                                        _tmp_path,
                                        f"evaluation_plots_temp/{split_ctx.split}",
                                    )
                                logger.trace(
                                    "temp eval: progression plot uploaded split={} checkpoints={}",
                                    split_ctx.split, len(_prog_history),
                                )
                        except Exception:
                            logger.error(
                                "temp eval: progression plot failed split={}",
                                split_ctx.split, exc_info=True,
                            )

                    mlflow.log_metric(
                        f"eval_{split_ctx.split}_reward",
                        final_reward,
                        step=step_number,
                    )
                    if metric_report is not None:
                        for key, value in metric_report.to_dict().items():
                            if isinstance(value, float) and math.isfinite(value):
                                mlflow.log_metric(
                                    f"eval_{split_ctx.split}_{key}",
                                    value,
                                    step=step_number,
                                )
                    if metric_report is not None:
                        try:
                            import os
                            import tempfile
                            from trading_rl.evaluation.plots import create_metrics_table_figure
                            fig = create_metrics_table_figure(
                                metric_report,
                                step=step_number,
                                split=split_ctx.split,
                                df=split_ctx.df,
                                max_steps=split_ctx.max_steps,
                            )
                            with tempfile.TemporaryDirectory() as _tmpdir:
                                _tbl_path = os.path.join(_tmpdir, "metrics_table.png")
                                fig.savefig(_tbl_path, dpi=150, bbox_inches="tight")
                                import matplotlib.pyplot as _plt
                                _plt.close(fig)
                                mlflow.log_artifact(_tbl_path, artifact_prefix)
                            logger.debug(
                                "temp eval: metrics table saved split={} step={}",
                                split_ctx.split, step_number,
                            )
                        except Exception:
                            logger.warning(
                                "temp eval: metrics table failed split={} step={}",
                                split_ctx.split, step_number, exc_info=True,
                            )
                    if hook.config.training.temp_eval.log_data:
                        last_result = last_eval_result
                        if last_result is not None:
                            try:
                                import numpy as _np
                                from pathlib import Path as _Path
                                from trading_rl.callbacks.artifacts import save_eval_rollout_artifact
                                save_eval_rollout_artifact(
                                    split=f"{split_ctx.split}_step_{step_number:08d}",
                                    last_positions=getattr(last_result, "last_positions", []),
                                    simple_returns=getattr(last_result, "simple_returns", _np.array([])),
                                    cumulative_returns=getattr(last_result, "cumulative_returns", None),
                                    df_index=split_ctx.df.index,
                                    output_dir=_Path(hook.config.logging.log_dir) / ArtifactPaths.EVAL_DATA_TEMP,
                                    artifact_path_prefix=ArtifactPaths.eval_data_temp(split_ctx.split, step_number),
                                )
                            except Exception:
                                logger.opt(exception=True).warning("temp eval: rollout data artifact failed split={} step={}", split_ctx.split, step_number)
                    _elapsed = time.monotonic() - _t_split
                    logger.info(
                        "temp eval complete split={} reward={} artifacts={} elapsed_s={}",
                        split_ctx.split, final_reward, artifact_prefix, _elapsed,
                    )
                else:
                    logger.warning(
                        "no active mlflow run, skip temp evaluation logging split={}",
                        split_ctx.split,
                    )
            except Exception:
                self._eval_consecutive_failures += 1
                logger.error(
                    "temp eval failed split={} step={} elapsed_s={:.2f} (failure {}/{})",
                    split_ctx.split, step_number, time.monotonic() - _t_split,
                    self._eval_consecutive_failures, self._MAX_HOOK_FAILURES,
                )
                if self._eval_consecutive_failures >= self._MAX_HOOK_FAILURES:
                    raise RuntimeError(
                        f"Periodic eval hook failed {self._MAX_HOOK_FAILURES} consecutive times; "
                        "aborting training. Check logs above for the root cause."
                    )
            else:
                self._eval_consecutive_failures = 0
                # Commit the staged progression entry now that the full
                # evaluation succeeded — history is only modified on success.
                if _staged_prog_entry is not None:
                    self._progression_history.setdefault(split_ctx.split, []).append(
                        _staged_prog_entry
                    )
                    logger.trace(
                        "temp eval: progression entry committed split={} step={} n={}",
                        split_ctx.split, step_number,
                        len(self._progression_history[split_ctx.split]),
                    )
                _dense_signal = (
                    metric_report.total_return
                    if metric_report is not None
                    else final_reward
                )
                self._update_eval_density(hook, split_ctx.split, _dense_signal, step_number)

        # Create and upload train/val progression plot (if we have both splits)
        if mlflow.active_run():
            train_history = self._progression_history.get("train", [])
            val_history = self._progression_history.get("val", [])
            if len(train_history) + len(val_history) >= 2:
                try:
                    import os
                    import tempfile
                    from trading_rl.evaluation.plots import create_train_val_progression_plot
                    from trading_rl.evaluation.thesis_theme import (
                        FIGURE_HEIGHT,
                        FIGURE_WIDTH,
                        PLOT_DPI,
                        save_plot as _save_plot,
                    )
                    _prog_plot = create_train_val_progression_plot(
                        train_history,
                        val_history,
                        initial_portfolio_value=hook.config.env.initial_portfolio_value,
                        metric="total_return",
                    )
                    if _prog_plot is not None:
                        with tempfile.TemporaryDirectory() as _tmpdir:
                            _tmp_path = os.path.join(_tmpdir, "train_val_progression.png")
                            _save_plot(
                                _prog_plot, _tmp_path,
                                width=FIGURE_WIDTH * 2,
                                height=FIGURE_HEIGHT * 1.5,
                                dpi=PLOT_DPI,
                            )
                            mlflow.log_artifact(
                                _tmp_path,
                                "evaluation_plots_temp",
                            )
                        logger.trace(
                            "temp eval: train/val progression plot uploaded train_checkpoints={} val_checkpoints={}",
                            len(train_history), len(val_history),
                        )
                except Exception:
                    logger.error(
                        "temp eval: train/val progression plot failed",
                    )

        logger.info(
            "temp eval all splits done step={} total_elapsed_s={}",
            step_number, time.monotonic() - _t_total,
        )

    def _update_eval_density(
        self,
        hook: PeriodicEvaluationHook,
        split: str,
        signal: float,
        step: int,
    ) -> None:
        """Shorten or restore the eval interval based on recent reward trend.

        Dense mode triggers when the most recent `window` evaluations show a
        relative improvement of at least `threshold` over the preceding `window`
        evaluations.  The cadence is restored once improvement flattens out.
        """
        import math

        cfg = hook.config.training.temp_eval
        if not cfg.dense_enabled:
            return
        if not math.isfinite(signal):
            return

        window: int = cfg.dense_window
        threshold: float = cfg.dense_improvement_threshold
        min_evals: int = cfg.dense_min_evals
        base_interval: int = cfg.interval

        history = hook.dense_reward_history.setdefault(split, [])
        history.append((step, signal))

        if len(history) < min_evals:
            return

        recent_vals = [r for _, r in history[-window:]]
        earlier_start = max(0, len(history) - 2 * window)
        earlier_end = max(0, len(history) - window)
        earlier_vals = [r for _, r in history[earlier_start:earlier_end]]

        if not earlier_vals:
            return

        recent_mean = sum(recent_vals) / len(recent_vals)
        earlier_mean = sum(earlier_vals) / len(earlier_vals)
        denominator = abs(earlier_mean) + 1e-12
        relative_change = (recent_mean - earlier_mean) / denominator

        dense_interval = cfg.dense_interval if cfg.dense_interval is not None else max(1, base_interval // 2)

        if relative_change > threshold:
            if hook.effective_interval != dense_interval:
                hook.effective_interval = dense_interval
                logger.info(
                    "dense eval: improvement {:.1%} > threshold {:.1%} split={} "
                    "interval {} → {}",
                    relative_change, threshold, split, base_interval, dense_interval,
                )
        elif len(history) >= 2 * window and hook.effective_interval != base_interval:
            hook.effective_interval = base_interval
            logger.info(
                "dense eval: plateau detected split={} interval restored to {}",
                split, base_interval,
            )

    def _run_temporary_explainability(
        self,
        step_number: int,
        hook: PeriodicExplainabilityHook,
    ) -> None:
        """Run explainability during training and log artifacts without affecting control flow."""
        logger.info(
            "Running temporary explainability analysis at step {}...",
            step_number,
        )

        try:
            from trading_rl.evaluation import EvaluationContext
            from trading_rl.pipeline.explainability import (
                run_explainability_analysis,
            )

            eval_ctx = EvaluationContext(
                split="temp",
                df=hook.df,
                env=hook.eval_env if hook.eval_env else self.trainer.env,
                max_steps=hook.max_steps,
            )
            artifact_prefix = f"explainability_temp/step_{step_number:08d}"
            run_explainability_analysis(
                config=hook.config,
                trainer=self.trainer,
                eval_ctx=eval_ctx,
                train_df=hook.df,
                logger=logger,
                artifact_path_prefix=artifact_prefix,
            )
            logger.info(
                "Temporary explainability complete: plots saved to {}",
                artifact_prefix,
            )
        except Exception:
            self._explainability_consecutive_failures += 1
            logger.error(
                "Temporary explainability failed at step {} (failure {}/{})",
                step_number,
                self._explainability_consecutive_failures, self._MAX_HOOK_FAILURES,
            )
            if self._explainability_consecutive_failures >= self._MAX_HOOK_FAILURES:
                raise RuntimeError(
                    f"Periodic explainability hook failed {self._MAX_HOOK_FAILURES} consecutive times; "
                    "aborting training. Check logs above for the root cause."
                )
        else:
            self._explainability_consecutive_failures = 0
