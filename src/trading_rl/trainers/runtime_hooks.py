"""Runtime hook orchestration for periodic trainer side effects."""

from __future__ import annotations

from dataclasses import dataclass
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
        eval_interval = getattr(config.training, "temp_eval_interval", None)
        if eval_interval is None or eval_interval <= 0:
            logger.info("periodic evaluation disabled (temp_eval_interval not set)")
            self._evaluation = None
            return

        self._evaluation = PeriodicEvaluationHook(
            splits=splits,
            config=config,
            algorithm=algorithm,
        )
        split_names = [s.split for s in splits]
        logger.info(
            "periodic evaluation enabled interval=%s splits=%s",
            eval_interval, split_names,
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
            "Periodic explainability enabled: will analyze every %s training steps",
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

        eval_interval = hook.config.training.temp_eval_interval
        if step_number - hook.last_step < eval_interval:
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
                    "price plot: column %r not found in split=%s df (columns=%s)",
                    price_column, split_ctx.split, list(split_ctx.df.columns)[:10],
                )
                return
            artifact_dir = f"{ArtifactPaths.EVAL_PLOTS_TEMP}/{split_ctx.split}"
            with tempfile.TemporaryDirectory() as tmpdir:
                path = os.path.join(tmpdir, "close_price.png")
                _save_plot(plot, path, width=FIGURE_WIDTH * 2, height=FIGURE_HEIGHT * 1.5, dpi=PLOT_DPI)
                mlflow.log_artifact(path, artifact_dir)
            logger.info("price plot logged split=%s column=%s", split_ctx.split, price_column)
        except Exception:
            logger.warning("price plot failed split=%s", split_ctx.split, exc_info=True)

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
            "run temporary evaluation step=%s splits=%s",
            step_number, split_names,
        )

        for split_ctx in hook.splits:
            _t_split = time.monotonic()
            logger.info(
                "temp eval split=%s max_steps=%s df_rows=%s",
                split_ctx.split, split_ctx.max_steps, len(split_ctx.df),
            )
            try:
                _t = time.monotonic()
                (
                    reward_plot,
                    action_plot,
                    action_probs_plot,
                    final_reward,
                    _last_positions,
                    equity_curve_plot,
                    merged_plot,
                ) = self.trainer.evaluate(
                    df=split_ctx.df,
                    max_steps=split_ctx.max_steps,
                    config=hook.config,
                    algorithm=hook.algorithm,
                    eval_env=split_ctx.eval_env,
                )
                logger.info(
                    "temp eval: trainer.evaluate split=%s elapsed_s=%.2f",
                    split_ctx.split, time.monotonic() - _t,
                )

                # Compute full MetricReport from the completed rollout
                try:
                    from trading_rl.evaluation.report import build_evaluation_report_for_trainer
                    from trading_rl.pipeline.evaluation import (
                        _last_evaluation_rollout,
                        _last_strategy_returns,
                    )
                    metric_report = build_evaluation_report_for_trainer(
                        trainer=self.trainer,
                        df_prices=split_ctx.df,
                        max_steps=split_ctx.max_steps,
                        config=hook.config,
                        eval_env=split_ctx.eval_env,
                        rollout=_last_evaluation_rollout(self.trainer),
                        strategy_simple_returns=_last_strategy_returns(self.trainer),
                    )
                except Exception:
                    logger.warning("temp eval: MetricReport failed split=%s", split_ctx.split, exc_info=True)
                    metric_report = None

                if metric_report is not None:
                    _sharpe = metric_report.sharpe_ratio
                    _sortino = metric_report.sortino_ratio
                    _tot_ret = metric_report.total_return
                    logger.info(
                        "temp eval metrics split=%s sharpe=%.3f sortino=%.3f total_return=%.4f",
                        split_ctx.split,
                        _sharpe if math.isfinite(_sharpe) else float("nan"),
                        _sortino if math.isfinite(_sortino) else float("nan"),
                        _tot_ret if math.isfinite(_tot_ret) else float("nan"),
                    )
                else:
                    logger.warning(
                        "temp eval: metric report unavailable split=%s (Sharpe/Sortino will not be logged)",
                        split_ctx.split,
                    )

                # Collect a ReturnSeries for the equity progression plot
                try:
                    from trading_rl.evaluation.returns import ReturnKind, ReturnSeries
                    _eval_result = getattr(self.trainer, "_last_evaluation_result", None)
                    if _eval_result is not None:
                        _rs = getattr(_eval_result, "return_series", None)
                        if _rs is None:
                            _simple = getattr(_eval_result, "simple_returns", None)
                            if _simple is not None and len(_simple) > 0:
                                _rs = ReturnSeries(_simple, ReturnKind.SIMPLE)
                        if _rs is not None:
                            self._progression_history.setdefault(split_ctx.split, []).append(
                                (step_number, _rs)
                            )
                            logger.debug(
                                "temp eval: progression history appended split=%s step=%s n=%d",
                                split_ctx.split, step_number,
                                len(self._progression_history[split_ctx.split]),
                            )
                except Exception:
                    logger.debug(
                        "temp eval: progression history collect failed split=%s",
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
                    MLflowTrainingCallback.log_evaluation_plots(
                        reward_plot=reward_plot,
                        action_plot=action_plot,
                        action_probs_plot=action_probs_plot,
                        equity_curve_plot=equity_curve_plot,
                        logs=None,
                        merged_plot=merged_plot,
                        artifact_path_prefix=artifact_prefix,
                        debug=getattr(hook.config.logging, "debug_plots", False),
                    )
                    logger.debug(
                        "temp eval: mlflow upload split=%s elapsed=%.2fs",
                        split_ctx.split, time.monotonic() - _t,
                    )
                    # Upload equity progression plot (all checkpoints so far)
                    _prog_history = self._progression_history.get(split_ctx.split, [])
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
                                logger.debug(
                                    "temp eval: progression plot uploaded split=%s checkpoints=%d",
                                    split_ctx.split, len(_prog_history),
                                )
                        except Exception:
                            logger.error(
                                "temp eval: progression plot failed split=%s",
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
                    if getattr(hook.config.training, "temp_eval_log_data", False):
                        last_result = getattr(self.trainer, "_last_evaluation_result", None)
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
                                logger.warning("temp eval: rollout data artifact failed split=%s step=%s", split_ctx.split, step_number, exc_info=True)
                    _elapsed = time.monotonic() - _t_split
                    logger.info(
                        "temp eval complete split=%s reward=%.4f artifacts=%s elapsed_s=%.2f",
                        split_ctx.split, final_reward, artifact_prefix, _elapsed,
                    )
                else:
                    logger.warning(
                        "no active mlflow run, skip temp evaluation logging split=%s",
                        split_ctx.split,
                    )
            except Exception:
                self._eval_consecutive_failures += 1
                logger.error(
                    "temp eval failed split=%s step=%s elapsed=%.2fs (failure %d/%d)",
                    split_ctx.split, step_number, time.monotonic() - _t_split,
                    self._eval_consecutive_failures, self._MAX_HOOK_FAILURES,
                    exc_info=True,
                )
                if self._eval_consecutive_failures >= self._MAX_HOOK_FAILURES:
                    raise RuntimeError(
                        f"Periodic eval hook failed {self._MAX_HOOK_FAILURES} consecutive times; "
                        "aborting training. Check logs above for the root cause."
                    )
            else:
                self._eval_consecutive_failures = 0

        logger.info(
            "temp eval all splits done step=%s total_elapsed_s=%.2f",
            step_number, time.monotonic() - _t_total,
        )

    def _run_temporary_explainability(
        self,
        step_number: int,
        hook: PeriodicExplainabilityHook,
    ) -> None:
        """Run explainability during training and log artifacts without affecting control flow."""
        logger.info(
            "Running temporary explainability analysis at step %s...",
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
                "Temporary explainability complete: plots saved to %s",
                artifact_prefix,
            )
        except Exception:
            self._explainability_consecutive_failures += 1
            logger.error(
                "Temporary explainability failed at step %s (failure %d/%d)",
                step_number,
                self._explainability_consecutive_failures, self._MAX_HOOK_FAILURES,
                exc_info=True,
            )
            if self._explainability_consecutive_failures >= self._MAX_HOOK_FAILURES:
                raise RuntimeError(
                    f"Periodic explainability hook failed {self._MAX_HOOK_FAILURES} consecutive times; "
                    "aborting training. Check logs above for the root cause."
                )
        else:
            self._explainability_consecutive_failures = 0
