"""Runtime hook orchestration for periodic trainer side effects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlflow

from logger import get_logger

logger = get_logger(__name__)

_C = "\033[36m"   # cyan — used to highlight log values
_R = "\033[0m"    # reset


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

    def __init__(self, trainer: Any):
        self.trainer = trainer
        self._evaluation: PeriodicEvaluationHook | None = None
        self._explainability: PeriodicExplainabilityHook | None = None

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

    def _run_temporary_evaluation(
        self,
        step_number: int,
        hook: PeriodicEvaluationHook,
    ) -> None:
        """Run evaluation during training and log artifacts without affecting control flow."""
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
                    actual_returns_plot,
                    merged_plot,
                ) = self.trainer.evaluate(
                    df=split_ctx.df,
                    max_steps=split_ctx.max_steps,
                    config=hook.config,
                    algorithm=hook.algorithm,
                    eval_env=split_ctx.eval_env,
                )
                logger.info(
                    "temp eval: trainer.evaluate split=%s elapsed=%.2fs",
                    split_ctx.split, time.monotonic() - _t,
                )

                if mlflow.active_run():
                    from trading_rl.callbacks import MLflowTrainingCallback

                    artifact_prefix = (
                        f"evaluation_plots_temp/step_{step_number:08d}/{split_ctx.split}"
                    )
                    _t = time.monotonic()
                    MLflowTrainingCallback.log_evaluation_plots(
                        reward_plot=reward_plot,
                        action_plot=action_plot,
                        action_probs_plot=action_probs_plot,
                        actual_returns_plot=actual_returns_plot,
                        logs=None,
                        merged_plot=merged_plot,
                        artifact_path_prefix=artifact_prefix,
                    )
                    logger.debug(
                        "temp eval: mlflow upload split=%s elapsed=%.2fs",
                        split_ctx.split, time.monotonic() - _t,
                    )
                    mlflow.log_metric(
                        f"temp_eval_reward_{split_ctx.split}",
                        final_reward,
                        step=step_number,
                    )
                    _elapsed = time.monotonic() - _t_split
                    logger.info(
                        "temp eval complete split=%s%s%s reward=%s%.4f%s"
                        " artifacts=%s%s%s elapsed_s=%s%.2f%s",
                        _C, split_ctx.split, _R,
                        _C, final_reward, _R,
                        _C, artifact_prefix, _R,
                        _C, _elapsed, _R,
                    )
                else:
                    logger.warning(
                        "no active mlflow run, skip temp evaluation logging split=%s",
                        split_ctx.split,
                    )
            except Exception as exc:
                logger.error(
                    "temp eval failed split=%s step=%s elapsed=%.2fs err=%s",
                    split_ctx.split, step_number, time.monotonic() - _t_split, exc,
                )

        logger.info(
            "temp eval all splits done step=%s total_elapsed=%.2fs",
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
        except Exception as exc:
            logger.error(
                "Temporary explainability failed at step %s: %s",
                step_number,
                exc,
            )
