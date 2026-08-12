"""Training loop extracted from BaseTrainer._run_training_loop."""

from __future__ import annotations

import contextlib
import signal
import time
from collections.abc import Callable
from typing import Any

from logger import get_logger, is_level_enabled, log_banner
from trading_rl.profiler import get_profiler

logger = get_logger(__name__)


@contextlib.contextmanager
def _signal_guard():
    """Install a clean SIGINT handler and restore the original on exit."""
    original = signal.getsignal(signal.SIGINT)

    def _handler(sig, frame):
        logger.info("sigint received raising KeyboardInterrupt")
        signal.signal(signal.SIGINT, original)
        raise KeyboardInterrupt()

    signal.signal(signal.SIGINT, _handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original)


class TrainingLoop:
    """The collect-optimize-checkpoint loop.

    Owns the loop structure (iteration, stopping conditions, interrupt
    handling) while delegating every algorithm-specific decision back to the
    trainer via its abstract interface.
    """

    def run(
        self,
        trainer: Any,
        callback: Any = None,
        *,
        start_message: str = "Starting training",
        completion_prefix: str = "Training complete",
        on_batch_start: Callable[[int, Any], None] | None = None,
        on_batch_end: Callable[[int, Any], None] | None = None,
        on_train_end: Callable[[], None] | None = None,
    ) -> dict[str, list]:
        """Run the training loop and return the logs dict."""
        log_banner(logger, f"TRAINING START  {start_message}")
        t0 = time.time()
        trainer.callback = callback
        trainer._log_step_offset = max(
            len(trainer.logs.get("loss_actor", [])),
            len(trainer.logs.get("loss_value", [])),
        )

        _profiler = get_profiler()
        with _signal_guard():
            try:
                for i, data in enumerate(trainer.collector):
                    if on_batch_start is not None:
                        on_batch_start(i, data)

                    trainer._current_batch = data

                    with _profiler.stage("buffer_extend", 2):
                        if trainer._use_replay_buffer:
                            trainer.replay_buffer.extend(data)
                            max_length = trainer.replay_buffer[:]["next", "step_count"].max()
                            buffer_len = len(trainer.replay_buffer)
                        else:
                            max_length = data["next", "step_count"].max()
                            buffer_len = data.numel()

                    trainer.total_count += data.numel()
                    if is_level_enabled("TRACE"):
                        logger.trace(
                            "batch collected batch={} n_frames={} total_count={} buffer_len={}",
                            i,
                            data.numel(),
                            trainer.total_count,
                            buffer_len,
                        )

                    collected_steps = (
                        trainer.total_count if not trainer._use_replay_buffer else buffer_len
                    )
                    if collected_steps > trainer.config.init_rand_steps:
                        with _profiler.stage("optimization", 2):
                            trainer._optimization_step(i, max_length, buffer_len)

                    episodes_in_batch = int(data["next", "done"].sum().item())
                    trainer.total_episodes += episodes_in_batch
                    if is_level_enabled("TRACE") and episodes_in_batch > 0:
                        logger.trace(
                            "episodes completed batch={} n_episodes={} total_episodes={}",
                            i,
                            episodes_in_batch,
                            trainer.total_episodes,
                        )

                    with _profiler.stage("checkpoint", 2):
                        trainer.checkpoint_manager.maybe_save(
                            trainer.total_count, trainer._snapshot
                        )

                    with _profiler.stage("periodic_hooks", 2):
                        trainer.runtime_hooks.maybe_run(trainer.total_count)

                    if trainer.callback and hasattr(trainer.callback, "log_episode_stats"):
                        trainer._log_episode_stats(data, trainer.callback)

                    finding = trainer.health_monitor.check()
                    if finding is not None:
                        logger.warning(
                            "runtime guardrail {} [{}] {} | suggestion: {}",
                            finding.severity.value,
                            finding.parameter,
                            finding.message,
                            finding.suggestion,
                        )
                        trainer.logs["early_stop_reason"].append(
                            f"{finding.severity.value}:{finding.parameter}:{finding.message}"
                        )
                        break

                    if on_batch_end is not None:
                        on_batch_end(i, data)

                    if trainer.total_count >= trainer.config.max_steps:
                        logger.info("training stopped max_steps={}", trainer.config.max_steps)
                        break
                    if (
                        trainer.config.max_train_seconds is not None
                        and (time.time() - t0) >= trainer.config.max_train_seconds
                    ):
                        logger.info(
                            "training stopped max_train_seconds={} elapsed_s={:.1f}",
                            trainer.config.max_train_seconds,
                            time.time() - t0,
                        )
                        break
            except KeyboardInterrupt:
                logger.warning("training interrupted by user saving checkpoint")
                checkpoint_path = trainer.checkpoint_manager.save_interrupt(
                    trainer.total_count, trainer._snapshot
                )
                if checkpoint_path:
                    logger.info("interrupt checkpoint saved path={}", checkpoint_path)
                raise

        if on_train_end is not None:
            on_train_end()

        elapsed = time.time() - t0
        early_stop_reasons = trainer.logs.get("early_stop_reason", [])
        if early_stop_reasons:
            logger.warning(
                "training ended early reason={} steps={}/{}",
                early_stop_reasons[-1],
                trainer.total_count,
                trainer.config.max_steps,
            )
        log_banner(
            logger,
            f"TRAINING END  {trainer.total_count} steps  "
            f"{trainer.total_episodes} episodes  {elapsed:.2f}s",
        )
        trainer.logs["training_duration_s"].append(elapsed)
        return dict(trainer.logs)
