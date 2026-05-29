"""Base trainer and utilities."""

import contextlib
import signal
import time
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.multiprocessing as mp
import torchrl.collectors.collectors as torchrl_collectors
from tensordict.nn import set_composite_lp_aggregate
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer

from logger import get_logger, is_level_enabled, log_banner
from trading_rl.config import EvaluationConfig, TrainingConfig
from trading_rl.constants import BenchmarkName
from trading_rl.evaluation.benchmarks import benchmarks_from_config
from trading_rl.evaluation.returns import ReturnKind, ReturnSeries
from trading_rl.profiler import get_profiler
from trading_rl.trainers.checkpointing import CheckpointManager
from trading_rl.trainers.episode_stats import EpisodeStatsTracker
from trading_rl.trainers.health_monitor import TrainingHealthMonitor
from trading_rl.trainers.runtime_hooks import TrainerRuntimeHooks

_MIN_BATCH_SUCCESS_RATE = 70.0  # Warn if fewer than this % of optimization batches succeed


def _log_network_stats(log, algo: str, actor: torch.nn.Module, critic: torch.nn.Module) -> None:
    """Emit a TRACE line with parameter and gradient statistics for actor and critic."""
    def _stats(net: torch.nn.Module) -> tuple[float, float, float, int]:
        params = list(net.parameters())
        abs_sum = sum(p.detach().abs().sum().item() for p in params)
        norm = sum(p.detach().pow(2).sum().item() for p in params) ** 0.5
        grad_norm = sum(
            p.grad.detach().pow(2).sum().item()
            for p in params if p.grad is not None
        ) ** 0.5
        n = sum(p.numel() for p in params)
        return abs_sum, norm, grad_norm, n

    a_abs, a_norm, a_gnorm, a_n = _stats(actor)
    c_abs, c_norm, c_gnorm, c_n = _stats(critic)
    log.trace(
        "%s network_stats "
        "actor_abs_sum=%.4f actor_norm=%.4f actor_grad_norm=%.4f actor_n_params=%d "
        "critic_abs_sum=%.4f critic_norm=%.4f critic_grad_norm=%.4f critic_n_params=%d",
        algo,
        a_abs, a_norm, a_gnorm, a_n,
        c_abs, c_norm, c_gnorm, c_n,
    )



class _LocalTrajectoryPool:
    """Minimal trajectory pool that avoids shared memory requirements."""

    def __init__(self, ctx=None, lock: bool = False):
        self.ctx = ctx
        self._traj_id = torch.zeros((), device="cpu", dtype=torch.int)
        if lock:
            self.lock = (ctx or mp).RLock()
        else:
            self.lock = contextlib.nullcontext()

    def get_traj_and_increment(self, n: int = 1, device=None):
        with self.lock:
            start = int(self._traj_id.item())
            out = torch.arange(start, start + n, device=device)
            self._traj_id.copy_(torch.tensor(out[-1].item() + 1))
        return out


logger = get_logger(__name__)


def _collect_mlflow_meta() -> dict:
    """Collect MLflow metadata; always includes tracking_uri when mlflow is available."""
    try:
        import mlflow
        meta: dict = {"tracking_uri": mlflow.get_tracking_uri()}
        run = mlflow.active_run()
        if run is None:
            return meta
        experiment = mlflow.get_experiment(run.info.experiment_id)
        meta.update({
            "run_id": run.info.run_id,
            "run_name": run.data.tags.get("mlflow.runName"),
            "experiment_id": run.info.experiment_id,
            "experiment_name": experiment.name if experiment else None,
        })
        return meta
    except Exception:
        logger.opt(exception=True).debug("_collect_mlflow_meta failed; checkpoint will have no mlflow metadata")
        return {}


def _patch_torchrl_trajectory_pool() -> None:
    """Replace TorchRL's _TrajectoryPool with a shared-memory-free implementation.

    Called lazily from BaseTrainer.__init__ so importing this module does not
    mutate TorchRL's global state.  The patch is idempotent — applying it
    twice is safe.

    TorchRL ≤0.11 exposed _TrajectoryPool on torchrl.collectors.collectors;
    TorchRL 0.12+ moved it to torchrl.collectors.utils.  We patch whichever
    module currently owns the name.
    """
    import torchrl.collectors.utils as _tc_utils

    patched = False
    for _mod in (torchrl_collectors, _tc_utils):
        if hasattr(_mod, "_TrajectoryPool") and _mod._TrajectoryPool is not _LocalTrajectoryPool:
            _mod._TrajectoryPool = _LocalTrajectoryPool
            patched = True
    if patched:
        logger.debug("patched torchrl _TrajectoryPool -> _LocalTrajectoryPool")


def _run_evaluation(
    trainer: Any,
    df: Any,
    max_steps: int,
    config: Any = None,
    algorithm: str | None = None,  # noqa: ARG001 — kept for API symmetry
    eval_env: Any | None = None,
) -> tuple[Any, ...]:
    """Run a policy evaluation rollout and build result plots.

    Extracted from BaseTrainer.evaluate() so the logic is testable without
    subclassing and so subclass overrides remain thin.

    Returns:
        (reward_plot, action_plot, None, final_reward, last_positions,
         equity_curve_plot, merged_plot)
    """
    from trading_rl.config import DEFAULT_INITIAL_PORTFOLIO_VALUE
    from trading_rl.evaluation.evaluator import StrategyEvaluatorConfig, StrategyEvaluator
    from trading_rl.utils import create_equity_curve_plot, create_merged_comparison_plot

    env_to_use = eval_env or trainer.env
    eval_config_kwargs: dict[str, Any] = {}

    if config:
        from trading_rl.evaluation.evaluator import EvaluatorEnvConfig
        eval_config_kwargs = {
            "reward_type": config.env.reward_type,
            "backend": config.env.backend,
            "price_column": config.env.price_column,
            "max_steps": max_steps,
            "enable_plots": True,
            "enable_metrics": False,
            "max_plot_points": config.training.max_plot_points,
            "show_allocation_ma": config.training.show_allocation_ma,
            "allocation_ma_window": config.training.allocation_ma_window,
            "eval_plots": tuple(config.evaluation.eval_plots),
            "training_steps": int(trainer.total_count) if trainer is not None else None,
            "training_episodes": int(trainer.total_episodes) if trainer is not None else None,
            "benchmarks": benchmarks_from_config(config.benchmarks),
            "env": EvaluatorEnvConfig(
                name=config.env.name,
                positions=config.env.positions,
                mode=config.env.mode,
                trading_fees=config.env.trading_fees,
                borrow_interest_rate=config.env.borrow_interest_rate,
                initial_portfolio_value=config.env.initial_portfolio_value,
                price_column=config.env.price_column or "close",
            ),
        }

    eval_config = StrategyEvaluatorConfig(**eval_config_kwargs)
    profiler = get_profiler()

    evaluator = StrategyEvaluator(
        env_factory=lambda _df, _cfg: env_to_use,
        policy=trainer.actor,
        config=eval_config,
    )

    _t = time.monotonic()
    with profiler.stage("agent_rollout", 2):
        result = evaluator.evaluate_split("eval", df, env=env_to_use)
    trainer._last_evaluation_result = result
    logger.trace("evaluate.rollout_and_metrics elapsed={:.2f}s", time.monotonic() - _t)

    _enabled_plots = set(eval_config.eval_plots)
    reward_plot = result.plots.get("reward_plot") if result.plots else None
    action_plot = result.plots.get("action_plot") if result.plots else None

    equity_curve_plot = None
    if "portfolio_value" in _enabled_plots:
        with profiler.stage("plot_equity_curve", 2):
            _t = time.monotonic()
            logger.trace("create_equity_curve_plot start n_steps={}", max_steps)
            _data_paths = config.data.data_paths if config else None
            plot_series = result.return_series or ReturnSeries(result.simple_returns, ReturnKind.SIMPLE)
            equity_curve_plot = create_equity_curve_plot(
                None,
                max_steps,
                df_prices=df,
                env=env_to_use,
                actual_returns_list=[plot_series],
                initial_portfolio_value=(
                    float(config.env.initial_portfolio_value)
                    if config else DEFAULT_INITIAL_PORTFOLIO_VALUE
                ),
                benchmark_price_column=config.env.price_column or "close" if config else "close",
                benchmarks=benchmarks_from_config(config.benchmarks) if config else frozenset({BenchmarkName.BUY_AND_HOLD}),
                training_steps=trainer.total_count,
                training_episodes=trainer.total_episodes,
                n_total_symbols=len(_data_paths) if _data_paths else None,
                max_plot_points=config.training.max_plot_points if config else None,
                reward_type=str(config.env.reward_type) if config else None,
            )
            logger.trace("evaluate.plot_equity_curve elapsed={:.2f}s", time.monotonic() - _t)

    merged_plot = None
    if reward_plot is not None and action_plot is not None:
        with profiler.stage("plot_merged", 2):
            _t = time.monotonic()
            merged_plot = create_merged_comparison_plot(reward_plot, action_plot, equity_curve_plot)
            logger.trace("evaluate.plot_merged elapsed={:.2f}s", time.monotonic() - _t)

    return (
        reward_plot,
        action_plot,
        None,  # action_probs_plot — PPO-specific, filled in by PPOTrainer.evaluate()
        float(result.final_reward),
        result.last_positions,
        equity_curve_plot,
        merged_plot,
    )


@contextlib.contextmanager
def _signal_guard():
    """Context manager that installs a clean SIGINT handler and restores the original on exit.

    Converts Ctrl-C into a plain KeyboardInterrupt so callers can catch it and
    save a checkpoint before re-raising.
    """
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


def _build_sync_data_collector(
    *,
    env: Any,
    actor: Any,
    config: TrainingConfig,
) -> SyncDataCollector:
    """Construct TorchRL's collector while suppressing its current deprecation noise."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="SyncDataCollector has been deprecated.*",
            category=DeprecationWarning,
        )
        return SyncDataCollector(
            create_env_fn=lambda: env,
            policy=actor,
            frames_per_batch=config.frames_per_batch,
            total_frames=config.max_steps,
        )


def _cumulative_log_returns_for_plot(
    simple_returns: np.ndarray,
    cumulative_returns: np.ndarray | None,
) -> np.ndarray:
    """Return one cumulative log-return value per plotted step."""
    series = (
        ReturnSeries(
            cumulative_returns,
            ReturnKind.CUMULATIVE_LOG,
            includes_initial=True,
        )
        if cumulative_returns is not None
        else ReturnSeries(simple_returns, ReturnKind.SIMPLE)
    )
    return series.to_cumulative_log(include_initial=False).values


class BaseTrainer(ABC):
    """Common utilities shared by RL trainers."""

    def __init__(
        self,
        actor: Any,
        value_net: Any,
        env: Any,
        config: TrainingConfig,
        *,
        eval_config: EvaluationConfig | None = None,
        enable_composite_lp: bool = False,
        checkpoint_dir: str | None = None,
        checkpoint_prefix: str | None = None,
    ):
        _patch_torchrl_trajectory_pool()
        self.actor = actor
        self.value_net = value_net
        self.env = env
        self.config = config
        self.eval_config = eval_config or EvaluationConfig()
        self.callback = None
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_prefix = checkpoint_prefix

        # Replay buffer — skipped for on-policy algorithms (e.g. PPO) that set
        # _use_replay_buffer = False before calling super().__init__().
        if getattr(self, "_use_replay_buffer", True):
            self.replay_buffer = ReplayBuffer(storage=LazyTensorStorage(config.buffer_size))
        else:
            self.replay_buffer = None
        self.collector = _build_sync_data_collector(
            env=env,
            actor=actor,
            config=config,
        )

        # Set by the pipeline after construction when val data length is known.
        # Used by _evaluate to resolve eval_fraction against actual data size.
        self._eval_data_len: int | None = None

        # Set by the pipeline after construction for checkpoint portability.
        self.n_obs: int | None = None
        self.n_act: int | None = None
        self.actor_hidden_dims: list[int] | None = None
        self.value_hidden_dims: list[int] | None = None

        # Optional dedicated evaluation environment.  When set, periodic _evaluate()
        # calls use this env instead of self.env, preventing SyncDataCollector
        # state corruption.  Set by the pipeline after construction.
        self._eval_env: Any | None = None
        self._last_evaluation_result: Any | None = None

        # Training state
        self.total_count = 0
        self.total_episodes = 0
        self.logs = defaultdict(list)
        self.checkpoint_manager = CheckpointManager(self)
        self.runtime_hooks = TrainerRuntimeHooks(self)
        self.health_monitor = TrainingHealthMonitor(
            stale_policy_min_ratio=getattr(config, "es_stale_policy_min_ratio", 0.0),
            stale_policy_window=getattr(config, "es_stale_policy_window", 20),
            saturation_max_rate=getattr(config, "es_saturation_max_rate", 0.0),
            saturation_window=getattr(config, "es_saturation_window", 20),
        )
        self.episode_stats = EpisodeStatsTracker(
            env=env,
            logs=self.logs,
            compute_exploration_ratio=self._compute_exploration_ratio,
            get_last_episode_final_nlv=self._get_last_episode_final_nlv,
            get_current_episode_context=self._get_current_episode_context,
            health_monitor=self.health_monitor,
        )

        # On-policy vs off-policy handling
        # Off-policy algorithms (TD3, DDPG) accumulate experiences in replay buffer
        # On-policy algorithms (PPO) only train on fresh data
        if not hasattr(self, "_use_replay_buffer"):
            self._use_replay_buffer = True  # Default for off-policy algorithms
        self._current_batch = None  # Stores fresh batch for on-policy algorithms

        if enable_composite_lp:
            set_composite_lp_aggregate(True).set()

    def _global_optimization_step(
        self, batch_idx: int, inner_idx: int, steps_per_batch: int
    ) -> int:
        """Compute stable global optimization step index."""
        offset = getattr(self, "_log_step_offset", 0)
        return offset + (batch_idx * steps_per_batch + inner_idx)

    def _should_log_step(self, step: int) -> bool:
        """Return True when progress logging should run at this optimization step."""
        return step % max(1, self.config.log_interval) == 0

    def _should_eval_step(self, step: int) -> bool:
        """Return True when policy evaluation should run at this optimization step."""
        interval = getattr(self.config, "eval_interval", 0)
        return interval > 0 and step % interval == 0

    @staticmethod
    @abstractmethod
    def build_models(n_obs: int, n_act: int, config: Any, env: Any):
        """Factory method that returns the actor and value/Q networks for the trainer."""

    @abstractmethod
    def _optimization_step(
        self, batch_idx: int, max_length: int, buffer_len: int
    ) -> None:
        """Run optimization for a batch."""

    @abstractmethod
    def _evaluate(self) -> None:
        """Evaluate current policy."""

    def _compute_exploration_ratio(self) -> float:
        """Algorithm-specific exploration metric."""
        return 0.0

    def _get_last_episode_final_nlv(self) -> tuple[float | None, int | None]:
        """Return (final_nlv, n_steps) of the most recently completed training episode."""
        obj = self.env
        for _ in range(10):
            if hasattr(obj, "_last_episode_final_nlv"):
                return obj._last_episode_final_nlv, getattr(obj, "_last_episode_steps", None)
            obj = getattr(obj, "_env", None) or getattr(obj, "env", None)
            if obj is None:
                break
        return None, None

    def _get_current_episode_context(self) -> tuple[str | None, str | None, str | None]:
        """Return (symbol, start_ts, end_ts) of the episode currently running in the training env."""
        obj = self.env
        for _ in range(10):
            if hasattr(obj, "_current_episode_symbol"):
                return (
                    obj._current_episode_symbol,
                    obj._current_episode_start_ts,
                    obj._current_episode_end_ts,
                )
            obj = getattr(obj, "_env", None) or getattr(obj, "env", None)
            if obj is None:
                break
        return None, None, None

    def create_action_probabilities_plot(
        self, max_steps: int, df: Any = None, config: Any = None, eval_env: Any | None = None
    ) -> Any:
        """Optional action-probability visualization; default is not implemented."""
        return None

    def _initialize_offpolicy_collection_policy(
        self,
        exploration_policy: Any,
        action_spec: Any,
        *,
        algorithm_label: str = "Off-policy",
    ) -> None:
        """Configure collector policy for off-policy warmup and resume.

        Uses random actions until ``init_rand_steps`` is reached, then switches to
        the provided exploration policy. On resume, if warmup is already complete,
        it starts directly with the exploration policy to avoid collecting an extra
        random batch.
        """
        self._offpolicy_exploration_policy = exploration_policy
        warmup_steps = int(getattr(self.config, "init_rand_steps", 0))

        if self.total_count >= warmup_steps:
            self.collector.policy = exploration_policy
            self.random_exploration_done = True
            logger.info(
                "%s random warmup already complete at %s steps; starting with exploration policy.",
                algorithm_label,
                self.total_count,
            )
            return

        from torchrl.envs.utils import RandomPolicy
        self.collector.policy = RandomPolicy(action_spec)
        self.random_exploration_done = False
        logger.info(
            "%s using random policy for first %s steps",
            algorithm_label,
            warmup_steps,
        )

    def _maybe_switch_from_random_warmup(
        self,
        *,
        algorithm_label: str = "Off-policy",
    ) -> None:
        """Switch collector from random warmup to exploration policy once ready."""
        if getattr(self, "random_exploration_done", True):
            return

        warmup_steps = int(getattr(self.config, "init_rand_steps", 0))
        if self.total_count < warmup_steps:
            return

        exploration_policy = getattr(self, "_offpolicy_exploration_policy", None)
        if exploration_policy is None:
            logger.warning(
                "%s warmup threshold reached but no exploration policy configured.",
                algorithm_label,
            )
            return

        buffer_len = len(self.replay_buffer) if getattr(self, "_use_replay_buffer", True) else 0
        logger.info(
            "%s random exploration finished at %s steps. Switching to exploration policy.",
            algorithm_label,
            self.total_count,
        )
        logger.trace("  Buffer now contains {} transitions", buffer_len)

        self.collector.policy = exploration_policy
        self.random_exploration_done = True

    # ------------------------------------------------------------------ #
    # Episode stats — delegated to EpisodeStatsTracker                   #
    # ------------------------------------------------------------------ #

    @property
    def _pending_episode_rewards(self) -> list[float]:
        return self.episode_stats._pending_rewards

    @_pending_episode_rewards.setter
    def _pending_episode_rewards(self, value: list[float]) -> None:
        self.episode_stats._pending_rewards = value

    @property
    def _pending_episode_actions(self) -> list[Any]:
        return self.episode_stats._pending_actions

    @_pending_episode_actions.setter
    def _pending_episode_actions(self, value: list[Any]) -> None:
        self.episode_stats._pending_actions = value

    def _log_episode_stats(self, data: Any, callback: Any) -> None:
        self.episode_stats.process_batch(data, callback)

    def _extract_logged_actions(
        self, actions_tensor: Any, callback: Any | None = None
    ) -> list[Any]:
        return self.episode_stats.extract_logged_actions(actions_tensor, callback)

    def _log_sample_transitions(self, data: Any, n: int = 3) -> None:
        self.episode_stats.log_sample_transitions(data, n)

    def evaluate(
        self,
        df: Any,
        max_steps: int,
        config: Any = None,
        algorithm: str | None = None,
        eval_env: Any | None = None,
    ) -> tuple[Any, ...]:
        """Run policy evaluation and return result plots and metrics."""
        return _run_evaluation(self, df, max_steps, config, algorithm, eval_env)

    def setup_periodic_evaluation(
        self,
        splits: Any,
        config: Any,
        algorithm: str,
    ) -> None:
        """Setup parameters for periodic evaluation during training.

        Call this before train() to enable temporary evaluations every N steps.

        Args:
            splits: List of SplitEvalContext objects (one per split to evaluate).
            config: Experiment configuration (must have training.temp_eval_interval set).
            algorithm: Algorithm name.
        """
        self.runtime_hooks.configure_periodic_evaluation(
            splits=splits,
            config=config,
            algorithm=algorithm,
        )

    def setup_periodic_explainability(
        self,
        df: Any,
        max_steps: int,
        config: Any,
        eval_env: Any = None,
    ) -> None:
        """Setup parameters for periodic explainability analysis during training.

        Call this before train() to enable temporary explainability every N steps.

        Args:
            df: DataFrame with evaluation data
            max_steps: Maximum steps for explainability rollout
            config: Experiment configuration (must have explainability.temp_explainability_interval set)
            eval_env: Optional dedicated evaluation environment
        """
        self.runtime_hooks.configure_periodic_explainability(
            df=df,
            max_steps=max_steps,
            config=config,
            eval_env=eval_env,
        )

    def _run_training_loop(
        self,
        callback: Any = None,
        *,
        start_message: str = "Starting training",
        completion_prefix: str = "Training complete",
        on_batch_start: Callable[[int, Any], None] | None = None,
        on_batch_end: Callable[[int, Any], None] | None = None,
        on_train_end: Callable[[], None] | None = None,
    ) -> dict[str, list]:
        """Shared training loop with optional algorithm-specific hooks."""
        log_banner(logger, f"TRAINING START  {start_message}")
        t0 = time.time()
        self.callback = callback
        self._log_step_offset = max(
            len(self.logs.get("loss_actor", [])),
            len(self.logs.get("loss_value", [])),
        )

        _profiler = get_profiler()
        with _signal_guard():
            try:
                for i, data in enumerate(self.collector):
                    if on_batch_start is not None:
                        on_batch_start(i, data)

                    self._current_batch = data

                    with _profiler.stage("buffer_extend", 2):
                        if self._use_replay_buffer:
                            self.replay_buffer.extend(data)
                            max_length = self.replay_buffer[:]["next", "step_count"].max()
                            buffer_len = len(self.replay_buffer)
                        else:
                            max_length = data["next", "step_count"].max()
                            buffer_len = data.numel()

                    self.total_count += data.numel()
                    if is_level_enabled("TRACE"):
                        logger.trace(
                            "batch collected batch=%d n_frames=%d total_count=%d buffer_len=%d",
                            i, data.numel(), self.total_count, buffer_len,
                        )

                    collected_steps = self.total_count if not self._use_replay_buffer else buffer_len
                    if collected_steps > self.config.init_rand_steps:
                        with _profiler.stage("optimization", 2):
                            self._optimization_step(i, max_length, buffer_len)

                    episodes_in_batch = int(data["next", "done"].sum().item())
                    self.total_episodes += episodes_in_batch
                    if is_level_enabled("TRACE") and episodes_in_batch > 0:
                        logger.trace(
                            "episodes completed batch=%d n_episodes=%d total_episodes=%d",
                            i, episodes_in_batch, self.total_episodes,
                        )

                    with _profiler.stage("checkpoint", 2):
                        self.checkpoint_manager.maybe_save_checkpoint()

                    with _profiler.stage("periodic_hooks", 2):
                        self.runtime_hooks.maybe_run(self.total_count)

                    if self.callback and hasattr(self.callback, "log_episode_stats"):
                        self._log_episode_stats(data, self.callback)

                    finding = self.health_monitor.check()
                    if finding is not None:
                        logger.warning(
                            "runtime guardrail %s [%s] %s | suggestion: %s",
                            finding.severity.value, finding.parameter,
                            finding.message, finding.suggestion,
                        )
                        self.logs["early_stop_reason"].append(
                            f"{finding.severity.value}:{finding.parameter}:{finding.message}"
                        )
                        break

                    if on_batch_end is not None:
                        on_batch_end(i, data)

                    if self.total_count >= self.config.max_steps:
                        logger.info("training stopped max_steps={}", self.config.max_steps)
                        break
                    if (
                        self.config.max_train_seconds is not None
                        and (time.time() - t0) >= self.config.max_train_seconds
                    ):
                        logger.info(
                            "training stopped max_train_seconds=%d elapsed=%.1fs",
                            self.config.max_train_seconds,
                            time.time() - t0,
                        )
                        break
            except KeyboardInterrupt:
                logger.warning("training interrupted by user saving checkpoint")
                checkpoint_path = self.checkpoint_manager.save_interrupt_checkpoint()
                if checkpoint_path:
                    logger.info("interrupt checkpoint saved path={}", checkpoint_path)
                raise

        if on_train_end is not None:
            on_train_end()

        t1 = time.time()
        elapsed = t1 - t0
        early_stop_reasons = self.logs.get("early_stop_reason", [])
        if early_stop_reasons:
            logger.warning(
                "training ended early reason=%s steps=%d/%d",
                early_stop_reasons[-1], self.total_count, self.config.max_steps,
            )
        log_banner(logger, f"TRAINING END  {self.total_count} steps  {self.total_episodes} episodes  {elapsed:.2f}s")
        self.logs["training_duration_s"].append(elapsed)
        return dict(self.logs)

    def train(self, callback: Any = None) -> dict[str, list]:
        """Run training loop for RL agent."""
        return self._run_training_loop(callback)
