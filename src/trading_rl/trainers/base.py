"""Base trainer and utilities."""

import contextlib
import time
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.multiprocessing as mp
import torchrl.collectors.collectors as torchrl_collectors
from tensordict.nn import InteractionType, set_composite_lp_aggregate
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.envs.utils import set_exploration_type

from logger import get_logger, is_level_enabled
from trading_rl.config import EvaluationConfig, TrainingConfig
from trading_rl.constants import BenchmarkName
from trading_rl.evaluation.benchmarks import benchmarks_from_config
from trading_rl.evaluation.returns import ReturnKind, ReturnSeries
from trading_rl.profiler import get_profiler
from trading_rl.trainers.checkpointing import CheckpointManager, TrainingCheckpoint
from trading_rl.trainers.episode_stats import EpisodeStatsTracker
from trading_rl.trainers.health_monitor import TrainingHealthMonitor
from trading_rl.trainers.runtime_hooks import TrainerRuntimeHooks
from trading_rl.trainers.training_loop import TrainingLoop
from trading_rl.trainers.warmup import WarmupController

_MIN_BATCH_SUCCESS_RATE = (
    70.0  # Warn if fewer than this % of optimization batches succeed
)
_MAX_CONSECUTIVE_SKIPPED_BATCHES = 10


@dataclass(frozen=True)
class EvaluationOutput:
    """Evaluation return value with tuple-compatible plot/metric payload."""

    reward_plot: Any
    action_plot: Any
    action_probs_plot: Any
    final_reward: float
    last_positions: Any
    equity_curve_plot: Any
    merged_plot: Any
    result: Any

    def __iter__(self):
        yield self.reward_plot
        yield self.action_plot
        yield self.action_probs_plot
        yield self.final_reward
        yield self.last_positions
        yield self.equity_curve_plot
        yield self.merged_plot


def _log_network_stats(
    log, algo: str, actor: torch.nn.Module, critic: torch.nn.Module
) -> None:
    """Emit a TRACE line with parameter and gradient statistics for actor and critic."""

    def _stats(net: torch.nn.Module) -> tuple[float, float, float, int]:
        params = list(net.parameters())
        abs_sum = sum(p.detach().abs().sum().item() for p in params)
        norm = sum(p.detach().pow(2).sum().item() for p in params) ** 0.5
        grad_norm = (
            sum(
                p.grad.detach().pow(2).sum().item()
                for p in params
                if p.grad is not None
            )
            ** 0.5
        )
        n = sum(p.numel() for p in params)
        return abs_sum, norm, grad_norm, n

    a_abs, a_norm, a_gnorm, a_n = _stats(actor)
    c_abs, c_norm, c_gnorm, c_n = _stats(critic)
    log.trace(
        "{} network_stats "
        "actor_abs_sum=%.4f actor_norm=%.4f actor_grad_norm=%.4f actor_n_params=%d "
        "critic_abs_sum=%.4f critic_norm=%.4f critic_grad_norm=%.4f critic_n_params=%d",
        algo,
        a_abs,
        a_norm,
        a_gnorm,
        a_n,
        c_abs,
        c_norm,
        c_gnorm,
        c_n,
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
        meta.update(
            {
                "run_id": run.info.run_id,
                "run_name": run.data.tags.get("mlflow.runName"),
                "experiment_id": run.info.experiment_id,
                "experiment_name": experiment.name if experiment else None,
            }
        )
        return meta
    except Exception:
        logger.opt(exception=True).warning(
            "_collect_mlflow_meta failed; checkpoint will have no mlflow metadata"
        )
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
        if (
            hasattr(_mod, "_TrajectoryPool")
            and _mod._TrajectoryPool is not _LocalTrajectoryPool
        ):
            _mod._TrajectoryPool = _LocalTrajectoryPool
            patched = True
    if patched:
        logger.debug("patched torchrl _TrajectoryPool -> _LocalTrajectoryPool")


def _run_evaluation(
    trainer: Any,
    df: Any,
    max_steps: int,
    config: Any = None,
    algorithm: str | None = None,
    eval_env: Any | None = None,
) -> EvaluationOutput:
    """Run a policy evaluation rollout and build result plots.

    Extracted from BaseTrainer.evaluate() so the logic is testable without
    subclassing and so subclass overrides remain thin.

    Returns:
        (reward_plot, action_plot, None, final_reward, last_positions,
         equity_curve_plot, merged_plot)
    """
    from trading_rl.config import DEFAULT_INITIAL_PORTFOLIO_VALUE
    from trading_rl.evaluation.evaluator import (
        StrategyEvaluator,
        StrategyEvaluatorConfig,
    )
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
            "training_episodes": int(trainer.total_episodes)
            if trainer is not None
            else None,
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
    logger.trace("evaluate.rollout_and_metrics elapsed_s={:.2f}", time.monotonic() - _t)

    _enabled_plots = set(eval_config.eval_plots)
    reward_plot = result.plots.get("reward_plot") if result.plots else None
    action_plot = result.plots.get("action_plot") if result.plots else None

    equity_curve_plot = None
    if "portfolio_value" in _enabled_plots:
        with profiler.stage("plot_equity_curve", 2):
            _t = time.monotonic()
            logger.trace("create_equity_curve_plot start n_steps={}", max_steps)
            _data_paths = config.data.data_paths if config else None
            plot_series = result.return_series or ReturnSeries(
                result.simple_returns, ReturnKind.SIMPLE
            )
            equity_curve_plot = create_equity_curve_plot(
                None,
                max_steps,
                df_prices=df,
                env=env_to_use,
                actual_returns_list=[plot_series],
                initial_portfolio_value=(
                    float(config.env.initial_portfolio_value)
                    if config
                    else DEFAULT_INITIAL_PORTFOLIO_VALUE
                ),
                benchmark_price_column=config.env.price_column or "close"
                if config
                else "close",
                benchmarks=benchmarks_from_config(config.benchmarks)
                if config
                else frozenset({BenchmarkName.BUY_AND_HOLD}),
                training_steps=trainer.total_count,
                training_episodes=trainer.total_episodes,
                n_total_symbols=len(_data_paths) if _data_paths else None,
                max_plot_points=config.training.max_plot_points if config else None,
                reward_type=str(config.env.reward_type) if config else None,
            )
            logger.trace(
                "evaluate.plot_equity_curve elapsed_s={:.2f}", time.monotonic() - _t
            )

    merged_plot = None
    if reward_plot is not None and action_plot is not None:
        with profiler.stage("plot_merged", 2):
            _t = time.monotonic()
            merged_plot = create_merged_comparison_plot(
                reward_plot, action_plot, equity_curve_plot
            )
            logger.trace("evaluate.plot_merged elapsed_s={:.2f}", time.monotonic() - _t)

    return EvaluationOutput(
        reward_plot=reward_plot,
        action_plot=action_plot,
        action_probs_plot=None,
        final_reward=float(result.final_reward),
        last_positions=result.last_positions,
        equity_curve_plot=equity_curve_plot,
        merged_plot=merged_plot,
        result=result,
    )


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
        n_obs: int,
        n_act: int,
        actor_hidden_dims: list[int],
        value_hidden_dims: list[int],
        eval_config: EvaluationConfig | None = None,
        eval_env: Any | None = None,
        eval_data_len: int | None = None,
        enable_composite_lp: bool = False,
        checkpoint_dir: str | None = None,
        checkpoint_prefix: str | None = None,
        use_replay_buffer: bool = True,
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

        self.n_obs = n_obs
        self.n_act = n_act
        self.actor_hidden_dims = actor_hidden_dims
        self.value_hidden_dims = value_hidden_dims
        self._eval_env = eval_env
        self._eval_data_len = eval_data_len

        self._use_replay_buffer = use_replay_buffer
        self.replay_buffer = (
            ReplayBuffer(storage=LazyTensorStorage(config.buffer_size))
            if use_replay_buffer
            else None
        )
        self.collector = _build_sync_data_collector(
            env=env,
            actor=actor,
            config=config,
        )

        # Training state
        self.total_count = 0
        self.total_episodes = 0
        self._replay_buffer_max_step_count = 0
        self.logs = defaultdict(list)
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            checkpoint_prefix=checkpoint_prefix,
            interval=getattr(config, "checkpoint_interval", 0),
            save_buffer=getattr(config, "save_buffer", False),
        )
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
        self.warmup_controller = WarmupController(
            collector=self.collector,
            init_rand_steps=int(getattr(config, "init_rand_steps", 0)),
            replay_buffer=self.replay_buffer,
            use_replay_buffer=use_replay_buffer,
        )
        self._current_batch = None

        if enable_composite_lp:
            set_composite_lp_aggregate(True).set()

        # Counters used by off-policy trainers; harmless for on-policy ones.
        self.successful_batches = 0
        self.skipped_batches = 0
        self._consecutive_skips = 0

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

    @property
    @abstractmethod
    def _algo_label(self) -> str:
        """Short algorithm identifier used in log messages and checkpoints."""

    @property
    def _value_loss_key(self) -> str:
        """Key for the value/critic loss in the loss_vals dict. DDPG overrides to 'loss_value'."""
        return "loss_qvalue"

    @abstractmethod
    def _get_checkpoint_network_state(self) -> dict:
        """Return algorithm-specific entries to merge into the checkpoint dict.

        Must include at minimum ``actor_params_state``, ``value_params_state``,
        and all optimizer state dicts.  Algorithm-specific extras (e.g. SAC's
        ``log_alpha``, DDPG's target params) go here too.
        """

    @abstractmethod
    def _load_checkpoint_network_state(self, checkpoint: dict) -> None:
        """Restore the algorithm-specific network and optimizer states from *checkpoint*."""

    # ------------------------------------------------------------------
    # Shared off-policy utilities (harmless stubs for on-policy trainers)
    # ------------------------------------------------------------------

    def _record_skipped_batch(
        self, reason: str, exc: RuntimeError | None = None
    ) -> None:
        """Track skipped optimization batches and raise after too many consecutive failures."""
        self.skipped_batches += 1
        self._consecutive_skips += 1
        if exc is None:
            logger.warning("{} skipping batch reason={}", self._algo_label, reason)
        else:
            logger.warning(
                "{} skipping batch reason={} err={}", self._algo_label, reason, exc
            )
        if (
            self._consecutive_skips >= _MAX_CONSECUTIVE_SKIPPED_BATCHES
            and self.successful_batches == 0
        ):
            error = RuntimeError(
                f"{self._algo_label.upper()}: {self._consecutive_skips} consecutive optimization "
                "batches skipped with zero successful updates. Training cannot proceed — "
                "check environment or replay buffer tensor shapes."
            )
            if exc is not None:
                raise error from exc
            raise error

    def _log_batch_summary(self) -> None:
        """Log successful vs skipped optimization batch counts at training end."""
        total = self.successful_batches + self.skipped_batches
        if total > 0:
            rate = (self.successful_batches / total) * 100
            _log = logger.warning if rate < _MIN_BATCH_SUCCESS_RATE else logger.info
            _log(
                "{} batch summary successful={}/{} success_rate={:.1f}% skipped={}",
                self._algo_label,
                self.successful_batches,
                total,
                rate,
                self.skipped_batches,
            )
        else:
            logger.warning("{} no optimization batches attempted", self._algo_label)

    def _evaluate(self) -> None:
        """Periodic policy evaluation shared by DDPG, TD3, and SAC.

        Runs a deterministic rollout on the dedicated eval env, logs scalar
        metrics, and calls ``_post_eval_trace_hook`` for algorithm-specific
        TRACE logging (e.g. TD3 action-distribution stats).
        """
        with set_exploration_type(InteractionType.DETERMINISTIC), torch.no_grad():
            n_eval = (
                self.eval_config.resolve_eval_steps(self._eval_data_len)
                if self._eval_data_len is not None
                else self.eval_config.eval_steps
            )
            if self._eval_env is None:
                logger.warning(
                    "{} _evaluate: no dedicated eval env set; skipping periodic eval "
                    "to avoid corrupting SyncDataCollector state",
                    self._algo_label,
                )
                return
            eval_rollout = self._eval_env.rollout(n_eval, self.actor)

            mean_reward = eval_rollout["next", "reward"].mean().item()
            sum_reward = eval_rollout["next", "reward"].sum().item()
            max_steps = eval_rollout["step_count"].max().item()

            self.logs["eval_reward_mean"].append(mean_reward)
            self.logs["eval_reward_sum"].append(sum_reward)
            self.logs["eval_step_count"].append(max_steps)

            self._post_eval_trace_hook(eval_rollout)

            eval_data_len = (
                self._eval_data_len if self._eval_data_len is not None else "?"
            )
            logger.info(
                "{} eval mean_reward={:.4f} sum_reward={:.4f} eval_steps={} eval_data_len={}",
                self._algo_label,
                mean_reward,
                sum_reward,
                max_steps,
                eval_data_len,
            )
            del eval_rollout

    def _post_eval_trace_hook(self, eval_rollout: Any) -> None:  # noqa: B027 -- intentionally optional, not required for every trainer
        """Optional TRACE-level hook called after each periodic eval rollout."""

    def _log_progress(
        self,
        max_length: int,
        buffer_len: int,
        loss_vals: dict,
        log_actor: bool = True,
        actor_loss: float | None = None,
        value_loss: float | None = None,
    ) -> None:
        """Log one optimization step; used by DDPG and TD3. SAC overrides with extra fields.

        ``actor_loss``/``value_loss``, when given, override the corresponding
        ``loss_vals`` entry — TD3/DDPG each run two forward passes (critic update,
        then actor update against the now-updated critic), so a single ``loss_vals``
        dict never holds the fresh value for both losses at once.
        """
        curr_loss_value = (
            value_loss
            if value_loss is not None
            else loss_vals[self._value_loss_key].item()
        )
        if log_actor:
            curr_loss_actor = (
                actor_loss if actor_loss is not None else loss_vals["loss_actor"].item()
            )
            logger.info(
                "{} step max_steps={} buffer_size={} loss_value={:.4f} loss_actor={:.4f}",
                self._algo_label,
                max_length,
                buffer_len,
                curr_loss_value,
                curr_loss_actor,
            )
        else:
            logger.info(
                "{} step max_steps={} buffer_size={} loss_value={:.4f}",
                self._algo_label,
                max_length,
                buffer_len,
                curr_loss_value,
            )
        if is_level_enabled("TRACE"):
            _log_network_stats(logger, self._algo_label, self.actor, self.value_net)

    # ------------------------------------------------------------------
    # Shared checkpoint save / load
    # ------------------------------------------------------------------

    def _snapshot(
        self,
        feature_pipeline_state: dict[str, dict[str, float]] | None = None,
        mlflow_meta: dict | None = None,
    ) -> TrainingCheckpoint:
        """Assemble current training state into a portable snapshot."""
        from trading_rl.models import _extract_action_bounds_from_spec

        if mlflow_meta is None:
            mlflow_meta = _collect_mlflow_meta()
        _env = getattr(self, "env", None)
        _bounds = _extract_action_bounds_from_spec(getattr(_env, "action_spec", None))
        return TrainingCheckpoint(
            algorithm=self._algo_label,
            n_obs=self.n_obs,
            n_act=self.n_act,
            actor_hidden_dims=self.actor_hidden_dims,
            value_hidden_dims=self.value_hidden_dims,
            action_low=_bounds[0].tolist() if _bounds is not None else None,
            action_high=_bounds[1].tolist() if _bounds is not None else None,
            actor_state_dict=self.actor.state_dict(),
            value_net_state_dict=self.value_net.state_dict(),
            total_count=self.total_count,
            total_episodes=self.total_episodes,
            replay_buffer_max_step_count=int(self._replay_buffer_max_step_count),
            episode_log_count=(
                int(self.logs.get("episode_log_count", [0])[-1])
                if self.logs.get("episode_log_count")
                else 0
            ),
            logs=dict(self.logs),
            mlflow_run_id=mlflow_meta.get("run_id"),
            mlflow_run_name=mlflow_meta.get("run_name"),
            mlflow_tracking_uri=mlflow_meta.get("tracking_uri"),
            mlflow_experiment_id=mlflow_meta.get("experiment_id"),
            mlflow_experiment_name=mlflow_meta.get("experiment_name"),
            feature_pipeline_state=feature_pipeline_state,
            network_state=self._get_checkpoint_network_state(),
        )

    def _restore(self, checkpoint: TrainingCheckpoint) -> None:
        """Restore training state from a snapshot."""
        self._load_checkpoint_network_state(checkpoint.network_state)

        # Backward-compat: restore visible module weights when present.
        if checkpoint.actor_state_dict:
            self.actor.load_state_dict(checkpoint.actor_state_dict)
        if checkpoint.value_net_state_dict:
            self.value_net.load_state_dict(checkpoint.value_net_state_dict)

        self.total_count = checkpoint.total_count
        self.total_episodes = checkpoint.total_episodes
        self._replay_buffer_max_step_count = checkpoint.replay_buffer_max_step_count
        self.logs = defaultdict(list, checkpoint.logs)
        self.mlflow_run_id = checkpoint.mlflow_run_id
        self.mlflow_run_name = checkpoint.mlflow_run_name
        self.mlflow_tracking_uri = checkpoint.mlflow_tracking_uri
        self.mlflow_experiment_id = checkpoint.mlflow_experiment_id
        self.mlflow_experiment_name = checkpoint.mlflow_experiment_name

    def save_checkpoint(
        self,
        path: str,
        feature_pipeline_state: dict[str, dict[str, float]] | None = None,
        mlflow_meta: dict | None = None,
    ) -> None:
        """Save a training checkpoint."""
        from pathlib import Path

        snapshot = self._snapshot(feature_pipeline_state, mlflow_meta)

        if (
            getattr(self, "replay_buffer", None) is not None
            and self.checkpoint_manager.save_buffer
        ):
            path_obj = Path(path)
            buffer_dir = path_obj.with_suffix("").with_name(f"{path_obj.stem}_buffer")
            try:
                self.replay_buffer.dumps(buffer_dir)
            except Exception:
                logger.exception(
                    "failed to save replay buffer; checkpoint will not include buffer"
                )
            else:
                snapshot.replay_buffer_path = str(buffer_dir)
                snapshot.buffer_metadata = {
                    "buffer_size": len(self.replay_buffer),
                    "max_size": self.replay_buffer._storage.max_size,
                }
                logger.info(
                    "save replay buffer path={} n_experiences={}",
                    buffer_dir,
                    len(self.replay_buffer),
                )

        self.checkpoint_manager.save(path, snapshot)

    def load_checkpoint(self, path: str) -> None:
        """Load a training checkpoint."""
        from pathlib import Path

        checkpoint = self.checkpoint_manager.load(path)

        if is_level_enabled("TRACE"):
            logger.trace(
                "{} checkpoint algorithm={}", self._algo_label, checkpoint.algorithm
            )

        self._restore(checkpoint)

        replay_buffer = getattr(self, "replay_buffer", None)
        if replay_buffer is not None:
            buffer_path = checkpoint.replay_buffer_path
            if buffer_path and Path(buffer_path).exists():
                try:
                    replay_buffer.loads(buffer_path)
                    logger.info(
                        "load replay buffer path={} n_experiences={}",
                        buffer_path,
                        len(replay_buffer),
                    )
                except Exception:
                    logger.exception(
                        "Failed to load replay buffer from {}", buffer_path
                    )
            else:
                logger.info("no replay buffer in checkpoint start_fresh=true")

        logger.debug(
            "load checkpoint state total_count={} total_episodes={} mlflow_run_id={} "
            "mlflow_run_name={} experiment={}",
            self.total_count,
            self.total_episodes,
            self.mlflow_run_id,
            self.mlflow_run_name,
            self.mlflow_experiment_name,
        )
        logger.info("load checkpoint path={}", path)

    def _compute_exploration_ratio(self) -> float:
        """Algorithm-specific exploration metric."""
        return 0.0

    def _get_last_episode_final_nlv(self) -> tuple[float | None, int | None]:
        """Pop and return (final_nlv, n_steps) of the next unconsumed completed
        training episode, in completion order.

        Backed by a FIFO queue rather than a single scalar so that a collector
        batch spanning multiple episode boundaries matches each completed
        episode to its own NLV instead of always the most recent one.
        """
        obj = self.env
        for _ in range(10):
            queue = getattr(obj, "_episode_final_nlv_queue", None)
            if queue is not None:
                return queue.pop(0) if queue else (None, None)
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
        self,
        max_steps: int,
        df: Any = None,
        config: Any = None,
        eval_env: Any | None = None,
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
        """Delegate to WarmupController."""
        self.warmup_controller.initialize(
            exploration_policy,
            action_spec,
            total_count=self.total_count,
            algorithm_label=algorithm_label,
        )

    def _maybe_switch_from_random_warmup(
        self,
        *,
        algorithm_label: str = "Off-policy",
    ) -> None:
        """Delegate to WarmupController."""
        self.warmup_controller.maybe_switch(
            self.total_count, algorithm_label=algorithm_label
        )

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
        """Delegate to TrainingLoop."""
        return TrainingLoop().run(
            self,
            callback,
            start_message=start_message,
            completion_prefix=completion_prefix,
            on_batch_start=on_batch_start,
            on_batch_end=on_batch_end,
            on_train_end=on_train_end,
        )

    def train(self, callback: Any = None) -> dict[str, list]:
        """Run training loop for RL agent."""
        return self._run_training_loop(callback)
