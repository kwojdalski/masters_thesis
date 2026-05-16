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

from logger import get_logger, log_banner
from trading_rl.profiler import get_profiler
from trading_rl.config import DEFAULT_INITIAL_PORTFOLIO_VALUE, TrainingConfig
from trading_rl.constants import RewardType
from trading_rl.evaluation.returns import ReturnKind, ReturnSeries
from trading_rl.trainers.checkpointing import CheckpointManager
from trading_rl.trainers.runtime_hooks import TrainerRuntimeHooks

_MIN_BATCH_SUCCESS_RATE = 70.0  # Warn if fewer than this % of optimization batches succeed


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


# Monkey patch TorchRL to avoid torch_shm_manager in sandboxed environments
torchrl_collectors._TrajectoryPool = _LocalTrajectoryPool

logger = get_logger(__name__)


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
        enable_composite_lp: bool = False,
        checkpoint_dir: str | None = None,
        checkpoint_prefix: str | None = None,
    ):
        self.actor = actor
        self.value_net = value_net
        self.env = env
        self.config = config
        self.callback = None
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_prefix = checkpoint_prefix

        # Replay buffer and data collector shared by both algorithms
        self.replay_buffer = ReplayBuffer(storage=LazyTensorStorage(config.buffer_size))
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

        # Training state
        self.total_count = 0
        self.total_episodes = 0
        self.logs = defaultdict(list)
        self.checkpoint_manager = CheckpointManager(self)
        self.runtime_hooks = TrainerRuntimeHooks(self)

        # On-policy vs off-policy handling
        # Off-policy algorithms (TD3, DDPG) accumulate experiences in replay buffer
        # On-policy algorithms (PPO) only train on fresh data
        self._use_replay_buffer = True  # Default for off-policy algorithms
        self._current_batch = None  # Stores fresh batch for on-policy algorithms

        if enable_composite_lp:
            set_composite_lp_aggregate(True).set()

    def _maybe_save_checkpoint(self) -> None:
        self.checkpoint_manager.maybe_save_checkpoint()

    def _save_interrupt_checkpoint(self) -> str | None:
        """Persist an emergency checkpoint when training is interrupted."""
        return self.checkpoint_manager.save_interrupt_checkpoint()

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

    def create_action_probabilities_plot(
        self, max_steps: int, df: Any = None, config: Any = None
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
        logger.debug("  Buffer now contains %s transitions", buffer_len)

        self.collector.policy = exploration_policy
        self.random_exploration_done = True

    def _log_episode_stats(self, data: Any, callback: Any) -> None:
        """Log episode statistics to provided callback."""
        episode_reward = data["next", "reward"].sum().item()
        # Reset portfolio value to starting amount at the beginning of each episode
        initial_val = getattr(callback, "initial_portfolio_value", DEFAULT_INITIAL_PORTFOLIO_VALUE)
        reward_type = getattr(callback, "reward_type", "log_return")

        # Determine actual portfolio valuation based on reward type
        if reward_type == RewardType.LOG_RETURN:
            # For log_return, reward matches portfolio growth
            portfolio_valuation = initial_val * np.exp(episode_reward)
        elif reward_type == RewardType.DIFFERENTIAL_SHARPE:
            # For DSR, extract actual dollar returns from environment broker
            from trading_rl.evaluation.returns import extract_tradingenv_return_series

            # Extract the broker equity path (ignores the DSR RL reward).
            actual_returns = extract_tradingenv_return_series(self.env, data.numel())
            if actual_returns is not None and actual_returns.values.size > 0:
                portfolio_valuation = float(actual_returns.values[-1])
            else:
                # Fallback to cumulative reward if extraction fails
                logger.warning("failed to extract actual returns for dsr, falling back to reward sum")
                portfolio_valuation = initial_val * np.exp(episode_reward)
        else:
            # Fallback for any other custom reward types
            logger.debug("unrecognized reward_type=%s assuming log-return for valuation", reward_type)
            portfolio_valuation = initial_val * np.exp(episode_reward)

        callback._portfolio_value = portfolio_valuation
        actions_tensor = data.get("action")
        if isinstance(actions_tensor, torch.Tensor):
            if actions_tensor.ndim > 1 and actions_tensor.shape[-1] > 1:
                actions = actions_tensor.argmax(dim=-1).reshape(-1).tolist()
            else:
                actions = actions_tensor.reshape(-1).tolist()
        else:
            actions = []

        exploration_ratio = self._compute_exploration_ratio()

        callback.log_episode_stats(
            episode_reward=episode_reward,
            portfolio_valuation=portfolio_valuation,
            actions=actions,
            exploration_ratio=exploration_ratio,
        )

        # Log episode completion metrics (replaces gym_trading_env verbose output)
        if hasattr(callback, "_episode_count"):
            callback._episode_count += 1
        else:
            callback._episode_count = 1
        self.logs["episode_log_count"].append(int(callback._episode_count))

        # Calculate portfolio return in percent (standardized formula)
        # Use initial_val from callback and portfolio_valuation calculated above
        portfolio_return = 100 * (portfolio_valuation / initial_val - 1)

        logger.info(
            "n_episode=%d portfolio_return_pct=%.2f portfolio_value=%.2f",
            callback._episode_count, portfolio_return, portfolio_valuation,
        )

    def _log_sample_transitions(self, data: Any, n: int = 3) -> None:
        """Log n sample (s, a, r, s') tuples from a collected batch at DEBUG level."""
        import numpy as np
        try:
            size = data.numel()
            indices = np.linspace(0, size - 1, min(n, size), dtype=int)
            for idx in indices:
                td = data[int(idx)]
                obs = td.get("observation")
                next_obs = td.get(("next", "observation"))
                action = td.get("action")
                reward = td.get(("next", "reward"))

                def _fmt(t):
                    if t is None:
                        return "?"
                    arr = t.reshape(-1).cpu().tolist()
                    return "[" + ", ".join(f"{v:.4f}" for v in arr) + "]"

                logger.debug(
                    "transition idx=%d  s=%s  a=%s  r=%.6f  s'=%s",
                    idx, _fmt(obs), _fmt(action),
                    float(reward) if reward is not None else float("nan"),
                    _fmt(next_obs),
                )
        except Exception as exc:
            logger.debug("_log_sample_transitions failed: %s", exc)

    def _is_portfolio_backend(self, config: Any) -> bool:
        """Detect if backend uses continuous portfolio weights.

        Args:
            config: Experiment configuration

        Returns:
            True if backend is portfolio-based (continuous weights), False otherwise
        """
        if config is None:
            return False
        backend = getattr(config.env, "backend", None)
        return backend == "tradingenv"

    def _extract_actions(self, rollout: Any, is_portfolio: bool) -> Any:
        """Extract actions from rollout based on backend type.

        Args:
            rollout: TensorDict containing rollout data
            is_portfolio: Whether backend uses portfolio weights

        Returns:
            Tensor of actions (continuous weights or discrete positions)
        """
        action_tensor = rollout["action"].squeeze()

        if is_portfolio:
            # Portfolio weights: keep continuous values as-is
            if action_tensor.ndim > 1:
                # Multi-asset: return all weights [batch, n_assets]
                return action_tensor
            else:
                # Single asset: return weights [batch]
                return action_tensor
        else:
            # Discrete positions: convert to position indices
            if action_tensor.ndim > 1 and action_tensor.shape[-1] > 1:
                # One-hot encoded -> argmax
                return action_tensor.argmax(dim=-1)
            else:
                # Already discrete
                return action_tensor

    def evaluate(
        self,
        df: Any,
        max_steps: int,
        config: Any = None,
        algorithm: str | None = None,
        eval_env: Any | None = None,
    ) -> tuple[Any, ...]:
        """Delegated evaluation - creates plots and results using StrategyEvaluator.

        This method now delegates to StrategyEvaluator for evaluation logic,
        decoupling training from evaluation. The original implementation
        remains for backward compatibility during transition.

        Args:
            df: Price/OHLCV data for evaluation
            max_steps: Maximum steps to evaluate
            config: Experiment configuration (optional, uses default if None)
            algorithm: Algorithm name (for logging only)
            eval_env: Optional dedicated evaluation environment

        Returns:
            Tuple of (reward_plot, action_plot, action_probs_plot, final_reward,
                     last_positions, actual_returns_plot, merged_plot)
        """
        from trading_rl.config import DEFAULT_INITIAL_PORTFOLIO_VALUE
        from trading_rl.utils import (
            create_actual_returns_plot,
            create_merged_comparison_plot,
        )

        env_to_use = eval_env or self.env

        # Build evaluation config from training config (or use defaults)
        eval_config_kwargs = {}
        if config:
            eval_config_kwargs = {
                "reward_type": getattr(config.env, "reward_type", "log_return"),
                "backend": getattr(config.env, "backend", "tradingenv"),
                "price_column": getattr(config.env, "price_column", None),
                "max_steps": max_steps,
                "enable_plots": True,
                "enable_metrics": False,  # Metrics computed separately
            }

        from trading_rl.evaluation.evaluator import EvaluationConfig, StrategyEvaluator

        # Add environment configuration to eval_config_kwargs
        if config:
            from trading_rl.evaluation.evaluator import EnvConfig

            eval_config_kwargs["env"] = EnvConfig(
                name=getattr(config.env, "name", ""),
                positions=getattr(config.env, "positions", None),
                mode=getattr(config.env, "mode", "mft"),
                trading_fees=getattr(config.env, "trading_fees", 0.0),
                borrow_interest_rate=getattr(config.env, "borrow_interest_rate", 0.0),
                initial_portfolio_value=getattr(config.env, "initial_portfolio_value", DEFAULT_INITIAL_PORTFOLIO_VALUE),
                price_column=getattr(config.env, "price_column", "close"),
            )

        eval_config = EvaluationConfig(**eval_config_kwargs)

        def env_factory(_df: Any, _config: Any) -> Any:
            return env_to_use

        profiler = get_profiler()

        evaluator = StrategyEvaluator(
            env_factory=env_factory,
            policy=self.actor,
            config=eval_config,
        )

        with profiler.stage("agent_rollout", 2):
            result = evaluator.evaluate_split("eval", df, env=env_to_use)

        reward_plot = result.plots["reward_plot"] if result.plots else None
        action_plot = result.plots["action_plot"] if result.plots else None
        plot_series = result.return_series or ReturnSeries(
            result.simple_returns,
            ReturnKind.SIMPLE,
        )

        with profiler.stage("plot_actual_returns", 2):
            _t = time.monotonic()
            logger.debug("create_actual_returns_plot start n_steps=%d", max_steps)
            actual_returns_plot = create_actual_returns_plot(
                None,
                max_steps,
                df_prices=df,
                env=env_to_use,
                actual_returns_list=[plot_series],
                initial_portfolio_value=(
                    float(getattr(config.env, "initial_portfolio_value", DEFAULT_INITIAL_PORTFOLIO_VALUE))
                    if config
                    else DEFAULT_INITIAL_PORTFOLIO_VALUE
                ),
                benchmark_price_column=getattr(config.env, "price_column", None) if config else "close",
                show_max_profit=config.benchmarks.show_max_profit if config else True,
                training_steps=self.total_count,
                training_episodes=self.total_episodes,
            )
            logger.debug("create_actual_returns_plot done elapsed=%.2fs", time.monotonic() - _t)

        with profiler.stage("plot_merged", 2):
            _t = time.monotonic()
            logger.debug("create_merged_comparison_plot start")
            merged_plot = create_merged_comparison_plot(reward_plot, action_plot)
            logger.debug("create_merged_comparison_plot done elapsed=%.2fs", time.monotonic() - _t)

        # Use final_reward and last_positions from SplitEvaluationResult
        final_reward = float(result.final_reward)
        last_positions = result.last_positions

        return (
            reward_plot,
            action_plot,
            None,  # Third plot (action_probs_plot) - PPO-specific
            final_reward,
            last_positions,
            actual_returns_plot,
            merged_plot,
        )

    def setup_periodic_evaluation(
        self,
        df: Any,
        max_steps: int,
        config: Any,
        algorithm: str,
        eval_env: Any = None,
    ) -> None:
        """Setup parameters for periodic evaluation during training.

        Call this before train() to enable temporary evaluations every N steps.

        Args:
            df: DataFrame with evaluation data
            max_steps: Maximum steps for evaluation rollout
            config: Experiment configuration (must have training.temp_eval_interval set)
            algorithm: Algorithm name
            eval_env: Optional dedicated evaluation environment
        """
        self.runtime_hooks.configure_periodic_evaluation(
            df=df,
            max_steps=max_steps,
            config=config,
            algorithm=algorithm,
            eval_env=eval_env,
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

        # Set up explicit signal handling to ensure interrupts are processed
        original_sigint_handler = signal.getsignal(signal.SIGINT)

        def signal_handler(sig, frame):
            logger.info("sigint received raising KeyboardInterrupt")
            signal.signal(signal.SIGINT, original_sigint_handler)
            raise KeyboardInterrupt()

        signal.signal(signal.SIGINT, signal_handler)

        _profiler = get_profiler()
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

                collected_steps = self.total_count if not self._use_replay_buffer else buffer_len
                if collected_steps > self.config.init_rand_steps:
                    with _profiler.stage("optimization", 2):
                        self._optimization_step(i, max_length, buffer_len)

                episodes_in_batch = int(data["next", "done"].sum().item())
                self.total_episodes += episodes_in_batch

                with _profiler.stage("checkpoint", 2):
                    self._maybe_save_checkpoint()

                with _profiler.stage("periodic_hooks", 2):
                    self.runtime_hooks.maybe_run(self.total_count)

                if self.callback and hasattr(self.callback, "log_episode_stats") and episodes_in_batch > 0:
                    self._log_episode_stats(data, self.callback)

                if on_batch_end is not None:
                    on_batch_end(i, data)

                if self.total_count >= self.config.max_steps:
                    logger.info("training stopped max_steps=%d", self.config.max_steps)
                    break
        except KeyboardInterrupt:
            logger.warning("training interrupted by user saving checkpoint")
            checkpoint_path = self._save_interrupt_checkpoint()
            if checkpoint_path:
                logger.info("interrupt checkpoint saved path=%s", checkpoint_path)
            raise
        finally:
            signal.signal(signal.SIGINT, original_sigint_handler)

        if on_train_end is not None:
            on_train_end()

        t1 = time.time()
        elapsed = t1 - t0
        log_banner(logger, f"TRAINING END  {self.total_count} steps  {self.total_episodes} episodes  {elapsed:.2f}s")
        self.logs["training_duration_s"].append(elapsed)
        return dict(self.logs)

    def train(self, callback: Any = None) -> dict[str, list]:
        """Run training loop for RL agent."""
        return self._run_training_loop(callback)
