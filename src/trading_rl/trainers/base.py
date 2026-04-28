"""Base trainer and utilities."""

import contextlib
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
from tensordict.nn import InteractionType, set_composite_lp_aggregate
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.envs.utils import set_exploration_type

from logger import get_logger
from trading_rl.config import DEFAULT_INITIAL_PORTFOLIO_VALUE, TrainingConfig
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
        if reward_type == "log_return":
            # For log_return, reward matches portfolio growth
            portfolio_valuation = initial_val * np.exp(episode_reward)
        elif reward_type == "differential_sharpe":
            # For DSR, extract actual dollar returns from environment broker
            from trading_rl.evaluation.returns import extract_tradingenv_returns

            # Extract cumulative returns from broker (ignores the DSR RL reward)
            actual_returns = extract_tradingenv_returns(self.env, data.numel())
            if actual_returns is not None and len(actual_returns) > 0:
                portfolio_valuation = initial_val * np.exp(actual_returns[-1])
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

        # Compute buy-and-hold benchmark if price series is available on the callback
        market_return = None
        price_series = getattr(callback, "price_series", None)
        episode_length = len(actions)
        if price_series is not None and episode_length > 0:
            end_idx = min(episode_length, len(price_series) - 1)
            start_price = price_series.iloc[0]
            end_price = price_series.iloc[end_idx]
            market_return = 100 * (end_price / start_price - 1)

        logger.info(
            f"Episode {callback._episode_count} complete | "
            f"Market Return: {market_return:5.2f}% | "
            f"Portfolio Return: {portfolio_return:5.2f}% | "
            f"Portfolio Value: ${portfolio_valuation:,.2f}"
        )

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
        """Default evaluation: deterministic vs random rollout comparison."""
        import numpy as np
        import pandas as pd
        from plotnine import (
            aes,
            element_text,
            geom_line,
            ggplot,
            guide_legend,
            guides,
            labs,
            scale_color_manual,
            theme,
        )

        from trading_rl.utils import (
            compare_rollouts,
            create_actual_returns_plot,
            create_merged_comparison_plot,
        )

        logger = get_logger(__name__)

        env_to_use = eval_env or self.env

        # Deterministic rollout
        logger.debug("eval deterministic max_steps=%d", max_steps)
        try:
            with set_exploration_type(InteractionType.MODE):
                rollout_deterministic = env_to_use.rollout(
                    max_steps=max_steps, policy=self.actor
                )
        except RuntimeError:
            # Fallback for distributions without analytical mode (e.g. TanhNormal)
            logger.debug("mode exploration failed, falling back to mean/deterministic")
            with set_exploration_type(InteractionType.DETERMINISTIC):
                rollout_deterministic = env_to_use.rollout(
                    max_steps=max_steps, policy=self.actor
                )

        # Extract actual returns immediately (before next rollout overwrites broker state)
        from trading_rl.evaluation.returns import extract_tradingenv_returns

        actual_returns_deterministic = extract_tradingenv_returns(env_to_use, max_steps)

        # Random rollout (can be overridden in subclasses)
        logger.debug("eval random max_steps=%d", max_steps)
        with set_exploration_type(InteractionType.RANDOM):
            rollout_random = env_to_use.rollout(max_steps=max_steps, policy=self.actor)

        # Extract actual returns immediately (for random rollout)
        actual_returns_random = extract_tradingenv_returns(env_to_use, max_steps)

        # Detect backend type for proper plot labeling
        is_portfolio = self._is_portfolio_backend(config)

        reward_plot, action_plot = compare_rollouts(
            [rollout_deterministic, rollout_random],
            n_obs=max_steps,
            is_portfolio=is_portfolio,
        )

        benchmark_price_column = "close"
        if config:
            configured_price_column = getattr(config.env, "price_column", None)
            if isinstance(configured_price_column, str) and configured_price_column:
                benchmark_price_column = configured_price_column

        if benchmark_price_column in df.columns:
            benchmark_series = df[benchmark_price_column]
        elif "close" in df.columns:
            logger.warning(
                "Benchmark column '%s' missing in evaluation frame; falling back to 'close'.",
                benchmark_price_column,
            )
            benchmark_series = df["close"]
            benchmark_price_column = "close"
        else:
            raise ValueError(
                "Evaluation benchmarks require env.price_column or 'close' in dataframe."
            )

        # Create actual returns plot with pre-extracted returns
        actual_returns_plot = create_actual_returns_plot(
            [rollout_deterministic, rollout_random],
            n_obs=max_steps,
            df_prices=df,
            env=None,  # Don't pass env, use pre-extracted returns
            actual_returns_list=[actual_returns_deterministic, actual_returns_random],
            initial_portfolio_value=(
                float(getattr(config.env, "initial_portfolio_value", DEFAULT_INITIAL_PORTFOLIO_VALUE))
                if config
                else DEFAULT_INITIAL_PORTFOLIO_VALUE
            ),
            benchmark_price_column=benchmark_price_column,
        )

        # Add benchmarks based on reward type
        reward_type = getattr(config.env, "reward_type", "log_return") if config else "log_return"
        benchmark_data = []

        if reward_type == "differential_sharpe":
            # For DSR, calculate DSR benchmarks
            from trading_rl.utils import calculate_benchmark_dsr

            # Get DSR parameters from config (reward_eta is the standard name)
            dsr_eta = getattr(config.env, "reward_eta", 0.01) if config else 0.01

            logger.debug("calc dsr benchmarks eta=%s", dsr_eta)

            # Calculate DSR for buy-and-hold
            bh_dsr, _ = calculate_benchmark_dsr(
                df,
                strategy="buy_and_hold",
                eta=dsr_eta,
                max_steps=max_steps,
                price_column=benchmark_price_column,
                initial_portfolio_value=float(getattr(config.env, "initial_portfolio_value", DEFAULT_INITIAL_PORTFOLIO_VALUE)),
            )

            # Calculate DSR for max profit
            mp_dsr, _ = calculate_benchmark_dsr(
                df,
                strategy="max_profit",
                eta=dsr_eta,
                max_steps=max_steps,
                price_column=benchmark_price_column,
                initial_portfolio_value=float(getattr(config.env, "initial_portfolio_value", DEFAULT_INITIAL_PORTFOLIO_VALUE)),
            )

            # Add to benchmark data
            for step, (bh_val, mp_val) in enumerate(zip(bh_dsr, mp_dsr, strict=False)):
                benchmark_data.extend(
                    [
                        {"Steps": step, "Cumulative_Reward": bh_val, "Run": "Buy-and-Hold"},
                        {
                            "Steps": step,
                            "Cumulative_Reward": mp_val,
                            "Run": "Max Profit (Unleveraged)",
                        },
                    ]
                )

            y_label = "Cumulative DSR"
        else:
            # For log_return and other rewards, use log return benchmarks
            benchmark_df = pd.DataFrame(
                {
                    "x": range(max_steps),
                    "buy_and_hold": np.log(benchmark_series / benchmark_series.shift(1))
                    .fillna(0)
                    .cumsum()[:max_steps],
                    "max_profit": np.log(
                        abs(benchmark_series / benchmark_series.shift(1) - 1) + 1
                    )
                    .fillna(0)
                    .cumsum()[:max_steps],
                }
            )
            for step, (bh_val, mp_val) in enumerate(
                zip(benchmark_df["buy_and_hold"], benchmark_df["max_profit"], strict=False)
            ):
                benchmark_data.extend(
                    [
                        {"Steps": step, "Cumulative_Reward": bh_val, "Run": "Buy-and-Hold"},
                        {
                            "Steps": step,
                            "Cumulative_Reward": mp_val,
                            "Run": "Max Profit (Unleveraged)",
                        },
                    ]
                )

            y_label = "Cumulative Reward"

        existing_data = reward_plot.data
        combined_data = pd.concat(
            [existing_data, pd.DataFrame(benchmark_data)], ignore_index=True
        )
        reward_plot = (
            ggplot(combined_data, aes(x="Steps", y="Cumulative_Reward", color="Run"))
            + geom_line()
            + labs(
                title="Cumulative Rewards Comparison", x="Steps", y=y_label
            )
            + scale_color_manual(
                values={
                    "Deterministic": "#F8766D",
                    "Random": "#00BFC4",
                    "Buy-and-Hold": "violet",
                    "Max Profit (Unleveraged)": "green",
                }
            )
            + theme(
                figure_size=(13, 7.8),
                legend_position="bottom",
                legend_title=element_text(weight="bold", size=11),
                legend_text=element_text(size=10),
                subplots_adjust={'left': 0.10, 'right': 0.95},
            )
            + guides(color=guide_legend(title="Strategy"))
        )

        # Create merged comparison plot (rewards + actions stacked vertically)
        merged_plot = create_merged_comparison_plot(reward_plot, action_plot)

        # Final metrics
        final_reward = float(rollout_deterministic["next"]["reward"].sum().item())

        # Extract actions appropriately (is_portfolio already detected above)
        actions = self._extract_actions(rollout_deterministic, is_portfolio)

        if is_portfolio:
            # Store portfolio weights (continuous values 0-1)
            if actions.ndim > 1:
                # Multi-asset: store mean allocation per asset
                last_positions = actions.mean(dim=0).tolist()
            else:
                # Single asset: store allocation over time
                last_positions = actions.flatten().tolist()
        else:
            # Store discrete positions (e.g., [-1, 0, 1])
            actions_flat = (
                actions.flatten().tolist() if hasattr(actions, "flatten") else []
            )
            last_positions = [int(a) - 1 for a in actions_flat] if actions_flat else []

        return (
            reward_plot,
            action_plot,
            None,
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
        logger.info(start_message)
        t0 = time.time()
        self.callback = callback
        self._log_step_offset = max(
            len(self.logs.get("loss_actor", [])),
            len(self.logs.get("loss_value", [])),
        )

        try:
            for i, data in enumerate(self.collector):
                if on_batch_start is not None:
                    on_batch_start(i, data)

                # Store current batch for on-policy algorithms (PPO)
                self._current_batch = data

                # Off-policy algorithms (TD3, DDPG) accumulate in replay buffer
                # On-policy algorithms (PPO) skip buffer and train on fresh data only
                if self._use_replay_buffer:
                    self.replay_buffer.extend(data)
                    max_length = self.replay_buffer[:]["next", "step_count"].max()
                    buffer_len = len(self.replay_buffer)
                else:
                    # On-policy: get max_length from current batch, buffer stays empty
                    max_length = data["next", "step_count"].max()
                    buffer_len = data.numel()

                self.total_count += data.numel()

                # Check if we've collected enough experience to start training
                # For off-policy: check replay buffer size
                # For on-policy: check total steps collected
                collected_steps = self.total_count if not self._use_replay_buffer else buffer_len
                if collected_steps > self.config.init_rand_steps:
                    self._optimization_step(i, max_length, buffer_len)
                self.total_episodes += data["next", "done"].sum()
                self._maybe_save_checkpoint()
                self.runtime_hooks.maybe_run(self.total_count)

                if self.callback and hasattr(self.callback, "log_episode_stats"):
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

        if on_train_end is not None:
            on_train_end()

        t1 = time.time()
        logger.info(
            f"{completion_prefix}: {self.total_count} steps, "
            f"{self.total_episodes} episodes, {t1 - t0:.2f}s"
        )
        return dict(self.logs)

    def train(self, callback: Any = None) -> dict[str, list]:
        """Run training loop for RL agent."""
        return self._run_training_loop(callback)
