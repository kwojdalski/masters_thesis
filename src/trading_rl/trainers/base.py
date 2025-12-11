"""Base trainer and utilities."""

import contextlib
import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any

import numpy as np
import torch
import torch.multiprocessing as mp
import torchrl.collectors.collectors as torchrl_collectors
from tensordict.nn import set_composite_lp_aggregate
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer

from logger import get_logger
from trading_rl.config import TrainingConfig


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
    ):
        self.actor = actor
        self.value_net = value_net
        self.env = env
        self.config = config
        self.callback = None

        # Replay buffer and data collector shared by both algorithms
        self.replay_buffer = ReplayBuffer(storage=LazyTensorStorage(config.buffer_size))
        self.collector = SyncDataCollector(
            create_env_fn=lambda: env,
            policy=actor,
            frames_per_batch=config.frames_per_batch,
            total_frames=config.max_steps,
        )

        # Training state
        self.total_count = 0
        self.total_episodes = 0
        self.logs = defaultdict(list)

        if enable_composite_lp:
            set_composite_lp_aggregate(True).set()

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

    def create_action_probabilities_plot(self, max_steps: int, df=None, config=None):
        """Optional action-probability visualization; default is not implemented."""
        return None

    def _log_episode_stats(self, data, callback) -> None:
        """Log episode statistics to provided callback."""
        episode_reward = data["next", "reward"].sum().item()

        # Reset portfolio value to starting amount at the beginning of each episode
        callback._portfolio_value = 10000.0  # Starting portfolio value

        # Update portfolio value based on episode reward (cumulative log return)
        if episode_reward != 0:
            callback._portfolio_value = 10000.0 * np.exp(episode_reward)

        portfolio_valuation = callback._portfolio_value
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

        # Calculate returns
        initial_portfolio = 10000.0
        portfolio_return = 100 * (portfolio_valuation / initial_portfolio - 1)

        # Market return would need market data, for now we use episode reward as proxy
        # Episode reward is cumulative log return, convert to percentage
        market_return = 100 * (np.exp(episode_reward) - 1)

        logger.info(
            f"Episode {callback._episode_count} complete | "
            f"Market Return: {market_return:5.2f}% | "
            f"Portfolio Return: {portfolio_return:5.2f}% | "
            f"Portfolio Value: ${portfolio_valuation:,.2f}"
        )

    def evaluate(
        self, df, max_steps: int, config=None, algorithm: str | None = None
    ) -> tuple:
        """Default evaluation: deterministic vs random rollout comparison."""
        from trading_rl.utils import compare_rollouts
        import pandas as pd
        import numpy as np
        from plotnine import aes, geom_line, ggplot, labs, scale_color_manual

        logger = get_logger(__name__)

        # Deterministic rollout
        logger.debug(f"Running deterministic evaluation for {max_steps} steps")
        with set_exploration_type(InteractionType.MODE):
            rollout_deterministic = self.env.rollout(max_steps=max_steps, policy=self.actor)

        # Random rollout (can be overridden in subclasses)
        logger.debug(f"Running random evaluation for {max_steps} steps")
        with set_exploration_type(InteractionType.RANDOM):
            rollout_random = self.env.rollout(max_steps=max_steps, policy=self.actor)

        reward_plot, action_plot = compare_rollouts(
            [rollout_deterministic, rollout_random],
            n_obs=max_steps,
        )

        # Benchmarks
        benchmark_df = pd.DataFrame(
            {
                "x": range(max_steps),
                "buy_and_hold": np.log(df["close"] / df["close"].shift(1))
                .fillna(0)
                .cumsum()[:max_steps],
                "max_profit": np.log(abs(df["close"] / df["close"].shift(1) - 1) + 1)
                .fillna(0)
                .cumsum()[:max_steps],
            }
        )
        benchmark_data = []
        for step, (bh_val, mp_val) in enumerate(
            zip(benchmark_df["buy_and_hold"], benchmark_df["max_profit"], strict=False)
        ):
            benchmark_data.extend(
                [
                    {"Steps": step, "Cumulative_Reward": bh_val, "Run": "Buy-and-Hold"},
                    {"Steps": step, "Cumulative_Reward": mp_val, "Run": "Max Profit"},
                ]
            )
        existing_data = reward_plot.data
        combined_data = pd.concat(
            [existing_data, pd.DataFrame(benchmark_data)], ignore_index=True
        )
        reward_plot = (
            ggplot(combined_data, aes(x="Steps", y="Cumulative_Reward", color="Run"))
            + geom_line()
            + labs(title="Cumulative Rewards Comparison", x="Steps", y="Cumulative Reward")
            + scale_color_manual(
                values={
                    "Deterministic": "#F8766D",
                    "Random": "#00BFC4",
                    "Buy-and-Hold": "violet",
                    "Max Profit": "green",
                }
            )
        )

        # Final metrics
        final_reward = float(rollout_deterministic["next"]["reward"].sum().item())
        action_tensor = rollout_deterministic["action"].squeeze()
        if action_tensor.ndim > 1 and action_tensor.shape[-1] > 1:
            actions = action_tensor.argmax(dim=-1)
        else:
            actions = action_tensor
        actions = actions.flatten().tolist() if hasattr(actions, "flatten") else []
        last_positions = [int(a) - 1 for a in actions] if actions else []

        return reward_plot, action_plot, None, final_reward, last_positions

    def train(self, callback=None) -> dict[str, list]:
        """Run training loop for RL agent."""
        logger.info("Starting training")
        t0 = time.time()
        self.callback = callback

        for i, data in enumerate(self.collector):
            self.replay_buffer.extend(data)

            max_length = self.replay_buffer[:]["next", "step_count"].max()
            buffer_len = len(self.replay_buffer)

            if buffer_len > self.config.init_rand_steps:
                self._optimization_step(i, max_length, buffer_len)

            self.total_count += data.numel()
            self.total_episodes += data["next", "done"].sum()

            if callback and hasattr(callback, "log_episode_stats"):
                self._log_episode_stats(data, callback)

            if self.total_count >= self.config.max_steps:
                logger.info(f"Training stopped after {self.config.max_steps} steps")
                break

        t1 = time.time()
        logger.info(
            f"Training complete: {self.total_count} steps, "
            f"{self.total_episodes} episodes, {t1 - t0:.2f}s"
        )

        return dict(self.logs)
