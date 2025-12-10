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
