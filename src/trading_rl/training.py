"""Training loop and utilities for trading RL."""

import logging
import time
from collections import defaultdict
from typing import Any

import torch
from tensordict.nn import InteractionType, set_composite_lp_aggregate
from torch.optim import Adam
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.envs.utils import set_exploration_type
from torchrl.objectives import DDPGLoss, SoftUpdate

from trading_rl.config import TrainingConfig

logger = logging.getLogger(__name__)


class DDPGTrainer:
    """Trainer for DDPG algorithm on trading environments."""

    def __init__(
        self,
        actor: Any,
        value_net: Any,
        env: Any,
        config: TrainingConfig,
    ):
        """Initialize DDPG trainer.

        Args:
            actor: Actor network
            value_net: Value network
            env: Trading environment
            config: Training configuration
        """
        self.actor = actor
        self.value_net = value_net
        self.env = env
        self.config = config

        # Initialize loss module
        self.ddpg_loss = DDPGLoss(
            actor_network=actor,
            value_network=value_net,
            loss_function=config.loss_function,
        )

        # Target network updater
        self.updater = SoftUpdate(self.ddpg_loss, tau=config.tau)

        # Optimizers
        self.optimizer_actor = Adam(
            self.ddpg_loss.actor_network_params.values(True, True),
            lr=config.actor_lr,
        )
        self.optimizer_value = Adam(
            self.ddpg_loss.value_network_params.values(True, True),
            lr=config.value_lr,
            weight_decay=config.value_weight_decay,
        )

        # Replay buffer
        self.replay_buffer = ReplayBuffer(storage=LazyTensorStorage(config.buffer_size))

        # Data collector
        self.collector = SyncDataCollector(
            create_env_fn=lambda: env,
            policy=actor,
            frames_per_batch=config.frames_per_batch,
            total_frames=config.max_training_steps,
        )

        # Training state
        self.total_count = 0
        self.total_episodes = 0
        self.logs = defaultdict(list)

        # Enable composite LP aggregate for better performance
        set_composite_lp_aggregate(True).set()

        logger.info("DDPG Trainer initialized")
        logger.info(f"Actor LR: {config.actor_lr}, Value LR: {config.value_lr}")
        logger.info(f"Buffer size: {config.buffer_size}, Tau: {config.tau}")

    def train(self, callback=None) -> dict[str, list]:
        """Run training loop.

        Args:
            callback: Optional callback for logging intermediate training statistics

        Returns:
            Dictionary containing training logs
        """
        logger.info("Starting training")
        t0 = time.time()
        self.callback = callback

        for i, data in enumerate(self.collector):
            # Add experience to replay buffer
            self.replay_buffer.extend(data)

            # Get metrics
            max_length = self.replay_buffer[:]["next", "step_count"].max()
            buffer_len = len(self.replay_buffer)

            # Start optimization after initial random steps
            if buffer_len > self.config.init_rand_steps:
                self._optimization_step(i, max_length, buffer_len)

            # Update counters
            self.total_count += data.numel()
            self.total_episodes += data["next", "done"].sum()

            # Log episode statistics to callback if provided
            if callback and hasattr(callback, "log_episode_stats"):
                episode_reward = data["next", "reward"].sum().item()
                # Extract portfolio value and actions if available
                portfolio_value = getattr(data.get("next", {}), "portfolio_value", 0.0)
                if hasattr(portfolio_value, "item"):
                    portfolio_value = portfolio_value.item()
                actions = data.get("action", torch.tensor([])).flatten().tolist()
                exploration_ratio = (
                    0.5  # Placeholder - could be calculated from exploration strategy
                )

                callback.log_episode_stats(
                    episode_reward=episode_reward,
                    portfolio_value=portfolio_value,
                    actions=actions,
                    exploration_ratio=exploration_ratio,
                )

            # Check stopping condition
            if self.total_count >= self.config.max_training_steps:
                logger.info(
                    f"Training stopped after {self.config.max_training_steps} steps"
                )
                break

        t1 = time.time()
        logger.info(
            f"Training complete: {self.total_count} steps, "
            f"{self.total_episodes} episodes, {t1 - t0:.2f}s"
        )

        return dict(self.logs)

    def _optimization_step(
        self, batch_idx: int, max_length: int, buffer_len: int
    ) -> None:
        """Perform optimization steps on sampled batches.

        Args:
            batch_idx: Current batch index
            max_length: Maximum episode length in buffer
            buffer_len: Current replay buffer size
        """
        for j in range(self.config.optim_steps_per_batch):
            # Sample from replay buffer
            sample = self.replay_buffer.sample(self.config.sample_size)

            # Compute losses
            loss_vals = self.ddpg_loss(sample)

            # Optimize actor
            loss_vals["loss_actor"].backward()
            self.optimizer_actor.step()
            self.optimizer_actor.zero_grad()

            # Optimize value network
            loss_vals["loss_value"].backward()
            self.optimizer_value.step()
            self.optimizer_value.zero_grad()

            # Update target networks
            self.updater.step()

            # Log losses
            actor_loss = loss_vals["loss_actor"].item()
            value_loss = loss_vals["loss_value"].item()
            self.logs["loss_value"].append(value_loss)
            self.logs["loss_actor"].append(actor_loss)

            # Log to callback if provided
            if (
                hasattr(self, "callback")
                and self.callback
                and hasattr(self.callback, "log_training_step")
            ):
                current_step = batch_idx * self.config.optim_steps_per_batch + j
                self.callback.log_training_step(current_step, actor_loss, value_loss)

            # Periodic logging and evaluation
            if j % self.config.log_interval == 0:
                self._log_progress(max_length, buffer_len, loss_vals)

            # Periodic evaluation
            if j % self.config.eval_interval == 0:
                self._evaluate()

    def _log_progress(self, max_length: int, buffer_len: int, loss_vals: dict) -> None:
        """Log training progress.

        Args:
            max_length: Maximum episode length
            buffer_len: Replay buffer size
            loss_vals: Current loss values
        """
        curr_loss_value = loss_vals["loss_value"].item()
        curr_loss_actor = loss_vals["loss_actor"].item()

        logger.info(f"Max steps: {max_length}, Buffer size: {buffer_len}")
        logger.info(f"Loss value: {curr_loss_value:.4f}")
        logger.info(f"Loss actor: {curr_loss_actor:.4f}")

    def _evaluate(self) -> None:
        """Evaluate current policy."""
        with set_exploration_type(InteractionType.DETERMINISTIC), torch.no_grad():
            eval_rollout = self.env.rollout(self.config.eval_steps, self.actor)

            # Log evaluation metrics
            mean_reward = eval_rollout["next", "reward"].mean().item()
            sum_reward = eval_rollout["next", "reward"].sum().item()
            max_steps = eval_rollout["step_count"].max().item()

            self.logs["eval_reward_mean"].append(mean_reward)
            self.logs["eval_reward_sum"].append(sum_reward)
            self.logs["eval_step_count"].append(max_steps)

            logger.info(
                f"Eval - Mean reward: {mean_reward:.4f}, "
                f"Sum reward: {sum_reward:.4f}, "
                f"Max steps: {max_steps}"
            )

            del eval_rollout

    def save_checkpoint(self, path: str) -> None:
        """Save training checkpoint.

        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            "actor_state_dict": self.actor.state_dict(),
            "value_net_state_dict": self.value_net.state_dict(),
            "optimizer_actor_state_dict": self.optimizer_actor.state_dict(),
            "optimizer_value_state_dict": self.optimizer_value.state_dict(),
            "total_count": self.total_count,
            "total_episodes": self.total_episodes,
            "logs": dict(self.logs),
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint.

        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.value_net.load_state_dict(checkpoint["value_net_state_dict"])
        self.optimizer_actor.load_state_dict(checkpoint["optimizer_actor_state_dict"])
        self.optimizer_value.load_state_dict(checkpoint["optimizer_value_state_dict"])
        self.total_count = checkpoint["total_count"]
        self.total_episodes = checkpoint["total_episodes"]
        self.logs = defaultdict(list, checkpoint["logs"])
        logger.info(f"Checkpoint loaded from {path}")
