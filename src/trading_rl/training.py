"""Training loop and utilities for trading RL."""

import logging
import time
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from tensordict.nn import InteractionType, set_composite_lp_aggregate
from torch.optim import Adam
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.envs.utils import set_exploration_type
from torchrl.objectives import ClipPPOLoss, DDPGLoss, SoftUpdate

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
            total_frames=config.max_steps,
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
                # Calculate portfolio valuation from cumulative rewards
                # Since reward = log(portfolio_val[t] / portfolio_val[t-1])
                # We can reconstruct portfolio value from cumulative log returns
                episode_reward = data["next", "reward"].sum().item()

                # Reset portfolio value to starting amount at the beginning of each episode
                callback._portfolio_value = 10000.0  # Starting portfolio value

                # Update portfolio value based on episode reward (cumulative log return)
                if episode_reward != 0:
                    # exp(log_return) = portfolio_val[t] / portfolio_val[t-1]
                    # For episode reward: portfolio_val_final = starting_val * exp(episode_reward)
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
                exploration_ratio = (
                    0.1  # Placeholder - could be calculated from exploration strategy
                )

                callback.log_episode_stats(
                    episode_reward=episode_reward,
                    portfolio_valuation=portfolio_valuation,
                    actions=actions,
                    exploration_ratio=exploration_ratio,
                )

            # Check stopping condition
            if self.total_count >= self.config.max_steps:
                logger.info(f"Training stopped after {self.config.max_steps} steps")
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


class PPOTrainer:
    """Trainer for PPO algorithm on trading environments."""

    def __init__(
        self,
        actor: Any,
        value_net: Any,
        env: Any,
        config: TrainingConfig,
    ):
        """Initialize PPO trainer.

        Args:
            actor: Actor network (categorical for discrete actions)
            value_net: Value network for state value estimation
            env: Trading environment
            config: Training configuration
        """
        self.actor = actor
        self.value_net = value_net
        self.env = env
        self.config = config

        # Initialize PPO loss module
        self.ppo_loss = ClipPPOLoss(
            actor_network=actor,
            critic_network=value_net,
            clip_epsilon=getattr(config, "clip_epsilon", 0.2),
            entropy_bonus=getattr(config, "entropy_bonus", 0.01),
            critic_coeff=getattr(config, "vf_coef", 0.5),
            loss_critic_type=getattr(config, "loss_function", "l2"),
            normalize_advantage=True,
        )

        # Single optimizer for both actor and critic (PPO style)
        self.optimizer = Adam(
            list(self.ppo_loss.actor_network_params.values(True, True))
            + list(self.ppo_loss.critic_network_params.values(True, True)),
            lr=config.actor_lr,  # Use actor_lr as base learning rate
            weight_decay=getattr(config, "value_weight_decay", 0.0),
        )

        # Replay buffer
        self.replay_buffer = ReplayBuffer(storage=LazyTensorStorage(config.buffer_size))

        # Data collector
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

        # Note: Don't set composite LP aggregate for PPO to avoid conflicts
        # with log_prob_key property

        logger.info("PPO Trainer initialized")
        logger.info(f"Learning rate: {config.actor_lr}")
        logger.info(f"Clip epsilon: {getattr(config, 'clip_epsilon', 0.2)}")
        logger.info(f"Entropy bonus: {getattr(config, 'entropy_bonus', 0.01)}")

    def train(self, callback=None) -> dict[str, list]:
        """Run PPO training loop.

        Args:
            callback: Optional callback for logging intermediate training statistics

        Returns:
            Dictionary containing training logs
        """
        logger.info("Starting PPO training")
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
                # Calculate portfolio valuation from cumulative rewards
                # Since reward = log(portfolio_val[t] / portfolio_val[t-1])
                # We can reconstruct portfolio value from cumulative log returns
                episode_reward = data["next", "reward"].sum().item()

                # Reset portfolio value to starting amount at the beginning of each episode
                callback._portfolio_value = 10000.0  # Starting portfolio value

                # Update portfolio value based on episode reward (cumulative log return)
                if episode_reward != 0:
                    # exp(log_return) = portfolio_val[t] / portfolio_val[t-1]
                    # For episode reward: portfolio_val_final = starting_val * exp(episode_reward)
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
                exploration_ratio = getattr(self.config, "entropy_bonus", 0.01)

                callback.log_episode_stats(
                    episode_reward=episode_reward,
                    portfolio_valuation=portfolio_valuation,
                    actions=actions,
                    exploration_ratio=exploration_ratio,
                )

            # Check stopping condition
            if self.total_count >= self.config.max_steps:
                logger.info(f"PPO training stopped after {self.config.max_steps} steps")
                break

        t1 = time.time()
        logger.info(
            f"PPO training complete: {self.total_count} steps, "
            f"{self.total_episodes} episodes, {t1 - t0:.2f}s"
        )

        return dict(self.logs)

    def _optimization_step(
        self, batch_idx: int, max_length: int, buffer_len: int
    ) -> None:
        """Perform PPO optimization steps on sampled batches.

        Args:
            batch_idx: Current batch index
            max_length: Maximum episode length in buffer
            buffer_len: Current replay buffer size
        """
        # PPO typically does multiple epochs per batch
        ppo_epochs = getattr(self.config, "ppo_epochs", 4)

        for j in range(ppo_epochs):
            # Sample from replay buffer
            sample = self.replay_buffer.sample(self.config.sample_size)

            # Compute PPO losses
            loss_vals = self.ppo_loss(sample)

            # Combined optimization step (actor + critic)
            total_loss = (
                loss_vals["loss_objective"]
                + loss_vals["loss_critic"]
                + loss_vals["loss_entropy"]
            )
            total_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Log losses
            actor_loss = loss_vals["loss_objective"].item()
            value_loss = loss_vals["loss_critic"].item()
            entropy_loss = loss_vals["loss_entropy"].item()

            self.logs["loss_actor"].append(actor_loss)
            self.logs["loss_value"].append(value_loss)
            self.logs["loss_entropy"].append(entropy_loss)

            # Log to callback if provided
            if (
                hasattr(self, "callback")
                and self.callback
                and hasattr(self.callback, "log_training_step")
            ):
                current_step = batch_idx * ppo_epochs + j
                self.callback.log_training_step(current_step, actor_loss, value_loss)

            # Periodic logging and evaluation
            if j % self.config.log_interval == 0:
                self._log_progress(max_length, buffer_len, loss_vals)

            # Periodic evaluation
            if j % self.config.eval_interval == 0:
                self._evaluate()

    def _log_progress(self, max_length: int, buffer_len: int, loss_vals: dict) -> None:
        """Log PPO training progress.

        Args:
            max_length: Maximum episode length
            buffer_len: Replay buffer size
            loss_vals: Current loss values
        """
        curr_loss_actor = loss_vals["loss_objective"].item()
        curr_loss_value = loss_vals["loss_critic"].item()
        curr_loss_entropy = loss_vals["loss_entropy"].item()

        logger.info(f"Max steps: {max_length}, Buffer size: {buffer_len}")
        logger.info(f"PPO Actor loss: {curr_loss_actor:.4f}")
        logger.info(f"PPO Value loss: {curr_loss_value:.4f}")
        logger.info(f"PPO Entropy loss: {curr_loss_entropy:.4f}")

    def _evaluate(self) -> None:
        """Evaluate current PPO policy."""
        with set_exploration_type(InteractionType.MODE), torch.no_grad():
            eval_rollout = self.env.rollout(self.config.eval_steps, self.actor)

            # Log evaluation metrics
            mean_reward = eval_rollout["next", "reward"].mean().item()
            sum_reward = eval_rollout["next", "reward"].sum().item()
            max_steps = eval_rollout["step_count"].max().item()

            self.logs["eval_reward_mean"].append(mean_reward)
            self.logs["eval_reward_sum"].append(sum_reward)
            self.logs["eval_step_count"].append(max_steps)

            logger.info(
                f"PPO Eval - Mean reward: {mean_reward:.4f}, "
                f"Sum reward: {sum_reward:.4f}, "
                f"Max steps: {max_steps}"
            )

            del eval_rollout

    def save_checkpoint(self, path: str) -> None:
        """Save PPO training checkpoint.

        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            "actor_state_dict": self.actor.state_dict(),
            "value_net_state_dict": self.value_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "total_count": self.total_count,
            "total_episodes": self.total_episodes,
            "logs": dict(self.logs),
        }
        torch.save(checkpoint, path)
        logger.info(f"PPO checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load PPO training checkpoint.

        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.value_net.load_state_dict(checkpoint["value_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_count = checkpoint["total_count"]
        self.total_episodes = checkpoint["total_episodes"]
        self.logs = defaultdict(list, checkpoint["logs"])
        logger.info(f"PPO checkpoint loaded from {path}")
