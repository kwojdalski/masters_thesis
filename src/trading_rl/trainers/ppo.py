"""PPO Trainer implementation."""

from collections import defaultdict
from typing import Any

import torch
from tensordict.nn import InteractionType
from torch.optim import Adam
from torchrl.envs.utils import set_exploration_type
from torchrl.objectives import ClipPPOLoss

from logger import get_logger
from trading_rl.config import TrainingConfig
from trading_rl.trainers.base import BaseTrainer

logger = get_logger(__name__)


class PPOTrainer(BaseTrainer):
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
        super().__init__(
            actor=actor,
            value_net=value_net,
            env=env,
            config=config,
            enable_composite_lp=False,
        )

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

        # Note: Don't set composite LP aggregate for PPO to avoid conflicts
        # with log_prob_key property

        logger.info("PPO Trainer initialized")
        logger.info(f"Learning rate: {config.actor_lr}")
        logger.info(f"Clip epsilon: {getattr(config, 'clip_epsilon', 0.2)}")
        logger.info(f"Entropy bonus: {getattr(config, 'entropy_bonus', 0.01)}")

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
                f"\033[92mPPO Eval\033[0m - \033[93mMean reward:\033[0m {mean_reward:.4f}, "
                f"\033[93mSum reward:\033[0m {sum_reward:.4f}, "
                f"\033[93mMax steps:\033[0m {max_steps}"
            )

            del eval_rollout

    def _compute_exploration_ratio(self) -> float:
        return getattr(self.config, "entropy_bonus", 0.01)

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
