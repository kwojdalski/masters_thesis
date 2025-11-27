"""Training loop and utilities for trading RL."""

import contextlib
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any

import numpy as np
import torch
import torch.multiprocessing as mp
import torchrl.collectors.collectors as torchrl_collectors
from tensordict.nn import InteractionType, set_composite_lp_aggregate
from torch.optim import Adam
from torchrl.collectors import SyncDataCollector
from torchrl.collectors.collectors import RandomPolicy
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.envs.utils import set_exploration_type
from torchrl.modules import AdditiveGaussianModule
from torchrl.objectives import ClipPPOLoss, DDPGLoss, SoftUpdate
from torchrl.objectives import TD3Loss as TorchRLTd3Loss

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


class TD3Loss(TorchRLTd3Loss):
    """Thin wrapper around TorchRL's TD3 loss to ensure consistent behavior."""
    
    actor_network: Any
    actor_network_params: Any
    target_actor_network_params: Any
    qvalue_network: Any
    qvalue_network_params: Any
    target_qvalue_network_params: Any
    
    @property
    def in_keys(self):
        return ["observation", "action", "next", "reward", "done", "terminated"]


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


class DDPGTrainer(BaseTrainer):
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
        super().__init__(
            actor=actor,
            value_net=value_net,
            env=env,
            config=config,
            enable_composite_lp=True,
        )

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

        logger.info("DDPG Trainer initialized")
        logger.info(f"Actor LR: {config.actor_lr}, Value LR: {config.value_lr}")
        logger.info(f"Buffer size: {config.buffer_size}, Tau: {config.tau}")

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

    def _compute_exploration_ratio(self) -> float:
        return 0.1  # Placeholder - could be calculated from exploration strategy

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
                f"PPO Eval - Mean reward: {mean_reward:.4f}, "
                f"Sum reward: {sum_reward:.4f}, "
                f"Max steps: {max_steps}"
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


class TD3Trainer(BaseTrainer):
    """Trainer for TD3 algorithm on trading environments."""

    def __init__(
        self,
        actor: Any,
        qvalue_nets: Any,  # Single twin Q-value network, not a list
        env: Any,
        config: TrainingConfig,
    ):
        super().__init__(
            actor=actor,
            value_net=qvalue_nets,
            env=env,
            config=config,
            enable_composite_lp=True,
        )

        # Continuous action spec in [-1, 1] to match the deterministic actor
        from torchrl.data import Bounded
        action_dim = self.env.action_spec.shape[-1]
        td3_action_spec = Bounded(
            low=-1.0,
            high=1.0,
            shape=(action_dim,),
            device=getattr(config, "device", "cpu"),
        )
        self.td3_action_spec = td3_action_spec

        # Gaussian exploration around the deterministic policy
        self.exploration_module = AdditiveGaussianModule(
            spec=td3_action_spec,
            sigma_init=getattr(config, "exploration_noise_std", 0.1),
            sigma_end=getattr(config, "exploration_noise_std", 0.1),
            annealing_num_steps=config.max_steps,
        )
        # Log the exploration noise std
        logger.info(
            "Exploration Noise Std: %.3f",
            getattr(config, "exploration_noise_std", 0.1),
        )

        # qvalue_nets is now a single ValueOperator that outputs 2 Q-values
        self.twin_qvalue_net = qvalue_nets
        num_qvalue_nets = 2

        self.td3_loss = TD3Loss(
            actor_network=actor,
            qvalue_network=self.twin_qvalue_net,
            action_spec=td3_action_spec,
            num_qvalue_nets=num_qvalue_nets,
            policy_noise=getattr(config, "policy_noise", 0.2),
            noise_clip=getattr(config, "noise_clip", 0.5),
            loss_function=getattr(config, "loss_function", "smooth_l1"),
            delay_actor=getattr(config, "delay_actor", True),
            delay_qvalue=getattr(config, "delay_qvalue", True),
            # Remove gamma parameter completely - let TD3Loss use defaults
        )

        for attr in ("actor_network_params", "qvalue_network_params"):
            params_td = getattr(self.td3_loss, attr, None)
            if params_td is not None and hasattr(params_td, "unlock_"):
                params_td.unlock_()

        self.updater = SoftUpdate(self.td3_loss, tau=config.tau)

        self.optimizer_actor = Adam(
            self.td3_loss.actor_network_params.values(True, True),
            lr=config.actor_lr,
        )
        self.optimizer_value = Adam(
            self.td3_loss.qvalue_network_params.values(True, True),
            lr=config.value_lr,
            weight_decay=config.value_weight_decay,
        )

        self.policy_delay = getattr(config, "policy_delay", 2)

        logger.info("TD3 Trainer initialized")
        logger.info(
            "Actor LR: %s, Value LR: %s, Noise: %.3f, Clip: %.3f, Delay: %d",
            config.actor_lr,
            config.value_lr,
            getattr(config, "policy_noise", 0.2),
            getattr(config, "noise_clip", 0.5),
            self.policy_delay,
        )

    def _optimization_step(
        self, batch_idx: int, max_length: int, buffer_len: int
    ) -> None:
        for j in range(self.config.optim_steps_per_batch):
            sample = self.replay_buffer.sample(self.config.sample_size)
            
            # Ensure sample has consistent shapes for TD3Loss
            # Check for any NaN or inf values that could cause shape issues
            if torch.isnan(sample["next", "reward"]).any() or torch.isinf(sample["next", "reward"]).any():
                logger.warning("Found NaN/inf in reward, skipping optimization step")
                continue
            
            # Ensure done and terminated have consistent shapes
            done = sample["next", "done"]
            terminated = sample["next", "terminated"]
            if done.shape != terminated.shape:
                logger.warning(f"Shape mismatch: done {done.shape} vs terminated {terminated.shape}")
                continue

            # 1. Update critics  
            try:
                loss_vals = self.td3_loss(sample)
            except RuntimeError as e:
                if "All input tensors" in str(e) and "must share a unique shape" in str(e):
                    logger.warning(f"TD3 tensor shape error: {e}, skipping this batch")
                    continue
                else:
                    raise e
            
            self.optimizer_value.zero_grad()
            loss_vals["loss_qvalue"].backward()
            self.optimizer_value.step()
            
            value_loss = loss_vals["loss_qvalue"].item()
            self.logs["loss_value"].append(value_loss)

            # 2. Delayed actor update
            update_actor = j % self.policy_delay == 0
            
            if update_actor:
                # Recompute loss for actor update (using updated critic weights)
                loss_vals_actor = self.td3_loss(sample)
                
                self.optimizer_actor.zero_grad()
                loss_vals_actor["loss_actor"].backward()
                self.optimizer_actor.step()

                # Update target networks
                self.updater.step()
                
                actor_loss = loss_vals_actor["loss_actor"].item()
            else:
                # For logging purposes, use the actor loss computed in the first pass
                actor_loss = loss_vals["loss_actor"].item()
            
            self.logs["loss_actor"].append(actor_loss)

            if (
                hasattr(self, "callback")
                and self.callback
                and hasattr(self.callback, "log_training_step")
            ):
                current_step = batch_idx * self.config.optim_steps_per_batch + j
                self.callback.log_training_step(current_step, actor_loss, value_loss)

            if j % self.config.log_interval == 0:
                self._log_progress(max_length, buffer_len, loss_vals)

            if j % self.config.eval_interval == 0:
                self._evaluate()

    def train(self, callback=None) -> dict[str, list]:
        """Run training loop for RL agent, with exploration for TD3."""
        logger.info("Starting TD3 training")
        t0 = time.time()
        self.callback = callback

        # Use a temporary random policy for initial steps
        initial_collector_policy = RandomPolicy(self.td3_action_spec)
        
        # Hot-swap the policy in the collector. 
        # Note: SyncDataCollector stores the policy in self.policy.
        # We need to ensure the collector uses this new policy.
        # For SyncDataCollector, direct assignment works if the collector is running in the same process (which it is).
        original_policy = self.collector.policy
        self.collector.policy = initial_collector_policy

        self.random_exploration_done = False
        
        # Create the noisy policy by chaining actor + exploration module
        # We do this once here to use when switching
        from tensordict.nn import TensorDictSequential
        self.noisy_policy = TensorDictSequential(self.actor, self.exploration_module)

        for i, data in enumerate(self.collector):
            self.replay_buffer.extend(data)

            max_length = self.replay_buffer[:]["next", "step_count"].max()
            buffer_len = len(self.replay_buffer)

            # Switch from random to noisy policy after initial steps
            if (
                not self.random_exploration_done
                and self.total_count >= self.config.init_rand_steps
            ):
                logger.info("Random exploration finished. Switching to noisy policy.")
                # Switch back to the trained policy (with noise)
                self.collector.policy = self.noisy_policy
                self.random_exploration_done = True

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
            f"TD3 Training complete: {self.total_count} steps, "
            f"{self.total_episodes} episodes, {t1 - t0:.2f}s"
        )

        return dict(self.logs)

    def _log_progress(self, max_length: int, buffer_len: int, loss_vals: dict) -> None:
        curr_loss_value = loss_vals["loss_qvalue"].item()
        curr_loss_actor = loss_vals["loss_actor"].item()

        logger.info(f"Max steps: {max_length}, Buffer size: {buffer_len}")
        logger.info(f"TD3 Value loss: {curr_loss_value:.4f}")
        logger.info(f"TD3 Actor loss: {curr_loss_actor:.4f}")

    def _evaluate(self) -> None:
        with set_exploration_type(InteractionType.DETERMINISTIC), torch.no_grad():
            eval_rollout = self.env.rollout(self.config.eval_steps, self.actor)

            mean_reward = eval_rollout["next", "reward"].mean().item()
            sum_reward = eval_rollout["next", "reward"].sum().item()
            max_steps = eval_rollout["step_count"].max().item()

            self.logs["eval_reward_mean"].append(mean_reward)
            self.logs["eval_reward_sum"].append(sum_reward)
            self.logs["eval_step_count"].append(max_steps)

            logger.info(
                f"TD3 Eval - Mean reward: {mean_reward:.4f}, "
                f"Sum reward: {sum_reward:.4f}, "
                f"Max steps: {max_steps}"
            )

            del eval_rollout

    def _compute_exploration_ratio(self) -> float:
        return getattr(self.config, "policy_noise", 0.2)

    def save_checkpoint(self, path: str) -> None:
        checkpoint = {
            "actor_state_dict": self.actor.state_dict(),
            "actor_params_state": self.td3_loss.actor_network_params.state_dict(),
            "qvalue_state_dict": self.value_net.state_dict(),
            "qvalue_params_state": self.td3_loss.qvalue_network_params.state_dict(),
            "optimizer_actor_state_dict": self.optimizer_actor.state_dict(),
            "optimizer_value_state_dict": self.optimizer_value.state_dict(),
            "total_count": self.total_count,
            "total_episodes": self.total_episodes,
            "logs": dict(self.logs),
        }
        torch.save(checkpoint, path)
        logger.info(f"TD3 checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path)

        # Restore functional params and sync modules
        if "actor_params_state" in checkpoint:
            self.td3_loss.actor_network_params.load_state_dict(
                checkpoint["actor_params_state"]
            )
            self.td3_loss.qvalue_network_params.load_state_dict(
                checkpoint["qvalue_params_state"]
            )
            # Sync back to modules for evaluation
            self.td3_loss.actor_network_params.to_module(self.actor)
            self.td3_loss.qvalue_network_params.to_module(self.twin_qvalue_net)
            self.value_net.load_state_dict(checkpoint["qvalue_state_dict"])
        else:
            self.actor.load_state_dict(checkpoint["actor_state_dict"])
            self.value_net.load_state_dict(checkpoint["qvalue_state_dict"])

        self.optimizer_actor.load_state_dict(checkpoint["optimizer_actor_state_dict"])
        self.optimizer_value.load_state_dict(checkpoint["optimizer_value_state_dict"])
        self.total_count = checkpoint["total_count"]
        self.total_episodes = checkpoint["total_episodes"]
        self.logs = defaultdict(list, checkpoint["logs"])
        logger.info(f"TD3 checkpoint loaded from {path}")
