"""TD3 Trainer implementation."""

import logging
import time
from collections import defaultdict
from typing import Any

import torch
from tensordict.nn import InteractionType, TensorDictSequential
from torch.optim import Adam
from torchrl.collectors.collectors import RandomPolicy
from torchrl.data import Bounded
from torchrl.envs.utils import set_exploration_type
from torchrl.modules import AdditiveGaussianModule
from torchrl.objectives import SoftUpdate
from torchrl.objectives import TD3Loss as TorchRLTd3Loss

from logger import get_logger
from trading_rl.config import TrainingConfig
from trading_rl.models import create_td3_actor, create_td3_qvalue_network
from trading_rl.trainers.base import BaseTrainer

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


class TD3Trainer(BaseTrainer):
    """Trainer for TD3 algorithm on trading environments."""

    def __init__(
        self,
        actor: Any,
        qvalue_nets: list[Any],
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

        # TD3 uses two critics; configure loss and optimizers
        base_qvalue_net = qvalue_nets[0]
        num_qvalue_nets = 2

        self.td3_loss = TD3Loss(
            actor_network=actor,
            qvalue_network=base_qvalue_net,
            action_spec=td3_action_spec,
            num_qvalue_nets=num_qvalue_nets,
            policy_noise=getattr(config, "policy_noise", 0.2),
            noise_clip=getattr(config, "noise_clip", 0.5),
            loss_function=getattr(config, "loss_function", "smooth_l1"),
            delay_actor=getattr(config, "delay_actor", True),
            delay_qvalue=getattr(config, "delay_qvalue", True),
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

        # Counters for tracking successful vs skipped batches
        self.successful_batches = 0
        self.skipped_batches = 0

        logger.info("TD3 Trainer initialized")
        logger.info(
            "Actor LR: %s, Value LR: %s, Noise: %.3f, Clip: %.3f, Delay: %d",
            config.actor_lr,
            config.value_lr,
            getattr(config, "policy_noise", 0.2),
            getattr(config, "noise_clip", 0.5),
            self.policy_delay,
        )

    @staticmethod
    def build_models(n_obs: int, n_act: int, config: Any, env: Any):
        """Factory for TD3 actor and Q-value networks."""
        actor = create_td3_actor(
            n_obs,
            n_act,
            hidden_dims=config.network.actor_hidden_dims,
            spec=env.action_spec,
        )
        qvalue_nets = [
            create_td3_qvalue_network(
                n_obs,
                n_act,
                hidden_dims=config.network.value_hidden_dims,
            ),
            create_td3_qvalue_network(
                n_obs,
                n_act,
                hidden_dims=config.network.value_hidden_dims,
            ),
        ]
        return actor, qvalue_nets

    def _optimization_step(
        self, batch_idx: int, max_length: int, buffer_len: int
    ) -> None:
        for j in range(self.config.optim_steps_per_batch):
            sample = self.replay_buffer.sample(self.config.sample_size)

            # DEBUG: Log sample statistics
            if logger.isEnabledFor(logging.DEBUG):
                actions = sample["action"]
                rewards = sample["next", "reward"]
                logger.debug(f"[Batch {batch_idx}, Step {j}] Sample stats:")
                logger.debug(
                    f"  Actions - mean: {actions.mean():.4f}, std: {actions.std():.4f}, min: {actions.min():.4f}, max: {actions.max():.4f}"
                )
                logger.debug(
                    f"  Rewards - mean: {rewards.mean():.4f}, std: {rewards.std():.4f}, min: {rewards.min():.4f}, max: {rewards.max():.4f}"
                )

            # Ensure sample has consistent shapes for TD3Loss
            # Check for any NaN or inf values that could cause shape issues
            if (
                torch.isnan(sample["next", "reward"]).any()
                or torch.isinf(sample["next", "reward"]).any()
            ):
                logger.warning("Found NaN/inf in reward, skipping optimization step")
                continue

            # Ensure done and terminated have consistent shapes
            done = sample["next", "done"]
            terminated = sample["next", "terminated"]
            if done.shape != terminated.shape:
                logger.warning(
                    f"Shape mismatch: done {done.shape} vs terminated {terminated.shape}"
                )
                continue

            # 1. Update critics
            try:
                loss_vals = self.td3_loss(sample)
                # If we get here, the batch was successful
                self.successful_batches += 1

                # DEBUG: Log loss values
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"  Q-value loss: {loss_vals['loss_qvalue'].item():.6f}"
                    )
                    logger.debug(f"  Actor loss: {loss_vals['loss_actor'].item():.6f}")

            except RuntimeError as e:
                if "All input tensors" in str(e) and "must share a unique shape" in str(
                    e
                ):
                    self.skipped_batches += 1
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

            # Log progress similar to PPO (info level)

            if j % max(1, self.config.log_interval) == 0:
                self._log_progress(max_length, buffer_len, loss_vals)

            if j % self.config.eval_interval == 0:
                self._evaluate()

    def train(self, callback=None) -> dict[str, list]:
        """Run training loop for RL agent, with exploration for TD3."""
        logger.info("Starting TD3 training")
        logger.debug("TD3 Training configuration:")
        logger.debug(f"  Max steps: {self.config.max_steps}")
        logger.debug(f"  Init random steps: {self.config.init_rand_steps}")
        logger.debug(f"  Frames per batch: {self.config.frames_per_batch}")
        logger.debug(f"  Buffer size: {self.config.buffer_size}")

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
        self.noisy_policy = TensorDictSequential(self.actor, self.exploration_module)

        logger.info(
            f"Using random policy for first {self.config.init_rand_steps} steps"
        )

        for i, data in enumerate(self.collector):
            self.replay_buffer.extend(data)

            max_length = self.replay_buffer[:]["next", "step_count"].max()
            buffer_len = len(self.replay_buffer)

            # DEBUG: Log data collection statistics
            if logger.isEnabledFor(logging.DEBUG) and i % 10 == 0:  # Every 10 batches
                episode_rewards = data["next", "reward"]
                logger.debug(
                    f"[Batch {i}] Collected {data.numel()} steps, buffer size: {buffer_len}"
                )
                logger.debug(
                    f"  Episode rewards - mean: {episode_rewards.mean():.4f}, std: {episode_rewards.std():.4f}"
                )

                # Log actions being collected
                collected_actions = data["action"]
                logger.debug(
                    f"  Collected actions - mean: {collected_actions.mean():.4f}, std: {collected_actions.std():.4f}"
                )

            # Switch from random to noisy policy after initial steps
            if (
                not self.random_exploration_done
                and self.total_count >= self.config.init_rand_steps
            ):
                logger.info(
                    f"✓ Random exploration finished at {self.total_count} steps. Switching to noisy policy."
                )
                logger.debug(
                    f"  Buffer now contains {buffer_len} transitions from random policy"
                )
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

        # Log batch success/failure summary
        total_batches = self.successful_batches + self.skipped_batches
        if total_batches > 0:
            success_rate = (self.successful_batches / total_batches) * 100
            summary_msg = (
                f"Batch processing summary: {self.successful_batches}/{total_batches} "
                f"batches successful ({success_rate:.1f}%), {self.skipped_batches} skipped due to tensor shape errors"
            )

            # Use warning if success rate is below 70%, otherwise info
            if success_rate < 70.0:
                logger.warning(summary_msg)
            else:
                logger.info(summary_msg)
        else:
            logger.warning("No optimization batches were attempted during training")

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

            # DEBUG: Log action distribution during evaluation
            if logger.isEnabledFor(logging.DEBUG):
                actions = eval_rollout["action"]
                logger.debug("[EVALUATION] Action statistics:")
                logger.debug(f"  Actions shape: {actions.shape}")
                logger.debug(
                    f"  Actions - mean: {actions.mean():.4f}, std: {actions.std():.4f}"
                )
                logger.debug(
                    f"  Actions - min: {actions.min():.4f}, max: {actions.max():.4f}"
                )

                # Show action distribution (histogram-like)
                actions_flat = actions.flatten().cpu().detach().numpy()
                import numpy as np

                unique_actions, counts = np.unique(
                    np.round(actions_flat, 2), return_counts=True
                )
                if len(unique_actions) <= 10:
                    logger.debug(
                        f"  Action distribution: {dict(zip(unique_actions, counts))}"
                    )
                else:
                    # Show percentiles if too many unique values
                    percentiles = np.percentile(actions_flat, [0, 25, 50, 75, 100])
                    logger.debug(
                        f"  Action percentiles [0,25,50,75,100]: {percentiles}"
                    )

                # Check if agent is stuck
                if actions.std() < 0.01:
                    logger.warning(
                        f"  ⚠️  Agent appears STUCK - action std is very low ({actions.std():.6f})"
                    )

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
            "qvalue_state_dict": [q.state_dict() for q in self.value_net],
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
            self.td3_loss.qvalue_network_params.to_module(self.value_net[0])
            # Load saved critic modules
            for q_net, state_dict in zip(
                self.value_net, checkpoint["qvalue_state_dict"], strict=False
            ):
                q_net.load_state_dict(state_dict)
        else:
            self.actor.load_state_dict(checkpoint["actor_state_dict"])
            for q_net, state_dict in zip(
                self.value_net, checkpoint["qvalue_state_dict"], strict=False
            ):
                q_net.load_state_dict(state_dict)

        self.optimizer_actor.load_state_dict(checkpoint["optimizer_actor_state_dict"])
        self.optimizer_value.load_state_dict(checkpoint["optimizer_value_state_dict"])
        self.total_count = checkpoint["total_count"]
        self.total_episodes = checkpoint["total_episodes"]
        self.logs = defaultdict(list, checkpoint["logs"])
        logger.info(f"TD3 checkpoint loaded from {path}")
