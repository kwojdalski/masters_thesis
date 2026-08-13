"""DDPG Trainer implementation."""

from typing import Any

import torch
from tensordict.nn import TensorDictSequential
from torch.optim import Adam
from torchrl.data import Bounded
from torchrl.modules import AdditiveGaussianModule
from torchrl.objectives import DDPGLoss, SoftUpdate

from logger import get_logger, is_level_enabled
from trading_rl.config import EvaluationConfig, TrainingConfig
from trading_rl.models import create_ddpg_actor, create_value_network
from trading_rl.trainers.base import BaseTrainer
from trading_rl.trainers.registry import register_trainer

logger = get_logger(__name__)


@register_trainer("DDPG", continuous=True)
class DDPGTrainer(BaseTrainer):
    """Trainer for DDPG algorithm on trading environments."""

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
        eval_config: "EvaluationConfig | None" = None,
        eval_env: Any | None = None,
        eval_data_len: int | None = None,
        checkpoint_dir: str | None = None,
        checkpoint_prefix: str | None = None,
    ):
        super().__init__(
            actor=actor,
            value_net=value_net,
            env=env,
            config=config,
            n_obs=n_obs,
            n_act=n_act,
            actor_hidden_dims=actor_hidden_dims,
            value_hidden_dims=value_hidden_dims,
            eval_config=eval_config,
            eval_env=eval_env,
            eval_data_len=eval_data_len,
            enable_composite_lp=True,
            checkpoint_dir=checkpoint_dir,
            checkpoint_prefix=checkpoint_prefix,
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
            weight_decay=config.actor_weight_decay,
        )
        self.optimizer_value = Adam(
            self.ddpg_loss.value_network_params.values(True, True),
            lr=config.value_lr,
            weight_decay=config.value_weight_decay,
        )

        # Prefer bounded env action spec for random warmup and exploration noise.
        env_action_spec = getattr(self.env, "action_spec", None)
        if isinstance(env_action_spec, Bounded):
            ddpg_action_spec = env_action_spec.to(torch.float32)
        else:
            action_dim = self.env.action_spec.shape[-1]
            ddpg_action_spec = Bounded(
                low=-1.0,
                high=1.0,
                shape=(action_dim,),
                device=getattr(config, "device", "cpu"),
                dtype=torch.float32,
            )
            logger.warning("action_spec not Bounded spec fallback bounds=[-1, 1]")
        self.ddpg_action_spec = ddpg_action_spec

        self.exploration_module = AdditiveGaussianModule(
            spec=ddpg_action_spec,
            sigma_init=config.td3.exploration_noise_std,
            sigma_end=config.td3.exploration_noise_std,
            annealing_num_steps=config.max_steps,
        )

        logger.info(
            "init ddpg trainer actor_lr={} value_lr={} buffer_size={} tau={} exploration_noise_std={:.3f}",
            config.actor_lr,
            config.value_lr,
            config.buffer_size,
            config.tau,
            config.td3.exploration_noise_std,
        )

    @staticmethod
    def build_models(n_obs: int, n_act: int, config: Any, env: Any):
        """Factory for DDPG actor and value network."""
        actor = create_ddpg_actor(
            n_obs,
            n_act,
            hidden_dims=config.network.actor_hidden_dims,
            spec=env.action_spec,
        )
        value_net = create_value_network(
            n_obs,
            n_act,
            hidden_dims=config.network.value_hidden_dims,
        )
        return actor, value_net

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
            current_step = self._global_optimization_step(
                batch_idx, j, self.config.optim_steps_per_batch
            )

            if (
                torch.isnan(sample["next", "reward"]).any()
                or torch.isinf(sample["next", "reward"]).any()
            ):
                self._record_skipped_batch("nan/inf in reward")
                continue

            # Ensure done and terminated have consistent shapes
            done = sample["next", "done"]
            terminated = sample["next", "terminated"]
            if done.shape != terminated.shape:
                self._record_skipped_batch(
                    "done/terminated shape mismatch "
                    f"done={done.shape} terminated={terminated.shape}"
                )
                continue

            # Compute losses with error handling
            try:
                loss_vals = self.ddpg_loss(sample)
                self.successful_batches += 1
                self._consecutive_skips = 0
            except RuntimeError as e:
                if "All input tensors" in str(e) and "must share a unique shape" in str(
                    e
                ):
                    self._record_skipped_batch("tensor shape error", exc=e)
                    continue
                else:
                    raise

            # Optimize value network
            self.optimizer_value.zero_grad()
            loss_vals["loss_value"].backward()
            if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.ddpg_loss.value_network_params.values(True, True),
                    self.config.max_grad_norm,
                )
            self.optimizer_value.step()

            # Sync functional value params back to the value module
            self.ddpg_loss.value_network_params.to_module(self.value_net)

            # Optimize actor against the updated critic.
            loss_vals_actor = self.ddpg_loss(sample)
            self.optimizer_actor.zero_grad()
            loss_vals_actor["loss_actor"].backward()
            if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.ddpg_loss.actor_network_params.values(True, True),
                    self.config.max_grad_norm,
                )
            self.optimizer_actor.step()

            # Sync functional actor params back to the actor module used by the collector/evaluator
            self.ddpg_loss.actor_network_params.to_module(self.actor)

            # Update target networks
            self.updater.step()

            # Log losses
            actor_loss = loss_vals_actor["loss_actor"].item()
            value_loss = loss_vals["loss_value"].item()
            self.logs["loss_value"].append(value_loss)
            self.logs["loss_actor"].append(actor_loss)

            # Log to callback if provided
            if (
                hasattr(self, "callback")
                and self.callback
                and hasattr(self.callback, "log_training_step")
            ):
                self.callback.log_training_step(current_step, actor_loss, value_loss)

            # Periodic logging and evaluation
            if self._should_log_step(current_step):
                self._log_progress(
                    max_length,
                    buffer_len,
                    loss_vals_actor,
                    actor_loss=actor_loss,
                    value_loss=value_loss,
                )

            # Periodic evaluation
            if self._should_eval_step(current_step):
                self._evaluate()

    def _compute_exploration_ratio(self) -> float:
        return self.config.td3.exploration_noise_std

    @property
    def _algo_label(self) -> str:
        return "ddpg"

    @property
    def _value_loss_key(self) -> str:
        return "loss_value"

    def _get_checkpoint_network_state(self) -> dict:
        return {
            "actor_params_state": self.ddpg_loss.actor_network_params.state_dict(),
            "value_params_state": self.ddpg_loss.value_network_params.state_dict(),
            "target_actor_params_state": self.ddpg_loss.target_actor_network_params.state_dict(),
            "target_value_params_state": self.ddpg_loss.target_value_network_params.state_dict(),
            "optimizer_actor_state_dict": self.optimizer_actor.state_dict(),
            "optimizer_value_state_dict": self.optimizer_value.state_dict(),
        }

    def _load_checkpoint_network_state(self, checkpoint: dict) -> None:
        self.ddpg_loss.actor_network_params.load_state_dict(
            checkpoint["actor_params_state"]
        )
        self.ddpg_loss.value_network_params.load_state_dict(
            checkpoint["value_params_state"]
        )
        target_actor = getattr(self.ddpg_loss, "target_actor_network_params", None)
        target_value = getattr(self.ddpg_loss, "target_value_network_params", None)
        if target_actor is not None and "target_actor_params_state" in checkpoint:
            target_actor.load_state_dict(checkpoint["target_actor_params_state"])
        if target_value is not None and "target_value_params_state" in checkpoint:
            target_value.load_state_dict(checkpoint["target_value_params_state"])
        self.ddpg_loss.actor_network_params.to_module(self.actor)
        self.ddpg_loss.value_network_params.to_module(self.value_net)
        self.optimizer_actor.load_state_dict(checkpoint["optimizer_actor_state_dict"])
        self.optimizer_value.load_state_dict(checkpoint["optimizer_value_state_dict"])

    def train(self, callback: Any = None) -> dict[str, list]:
        """Run training loop for DDPG agent with batch summary."""
        self.noisy_policy = TensorDictSequential(self.actor, self.exploration_module)
        self._initialize_offpolicy_collection_policy(
            self.noisy_policy,
            self.ddpg_action_spec,
            algorithm_label="DDPG",
        )

        def on_batch_start(i, data) -> None:
            if is_level_enabled("TRACE") and i % 10 == 0:
                episode_rewards = data["next", "reward"]
                buffer_len = len(self.replay_buffer)
                logger.trace(
                    "ddpg batch={} steps={} buffer_size={}", i, data.numel(), buffer_len
                )
                logger.trace(
                    "ddpg episode reward stats mean={} std={}",
                    episode_rewards.mean(),
                    episode_rewards.std(),
                )
                collected_actions = data["action"]
                logger.trace(
                    "ddpg collected action stats mean={} std={}",
                    collected_actions.mean(),
                    collected_actions.std(),
                )

        def on_batch_end(i, data) -> None:
            self._maybe_switch_from_random_warmup(algorithm_label="DDPG")

        return self._run_training_loop(
            callback,
            start_message="Starting DDPG training",
            completion_prefix="DDPG Training complete",
            on_batch_start=on_batch_start,
            on_batch_end=on_batch_end,
            on_train_end=self._log_batch_summary,
        )
