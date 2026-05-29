"""SAC Trainer implementation."""

from trading_rl.trainers.registry import register_trainer
from typing import Any

import torch
from torch.optim import Adam
from torchrl.data import Bounded
from torchrl.objectives import SACLoss, SoftUpdate

from logger import get_logger, is_level_enabled
from trading_rl.config import EvaluationConfig, TrainingConfig
from trading_rl.constants import LossFunction
from trading_rl.models import create_sac_actor, create_sac_qvalue_network
from trading_rl.trainers.base import _log_network_stats, BaseTrainer

logger = get_logger(__name__)


@register_trainer("SAC", continuous=True)
class SACTrainer(BaseTrainer):
    """Trainer for SAC (Soft Actor-Critic) algorithm on trading environments.

    SAC is an off-policy algorithm with entropy regularisation. The stochastic
    TanhNormal actor provides exploration without a separate noise module; the
    temperature parameter alpha is tuned automatically via a third optimizer.
    """

    def __init__(
        self,
        actor: Any,
        value_net: Any,
        env: Any,
        config: TrainingConfig,
        *,
        eval_config: "EvaluationConfig | None" = None,
        checkpoint_dir: str | None = None,
        checkpoint_prefix: str | None = None,
    ):
        """Initialize SAC trainer.

        Args:
            actor: Stochastic actor network (TanhNormal distribution)
            value_net: Q-value network (double Q by default)
            env: Trading environment
            config: Training configuration
            eval_config: Evaluation configuration (eval_steps, eval_fraction, log_data)
            checkpoint_dir: Directory for checkpoints
            checkpoint_prefix: Prefix for checkpoint filenames
        """
        super().__init__(
            actor=actor,
            value_net=value_net,
            env=env,
            config=config,
            eval_config=eval_config,
            enable_composite_lp=True,
            checkpoint_dir=checkpoint_dir,
            checkpoint_prefix=checkpoint_prefix,
        )

        env_action_spec = getattr(self.env, "action_spec", None)
        if isinstance(env_action_spec, Bounded):
            sac_action_spec = env_action_spec.to(torch.float32)
        else:
            action_dim = self.env.action_spec.shape[-1]
            sac_action_spec = Bounded(
                low=-1.0,
                high=1.0,
                shape=(action_dim,),
                device=getattr(config, "device", "cpu"),
                dtype=torch.float32,
            )
            logger.warning("action_spec not Bounded spec fallback bounds=[-1, 1]")
        self.sac_action_spec = sac_action_spec

        target_entropy = config.sac.target_entropy
        if target_entropy is None:
            target_entropy = -float(self.n_act)

        self.sac_loss = SACLoss(
            actor_network=actor,
            qvalue_network=value_net,
            num_qvalue_nets=2,
            loss_function=getattr(config, "loss_function", LossFunction.L2),
            alpha_init=config.sac.initial_alpha,
            fixed_alpha=False,
            target_entropy=target_entropy,
            delay_actor=False,
            delay_qvalue=True,
        )

        self.updater = SoftUpdate(self.sac_loss, tau=config.tau)

        self.optimizer_actor = Adam(
            self.sac_loss.actor_network_params.values(True, True),
            lr=config.actor_lr,
            weight_decay=config.actor_weight_decay,
        )
        self.optimizer_value = Adam(
            self.sac_loss.qvalue_network_params.values(True, True),
            lr=config.value_lr,
            weight_decay=config.value_weight_decay,
        )
        self.optimizer_alpha = Adam(
            [self.sac_loss.log_alpha],
            lr=config.sac.alpha_lr,
        )

        logger.info(
            "init sac trainer actor_lr={} value_lr={} alpha_lr={} buffer_size={} tau={} "
            "initial_alpha={} target_entropy={}",
            config.actor_lr,
            config.value_lr,
            config.sac.alpha_lr,
            config.buffer_size,
            config.tau,
            config.sac.initial_alpha,
            target_entropy,
        )

    @staticmethod
    def build_models(n_obs: int, n_act: int, config: Any, env: Any):
        """Factory for SAC actor and Q-value networks."""
        actor = create_sac_actor(
            n_obs,
            n_act,
            hidden_dims=config.network.actor_hidden_dims,
            spec=env.action_spec,
        )
        value_net = create_sac_qvalue_network(
            n_obs,
            n_act,
            hidden_dims=config.network.value_hidden_dims,
        )
        return actor, value_net

    def _optimization_step(
        self, batch_idx: int, max_length: int, buffer_len: int
    ) -> None:
        """Perform SAC optimization: critic update → actor update → alpha update.

        Three separate forward passes avoid computation-graph conflicts between
        the critic, actor, and temperature objectives.

        Args:
            batch_idx: Current batch index
            max_length: Maximum episode length in buffer
            buffer_len: Current replay buffer size
        """
        for j in range(self.config.optim_steps_per_batch):
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

            done = sample["next", "done"]
            terminated = sample["next", "terminated"]
            if done.shape != terminated.shape:
                self._record_skipped_batch(
                    f"done/terminated shape mismatch done={done.shape} terminated={terminated.shape}"
                )
                continue

            # 1. Critic (Q-value) update
            try:
                loss_vals = self.sac_loss(sample)
                self.successful_batches += 1
                self._consecutive_skips = 0
            except RuntimeError as e:
                if "All input tensors" in str(e) and "must share a unique shape" in str(e):
                    self._record_skipped_batch("tensor shape error", exc=e)
                    continue
                else:
                    raise

            self.optimizer_value.zero_grad()
            loss_vals["loss_qvalue"].backward()
            if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.sac_loss.qvalue_network_params.values(True, True),
                    self.config.max_grad_norm,
                )
            self.optimizer_value.step()
            self.sac_loss.qvalue_network_params.to_module(self.value_net)

            # 2. Actor update (fresh forward pass)
            loss_vals_actor = self.sac_loss(sample)

            self.optimizer_actor.zero_grad()
            loss_vals_actor["loss_actor"].backward()
            if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.sac_loss.actor_network_params.values(True, True),
                    self.config.max_grad_norm,
                )
            self.optimizer_actor.step()
            self.sac_loss.actor_network_params.to_module(self.actor)

            # 3. Alpha (temperature) update (fresh forward pass)
            loss_vals_alpha = self.sac_loss(sample)

            self.optimizer_alpha.zero_grad()
            loss_vals_alpha["loss_alpha"].backward()
            self.optimizer_alpha.step()

            # Update target Q networks after all three updates
            self.updater.step()

            # Logging
            value_loss = loss_vals["loss_qvalue"].item()
            actor_loss = loss_vals_actor["loss_actor"].item()
            alpha_loss = loss_vals_alpha["loss_alpha"].item()
            alpha_val = loss_vals_alpha["alpha"].item()
            entropy_tensor = loss_vals_alpha.get("entropy", None)
            entropy_val = entropy_tensor.item() if entropy_tensor is not None else float("nan")

            self.logs["loss_value"].append(value_loss)
            self.logs["loss_actor"].append(actor_loss)
            self.logs["loss_alpha"].append(alpha_loss)
            self.logs["alpha"].append(alpha_val)
            self.logs["entropy"].append(entropy_val)

            if (
                hasattr(self, "callback")
                and self.callback
                and hasattr(self.callback, "log_training_step")
            ):
                self.callback.log_training_step(current_step, actor_loss, value_loss)

            if self._should_log_step(current_step):
                self._log_progress(max_length, buffer_len, loss_vals, loss_vals_actor, loss_vals_alpha)

            if self._should_eval_step(current_step):
                self._evaluate()

    def _log_progress(
        self,
        max_length: int,
        buffer_len: int,
        loss_vals: dict,
        loss_vals_actor: dict,
        loss_vals_alpha: dict,
    ) -> None:
        curr_loss_value = loss_vals["loss_qvalue"].item()
        curr_loss_actor = loss_vals_actor["loss_actor"].item()
        curr_loss_alpha = loss_vals_alpha["loss_alpha"].item()
        curr_alpha = loss_vals_alpha["alpha"].item()

        logger.info(
            "sac step max_steps={} buffer_size={} loss_value={:.4f} loss_actor={:.4f} "
            "loss_alpha={} alpha={}",
            max_length, buffer_len, curr_loss_value, curr_loss_actor,
            curr_loss_alpha, curr_alpha,
        )

        if is_level_enabled("TRACE"):
            _log_network_stats(logger, "sac", self.actor, self.value_net)

    def _compute_exploration_ratio(self) -> float:
        # Current alpha temperature serves as a proxy for exploration intensity
        return float(self.sac_loss.log_alpha.exp().item())

    @property
    def _algo_label(self) -> str:
        return "sac"

    def _get_checkpoint_network_state(self) -> dict:
        return {
            "actor_params_state": self.sac_loss.actor_network_params.state_dict(),
            "value_params_state": self.sac_loss.qvalue_network_params.state_dict(),
            "target_value_params_state": self.sac_loss.target_qvalue_network_params.state_dict(),
            "log_alpha": self.sac_loss.log_alpha.item(),
            "optimizer_actor_state_dict": self.optimizer_actor.state_dict(),
            "optimizer_value_state_dict": self.optimizer_value.state_dict(),
            "optimizer_alpha_state_dict": self.optimizer_alpha.state_dict(),
        }

    def _load_checkpoint_network_state(self, checkpoint: dict) -> None:
        self.sac_loss.actor_network_params.load_state_dict(checkpoint["actor_params_state"])
        self.sac_loss.qvalue_network_params.load_state_dict(checkpoint["value_params_state"])
        target_value = getattr(self.sac_loss, "target_qvalue_network_params", None)
        if target_value is not None and "target_value_params_state" in checkpoint:
            target_value.load_state_dict(checkpoint["target_value_params_state"])
        self.sac_loss.actor_network_params.to_module(self.actor)
        self.sac_loss.qvalue_network_params.to_module(self.value_net)
        log_alpha_val = checkpoint.get("log_alpha")
        if log_alpha_val is not None:
            with torch.no_grad():
                self.sac_loss.log_alpha.fill_(log_alpha_val)
        self.optimizer_actor.load_state_dict(checkpoint["optimizer_actor_state_dict"])
        self.optimizer_value.load_state_dict(checkpoint["optimizer_value_state_dict"])
        self.optimizer_alpha.load_state_dict(checkpoint["optimizer_alpha_state_dict"])

    def train(self, callback: Any = None) -> dict[str, list]:
        """Run training loop for SAC agent."""
        # SAC actor is stochastic — no separate exploration module is needed.
        self._initialize_offpolicy_collection_policy(
            self.actor,
            self.sac_action_spec,
            algorithm_label="SAC",
        )

        def on_batch_start(i, data) -> None:
            if is_level_enabled("TRACE") and i % 10 == 0:
                episode_rewards = data["next", "reward"]
                buffer_len = len(self.replay_buffer)
                logger.trace(
                    "sac batch={} steps={} buffer_size={}", i, data.numel(), buffer_len
                )
                logger.trace(
                    "sac episode reward stats mean={} std={}",
                    episode_rewards.mean(), episode_rewards.std(),
                )
                collected_actions = data["action"]
                logger.trace(
                    "sac collected action stats mean={} std={}",
                    collected_actions.mean(), collected_actions.std(),
                )

        def on_batch_end(i, data) -> None:
            self._maybe_switch_from_random_warmup(algorithm_label="SAC")

        return self._run_training_loop(
            callback,
            start_message="Starting SAC training",
            completion_prefix="SAC Training complete",
            on_batch_start=on_batch_start,
            on_batch_end=on_batch_end,
            on_train_end=self._log_batch_summary,
        )

