"""TD3 Trainer implementation."""

from trading_rl.trainers.registry import register_trainer
from typing import Any

import torch
from tensordict.nn import TensorDictSequential
from torch.optim import Adam
from torchrl.data import Bounded
from torchrl.modules import AdditiveGaussianModule
from torchrl.objectives import SoftUpdate
from torchrl.objectives import TD3Loss as TorchRLTd3Loss

from logger import get_logger, is_level_enabled
from trading_rl.config import EvaluationConfig, TrainingConfig
from trading_rl.constants import LossFunction
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


@register_trainer("TD3", continuous=True)
class TD3Trainer(BaseTrainer):
    """Trainer for TD3 algorithm on trading environments."""

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

        # Prefer the environment's bounded action spec so exploration and TD3
        # target action clipping are defined in the same domain as the env.
        env_action_spec = getattr(self.env, "action_spec", None)
        if isinstance(env_action_spec, Bounded):
            # Ensure dtype is float32 to match network parameters
            td3_action_spec = env_action_spec.to(torch.float32)
        else:
            action_dim = self.env.action_spec.shape[-1]
            td3_action_spec = Bounded(
                low=-1.0,
                high=1.0,
                shape=(action_dim,),
                device=getattr(config, "device", "cpu"),
                dtype=torch.float32,
            )
            logger.warning(
                "action_spec not Bounded spec fallback bounds=[-1, 1]"
            )
        self.td3_action_spec = td3_action_spec

        # Gaussian exploration around the deterministic policy
        self.exploration_module = AdditiveGaussianModule(
            spec=td3_action_spec,
            sigma_init=config.td3.exploration_noise_std,
            sigma_end=config.td3.exploration_noise_std,
            annealing_num_steps=config.max_steps,
        )

        # TD3 uses two critics; configure loss and optimizers
        self.td3_loss = TD3Loss(
            actor_network=actor,
            qvalue_network=value_net,
            action_spec=td3_action_spec,
            num_qvalue_nets=2,
            policy_noise=config.td3.policy_noise,
            noise_clip=config.td3.noise_clip,
            loss_function=getattr(config, "loss_function", LossFunction.L2),
            delay_actor=config.td3.delay_actor,
            delay_qvalue=config.td3.delay_qvalue,
        )

        for attr in ("actor_network_params", "qvalue_network_params"):
            params_td = getattr(self.td3_loss, attr, None)
            if params_td is not None and hasattr(params_td, "unlock_"):
                params_td.unlock_()

        self.updater = SoftUpdate(self.td3_loss, tau=config.tau)

        self.optimizer_actor = Adam(
            self.td3_loss.actor_network_params.values(True, True),
            lr=config.actor_lr,
            weight_decay=config.actor_weight_decay,
        )
        self.optimizer_value = Adam(
            self.td3_loss.qvalue_network_params.values(True, True),
            lr=config.value_lr,
            weight_decay=config.value_weight_decay,
        )

        self.policy_delay = config.td3.policy_delay

        logger.info(
            "init td3 trainer actor_lr={} value_lr={} exploration_noise_std=%.3f policy_noise=%.3f noise_clip=%.3f policy_delay=%d",
            config.actor_lr,
            config.value_lr,
            config.td3.exploration_noise_std,
            config.td3.policy_noise,
            config.td3.noise_clip,
            self.policy_delay,
        )

    def _should_update_actor(self, current_step: int) -> bool:
        """Return True after policy_delay critic updates have completed."""
        return current_step > 0 and current_step % self.policy_delay == 0

    @staticmethod
    def build_models(n_obs: int, n_act: int, config: Any, env: Any):
        """Factory for TD3 actor and Q-value networks."""
        actor = create_td3_actor(
            n_obs,
            n_act,
            hidden_dims=config.network.actor_hidden_dims,
            spec=env.action_spec,
        )
        value_net = create_td3_qvalue_network(
            n_obs,
            n_act,
            hidden_dims=config.network.value_hidden_dims,
        )
        return actor, value_net

    def _normalize_batch_shapes(self, sample) -> None:
        """Ensure reward/done/terminated have consistent 2-D shapes for TD3Loss."""
        for key in [("next", "reward"), ("next", "done"), ("next", "terminated")]:
            tensor = sample.get(key)
            if tensor is None:
                continue
            if tensor.ndim == 0:
                tensor = tensor.unsqueeze(0).unsqueeze(-1)
            elif tensor.ndim == 1:
                tensor = tensor.unsqueeze(-1)
            sample.set(key, tensor)

    def _update_critics(self, sample) -> tuple[Any, float] | None:
        """Run one critic gradient step. Returns (loss_vals, value_loss) or None on skip."""
        try:
            loss_vals = self.td3_loss(sample)
            self.successful_batches += 1
            self._consecutive_skips = 0
            if is_level_enabled("TRACE"):
                logger.trace(
                    "td3 losses loss_qvalue={} loss_actor={}",
                    loss_vals["loss_qvalue"].item(), loss_vals["loss_actor"].item(),
                )
        except RuntimeError as e:
            if "All input tensors" in str(e) and "must share a unique shape" in str(e):
                self._record_skipped_batch("tensor shape error", exc=e)
                return None
            raise

        self.optimizer_value.zero_grad()
        loss_vals["loss_qvalue"].backward()
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.td3_loss.qvalue_network_params.values(True, True),
                self.config.max_grad_norm,
            )
        self.optimizer_value.step()
        self.td3_loss.qvalue_network_params.to_module(self.value_net)

        value_loss = loss_vals["loss_qvalue"].item()
        self.logs["loss_value"].append(value_loss)
        return loss_vals, value_loss

    def _update_actor_and_targets(self, sample) -> tuple[float, dict | None]:
        """Run delayed actor gradient step and target network update.

        TD3 recomputes the loss on the same batch after critic weights have been
        updated so the actor gradient uses the latest Q-values.

        Returns (actor_loss, extra_metrics).
        """
        loss_vals_actor = self.td3_loss(sample)

        self.optimizer_actor.zero_grad()
        loss_vals_actor["loss_actor"].backward()
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.td3_loss.actor_network_params.values(True, True),
                self.config.max_grad_norm,
            )
        self.optimizer_actor.step()
        self.td3_loss.actor_network_params.to_module(self.actor)
        self.updater.step()

        actor_loss = loss_vals_actor["loss_actor"].item()

        extra_metrics: dict | None = None
        if is_level_enabled("TRACE"):
            actor_sum = float(sum(p.abs().sum().item() for p in self.actor.parameters()))
            critic_sum = float(sum(p.abs().sum().item() for p in self.value_net.parameters()))
            extra_metrics = {
                "actor_param_abs_sum": actor_sum,
                "critic_param_abs_sum": critic_sum,
            }

            params = self.td3_loss.qvalue_network_params
            if getattr(params, "batch_size", None) and params.batch_size[0] >= 2:
                params0, params1 = params[0], params[1]
                max_diff = max(
                    (p0 - params1.get(key)).abs().max().item()
                    for key, p0 in params0.items(True, True)
                    if isinstance(p0, torch.Tensor) and isinstance(params1.get(key), torch.Tensor)
                )
                extra_metrics["critic_qvalue_params_max_diff"] = max_diff

        return actor_loss, extra_metrics

    def _optimization_step(
        self, batch_idx: int, max_length: int, buffer_len: int
    ) -> None:
        for j in range(self.config.optim_steps_per_batch):
            sample = self.replay_buffer.sample(self.config.sample_size)
            current_step = self._global_optimization_step(
                batch_idx, j, self.config.optim_steps_per_batch
            )

            self._normalize_batch_shapes(sample)

            if is_level_enabled("TRACE"):
                actions = sample["action"]
                rewards = sample["next", "reward"]
                logger.trace(
                    "td3 batch sample stats batch=%d step=%d "
                    "action_mean={} action_std={} action_min={} action_max={} "
                    "reward_mean={} reward_std={} reward_min={} reward_max={}",
                    batch_idx, j,
                    actions.mean(), actions.std(), actions.min(), actions.max(),
                    rewards.mean(), rewards.std(), rewards.min(), rewards.max(),
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

            result = self._update_critics(sample)
            if result is None:
                continue
            loss_vals, value_loss = result

            if self._should_update_actor(current_step):
                actor_loss, extra_metrics = self._update_actor_and_targets(sample)
            else:
                actor_loss = loss_vals["loss_actor"].item()
                extra_metrics = None

            self.logs["loss_actor"].append(actor_loss)

            if (
                hasattr(self, "callback")
                and self.callback
                and hasattr(self.callback, "log_training_step")
            ):
                self.callback.log_training_step(
                    current_step, actor_loss, value_loss, extra_metrics=extra_metrics
                )

            if self._should_log_step(current_step):
                self._log_progress(max_length, buffer_len, loss_vals)

            if self._should_eval_step(current_step):
                self._evaluate()

    def _compute_exploration_ratio(self) -> float:
        return self.config.td3.policy_noise

    @property
    def _algo_label(self) -> str:
        return "td3"

    def _get_checkpoint_network_state(self) -> dict:
        return {
            "actor_params_state": self.td3_loss.actor_network_params.state_dict(),
            "target_actor_params_state": self.td3_loss.target_actor_network_params.state_dict(),
            "value_params_state": self.td3_loss.qvalue_network_params.state_dict(),
            "target_value_params_state": self.td3_loss.target_qvalue_network_params.state_dict(),
            "optimizer_actor_state_dict": self.optimizer_actor.state_dict(),
            "optimizer_value_state_dict": self.optimizer_value.state_dict(),
        }

    def _load_checkpoint_network_state(self, checkpoint: dict) -> None:
        self.td3_loss.actor_network_params.load_state_dict(checkpoint["actor_params_state"])
        self.td3_loss.qvalue_network_params.load_state_dict(checkpoint["value_params_state"])
        self.td3_loss.target_actor_network_params.load_state_dict(checkpoint["target_actor_params_state"])
        self.td3_loss.target_qvalue_network_params.load_state_dict(checkpoint["target_value_params_state"])
        self.td3_loss.actor_network_params.to_module(self.actor)
        self.td3_loss.qvalue_network_params.to_module(self.value_net)
        self.optimizer_actor.load_state_dict(checkpoint["optimizer_actor_state_dict"])
        self.optimizer_value.load_state_dict(checkpoint["optimizer_value_state_dict"])

    def _post_eval_trace_hook(self, eval_rollout: Any) -> None:
        if not is_level_enabled("TRACE"):
            return
        import numpy as np
        actions = eval_rollout["action"]
        logger.trace(
            "td3 eval action stats n=%d mean=%.4f std=%.4f",
            actions.numel(), actions.mean(), actions.std(),
        )
        logger.trace("td3 eval action min={} max={}", actions.min(), actions.max())
        actions_flat = actions.flatten().cpu().detach().numpy()
        unique_actions, counts = np.unique(np.round(actions_flat, 2), return_counts=True)
        if len(unique_actions) <= 10:
            logger.trace(
                "td3 eval action distribution={}",
                dict(zip(unique_actions, counts, strict=False)),
            )
        else:
            percentiles = np.percentile(actions_flat, [0, 25, 50, 75, 100])
            logger.trace("td3 eval action percentiles={}", percentiles.tolist())
        if actions.std() < 0.01:
            logger.warning("td3 eval agent stuck action_std={:.6f}", actions.std())

    def train(self, callback: Any = None) -> dict[str, list]:
        """Run training loop for RL agent, with exploration for TD3."""
        logger.debug(
            "td3 train config max_steps=%d init_rand_steps=%d frames_per_batch=%d buffer_size=%d",
            self.config.max_steps, self.config.init_rand_steps,
            self.config.frames_per_batch, self.config.buffer_size,
        )

        # Create the noisy policy by chaining actor + exploration module
        # We do this once here to use when switching
        self.noisy_policy = TensorDictSequential(self.actor, self.exploration_module)
        self._initialize_offpolicy_collection_policy(
            self.noisy_policy,
            self.td3_action_spec,
            algorithm_label="TD3",
        )

        def on_batch_start(i, data) -> None:
            if is_level_enabled("TRACE") and i % 10 == 0:
                episode_rewards = data["next", "reward"]
                buffer_len = len(self.replay_buffer)
                logger.trace(
                    "batch=%d steps=%d buffer_size=%d", i, data.numel(), buffer_len
                )
                logger.trace(
                    "episode reward stats mean={} std={}",
                    episode_rewards.mean(), episode_rewards.std(),
                )
                collected_actions = data["action"]
                logger.trace(
                    "collected action stats mean={} std={}",
                    collected_actions.mean(), collected_actions.std(),
                )
                self._log_sample_transitions(data, n=3)

        def on_batch_end(i, data) -> None:
            self._maybe_switch_from_random_warmup(algorithm_label="TD3")

        return self._run_training_loop(
            callback,
            start_message="Starting TD3 training",
            completion_prefix="TD3 Training complete",
            on_batch_start=on_batch_start,
            on_batch_end=on_batch_end,
            on_train_end=self._log_batch_summary,
        )
