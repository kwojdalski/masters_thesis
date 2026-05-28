"""RecurrentPPO Trainer implementation."""

import logging
from trading_rl.trainers.registry import register_trainer
from typing import Any

import torch
from tensordict.nn import InteractionType
from torchrl.envs.utils import set_exploration_type

from logger import get_logger
from trading_rl.config import EvaluationConfig, TrainingConfig
from trading_rl.models import (
    create_recurrent_ppo_actor,
    create_recurrent_ppo_value_network,
)
from trading_rl.trainers.base import _log_network_stats
from trading_rl.trainers.ppo import PPOTrainerContinuous

logger = get_logger(__name__)


@register_trainer("RECURRENT_PPO", continuous=True)
class RecurrentPPOTrainer(PPOTrainerContinuous):
    """PPO trainer with GRU recurrent actor and value networks.

    Temporal structure of episode sequences is preserved: minibatch sampling
    is disabled in favour of full-batch updates so that GRU hidden states
    remain consistent across a rollout.

    The GRU hidden size is taken from ``config.network.actor_hidden_dims[0]``
    and the number of layers from ``config.network.gru_num_layers``.
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
        """Initialize RecurrentPPO trainer.

        Args:
            actor: GRU-backed stochastic actor network
            value_net: GRU-backed value network
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
            checkpoint_dir=checkpoint_dir,
            checkpoint_prefix=checkpoint_prefix,
        )

        logger.info(
            "init recurrent_ppo trainer lr=%s clip_epsilon=%.3f entropy_bonus=%.4f "
            "gru_num_layers=%d",
            config.actor_lr,
            getattr(config, "clip_epsilon", 0.2),
            getattr(config, "entropy_bonus", 0.01),
            getattr(config.network, "gru_num_layers", 1),
        )

    @staticmethod
    def build_models(n_obs: int, n_act: int, config: Any, env: Any):
        """Factory for RecurrentPPO GRU actor and value network."""
        actor = create_recurrent_ppo_actor(
            n_obs,
            n_act,
            hidden_dims=config.network.actor_hidden_dims,
            spec=env.action_spec,
            gru_num_layers=getattr(config.network, "gru_num_layers", 1),
        )
        value_net = create_recurrent_ppo_value_network(
            n_obs,
            hidden_dims=config.network.value_hidden_dims,
            gru_num_layers=getattr(config.network, "gru_num_layers", 1),
        )
        return actor, value_net

    def _optimization_step(
        self, batch_idx: int, max_length: int, buffer_len: int
    ) -> None:
        """Perform recurrent PPO optimization over full ordered sequences.

        Unlike the base PPO trainer, minibatch sampling is skipped: the full
        collected rollout is used in temporal order each epoch so that GRU
        hidden states computed during the forward pass remain valid.

        Args:
            batch_idx: Current batch index
            max_length: Maximum episode length in buffer
            buffer_len: Current replay buffer size
        """
        ppo_epochs = getattr(self.config, "ppo_epochs", 4)

        # GAE / value targets computed on the full ordered rollout before any updates.
        ppo_batch = self._current_batch.clone()
        with torch.no_grad():
            self.ppo_loss.value_estimator(
                ppo_batch,
                params=self.ppo_loss._cached_critic_network_params_detached,
                target_params=getattr(self.ppo_loss, "target_critic_network_params", None),
            )

        self.optimizer.zero_grad()

        for j in range(ppo_epochs):
            current_step = self._global_optimization_step(batch_idx, j, ppo_epochs)

            # Pass the full ordered batch — temporal structure must not be broken
            # by random minibatch sampling when the policy contains GRU layers.
            loss_vals = self.ppo_loss(ppo_batch)

            total_loss = (
                loss_vals["loss_objective"]
                + loss_vals["loss_critic"]
                + loss_vals["loss_entropy"]
            )
            total_loss.backward()
            if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for group in self.optimizer.param_groups for p in group["params"]],
                    self.config.max_grad_norm,
                )
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.ppo_loss.actor_network_params.to_module(self.actor)
            self.ppo_loss.critic_network_params.to_module(self.value_net)

            actor_loss = loss_vals["loss_objective"].item()
            value_loss = loss_vals["loss_critic"].item()
            entropy_loss = loss_vals["loss_entropy"].item()

            self.logs["loss_actor"].append(actor_loss)
            self.logs["loss_value"].append(value_loss)
            self.logs["loss_entropy"].append(entropy_loss)

            if (
                hasattr(self, "callback")
                and self.callback
                and hasattr(self.callback, "log_training_step")
            ):
                self.callback.log_training_step(current_step, actor_loss, value_loss)

            if self._should_log_step(current_step):
                self._log_progress(max_length, buffer_len, loss_vals)

            if self._should_eval_step(current_step):
                self._evaluate()

    def _log_progress(self, max_length: int, buffer_len: int, loss_vals: dict) -> None:
        curr_loss_actor = loss_vals["loss_objective"].item()
        curr_loss_value = loss_vals["loss_critic"].item()
        curr_loss_entropy = loss_vals["loss_entropy"].item()

        logger.info(
            "recurrent_ppo step max_steps=%d buffer_size=%d loss_value=%.4f "
            "loss_actor=%.4f loss_entropy=%.4f",
            max_length, buffer_len, curr_loss_value, curr_loss_actor, curr_loss_entropy,
        )

        if logger.isEnabledFor(logging.DEBUG):
            _log_network_stats(logger, "recurrent_ppo", self.actor, self.value_net)

    def _evaluate(self) -> None:
        """Evaluate current recurrent PPO policy."""
        with torch.no_grad():
            n_eval = (
                self.eval_config.resolve_eval_steps(self._eval_data_len)
                if self._eval_data_len is not None
                else self.eval_config.eval_steps
            )
            eval_env = self._eval_env or self.env
            if self._eval_env is None:
                logger.warning(
                    "recurrent_ppo _evaluate: no dedicated eval env set; using training env "
                    "which may corrupt SyncDataCollector state"
                )
            try:
                with set_exploration_type(InteractionType.MODE):
                    eval_rollout = eval_env.rollout(n_eval, self.actor)
            except RuntimeError:
                logger.debug("Mode not available for distribution, falling back to Mean")
                with set_exploration_type(InteractionType.DETERMINISTIC):
                    eval_rollout = eval_env.rollout(n_eval, self.actor)

            mean_reward = eval_rollout["next", "reward"].mean().item()
            sum_reward = eval_rollout["next", "reward"].sum().item()
            max_steps = eval_rollout["step_count"].max().item()

            self.logs["eval_reward_mean"].append(mean_reward)
            self.logs["eval_reward_sum"].append(sum_reward)
            self.logs["eval_step_count"].append(max_steps)

            eval_data_len = self._eval_data_len if self._eval_data_len is not None else "?"
            logger.info(
                "recurrent_ppo eval mean_reward=%.4f sum_reward=%.4f eval_steps=%d eval_data_len=%s",
                mean_reward, sum_reward, max_steps, eval_data_len,
            )

            del eval_rollout

    def train(self, callback: Any = None) -> dict[str, list]:
        """Run training loop for RecurrentPPO agent."""
        return self._run_training_loop(
            callback,
            start_message="Starting RecurrentPPO training",
            completion_prefix="RecurrentPPO Training complete",
        )
