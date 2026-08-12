"""PPO Trainer implementation."""

from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from plotnine import (
    aes,
    geom_area,
    geom_line,
    geom_ribbon,
    ggplot,
    labs,
    scale_fill_manual,
    scale_x_continuous,
)
from tensordict import TensorDict
from tensordict.nn import InteractionType
from torch.optim import Adam
from torchrl.envs.utils import set_exploration_type
from torchrl.objectives import ClipPPOLoss

from logger import get_logger, is_level_enabled
from trading_rl.config import EvaluationConfig, TrainingConfig
from trading_rl.constants import LossFunction, TradePosition
from trading_rl.evaluation.thesis_theme import FIGURE_HEIGHT, FIGURE_WIDTH, thesis_theme
from trading_rl.models import (
    create_continuous_ppo_actor,
    create_ppo_actor,
    create_ppo_value_network,
)
from trading_rl.trainers.base import BaseTrainer, EvaluationOutput, _log_network_stats
from trading_rl.trainers.registry import register_trainer

logger = get_logger(__name__)


def _run_viz_rollout(env, actor, max_steps: int, max_episode_length: int, process_step) -> None:
    """Shared environment loop for action-distribution visualizations.

    Drives ``env`` for ``max_steps`` steps.  At each step ``process_step(step,
    actor_output)`` is called — it should record whatever per-step data it needs
    and return the action tensor (batch-dim included, ready for env.step) or
    ``None`` to skip the step.
    """
    obs = env.reset()
    current_episode_steps = 0

    for step in range(max_steps):
        if current_episode_steps >= max_episode_length:
            obs = env.reset()
            current_episode_steps = 0

        current_obs = obs["observation"] if (hasattr(obs, "get") and "observation" in obs) else obs
        if isinstance(obs, TensorDict):
            actor_input = obs.clone()
        else:
            batch_size = (
                [current_obs.shape[0]]
                if hasattr(current_obs, "shape") and len(getattr(current_obs, "shape", [])) > 0
                else [1]
            )
            actor_input = TensorDict(
                {"observation": torch.as_tensor(current_obs)},
                batch_size=batch_size,
            )

        actor_output = actor(actor_input)
        action_tensor = process_step(step, actor_output)
        if action_tensor is None:
            continue

        expected_batch = obs.batch_size[0] if (hasattr(obs, "batch_size") and obs.batch_size) else 1
        action_tensor = torch.as_tensor(action_tensor)
        if action_tensor.dim() == 0:
            action_tensor = action_tensor.unsqueeze(0)
        if action_tensor.shape[0] != expected_batch:
            action_tensor = (
                action_tensor.expand(expected_batch, *action_tensor.shape[1:])
                if action_tensor.dim() > 1
                else action_tensor.expand(expected_batch)
            )

        if not (hasattr(obs, "clone") and hasattr(obs, "set")):
            raise RuntimeError("Environment observation is not a TensorDict")
        action_td = obs.clone()
        action_td.set("action", action_tensor)
        step_result = env.step(action_td)

        obs = step_result.get("next").clone() if "next" in step_result.keys() else step_result.clone()
        current_episode_steps += 1

        done_tensor = obs.get("done", torch.tensor([False])) if hasattr(obs, "get") else None
        if done_tensor is not None and torch.as_tensor(done_tensor).any():
            obs = env.reset()
            current_episode_steps = 0


@register_trainer("PPO")
class PPOTrainer(BaseTrainer):
    """Trainer for PPO algorithm on trading environments."""

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
            enable_composite_lp=False,
            checkpoint_dir=checkpoint_dir,
            checkpoint_prefix=checkpoint_prefix,
            use_replay_buffer=False,
        )

        # Initialize PPO loss module
        self.ppo_loss = ClipPPOLoss(
            actor_network=actor,
            critic_network=value_net,
            clip_epsilon=config.ppo.clip_epsilon,
            entropy_bonus=config.ppo.entropy_bonus,
            critic_coeff=config.ppo.vf_coef,
            loss_critic_type=getattr(config, "loss_function", LossFunction.L2),
            normalize_advantage=True,
        )

        # Separate optimizers with distinct weight decay for actor vs critic
        self.optimizer = Adam([
            {
                "params": list(self.ppo_loss.actor_network_params.values(True, True)),
                "lr": config.actor_lr,
                "weight_decay": getattr(config, "actor_weight_decay", 0.0),
            },
            {
                "params": list(self.ppo_loss.critic_network_params.values(True, True)),
                "lr": getattr(config, "value_lr", config.actor_lr),
                "weight_decay": getattr(config, "value_weight_decay", 0.0),
            },
        ])

        logger.info(
            "init ppo trainer lr={} clip_epsilon={} entropy_bonus={}",
            config.actor_lr,
            config.ppo.clip_epsilon,
            config.ppo.entropy_bonus,
        )

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
        ppo_epochs = self.config.ppo.epochs

        # Compute trajectory-based targets (GAE/value_target) on the full ordered
        # rollout before random minibatch sampling. Computing these on shuffled
        # minibatches breaks temporal structure and can destabilize PPO updates.
        ppo_batch = self._current_batch.clone()
        with torch.no_grad():
            self.ppo_loss.value_estimator(
                ppo_batch,
                params=self.ppo_loss._cached_critic_network_params_detached,
                target_params=getattr(self.ppo_loss, "target_critic_network_params", None),
            )

        # PPO is on-policy: create temporary buffer from fresh batch only.
        # SamplerWithoutReplacement ensures each transition is seen exactly once
        # per epoch — RandomSampler (the default) allows duplicates and skips
        # some transitions, violating the on-policy assumption.
        from torchrl.data import LazyTensorStorage, ReplayBuffer
        from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement

        fresh_buffer = ReplayBuffer(
            storage=LazyTensorStorage(self.config.frames_per_batch),
            sampler=SamplerWithoutReplacement(),
        )
        fresh_buffer.extend(ppo_batch)

        # Clear gradients before starting PPO epochs to prevent accumulation
        self.optimizer.zero_grad()

        for j in range(ppo_epochs):
            # Sample from FRESH batch only (not from accumulated old experiences)
            sample = fresh_buffer.sample(self.config.sample_size)
            current_step = self._global_optimization_step(batch_idx, j, ppo_epochs)

            # Compute PPO losses
            loss_vals = self.ppo_loss(sample)

            # Combined optimization step (actor + critic)
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
                self.callback.log_training_step(current_step, actor_loss, value_loss)

            # Periodic logging and evaluation
            if self._should_log_step(current_step):
                self._log_progress(max_length, buffer_len, loss_vals)

            # Periodic evaluation
            if self._should_eval_step(current_step):
                self._evaluate()

    def _log_progress(self, max_length: int, buffer_len: int, loss_vals: dict) -> None:
        curr_loss_actor = loss_vals["loss_objective"].item()
        curr_loss_value = loss_vals["loss_critic"].item()
        curr_loss_entropy = loss_vals["loss_entropy"].item()

        logger.info(
            "ppo step max_steps={} buffer_size={} loss_value={:.4f} loss_actor={:.4f} loss_entropy={:.4f}",
            max_length, buffer_len, curr_loss_value, curr_loss_actor, curr_loss_entropy,
        )

        if is_level_enabled("TRACE"):
            _log_network_stats(logger, "ppo", self.actor, self.value_net)

    def _evaluate(self) -> None:
        """Evaluate current PPO policy."""
        with torch.no_grad():
            n_eval = self.eval_config.resolve_eval_steps(self._eval_data_len) if self._eval_data_len is not None else self.eval_config.eval_steps
            eval_env = self._eval_env or self.env
            if self._eval_env is None:
                logger.warning(
                    "ppo _evaluate: no dedicated eval env set; using training env "
                    "which may corrupt SyncDataCollector state"
                )
            try:
                with set_exploration_type(InteractionType.MODE):
                    eval_rollout = eval_env.rollout(n_eval, self.actor)
            except RuntimeError:
                logger.debug(
                    "Mode not available for distribution, falling back to Mean"
                )
                with set_exploration_type(InteractionType.DETERMINISTIC):
                    eval_rollout = eval_env.rollout(n_eval, self.actor)
            except IndexError:
                # gym_anytrading raises IndexError when the episode (val_df) is too
                # short relative to window_size, so prices[tick+1] goes out of bounds.
                # Skip this eval step rather than crashing the training run.
                logger.warning(
                    "ppo _evaluate: eval env raised IndexError (episode too short for "
                    "gym_anytrading window); skipping this eval step"
                )
                return

            # Log evaluation metrics
            mean_reward = eval_rollout["next", "reward"].mean().item()
            sum_reward = eval_rollout["next", "reward"].sum().item()
            max_steps = eval_rollout["step_count"].max().item()

            self.logs["eval_reward_mean"].append(mean_reward)
            self.logs["eval_reward_sum"].append(sum_reward)
            self.logs["eval_step_count"].append(max_steps)

            eval_data_len = self._eval_data_len if self._eval_data_len is not None else "?"
            logger.info(
                "ppo eval mean_reward={:.4f} sum_reward={:.4f} eval_steps={} eval_data_len={}",
                mean_reward, sum_reward, max_steps, eval_data_len,
            )

            del eval_rollout

    def _compute_exploration_ratio(self) -> float:
        return self.config.ppo.entropy_bonus

    @property
    def _algo_label(self) -> str:
        return "ppo"

    def _get_checkpoint_network_state(self) -> dict:
        return {
            "actor_params_state": self.ppo_loss.actor_network_params.state_dict(),
            "value_params_state": self.ppo_loss.critic_network_params.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

    def _load_checkpoint_network_state(self, checkpoint: dict) -> None:
        self.ppo_loss.actor_network_params.load_state_dict(checkpoint["actor_params_state"])
        self.ppo_loss.critic_network_params.load_state_dict(checkpoint["value_params_state"])
        self.ppo_loss.actor_network_params.to_module(self.actor)
        self.ppo_loss.critic_network_params.to_module(self.value_net)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    def train(self, callback: Any = None) -> dict[str, list]:
        """Run training loop for PPO agent."""
        return self._run_training_loop(
            callback,
            start_message="Starting PPO training",
            completion_prefix="PPO Training complete",
        )

    @staticmethod
    def build_models(n_obs: int, n_act: int, config: Any, env: Any):
        """Factory for PPO actor and value network."""
        obs_ndim = len(env.observation_spec["observation"].shape)
        actor = create_ppo_actor(
            n_obs,
            n_act,
            hidden_dims=config.network.actor_hidden_dims,
            spec=env.action_spec,
            obs_ndim=obs_ndim,
        )
        value_net = create_ppo_value_network(
            n_obs,
            hidden_dims=config.network.value_hidden_dims,
            obs_ndim=obs_ndim,
        )
        return actor, value_net

    def create_action_probabilities_plot(
        self,
        max_steps: int,
        df=None,
        config=None,
        eval_env: Any | None = None,
    ):
        env_for_plot = eval_env or self._eval_env
        if env_for_plot is None:
            logger.warning(
                "skipping action probabilities plot: no eval env provided "
                "and using the training env risks corrupting SyncDataCollector state"
            )
            return None
        try:
            return self._build_action_probabilities_plot(
                env_for_plot, self.actor, max_steps, df, config
            )
        except Exception:
            logger.exception("action probabilities plot failed")
            return None

    def evaluate(
        self,
        df: Any,
        max_steps: int,
        config: Any = None,
        algorithm: str | None = None,
        eval_env: Any | None = None,
    ) -> EvaluationOutput:
        output = super().evaluate(
            df,
            max_steps,
            config=config,
            algorithm=algorithm,
            eval_env=eval_env,
        )
        action_probs_plot = self.create_action_probabilities_plot(
            max_steps=max_steps, df=df, config=config, eval_env=eval_env
        )
        return EvaluationOutput(
            reward_plot=output.reward_plot,
            action_plot=output.action_plot,
            action_probs_plot=action_probs_plot,
            final_reward=output.final_reward,
            last_positions=output.last_positions,
            equity_curve_plot=output.equity_curve_plot,
            merged_plot=output.merged_plot,
            result=output.result,
        )

    @staticmethod
    def _build_action_probabilities_plot(env, actor, max_steps, df=None, config=None):
        """Create a plot showing action probability distributions over time steps for PPO."""
        action_names = {i: pos.name.capitalize() for i, pos in enumerate(TradePosition)}
        n_actions = len(action_names)
        action_probs_data: list[dict] = []

        def _process_step(step: int, actor_output) -> torch.Tensor:
            probs = None
            if isinstance(actor_output, TensorDict):
                if "probs" in actor_output.keys():
                    probs = actor_output.get("probs")
                elif "logits" in actor_output.keys():
                    probs = torch.softmax(actor_output.get("logits"), dim=-1)
            if probs is None and hasattr(actor_output, "logits"):
                probs = torch.softmax(actor_output.logits, dim=-1)
            elif probs is None and isinstance(actor_output, tuple) and len(actor_output) >= 1:
                probs = actor_output[0]
            elif probs is None and hasattr(actor_output, "loc"):
                v = float(torch.clamp(actor_output.loc, -1, 1).item())
                probs = torch.tensor(
                    [0.7, 0.2, 0.1] if v < -0.33 else ([0.1, 0.2, 0.7] if v > 0.33 else [0.2, 0.6, 0.2])
                )
            if probs is None:
                probs = torch.tensor([0.33, 0.34, 0.33])

            probs = torch.as_tensor(probs).squeeze()
            if probs.dim() == 0 or len(probs) != n_actions:
                probs = torch.tensor([0.33, 0.34, 0.33])
            probs = probs / probs.sum()

            for action_idx, action_name in action_names.items():
                action_probs_data.append({"Step": step, "Action": action_name, "Probability": float(probs[action_idx])})

            if isinstance(actor_output, TensorDict) and "action" in actor_output.keys():
                action = actor_output.get("action")
            elif hasattr(actor_output, "sample"):
                action = actor_output.sample()
            elif isinstance(actor_output, tuple) and len(actor_output) >= 2:
                action = actor_output[1]
            else:
                action = F.one_hot(
                    torch.tensor(int(torch.multinomial(probs, 1).item())), num_classes=n_actions
                ).to(torch.float32)

            action_t = torch.as_tensor(action)
            if action_t.dim() == 0 or (action_t.dim() == 1 and action_t.shape[0] != n_actions):
                action_t = F.one_hot(action_t.long(), num_classes=n_actions).to(torch.float32)
            if action_t.dim() == 1 and action_t.shape[0] == n_actions:
                action_t = action_t.unsqueeze(0)
            if action_t.shape[-1] == n_actions:
                action_t = action_t.argmax(dim=-1)
            return action_t.to(torch.long)

        with torch.no_grad():
            _run_viz_rollout(env, actor, max_steps, 20, _process_step)

        df_probs = pd.DataFrame(action_probs_data)
        action_order = [pos.name.capitalize() for pos in TradePosition]
        df_probs["Action"] = pd.Categorical(
            df_probs["Action"], categories=action_order, ordered=True
        )

        plot = (
            ggplot(df_probs, aes(x="Step", y="Probability", fill="Action"))
            + geom_area(position="stack", alpha=0.8)
            + labs(
                title="Action Probability Distributions Over Time",
                x="Time Step",
                y="Probability",
                fill="Action",
            )
            + scale_fill_manual(
                name="Action",
                values={
                    "Short": "#F8766D",
                    "Hold": "#C0C0C0",
                    "Long": "#00BFC4",
                },
            )
            + scale_x_continuous(expand=(0, 0))
            + thesis_theme(figure_size=(FIGURE_WIDTH * 2, FIGURE_HEIGHT))
        )

        return plot


@register_trainer("PPO", continuous=True)
class PPOTrainerContinuous(PPOTrainer):
    """Trainer for PPO algorithm with continuous action spaces."""

    @staticmethod
    def build_models(n_obs: int, n_act: int, config: Any, env: Any):
        """Factory for Continuous PPO actor and value network."""
        actor = create_continuous_ppo_actor(
            n_obs,
            n_act,
            hidden_dims=config.network.actor_hidden_dims,
            spec=env.action_spec,
        )
        value_net = create_ppo_value_network(
            n_obs,
            hidden_dims=config.network.value_hidden_dims,
        )
        return actor, value_net

    @staticmethod
    def _build_action_probabilities_plot(env, actor, max_steps, df=None, config=None):
        """Create a plot showing continuous action distributions over time steps."""
        continuous_data: list[dict] = []

        def _process_step(step: int, actor_output) -> torch.Tensor | None:
            if not hasattr(actor_output, "loc"):
                return None
            mean = actor_output.loc.squeeze()
            scale = getattr(actor_output, "scale", torch.zeros_like(mean)).squeeze()
            if mean.ndim > 0:
                mean, scale = mean[0], scale[0]
            mean_val, std_val = float(mean.item()), float(scale.item())
            continuous_data.append({"Step": step, "Mean": mean_val, "Upper": mean_val + std_val, "Lower": mean_val - std_val})
            return actor_output.sample() if hasattr(actor_output, "sample") else actor_output.loc

        with torch.no_grad():
            _run_viz_rollout(env, actor, max_steps, 20, _process_step)

        df_cont = pd.DataFrame(continuous_data)
        if df_cont.empty:
            return None

        plot = (
            ggplot(df_cont, aes(x="Step"))
            + geom_ribbon(
                aes(ymin="Lower", ymax="Upper"), fill="#00BFC4", alpha=0.3
            )
            + geom_line(aes(y="Mean"), color="#00BFC4", size=1)
            + labs(
                title="Continuous Action Distribution (Mean ± Std)",
                x="Time Step",
                y="Action Value",
            )
            + thesis_theme(figure_size=(FIGURE_WIDTH * 2, FIGURE_HEIGHT))
        )
        return plot
