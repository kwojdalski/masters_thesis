"""PPO Trainer implementation."""

from collections import defaultdict
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from plotnine import (
    aes,
    element_text,
    geom_area,
    geom_line,
    geom_ribbon,
    ggplot,
    labs,
    scale_fill_manual,
    scale_x_continuous,
    theme,
)
from tensordict import TensorDict
from tensordict.nn import InteractionType
from torch.optim import Adam
from torchrl.envs.utils import set_exploration_type
from torchrl.objectives import ClipPPOLoss

from logger import get_logger
from trading_rl.config import TrainingConfig
from trading_rl.models import (
    create_continuous_ppo_actor,
    create_ppo_actor,
    create_ppo_value_network,
)
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
        checkpoint_dir: str | None = None,
        checkpoint_prefix: str | None = None,
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
            checkpoint_dir=checkpoint_dir,
            checkpoint_prefix=checkpoint_prefix,
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
        with torch.no_grad():
            try:
                with set_exploration_type(InteractionType.MODE):
                    eval_rollout = self.env.rollout(self.config.eval_steps, self.actor)
            except RuntimeError:
                logger.debug(
                    "Mode not available for distribution, falling back to Mean"
                )
                with set_exploration_type(InteractionType.DETERMINISTIC):
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
        import mlflow

        run = mlflow.active_run()
        tracking_uri = mlflow.get_tracking_uri()
        run_name = run.data.tags.get("mlflow.runName") if run else None
        experiment_name = None
        if run:
            experiment = mlflow.get_experiment(run.info.experiment_id)
            experiment_name = experiment.name if experiment else None
        checkpoint = {
            "actor_state_dict": self.actor.state_dict(),
            "value_net_state_dict": self.value_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "total_count": self.total_count,
            "total_episodes": self.total_episodes,
            "logs": dict(self.logs),
            "mlflow_run_id": run.info.run_id if run else None,
            "mlflow_run_name": run_name,
            "mlflow_tracking_uri": tracking_uri,
            "mlflow_experiment_id": run.info.experiment_id if run else None,
            "mlflow_experiment_name": experiment_name,
        }
        torch.save(checkpoint, path)
        logger.info(f"\033[95mPPO checkpoint saved to {path}\033[0m")

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
        self.mlflow_run_id = checkpoint.get("mlflow_run_id")
        self.mlflow_run_name = checkpoint.get("mlflow_run_name")
        self.mlflow_tracking_uri = checkpoint.get("mlflow_tracking_uri")
        self.mlflow_experiment_id = checkpoint.get("mlflow_experiment_id")
        self.mlflow_experiment_name = checkpoint.get("mlflow_experiment_name")
        logger.info(f"\033[95mPPO checkpoint loaded from {path}\033[0m")

    @staticmethod
    def build_models(n_obs: int, n_act: int, config: Any, env: Any):
        """Factory for PPO actor and value network."""
        actor = create_ppo_actor(
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

    def create_action_probabilities_plot(self, max_steps: int, df=None, config=None):
        return self._build_action_probabilities_plot(
            self.env, self.actor, max_steps, df, config
        )

    def evaluate(self, df, max_steps: int, config=None, algorithm: str | None = None):
        reward_plot, action_plot, _, final_reward, last_positions = super().evaluate(
            df, max_steps, config=config, algorithm=algorithm
        )
        action_probs_plot = self.create_action_probabilities_plot(
            max_steps=max_steps, df=df, config=config
        )
        return reward_plot, action_plot, action_probs_plot, final_reward, last_positions

    @staticmethod
    def _build_action_probabilities_plot(env, actor, max_steps, df=None, config=None):
        """Create a plot showing action probability distributions over time steps for PPO."""
        try:
            max_viz_steps = max_steps
            max_episode_length = 20

            with torch.no_grad():
                env_to_use = env
                obs = env_to_use.reset()
                action_names = {0: "Short", 1: "Hold", 2: "Long"}
                action_probs_data = []
                current_episode_steps = 0

                for step in range(max_viz_steps):
                    if current_episode_steps >= max_episode_length:
                        obs = env_to_use.reset()
                        current_episode_steps = 0

                    if hasattr(obs, "get") and "observation" in obs:
                        current_obs = obs["observation"]
                    else:
                        current_obs = obs

                    if isinstance(obs, TensorDict):
                        actor_input = obs.clone()
                    else:
                        batch_size = (
                            [current_obs.shape[0]]
                            if hasattr(current_obs, "shape")
                            and len(getattr(current_obs, "shape", [])) > 0
                            else [1]
                        )
                        actor_input = TensorDict(
                            {"observation": torch.as_tensor(current_obs)},
                            batch_size=batch_size,
                        )

                    actor_output = actor(actor_input)

                    probs = None
                    if isinstance(actor_output, TensorDict):
                        if "probs" in actor_output.keys():
                            probs = actor_output.get("probs")
                        elif "logits" in actor_output.keys():
                            logits = actor_output.get("logits")
                            probs = torch.softmax(logits, dim=-1)

                    if probs is None and hasattr(actor_output, "logits"):
                        logits = actor_output.logits
                        probs = torch.softmax(logits, dim=-1)
                    elif (
                        probs is None
                        and isinstance(actor_output, tuple)
                        and len(actor_output) >= 1
                    ):
                        probs = actor_output[0]
                    elif probs is None and hasattr(actor_output, "loc"):
                        action_val = torch.clamp(actor_output.loc, -1, 1)
                        if action_val < -0.33:
                            probs = torch.tensor([0.7, 0.2, 0.1])
                        elif action_val > 0.33:
                            probs = torch.tensor([0.1, 0.2, 0.7])
                        else:
                            probs = torch.tensor([0.2, 0.6, 0.2])

                    if probs is None:
                        probs = torch.tensor([0.33, 0.34, 0.33])

                    probs = torch.as_tensor(probs).squeeze()
                    if probs.dim() == 0 or len(probs) != 3:
                        probs = torch.tensor([0.33, 0.34, 0.33])
                    probs = probs / probs.sum()

                    for action_idx, action_name in action_names.items():
                        action_probs_data.append(
                            {
                                "Step": step,
                                "Action": action_name,
                                "Probability": float(probs[action_idx]),
                            }
                        )

                    if (
                        isinstance(actor_output, TensorDict)
                        and "action" in actor_output.keys()
                    ):
                        action = actor_output.get("action")
                    elif hasattr(actor_output, "sample"):
                        action = actor_output.sample()
                    elif isinstance(actor_output, tuple) and len(actor_output) >= 2:
                        action = actor_output[1]
                    else:
                        action_idx = torch.multinomial(probs, 1).item()
                        action = F.one_hot(
                            torch.tensor(action_idx), num_classes=len(action_names)
                        ).to(torch.float32)

                    action_tensor = torch.as_tensor(action)
                    if action_tensor.dim() == 0:
                        action_tensor = F.one_hot(
                            action_tensor.long(), num_classes=len(action_names)
                        ).to(torch.float32)
                    elif action_tensor.dim() == 1 and action_tensor.shape[0] != len(
                        action_names
                    ):
                        action_tensor = F.one_hot(
                            action_tensor.long(), num_classes=len(action_names)
                        ).to(torch.float32)

                    if action_tensor.dim() == 1 and action_tensor.shape[0] == len(
                        action_names
                    ):
                        action_tensor = action_tensor.unsqueeze(0)

                    if hasattr(obs, "batch_size") and obs.batch_size:
                        expected_batch = obs.batch_size[0]
                    else:
                        expected_batch = 1

                    if action_tensor.shape[0] != expected_batch:
                        action_tensor = action_tensor.expand(
                            expected_batch, action_tensor.shape[1]
                        )

                    if action_tensor.shape[-1] == len(action_names):
                        action_tensor = action_tensor.argmax(dim=-1)

                    action = action_tensor.to(torch.long)

                    if hasattr(obs, "clone") and hasattr(obs, "set"):
                        action_td = obs.clone()
                        action_td.set("action", action)
                        step_result = env_to_use.step(action_td)
                    else:
                        raise RuntimeError(
                            "Environment observation is not a TensorDict"
                        )

                    if "next" in step_result.keys():
                        next_obs = step_result.get("next").clone()
                    else:
                        next_obs = step_result.clone()
                    obs = next_obs
                    current_episode_steps += 1

                    done_tensor = None
                    if hasattr(obs, "get"):
                        done_tensor = obs.get("done", torch.tensor([False]))
                    if done_tensor is not None and torch.as_tensor(done_tensor).any():
                        break

            df_probs = pd.DataFrame(action_probs_data)
            action_order = ["Short", "Hold", "Long"]
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
                + theme(
                    figure_size=(12, 6),
                    axis_title=element_text(size=11),
                    legend_position="right",
                )
            )

            return plot

        except Exception as exc:  # pragma: no cover - logged for diagnostics
            logger.exception("Action probabilities plot failed", exc_info=exc)
            fallback_steps = min(max_steps, 500)
            fallback_data = []
            for step in range(fallback_steps):
                for action in ["Short", "Hold", "Long"]:
                    fallback_data.append(
                        {"Step": step, "Action": action, "Probability": 0.33}
                    )

            df_fallback = pd.DataFrame(fallback_data)
            df_fallback["Action"] = pd.Categorical(
                df_fallback["Action"],
                categories=["Short", "Hold", "Long"],
                ordered=True,
            )
            plot = (
                ggplot(df_fallback, aes(x="Step", y="Probability", fill="Action"))
                + geom_area(position="stack", alpha=0.8)
                + labs(
                    title="Action Probability Distribution (Fallback)",
                    x="Time Step",
                    y="Probability",
                )
                + scale_fill_manual(
                    name="Action",
                    values={
                        "Short": "#F8766D",
                        "Hold": "#C0C0C0",
                        "Long": "#00BFC4",
                    },
                )
            )

            return plot


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
        """Create a plot showing action distributions over time steps."""
        try:
            max_viz_steps = max_steps
            max_episode_length = 20

            # Data collection containers
            continuous_data = []  # List of dicts: {Step, Mean, Std_Upper, Std_Lower}

            with torch.no_grad():
                env_to_use = env
                obs = env_to_use.reset()
                current_episode_steps = 0

                for step in range(max_viz_steps):
                    if current_episode_steps >= max_episode_length:
                        obs = env_to_use.reset()
                        current_episode_steps = 0

                    if hasattr(obs, "get") and "observation" in obs:
                        current_obs = obs["observation"]
                    else:
                        current_obs = obs

                    if isinstance(obs, TensorDict):
                        actor_input = obs.clone()
                    else:
                        batch_size = (
                            [current_obs.shape[0]]
                            if hasattr(current_obs, "shape")
                            and len(getattr(current_obs, "shape", [])) > 0
                            else [1]
                        )
                        actor_input = TensorDict(
                            {"observation": torch.as_tensor(current_obs)},
                            batch_size=batch_size,
                        )

                    actor_output = actor(actor_input)

                    # --- Continuous Output Handling ---
                    if hasattr(actor_output, "loc"):
                        mean = actor_output.loc.squeeze()
                        scale = getattr(
                            actor_output, "scale", torch.zeros_like(mean)
                        ).squeeze()

                        # Handle multi-dimensional continuous actions (take first dim)
                        if mean.ndim > 0:
                            mean = mean[0]
                            scale = scale[0]

                        mean_val = float(mean.item())
                        std_val = float(scale.item())

                        continuous_data.append(
                            {
                                "Step": step,
                                "Mean": mean_val,
                                "Upper": mean_val + std_val,
                                "Lower": mean_val - std_val,
                            }
                        )

                        # Sample action for simulation
                        action = (
                            actor_output.sample()
                            if hasattr(actor_output, "sample")
                            else actor_output.loc
                        )
                    else:
                        # Fallback for unexpected output format
                        continue

                    # --- Environment Step ---
                    # Ensure action tensor is correctly shaped for the environment
                    action_tensor = torch.as_tensor(action)

                    if hasattr(obs, "batch_size") and obs.batch_size:
                        expected_batch = obs.batch_size[0]
                    else:
                        expected_batch = 1

                    # Ensure batch dimension
                    if action_tensor.dim() == 0:
                        action_tensor = action_tensor.unsqueeze(0)

                    # Fix batch size mismatch if necessary
                    if action_tensor.shape[0] != expected_batch:
                        if action_tensor.shape[0] == 1:
                            # Broadcast to batch size
                            if action_tensor.dim() == 1:
                                action_tensor = action_tensor.expand(expected_batch)
                            else:
                                action_tensor = action_tensor.expand(expected_batch, -1)
                        else:
                            # Just take the first one if we have too many
                            action_tensor = action_tensor[:expected_batch]

                    if hasattr(obs, "clone") and hasattr(obs, "set"):
                        action_td = obs.clone()
                        action_td.set("action", action_tensor)
                        step_result = env_to_use.step(action_td)
                    else:
                        raise RuntimeError(
                            "Environment observation is not a TensorDict"
                        )

                    if "next" in step_result.keys():
                        next_obs = step_result.get("next").clone()
                    else:
                        next_obs = step_result.clone()
                    obs = next_obs
                    current_episode_steps += 1

                    done_tensor = None
                    if hasattr(obs, "get"):
                        done_tensor = obs.get("done", torch.tensor([False]))
                    if done_tensor is not None and torch.as_tensor(done_tensor).any():
                        break

            # --- Plotting ---
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
                + theme(
                    figure_size=(12, 6),
                    axis_title=element_text(size=11),
                )
            )
            return plot

        except Exception as exc:  # pragma: no cover - logged for diagnostics
            logger.exception("Continuous action plot failed", exc_info=exc)
            return None
