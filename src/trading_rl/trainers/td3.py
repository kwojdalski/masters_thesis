"""TD3 Trainer implementation."""

import logging
from collections import defaultdict
from typing import Any

import torch
from tensordict.nn import InteractionType, TensorDictSequential
from torch.optim import Adam
from torchrl.envs.utils import RandomPolicy
from torchrl.data import Bounded
from torchrl.envs.utils import set_exploration_type
from torchrl.modules import AdditiveGaussianModule
from torchrl.objectives import SoftUpdate
from torchrl.objectives import TD3Loss as TorchRLTd3Loss

from logger import get_logger
from trading_rl.config import DEFAULT_INITIAL_PORTFOLIO_VALUE, TrainingConfig
from trading_rl.models import create_td3_actor, create_td3_qvalue_network
from trading_rl.trainers.base import BaseTrainer, _MIN_BATCH_SUCCESS_RATE

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
        qvalue_net: Any,
        env: Any,
        config: TrainingConfig,
        checkpoint_dir: str | None = None,
        checkpoint_prefix: str | None = None,
    ):
        super().__init__(
            actor=actor,
            value_net=qvalue_net,
            env=env,
            config=config,
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
                "Environment action_spec is not a Bounded spec; falling back to TD3 default [-1, 1] bounds."
            )
        self.td3_action_spec = td3_action_spec

        # Gaussian exploration around the deterministic policy
        self.exploration_module = AdditiveGaussianModule(
            spec=td3_action_spec,
            sigma_init=getattr(config, "exploration_noise_std", 0.2),
            sigma_end=getattr(config, "exploration_noise_std", 0.2),
            annealing_num_steps=config.max_steps,
        )
        # Log the exploration noise std

        # TD3 uses two critics; configure loss and optimizers
        self.td3_loss = TD3Loss(
            actor_network=actor,
            qvalue_network=qvalue_net,
            action_spec=td3_action_spec,
            num_qvalue_nets=2,
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
            weight_decay=config.actor_weight_decay,
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

        logger.info(
            "init td3 trainer actor_lr=%s value_lr=%s exploration_noise_std=%.3f policy_noise=%.3f noise_clip=%.3f policy_delay=%d",
            config.actor_lr,
            config.value_lr,
            getattr(config, "exploration_noise_std", 0.2),
            getattr(config, "policy_noise", 0.2),
            getattr(config, "noise_clip", 0.5),
            self.policy_delay,
        )

    def evaluate(
        self,
        df,
        max_steps: int,
        config=None,
        algorithm: str | None = None,
        eval_env: Any | None = None,
    ):
        import numpy as np
        import pandas as pd
        from plotnine import (
            aes,
            element_text,
            geom_line,
            ggplot,
            guide_legend,
            guides,
            labs,
            scale_color_manual,
            theme,
        )

        from trading_rl.utils import (
            compare_rollouts,
            create_actual_returns_plot,
            create_merged_comparison_plot,
        )

        logger = get_logger(__name__)

        env_to_use = eval_env or self.env

        logger.debug("eval deterministic max_steps=%d", max_steps)
        with set_exploration_type(InteractionType.MODE):
            rollout_deterministic = env_to_use.rollout(
                max_steps=max_steps, policy=self.actor
            )

        # Extract actual returns immediately (before next rollout overwrites broker state)
        from trading_rl.evaluation.returns import extract_tradingenv_returns

        actual_returns_deterministic = extract_tradingenv_returns(env_to_use, max_steps)

        logger.debug("eval random max_steps=%d", max_steps)
        # Use explicit random policy to ensure a distinct trajectory
        random_policy = RandomPolicy(self.td3_action_spec)
        with set_exploration_type(InteractionType.RANDOM):
            rollout_random = env_to_use.rollout(max_steps=max_steps, policy=random_policy)

        # Extract actual returns immediately (for random rollout)
        actual_returns_random = extract_tradingenv_returns(env_to_use, max_steps)

        reward_plot, action_plot = compare_rollouts(
            [rollout_deterministic, rollout_random],
            n_obs=max_steps,
        )

        benchmark_price_column = "close"
        if config:
            configured_price_column = getattr(config.env, "price_column", None)
            if isinstance(configured_price_column, str) and configured_price_column:
                benchmark_price_column = configured_price_column

        if benchmark_price_column in df.columns:
            benchmark_series = df[benchmark_price_column]
        elif "close" in df.columns:
            logger.warning(
                "Benchmark column '%s' missing in evaluation frame; falling back to 'close'.",
                benchmark_price_column,
            )
            benchmark_series = df["close"]
            benchmark_price_column = "close"
        else:
            raise ValueError(
                "Evaluation benchmarks require env.price_column or 'close' in dataframe."
            )

        # Create actual returns plot with pre-extracted returns
        actual_returns_plot = create_actual_returns_plot(
            [rollout_deterministic, rollout_random],
            n_obs=max_steps,
            df_prices=df,
            env=None,  # Don't pass env, use pre-extracted returns
            actual_returns_list=[actual_returns_deterministic, actual_returns_random],
            initial_portfolio_value=(
                float(getattr(config.env, "initial_portfolio_value", DEFAULT_INITIAL_PORTFOLIO_VALUE))
                if config
                else DEFAULT_INITIAL_PORTFOLIO_VALUE
            ),
            benchmark_price_column=benchmark_price_column,
        )

        # Add benchmarks based on reward type
        reward_type = getattr(config.env, "reward_type", "log_return") if config else "log_return"
        benchmark_data = []

        if reward_type == "differential_sharpe":
            # For DSR, calculate DSR benchmarks
            from trading_rl.utils import calculate_benchmark_dsr

            # Get DSR parameters from config (reward_eta is the standard name)
            dsr_eta = getattr(config.env, "reward_eta", 0.01) if config else 0.01

            logger.debug("calc dsr benchmarks eta=%s", dsr_eta)

            # Calculate DSR for buy-and-hold
            bh_dsr, _ = calculate_benchmark_dsr(
                df,
                strategy="buy_and_hold",
                eta=dsr_eta,
                max_steps=max_steps,
                price_column=benchmark_price_column,
                initial_portfolio_value=float(getattr(config.env, "initial_portfolio_value", DEFAULT_INITIAL_PORTFOLIO_VALUE)),
            )

            # Calculate DSR for max profit
            mp_dsr, _ = calculate_benchmark_dsr(
                df,
                strategy="max_profit",
                eta=dsr_eta,
                max_steps=max_steps,
                price_column=benchmark_price_column,
                initial_portfolio_value=float(getattr(config.env, "initial_portfolio_value", DEFAULT_INITIAL_PORTFOLIO_VALUE)),
            )

            # Add to benchmark data
            for step, (bh_val, mp_val) in enumerate(zip(bh_dsr, mp_dsr, strict=False)):
                benchmark_data.extend(
                    [
                        {"Steps": step, "Cumulative_Reward": bh_val, "Run": "Buy-and-Hold"},
                        {
                            "Steps": step,
                            "Cumulative_Reward": mp_val,
                            "Run": "Max Profit (Unleveraged)",
                        },
                    ]
                )

            y_label = "Cumulative DSR"
        else:
            # For log_return and other rewards, use log return benchmarks
            benchmark_df = pd.DataFrame(
                {
                    "x": range(max_steps),
                    "buy_and_hold": np.log(benchmark_series / benchmark_series.shift(1))
                    .fillna(0)
                    .cumsum()[:max_steps],
                    "max_profit": np.log(
                        abs(benchmark_series / benchmark_series.shift(1) - 1) + 1
                    )
                    .fillna(0)
                    .cumsum()[:max_steps],
                }
            )
            for step, (bh_val, mp_val) in enumerate(
                zip(benchmark_df["buy_and_hold"], benchmark_df["max_profit"], strict=False)
            ):
                benchmark_data.extend(
                    [
                        {"Steps": step, "Cumulative_Reward": bh_val, "Run": "Buy-and-Hold"},
                        {
                            "Steps": step,
                            "Cumulative_Reward": mp_val,
                            "Run": "Max Profit (Unleveraged)",
                        },
                    ]
                )

            y_label = "Cumulative Reward"

        existing_data = reward_plot.data
        combined_data = pd.concat(
            [existing_data, pd.DataFrame(benchmark_data)], ignore_index=True
        )
        reward_plot = (
            ggplot(combined_data, aes(x="Steps", y="Cumulative_Reward", color="Run"))
            + geom_line(alpha=0.5)
            + labs(
                title="Cumulative Rewards Comparison", x="Steps", y=y_label
            )
            + scale_color_manual(
                values={
                    "Deterministic": "#F8766D",
                    "Random": "#00BFC4",
                    "Buy-and-Hold": "violet",
                    "Max Profit (Unleveraged)": "green",
                }
            )
            + theme(
                figure_size=(13, 7.8),
                legend_position="bottom",
                legend_title=element_text(weight="bold", size=11),
                legend_text=element_text(size=10),
                subplots_adjust={'left': 0.10, 'right': 0.95},
            )
            + guides(color=guide_legend(title="Strategy"))
        )

        # Create merged comparison plot (rewards + actions stacked vertically)
        merged_plot = create_merged_comparison_plot(reward_plot, action_plot)

        final_reward = float(rollout_deterministic["next"]["reward"].sum().item())
        action_tensor = rollout_deterministic["action"].squeeze()
        if action_tensor.ndim > 1 and action_tensor.shape[-1] > 1:
            actions = action_tensor.argmax(dim=-1)
        else:
            actions = action_tensor
        actions = actions.flatten().tolist() if hasattr(actions, "flatten") else []
        last_positions = [float(a) for a in actions] if actions else []

        return (
            reward_plot,
            action_plot,
            None,
            final_reward,
            last_positions,
            actual_returns_plot,
            merged_plot,
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
        qvalue_net = create_td3_qvalue_network(
            n_obs,
            n_act,
            hidden_dims=config.network.value_hidden_dims,
        )
        return actor, qvalue_net

    def _optimization_step(
        self, batch_idx: int, max_length: int, buffer_len: int
    ) -> None:
        for j in range(self.config.optim_steps_per_batch):
            sample = self.replay_buffer.sample(self.config.sample_size)
            current_step = self._global_optimization_step(
                batch_idx, j, self.config.optim_steps_per_batch
            )

            # TorchRL's TD3Loss requires reward/done/terminated to have identical shapes.
            # Some env wrappers emit 1D tensors which then trigger a shape error; normalize here.
            for key in [("next", "reward"), ("next", "done"), ("next", "terminated")]:
                tensor = sample.get(key)
                if tensor is None:
                    continue
                if tensor.ndim == 0:
                    tensor = tensor.unsqueeze(0).unsqueeze(-1)
                elif tensor.ndim == 1:
                    tensor = tensor.unsqueeze(-1)
                sample.set(key, tensor)

            # DEBUG: Log sample statistics
            if logger.isEnabledFor(logging.DEBUG):
                actions = sample["action"]
                rewards = sample["next", "reward"]
                logger.debug(
                    "td3 batch sample stats batch=%d step=%d "
                    "action_mean=%.4f action_std=%.4f action_min=%.4f action_max=%.4f "
                    "reward_mean=%.4f reward_std=%.4f reward_min=%.4f reward_max=%.4f",
                    batch_idx, j,
                    actions.mean(), actions.std(), actions.min(), actions.max(),
                    rewards.mean(), rewards.std(), rewards.min(), rewards.max(),
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
                        "td3 losses loss_qvalue=%.6f loss_actor=%.6f",
                        loss_vals['loss_qvalue'].item(), loss_vals['loss_actor'].item(),
                    )

            except RuntimeError as e:
                if "All input tensors" in str(e) and "must share a unique shape" in str(
                    e
                ):
                    self.skipped_batches += 1
                    logger.warning("td3 tensor shape error skipping batch err=%s", e)
                    continue
                else:
                    raise e

            self.optimizer_value.zero_grad()
            loss_vals["loss_qvalue"].backward()
            self.optimizer_value.step()

            # Sync functional critic params back to critic module used for collection/eval
            self.td3_loss.qvalue_network_params.to_module(self.value_net)

            # for name, param in list(self.value_net.named_parameters())[:3]:
            #     logger.debug(f"{name}, {param.view(-1)[:5]}")

            value_loss = loss_vals["loss_qvalue"].item()
            self.logs["loss_value"].append(value_loss)

            # 2. Delayed actor update
            update_actor = current_step % self.policy_delay == 0

            extra_metrics = None
            if update_actor:
                # Recompute loss for actor update (using updated critic weights)
                loss_vals_actor = self.td3_loss(sample)

                self.optimizer_actor.zero_grad()
                loss_vals_actor["loss_actor"].backward()
                self.optimizer_actor.step()

                # Sync functional actor params back to the actor module the collector uses
                self.td3_loss.actor_network_params.to_module(self.actor)

                # Update target networks
                self.updater.step()

                actor_loss = loss_vals_actor["loss_actor"].item()

                # Optional debug: prepare parameter magnitude summaries
                if logger.isEnabledFor(logging.DEBUG):
                    actor_sum = float(
                        sum(p.abs().sum().item() for p in self.actor.parameters())
                    )
                    critic_sum = float(
                        sum(p.abs().sum().item() for p in self.value_net.parameters())
                    )
                    extra_metrics = {
                        "actor_param_abs_sum": actor_sum,
                        "critic_param_abs_sum": critic_sum,
                    }
            else:
                # For logging purposes, use the actor loss computed in the first pass
                actor_loss = loss_vals["loss_actor"].item()

            if logger.isEnabledFor(logging.DEBUG):
                critic_param_diff = None
                params = self.td3_loss.qvalue_network_params
                if getattr(params, "batch_size", None) and params.batch_size[0] >= 2:
                    params0 = params[0]
                    params1 = params[1]
                    max_diff = 0.0
                    for key, p0 in params0.items(True, True):
                        p1 = params1.get(key)
                        if isinstance(p0, torch.Tensor) and isinstance(
                            p1, torch.Tensor
                        ):
                            diff = (p0 - p1).abs().max().item()
                            if diff > max_diff:
                                max_diff = diff
                    critic_param_diff = max_diff

                if critic_param_diff is not None:
                    if extra_metrics is None:
                        extra_metrics = {}
                    extra_metrics["critic_qvalue_params_max_diff"] = critic_param_diff

            self.logs["loss_actor"].append(actor_loss)

            if (
                hasattr(self, "callback")
                and self.callback
                and hasattr(self.callback, "log_training_step")
            ):
                self.callback.log_training_step(
                    current_step, actor_loss, value_loss, extra_metrics=extra_metrics
                )

            # Log progress similar to PPO (info level)

            if self._should_log_step(current_step):
                self._log_progress(max_length, buffer_len, loss_vals)

            if self._should_eval_step(current_step):
                self._evaluate()

    def train(self, callback=None) -> dict[str, list]:
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
            # DEBUG: Log data collection statistics
            if logger.isEnabledFor(logging.DEBUG) and i % 10 == 0:  # Every 10 batches
                episode_rewards = data["next", "reward"]
                buffer_len = len(self.replay_buffer)
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

    def _log_batch_summary(self) -> None:
        """Log successful vs skipped optimization batch summary."""
        # Log batch success/failure summary
        total_batches = self.successful_batches + self.skipped_batches
        if total_batches > 0:
            success_rate = (self.successful_batches / total_batches) * 100
            _log = logger.warning if success_rate < _MIN_BATCH_SUCCESS_RATE else logger.info
            _log(
                "td3 batch summary successful=%d/%d success_rate=%.1f%% skipped=%d",
                self.successful_batches, total_batches, success_rate, self.skipped_batches,
            )
        else:
            logger.warning("td3 no optimization batches attempted")

    def _log_progress(self, max_length: int, buffer_len: int, loss_vals: dict) -> None:
        curr_loss_value = loss_vals["loss_qvalue"].item()
        curr_loss_actor = loss_vals["loss_actor"].item()

        logger.info(
            "td3 step max_steps=%d buffer_size=%d loss_value=%.4f loss_actor=%.4f",
            max_length, buffer_len, curr_loss_value, curr_loss_actor,
        )

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
                logger.debug(
                    "td3 eval action stats shape=%s mean=%.4f std=%.4f",
                    actions.shape, actions.mean(), actions.std(),
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
                        f"  Action distribution: {dict(zip(unique_actions, counts, strict=False))}"
                    )
                else:
                    # Show percentiles if too many unique values
                    percentiles = np.percentile(actions_flat, [0, 25, 50, 75, 100])
                    logger.debug(
                        f"  Action percentiles [0,25,50,75,100]: {percentiles}"
                    )

                # Check if agent is stuck
                if actions.std() < 0.01:
                    logger.warning("td3 eval agent stuck action_std=%.6f", actions.std())

            logger.info(
                "td3 eval mean_reward=%.4f sum_reward=%.4f max_steps=%d",
                mean_reward, sum_reward, max_steps,
            )

            del eval_rollout

    def _compute_exploration_ratio(self) -> float:
        return getattr(self.config, "policy_noise", 0.2)

    def save_checkpoint(self, path: str) -> None:
        from pathlib import Path

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
            "actor_params_state": self.td3_loss.actor_network_params.state_dict(),
            "target_actor_params_state": self.td3_loss.target_actor_network_params.state_dict(),
            "qvalue_state_dict": self.value_net.state_dict(),
            "qvalue_params_state": self.td3_loss.qvalue_network_params.state_dict(),
            "target_qvalue_params_state": self.td3_loss.target_qvalue_network_params.state_dict(),
            "optimizer_actor_state_dict": self.optimizer_actor.state_dict(),
            "optimizer_value_state_dict": self.optimizer_value.state_dict(),
            "total_count": self.total_count,
            "total_episodes": self.total_episodes,
            "episode_log_count": (
                int(self.logs.get("episode_log_count", [0])[-1])
                if self.logs.get("episode_log_count")
                else 0
            ),
            "logs": dict(self.logs),
            "mlflow_run_id": run.info.run_id if run else None,
            "mlflow_run_name": run_name,
            "mlflow_tracking_uri": tracking_uri,
            "mlflow_experiment_id": run.info.experiment_id if run else None,
            "mlflow_experiment_name": experiment_name,
        }

        # Optionally save replay buffer (can be very large)
        if getattr(self.config, "save_buffer", False):
            logger.info("save replay buffer")
            path_obj = Path(path)
            buffer_dir = path_obj.with_suffix("")
            buffer_dir = buffer_dir.with_name(f"{buffer_dir.name}_buffer")
            try:
                self.replay_buffer.dumps(buffer_dir)
                checkpoint["replay_buffer_path"] = str(buffer_dir)
                checkpoint["buffer_metadata"] = {
                    "buffer_size": len(self.replay_buffer),
                    "max_size": self.replay_buffer._storage.max_size,
                }
                logger.info("save replay buffer path=%s n_experiences=%s", buffer_dir, len(self.replay_buffer))
            except Exception:
                logger.exception("Failed to save replay buffer")

        torch.save(checkpoint, path)
        logger.info("save checkpoint path=%s", path)

    def load_checkpoint(self, path: str) -> None:
        # Load checkpoint with weights_only=False for TorchRL compatibility
        # TensorDict objects require custom unpickling that isn't in PyTorch's safe allowlist
        from pathlib import Path

        checkpoint = torch.load(path, weights_only=False)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("TD3 checkpoint keys: %s", sorted(checkpoint.keys()))

        # Restore functional params and sync modules
        if "actor_params_state" in checkpoint:
            self.td3_loss.actor_network_params.load_state_dict(
                checkpoint["actor_params_state"]
            )
            self.td3_loss.qvalue_network_params.load_state_dict(
                checkpoint["qvalue_params_state"]
            )
            self.td3_loss.target_actor_network_params.load_state_dict(
                checkpoint["target_actor_params_state"]
            )
            self.td3_loss.target_qvalue_network_params.load_state_dict(
                checkpoint["target_qvalue_params_state"]
            )
            # Sync back to modules for evaluation
            self.td3_loss.actor_network_params.to_module(self.actor)
            self.td3_loss.qvalue_network_params.to_module(self.value_net)
            # For modern checkpoints, functional params are the source of truth.
            # Do not overwrite the synced critic module with qvalue_state_dict.
        else:
            self.actor.load_state_dict(checkpoint["actor_state_dict"])
            qvalue_state = checkpoint.get("qvalue_state_dict")
            if isinstance(qvalue_state, list):
                if qvalue_state:
                    self.value_net.load_state_dict(qvalue_state[0])
            elif qvalue_state is not None:
                self.value_net.load_state_dict(qvalue_state)

        self.optimizer_actor.load_state_dict(checkpoint["optimizer_actor_state_dict"])
        self.optimizer_value.load_state_dict(checkpoint["optimizer_value_state_dict"])
        self.total_count = checkpoint["total_count"]
        self.total_episodes = checkpoint["total_episodes"]
        self.logs = defaultdict(list, checkpoint["logs"])
        self.mlflow_run_id = checkpoint.get("mlflow_run_id")
        self.mlflow_run_name = checkpoint.get("mlflow_run_name")
        self.mlflow_tracking_uri = checkpoint.get("mlflow_tracking_uri")
        self.mlflow_experiment_id = checkpoint.get("mlflow_experiment_id")
        self.mlflow_experiment_name = checkpoint.get("mlflow_experiment_name")

        # Optionally restore replay buffer if it was saved
        if "replay_buffer" in checkpoint:
            logger.info("restore replay buffer legacy n_experiences=%s", len(checkpoint["replay_buffer"]))
            self.replay_buffer = checkpoint["replay_buffer"]
        else:
            buffer_path = checkpoint.get("replay_buffer_path")
            if buffer_path and Path(buffer_path).exists():
                try:
                    self.replay_buffer.loads(buffer_path)
                    buffer_size = len(self.replay_buffer)
                    logger.info("load replay buffer path=%s n_experiences=%s", buffer_path, buffer_size)
                except Exception:
                    logger.exception(
                        "Failed to load replay buffer from %s", buffer_path
                    )
            else:
                logger.info("no replay buffer in checkpoint start_fresh=true")

        logger.debug(
            "load checkpoint state total_count=%s total_episodes=%s mlflow_run_id=%s mlflow_run_name=%s experiment=%s",
            self.total_count,
            self.total_episodes,
            self.mlflow_run_id,
            self.mlflow_run_name,
            self.mlflow_experiment_name,
        )
        logger.info("load checkpoint path=%s", path)
