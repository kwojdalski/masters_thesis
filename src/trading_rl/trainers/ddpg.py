"""DDPG Trainer implementation."""

import time
from collections import defaultdict
from typing import Any

import torch
from tensordict.nn import InteractionType
from torch.optim import Adam
from torchrl.envs.utils import set_exploration_type
from torchrl.objectives import DDPGLoss, SoftUpdate

from logger import get_logger
from trading_rl.config import TrainingConfig
from trading_rl.models import create_ddpg_actor, create_value_network
from trading_rl.trainers.base import BaseTrainer

logger = get_logger(__name__)


class DDPGTrainer(BaseTrainer):
    """Trainer for DDPG algorithm on trading environments."""

    def __init__(
        self,
        actor: Any,
        value_net: Any,
        env: Any,
        config: TrainingConfig,
        checkpoint_dir: str | None = None,
        checkpoint_prefix: str | None = None,
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
        )
        self.optimizer_value = Adam(
            self.ddpg_loss.value_network_params.values(True, True),
            lr=config.value_lr,
            weight_decay=config.value_weight_decay,
        )

        # Counters for tracking successful vs skipped batches
        self.successful_batches = 0
        self.skipped_batches = 0

        logger.info("DDPG Trainer initialized")
        logger.info(f"Actor LR: {config.actor_lr}, Value LR: {config.value_lr}")
        logger.info(f"Buffer size: {config.buffer_size}, Tau: {config.tau}")

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

    def evaluate(self, df, max_steps: int, config=None, algorithm: str | None = None):
        return super().evaluate(df, max_steps, config=config, algorithm=algorithm)

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

            # Ensure sample has consistent shapes for DDPG Loss
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

            # Compute losses with error handling
            try:
                loss_vals = self.ddpg_loss(sample)
                # If we get here, the batch was successful
                self.successful_batches += 1
            except RuntimeError as e:
                if "All input tensors" in str(e) and "must share a unique shape" in str(
                    e
                ):
                    self.skipped_batches += 1
                    logger.warning(f"DDPG tensor shape error: {e}, skipping this batch")
                    continue
                else:
                    raise e

            # Optimize actor
            loss_vals["loss_actor"].backward()
            self.optimizer_actor.step()
            self.optimizer_actor.zero_grad()

            # Sync functional actor params back to the actor module used by the collector/evaluator
            self.ddpg_loss.actor_network_params.to_module(self.actor)

            # Optimize value network
            loss_vals["loss_value"].backward()
            self.optimizer_value.step()
            self.optimizer_value.zero_grad()

            # Sync functional value params back to the value module
            self.ddpg_loss.value_network_params.to_module(self.value_net)

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
                offset = getattr(self, "_log_step_offset", 0)
                current_step = offset + (
                    batch_idx * self.config.optim_steps_per_batch + j
                )
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
        import mlflow
        from pathlib import Path

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
            "optimizer_actor_state_dict": self.optimizer_actor.state_dict(),
            "optimizer_value_state_dict": self.optimizer_value.state_dict(),
            "total_count": self.total_count,
            "total_episodes": self.total_episodes,
            "logs": dict(self.logs),
            "mlflow_run_id": run.info.run_id if run else None,
            "mlflow_run_name": run_name,
            "mlflow_tracking_uri": tracking_uri,
            "mlflow_experiment_id": run.info.experiment_id if run else None,
            "mlflow_experiment_name": experiment_name,
        }

        # Optionally save replay buffer (can be very large)
        if getattr(self.config, "save_buffer", False):
            logger.info("Saving replay buffer to disk (this may take a while)...")
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
                logger.info(
                    "Replay buffer saved to %s (%s experiences)",
                    buffer_dir,
                    len(self.replay_buffer),
                )
            except Exception:
                logger.exception("Failed to save replay buffer")

        torch.save(checkpoint, path)
        logger.info(f"\033[95mDDPG checkpoint saved to {path}\033[0m")

    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint.

        Args:
            path: Path to checkpoint
        """
        # Load checkpoint with weights_only=False for TorchRL compatibility
        # TensorDict objects require custom unpickling that isn't in PyTorch's safe allowlist
        from pathlib import Path

        checkpoint = torch.load(path, weights_only=False)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.value_net.load_state_dict(checkpoint["value_net_state_dict"])
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
            logger.info("Restoring replay buffer from checkpoint (legacy)...")
            self.replay_buffer = checkpoint["replay_buffer"]
            buffer_size = len(self.replay_buffer)
            logger.info("Replay buffer restored: %s experiences", buffer_size)
        else:
            buffer_path = checkpoint.get("replay_buffer_path")
            if buffer_path and Path(buffer_path).exists():
                try:
                    self.replay_buffer.loads(buffer_path)
                    buffer_size = len(self.replay_buffer)
                    logger.info(
                        "Replay buffer loaded from %s (%s experiences)",
                        buffer_path,
                        buffer_size,
                    )
                except Exception:
                    logger.exception(
                        "Failed to load replay buffer from %s", buffer_path
                    )
            else:
                logger.info("No replay buffer in checkpoint - starting with empty buffer")

        logger.info(f"\033[95mDDPG checkpoint loaded from {path}\033[0m")

    def train(self, callback=None) -> dict[str, list]:
        """Run training loop for DDPG agent with batch summary."""
        logger.info("Starting DDPG training")
        t0 = time.time()
        self.callback = callback
        self._log_step_offset = max(
            len(self.logs.get("loss_actor", [])),
            len(self.logs.get("loss_value", [])),
        )

        for i, data in enumerate(self.collector):
            self.replay_buffer.extend(data)

            max_length = self.replay_buffer[:]["next", "step_count"].max()
            buffer_len = len(self.replay_buffer)

            if buffer_len > self.config.init_rand_steps:
                self._optimization_step(i, max_length, buffer_len)

            self.total_count += data.numel()
            self.total_episodes += data["next", "done"].sum()
            self._maybe_save_checkpoint()

            if callback and hasattr(callback, "log_episode_stats"):
                self._log_episode_stats(data, callback)

            if self.total_count >= self.config.max_steps:
                logger.info(f"Training stopped after {self.config.max_steps} steps")
                break

        t1 = time.time()
        logger.info(
            f"DDPG Training complete: {self.total_count} steps, "
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
