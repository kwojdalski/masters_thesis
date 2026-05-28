"""SAC Trainer implementation."""

import logging
from trading_rl.trainers.registry import register_trainer
from collections import defaultdict
from typing import Any

import torch
from tensordict.nn import InteractionType
from torch.optim import Adam
from torchrl.data import Bounded
from torchrl.envs.utils import set_exploration_type
from torchrl.objectives import SACLoss, SoftUpdate

from logger import get_logger
from trading_rl.config import EvaluationConfig, TrainingConfig
from trading_rl.constants import LossFunction
from trading_rl.evaluation.asset_meta import write_asset_meta
from trading_rl.models import create_sac_actor, create_sac_qvalue_network
from trading_rl.trainers.base import _MIN_BATCH_SUCCESS_RATE, _collect_mlflow_meta, _log_network_stats, BaseTrainer

logger = get_logger(__name__)

_MAX_CONSECUTIVE_SKIPPED_BATCHES = 10


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

        # Counters for tracking successful vs skipped batches
        self.successful_batches = 0
        self.skipped_batches = 0
        self._consecutive_skips = 0

        logger.info(
            "init sac trainer actor_lr=%s value_lr=%s alpha_lr=%s buffer_size=%d tau=%s "
            "initial_alpha=%.3f target_entropy=%.3f",
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

    def _record_skipped_batch(self, reason: str, exc: RuntimeError | None = None) -> None:
        """Track skipped SAC optimization batches and fail fast if none succeed."""
        self.skipped_batches += 1
        self._consecutive_skips += 1

        if exc is None:
            logger.warning("sac skipping batch reason=%s", reason)
        else:
            logger.warning("sac skipping batch reason=%s err=%s", reason, exc)

        if (
            self._consecutive_skips >= _MAX_CONSECUTIVE_SKIPPED_BATCHES
            and self.successful_batches == 0
        ):
            error = RuntimeError(
                f"SAC: {self._consecutive_skips} consecutive optimization batches "
                "skipped with zero successful updates. Training cannot proceed - "
                "check environment or replay buffer tensor shapes."
            )
            if exc is not None:
                raise error from exc
            raise error

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
            "sac step max_steps=%d buffer_size=%d loss_value=%.4f loss_actor=%.4f "
            "loss_alpha=%.4f alpha=%.4f",
            max_length, buffer_len, curr_loss_value, curr_loss_actor,
            curr_loss_alpha, curr_alpha,
        )

        if logger.isEnabledFor(logging.DEBUG):
            _log_network_stats(logger, "sac", self.actor, self.value_net)

    def _evaluate(self) -> None:
        """Evaluate current SAC policy (deterministic mean action)."""
        with set_exploration_type(InteractionType.DETERMINISTIC), torch.no_grad():
            n_eval = (
                self.eval_config.resolve_eval_steps(self._eval_data_len)
                if self._eval_data_len is not None
                else self.eval_config.eval_steps
            )
            if self._eval_env is None:
                logger.warning(
                    "sac _evaluate: no dedicated eval env set; skipping periodic eval "
                    "to avoid corrupting SyncDataCollector state"
                )
                return
            eval_rollout = self._eval_env.rollout(n_eval, self.actor)

            mean_reward = eval_rollout["next", "reward"].mean().item()
            sum_reward = eval_rollout["next", "reward"].sum().item()
            max_steps = eval_rollout["step_count"].max().item()

            self.logs["eval_reward_mean"].append(mean_reward)
            self.logs["eval_reward_sum"].append(sum_reward)
            self.logs["eval_step_count"].append(max_steps)

            eval_data_len = self._eval_data_len if self._eval_data_len is not None else "?"
            logger.info(
                "sac eval mean_reward=%.4f sum_reward=%.4f eval_steps=%d eval_data_len=%s",
                mean_reward, sum_reward, max_steps, eval_data_len,
            )

            del eval_rollout

    def _compute_exploration_ratio(self) -> float:
        # Current alpha temperature serves as a proxy for exploration intensity
        return float(self.sac_loss.log_alpha.exp().item())

    def train(self, callback: Any = None) -> dict[str, list]:
        """Run training loop for SAC agent."""
        # SAC actor is stochastic — no separate exploration module is needed.
        self._initialize_offpolicy_collection_policy(
            self.actor,
            self.sac_action_spec,
            algorithm_label="SAC",
        )

        def on_batch_start(i, data) -> None:
            if logger.isEnabledFor(logging.DEBUG) and i % 10 == 0:
                episode_rewards = data["next", "reward"]
                buffer_len = len(self.replay_buffer)
                logger.debug(
                    "sac batch=%d steps=%d buffer_size=%d", i, data.numel(), buffer_len
                )
                logger.debug(
                    "sac episode reward stats mean=%.4f std=%.4f",
                    episode_rewards.mean(), episode_rewards.std(),
                )
                collected_actions = data["action"]
                logger.debug(
                    "sac collected action stats mean=%.4f std=%.4f",
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

    def _log_batch_summary(self) -> None:
        """Log successful vs skipped optimization batch summary."""
        total_batches = self.successful_batches + self.skipped_batches
        if total_batches > 0:
            success_rate = (self.successful_batches / total_batches) * 100
            _log = logger.warning if success_rate < _MIN_BATCH_SUCCESS_RATE else logger.info
            _log(
                "sac batch summary successful=%d/%d success_rate=%.1f%% skipped=%d",
                self.successful_batches, total_batches, success_rate, self.skipped_batches,
            )
        else:
            logger.warning("sac no optimization batches attempted")

    def save_checkpoint(
        self,
        path: str,
        feature_pipeline_state: dict[str, dict[str, float]] | None = None,
        mlflow_meta: dict | None = None,
    ) -> None:
        """Save SAC training checkpoint.

        Args:
            path: Path to save checkpoint
            feature_pipeline_state: Optional serialised scaler state for the feature pipeline.
            mlflow_meta: Optional MLflow run metadata dict. When None, the active
                MLflow run is queried automatically as a fallback.
        """
        from pathlib import Path

        if mlflow_meta is None:
            mlflow_meta = _collect_mlflow_meta()
        from trading_rl.models import _extract_action_bounds_from_spec
        _bounds = _extract_action_bounds_from_spec(getattr(self.env, "action_spec", None))
        checkpoint = {
            "algorithm": "sac",
            "n_obs": self.n_obs,
            "n_act": self.n_act,
            "actor_hidden_dims": self.actor_hidden_dims,
            "value_hidden_dims": self.value_hidden_dims,
            "action_low": _bounds[0].tolist() if _bounds is not None else None,
            "action_high": _bounds[1].tolist() if _bounds is not None else None,
            "actor_state_dict": self.actor.state_dict(),
            "value_net_state_dict": self.value_net.state_dict(),
            "actor_params_state": self.sac_loss.actor_network_params.state_dict(),
            "value_params_state": self.sac_loss.qvalue_network_params.state_dict(),
            "target_value_params_state": self.sac_loss.target_qvalue_network_params.state_dict(),
            "log_alpha": self.sac_loss.log_alpha.item(),
            "optimizer_actor_state_dict": self.optimizer_actor.state_dict(),
            "optimizer_value_state_dict": self.optimizer_value.state_dict(),
            "optimizer_alpha_state_dict": self.optimizer_alpha.state_dict(),
            "total_count": self.total_count,
            "total_episodes": self.total_episodes,
            "episode_log_count": (
                int(self.logs.get("episode_log_count", [0])[-1])
                if self.logs.get("episode_log_count")
                else 0
            ),
            "logs": dict(self.logs),
            "mlflow_run_id": mlflow_meta.get("run_id"),
            "mlflow_run_name": mlflow_meta.get("run_name"),
            "mlflow_tracking_uri": mlflow_meta.get("tracking_uri"),
            "mlflow_experiment_id": mlflow_meta.get("experiment_id"),
            "mlflow_experiment_name": mlflow_meta.get("experiment_name"),
            "feature_pipeline_state": feature_pipeline_state,
        }

        if getattr(self.config, "save_buffer", False):
            logger.info("save replay buffer")
            path_obj = Path(path)
            buffer_dir = path_obj.with_suffix("")
            buffer_dir = buffer_dir.with_name(f"{buffer_dir.name}_buffer")
            try:
                self.replay_buffer.dumps(buffer_dir)
            except Exception:
                logger.exception("failed to save replay buffer; checkpoint will not include buffer")
            else:
                checkpoint["replay_buffer_path"] = str(buffer_dir)
                checkpoint["buffer_metadata"] = {
                    "buffer_size": len(self.replay_buffer),
                    "max_size": self.replay_buffer._storage.max_size,
                }
                logger.info(
                    "save replay buffer path=%s n_experiences=%s",
                    buffer_dir,
                    len(self.replay_buffer),
                )

        torch.save(checkpoint, path)
        write_asset_meta(path, generator="trainers/sac.py")
        logger.info("save checkpoint path=%s", path)

    def load_checkpoint(self, path: str) -> None:
        """Load SAC training checkpoint.

        Args:
            path: Path to checkpoint
        """
        from pathlib import Path

        checkpoint = torch.load(path, weights_only=True)
        if "actor_params_state" not in checkpoint or "value_params_state" not in checkpoint:
            raise KeyError(
                "SAC checkpoint is missing functional parameter states "
                "(actor_params_state/value_params_state). "
                "Legacy module-only checkpoints are no longer supported."
            )

        self.sac_loss.actor_network_params.load_state_dict(checkpoint["actor_params_state"])
        self.sac_loss.qvalue_network_params.load_state_dict(checkpoint["value_params_state"])

        target_value_params = getattr(self.sac_loss, "target_qvalue_network_params", None)
        target_value_state = checkpoint.get("target_value_params_state")
        if target_value_params is not None and target_value_state is not None:
            target_value_params.load_state_dict(target_value_state)

        self.sac_loss.actor_network_params.to_module(self.actor)
        self.sac_loss.qvalue_network_params.to_module(self.value_net)

        # Restore temperature parameter
        log_alpha_val = checkpoint.get("log_alpha")
        if log_alpha_val is not None:
            with torch.no_grad():
                self.sac_loss.log_alpha.fill_(log_alpha_val)

        self.optimizer_actor.load_state_dict(checkpoint["optimizer_actor_state_dict"])
        self.optimizer_value.load_state_dict(checkpoint["optimizer_value_state_dict"])
        self.optimizer_alpha.load_state_dict(checkpoint["optimizer_alpha_state_dict"])
        self.total_count = checkpoint["total_count"]
        self.total_episodes = checkpoint["total_episodes"]
        self.logs = defaultdict(list, checkpoint["logs"])
        self.mlflow_run_id = checkpoint.get("mlflow_run_id")
        self.mlflow_run_name = checkpoint.get("mlflow_run_name")
        self.mlflow_tracking_uri = checkpoint.get("mlflow_tracking_uri")
        self.mlflow_experiment_id = checkpoint.get("mlflow_experiment_id")
        self.mlflow_experiment_name = checkpoint.get("mlflow_experiment_name")

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
                    logger.exception("Failed to load replay buffer from %s", buffer_path)
            else:
                logger.info("no replay buffer in checkpoint start_fresh=true")

        logger.info("load checkpoint path=%s", path)
