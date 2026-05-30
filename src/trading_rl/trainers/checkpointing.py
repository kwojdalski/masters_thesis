"""Checkpoint data model and persistence policy for trainers."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from logger import get_logger

logger = get_logger(__name__)


@dataclass
class TrainingCheckpoint:
    """All state needed to resume or evaluate a training run."""

    algorithm: str
    total_count: int
    total_episodes: int
    logs: dict
    # algorithm-specific: actor/value params, optimizer states — stored flat in the file
    network_state: dict
    n_obs: int | None = None
    n_act: int | None = None
    actor_hidden_dims: list[int] | None = None
    value_hidden_dims: list[int] | None = None
    action_low: list | None = None
    action_high: list | None = None
    actor_state_dict: dict = field(default_factory=dict)
    value_net_state_dict: dict = field(default_factory=dict)
    episode_log_count: int = 0
    mlflow_run_id: str | None = None
    mlflow_run_name: str | None = None
    mlflow_tracking_uri: str | None = None
    mlflow_experiment_id: str | None = None
    mlflow_experiment_name: str | None = None
    feature_pipeline_state: dict | None = None
    replay_buffer_path: str | None = None
    buffer_metadata: dict | None = None


# All fields except `network_state`, which is merged flat into the on-disk dict for
# backward-compat with existing checkpoint files and the PolicyLoader.
_COMMON_CHECKPOINT_KEYS: frozenset[str] = frozenset(
    f.name for f in dataclasses.fields(TrainingCheckpoint) if f.name != "network_state"
)


class CheckpointManager:
    """Encapsulate checkpoint persistence policy and I/O."""

    def __init__(
        self,
        checkpoint_dir: str | Path | None,
        checkpoint_prefix: str | None,
        interval: int,
        save_buffer: bool = False,
    ) -> None:
        self._dir = Path(checkpoint_dir) if checkpoint_dir else None
        self._prefix = checkpoint_prefix
        self._interval = interval
        self.save_buffer = save_buffer
        self._last_checkpoint_step = 0

    # --- I/O -----------------------------------------------------------------

    def save(self, path: str, checkpoint: TrainingCheckpoint) -> None:
        """Serialize checkpoint to disk as a flat dict compatible with weights_only=True."""
        from trading_rl.evaluation.asset_meta import write_asset_meta

        raw: dict[str, Any] = {k: getattr(checkpoint, k) for k in _COMMON_CHECKPOINT_KEYS}
        raw.update(checkpoint.network_state)  # merge algorithm-specific state flat
        torch.save(raw, path)
        write_asset_meta(path, generator="trainers/checkpointing.py")
        logger.info("save checkpoint path={}", path)

    def load(self, path: str) -> TrainingCheckpoint:
        """Deserialize and validate a checkpoint from disk."""
        raw: dict = torch.load(path, weights_only=True)
        logger.trace("checkpoint keys={}", sorted(raw.keys()))

        if "actor_params_state" not in raw or "value_params_state" not in raw:
            raise KeyError(
                "Checkpoint is missing required parameter states "
                "(actor_params_state / value_params_state). "
                "Legacy module-only checkpoints are no longer supported."
            )

        network_state = {k: v for k, v in raw.items() if k not in _COMMON_CHECKPOINT_KEYS}
        common = {k: raw.get(k) for k in _COMMON_CHECKPOINT_KEYS}

        if common.get("algorithm") is None:
            raise KeyError("Checkpoint missing required 'algorithm' key")

        # Fill safe defaults for optional counters absent in older files.
        common.setdefault("total_count", 0)
        common.setdefault("total_episodes", 0)
        common.setdefault("logs", {})
        common.setdefault("actor_state_dict", {})
        common.setdefault("value_net_state_dict", {})
        common.setdefault("episode_log_count", 0)

        return TrainingCheckpoint(**common, network_state=network_state)

    # --- Policy --------------------------------------------------------------

    def maybe_save(self, step: int, snapshot_fn: Callable[[], TrainingCheckpoint]) -> None:
        """Save a periodic checkpoint when the configured interval is reached.

        snapshot_fn is called lazily — only when the interval condition is met,
        so no snapshot is assembled on skipped steps.
        """
        if not self._should_save(step):
            return
        path = self._build_path(f"step_{step}")
        self.save(path, snapshot_fn())
        self._last_checkpoint_step = step
        logger.trace("checkpoint saved path={} step={}", path, step)

    def save_interrupt(
        self, step: int, snapshot_fn: Callable[[], TrainingCheckpoint]
    ) -> str | None:
        """Save an emergency checkpoint when training is interrupted."""
        if self._dir is None or self._prefix is None:
            logger.warning("Interrupted but checkpoint_dir/checkpoint_prefix not configured.")
            return None
        ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        path = self._build_path(f"interrupt_step_{step}_{ts}")
        self.save(path, snapshot_fn())
        logger.trace("interrupt checkpoint saved path={} step={}", path, step)
        return path

    def _should_save(self, step: int) -> bool:
        return (
            self._interval > 0
            and self._dir is not None
            and self._prefix is not None
            and (step - self._last_checkpoint_step) >= self._interval
        )

    def _build_path(self, suffix: str) -> str:
        assert self._dir is not None and self._prefix is not None
        p = self._dir / f"{self._prefix}_checkpoint_{suffix}.pt"
        p.parent.mkdir(parents=True, exist_ok=True)
        return str(p)
