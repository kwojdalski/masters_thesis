"""Checkpoint policy helpers for trainers."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from logger import get_logger

logger = get_logger(__name__)


class CheckpointManager:
    """Encapsulate periodic and interrupt checkpoint persistence policy."""

    def __init__(self, trainer: Any):
        self.trainer = trainer
        self._last_checkpoint_step = 0

    def maybe_save_checkpoint(self) -> None:
        """Persist a periodic checkpoint when the configured interval is reached."""
        interval = getattr(self.trainer.config, "checkpoint_interval", 0)
        checkpoint_dir = getattr(self.trainer, "checkpoint_dir", None)
        checkpoint_prefix = getattr(self.trainer, "checkpoint_prefix", None)

        if interval <= 0 or not checkpoint_dir or not checkpoint_prefix:
            return
        if self.trainer.total_count - self._last_checkpoint_step < interval:
            return

        checkpoint_path = (
            Path(checkpoint_dir)
            / f"{checkpoint_prefix}_checkpoint_step_{self.trainer.total_count}.pt"
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.trainer.save_checkpoint(str(checkpoint_path))
        self._last_checkpoint_step = self.trainer.total_count

    def save_interrupt_checkpoint(self) -> str | None:
        """Persist an emergency checkpoint when training is interrupted."""
        checkpoint_dir = getattr(self.trainer, "checkpoint_dir", None)
        checkpoint_prefix = getattr(self.trainer, "checkpoint_prefix", None)

        if not checkpoint_dir or not checkpoint_prefix:
            logger.warning(
                "Interrupted, but checkpoint_dir/checkpoint_prefix is not configured."
            )
            return None

        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        checkpoint_path = (
            Path(checkpoint_dir)
            / (
                f"{checkpoint_prefix}_checkpoint_interrupt_"
                f"step_{self.trainer.total_count}_{timestamp}.pt"
            )
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.trainer.save_checkpoint(str(checkpoint_path))
        return str(checkpoint_path)
