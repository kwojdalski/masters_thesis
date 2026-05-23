"""Tests for DDPG skipped-batch accounting."""

from __future__ import annotations

import pytest

from trading_rl.trainers.ddpg import DDPGTrainer


def _bare_ddpg_trainer() -> DDPGTrainer:
    trainer = object.__new__(DDPGTrainer)
    trainer.successful_batches = 0
    trainer.skipped_batches = 0
    trainer._consecutive_skips = 0
    return trainer


def test_ddpg_record_skipped_batch_updates_skip_counters() -> None:
    trainer = _bare_ddpg_trainer()

    trainer._record_skipped_batch("done/terminated shape mismatch")

    assert trainer.skipped_batches == 1
    assert trainer._consecutive_skips == 1


def test_ddpg_record_skipped_batch_aborts_after_repeated_zero_success_skips() -> None:
    trainer = _bare_ddpg_trainer()

    for _ in range(9):
        trainer._record_skipped_batch("done/terminated shape mismatch")

    with pytest.raises(RuntimeError, match="10 consecutive optimization batches"):
        trainer._record_skipped_batch("done/terminated shape mismatch")

    assert trainer.skipped_batches == 10
    assert trainer.successful_batches == 0


def test_ddpg_record_skipped_batch_does_not_abort_after_successful_batch() -> None:
    trainer = _bare_ddpg_trainer()
    trainer.successful_batches = 1

    for _ in range(10):
        trainer._record_skipped_batch("done/terminated shape mismatch")

    assert trainer.skipped_batches == 10
    assert trainer._consecutive_skips == 10
