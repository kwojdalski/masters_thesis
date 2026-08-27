"""Tests for training configuration preparation outside the CLI command."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from cli.services import (
    TrainingConfigRequest,
    TrainingConfigService,
    ValidationReport,
)
from trading_rl import ExperimentConfig


def _service() -> TrainingConfigService:
    return TrainingConfigService(validate=lambda _config: ValidationReport())


def _loader_calls(config: ExperimentConfig):
    calls = []

    def load(source, command, overrides):
        calls.append((source, command, overrides))
        return config

    return calls, load


def test_prepare_loads_source_and_applies_cli_overrides() -> None:
    config = ExperimentConfig()
    calls, load = _loader_calls(config)

    prepared = _service().prepare(
        TrainingConfigRequest(
            scenario="demo",
            config_overrides=["data.train_size=10"],
            experiment_name="renamed",
            seed=7,
            max_steps=123,
        ),
        load_config=load,
        resolve_seed=lambda seed: seed or 99,
    )

    assert calls == [("demo", "train", ["data.train_size=10"])]
    assert prepared.config.experiment_name == "renamed"
    assert prepared.config.seed == 7
    assert prepared.config.training.max_steps == 123
    assert prepared.checkpoint_path is None


def test_prepare_rejects_conflicting_sources() -> None:
    with pytest.raises(ValueError, match="both --config and --scenario"):
        _service().prepare(
            TrainingConfigRequest(config_file=Path("a.yaml"), scenario="demo"),
            load_config=lambda *_args: ExperimentConfig(),
            resolve_seed=lambda seed: seed or 1,
        )


def test_prepare_rejects_overrides_without_source() -> None:
    with pytest.raises(ValueError, match="requires --config or --scenario"):
        _service().prepare(
            TrainingConfigRequest(config_overrides=["training.max_steps=1"]),
            load_config=lambda *_args: ExperimentConfig(),
            resolve_seed=lambda seed: seed or 1,
        )


def test_prepare_resolves_explicit_checkpoint_alias() -> None:
    checkpoint = Path("checkpoint.pt")
    prepared = _service().prepare(
        TrainingConfigRequest(from_checkpoint=checkpoint),
        load_config=lambda *_args: ExperimentConfig(),
        resolve_seed=lambda seed: seed or 1,
    )

    assert prepared.checkpoint_path == checkpoint


def test_prepare_resolves_latest_checkpoint(tmp_path: Path) -> None:
    config = ExperimentConfig()
    config.experiment_name = "trial"
    config.logging.log_dir = str(tmp_path)
    older = tmp_path / "trial_checkpoint_step_1.pt"
    newer = tmp_path / "trial_checkpoint_step_2.pt"
    older.touch()
    newer.touch()
    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))

    prepared = _service().prepare(
        TrainingConfigRequest(scenario="demo", from_last_checkpoint=True),
        load_config=lambda *_args: config,
        resolve_seed=lambda seed: seed or 1,
    )

    assert prepared.checkpoint_path == newer


def test_prepare_resolves_latest_checkpoint_by_step_not_mtime(tmp_path: Path) -> None:
    """A later interrupt checkpoint at a low step must not be selected over
    an earlier periodic checkpoint at a much higher step just because its
    mtime is newer -- issue #438."""
    config = ExperimentConfig()
    config.experiment_name = "trial"
    config.logging.log_dir = str(tmp_path)
    high_step = tmp_path / "trial_checkpoint_step_3000000.pt"
    low_step_interrupt = (
        tmp_path / "trial_checkpoint_interrupt_step_40200_20260819_120000.pt"
    )
    high_step.touch()
    low_step_interrupt.touch()
    # Inverted mtime order: the low-step interrupt checkpoint was written
    # LATER (e.g. after restoring from backup, or an interrupted resume).
    os.utime(high_step, (1, 1))
    os.utime(low_step_interrupt, (2, 2))

    prepared = _service().prepare(
        TrainingConfigRequest(scenario="demo", from_last_checkpoint=True),
        load_config=lambda *_args: config,
        resolve_seed=lambda seed: seed or 1,
    )

    assert prepared.checkpoint_path == high_step


def test_prepare_falls_back_to_mtime_when_no_checkpoint_has_a_parseable_step(
    tmp_path: Path,
) -> None:
    config = ExperimentConfig()
    config.experiment_name = "trial"
    config.logging.log_dir = str(tmp_path)
    older = tmp_path / "trial_checkpoint_legacy.pt"
    newer = tmp_path / "trial_checkpoint_final.pt"
    older.touch()
    newer.touch()
    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))

    prepared = _service().prepare(
        TrainingConfigRequest(scenario="demo", from_last_checkpoint=True),
        load_config=lambda *_args: config,
        resolve_seed=lambda seed: seed or 1,
    )

    assert prepared.checkpoint_path == newer
