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
