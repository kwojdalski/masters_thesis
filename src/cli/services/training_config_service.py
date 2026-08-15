"""Prepare training configuration independently of CLI presentation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from trading_rl import ExperimentConfig

from .config_validation_service import ValidationReport, validate_experiment_config

ConfigLoader = Callable[[str | Path, str, list[str] | None], ExperimentConfig]
SeedResolver = Callable[[int | None], int]


@dataclass(frozen=True)
class TrainingConfigRequest:
    """Inputs that affect training configuration and checkpoint selection."""

    config_file: Path | None = None
    scenario: str | None = None
    config_overrides: list[str] | None = None
    experiment_name: str | None = None
    seed: int | None = None
    max_steps: int | None = None
    checkpoint_path: Path | None = None
    from_checkpoint: Path | None = None
    from_last_checkpoint: bool = False


@dataclass(frozen=True)
class PreparedTrainingConfig:
    """Prepared domain config and resolved runtime inputs."""

    config: ExperimentConfig
    checkpoint_path: Path | None
    validation: ValidationReport


class TrainingConfigService:
    """Load, override, validate, and resolve checkpoint inputs for training."""

    def __init__(
        self,
        validate: Callable[[ExperimentConfig], ValidationReport] = validate_experiment_config,
    ) -> None:
        self._validate = validate

    def prepare(
        self,
        request: TrainingConfigRequest,
        *,
        load_config: ConfigLoader,
        resolve_seed: SeedResolver,
    ) -> PreparedTrainingConfig:
        """Return a validated configuration and resolved checkpoint path."""
        config = self._load(request, load_config)
        if request.experiment_name:
            config.experiment_name = request.experiment_name
        if request.seed is not None:
            config.seed = request.seed
        if request.max_steps is not None:
            config.training.max_steps = request.max_steps
        config.seed = resolve_seed(config.seed)

        validation = self._validate(config)
        checkpoint_path = self._resolve_checkpoint(config, request)
        return PreparedTrainingConfig(config, checkpoint_path, validation)

    @staticmethod
    def _load(
        request: TrainingConfigRequest, load_config: ConfigLoader
    ) -> ExperimentConfig:
        if request.config_file and request.scenario:
            raise ValueError("Cannot specify both --config and --scenario.")
        source: str | Path | None = request.config_file or request.scenario
        if source is not None:
            return load_config(source, "train", request.config_overrides)
        if request.config_overrides:
            raise ValueError("--config-override requires --config or --scenario.")
        return ExperimentConfig()

    @staticmethod
    def _resolve_checkpoint(
        config: ExperimentConfig, request: TrainingConfigRequest
    ) -> Path | None:
        if request.from_checkpoint and request.from_last_checkpoint:
            raise ValueError(
                "Use only one of --from-checkpoint or --from-last-checkpoint."
            )
        if request.checkpoint_path:
            return request.checkpoint_path
        if request.from_checkpoint:
            return request.from_checkpoint
        if not request.from_last_checkpoint:
            return None

        log_dir = Path(config.logging.log_dir)
        matches = list(log_dir.rglob(f"{config.experiment_name}_checkpoint*.pt"))
        if not matches:
            raise FileNotFoundError(
                f"No checkpoints found for {config.experiment_name} in {log_dir}"
            )
        return max(matches, key=lambda path: path.stat().st_mtime)
