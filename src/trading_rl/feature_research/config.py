"""Configuration for offline feature research."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import yaml

try:
    from omegaconf import OmegaConf
except ImportError:  # pragma: no cover - optional dependency
    OmegaConf = None

from trading_rl.config import ExperimentConfig


@dataclass
class FeatureResearchDataConfig:
    """Data inputs for offline feature research."""

    data_path: str = "data/bitfinex2-BTCUSDT-1m.pkl"
    data_paths: list[str] | None = None
    train_size: int = 1000
    validation_size: int | None = None
    feature_config: str | None = None
    feature_cache_dir: str | None = ".cache/feature_transformation"

    def resolve_paths(self) -> list[str]:
        """Return the effective list of data paths.

        ``data_paths`` takes precedence; falls back to the single ``data_path``.
        """
        if self.data_paths:
            return list(self.data_paths)
        return [self.data_path]


class TargetType(StrEnum):
    SHARPE = "sharpe"
    RETURN = "return"


@dataclass
class FeatureResearchRunConfig:
    """Research procedure settings."""

    target_type: TargetType = TargetType.SHARPE
    horizons: list[int] = field(default_factory=lambda: [10, 50, 200])
    vol_window: int = 50          # rolling vol window (only used when target_type="sharpe")
    top_k: int = 10
    icir_threshold: float = 0.02
    window_size: int = 1000
    output_dir: str | None = None


@dataclass
class FeatureResearchConfig:
    """Full offline feature research configuration."""

    experiment_name: str = "feature_research"
    data: FeatureResearchDataConfig = field(default_factory=FeatureResearchDataConfig)
    research: FeatureResearchRunConfig = field(default_factory=FeatureResearchRunConfig)

    def __post_init__(self) -> None:
        """Validate configuration values."""
        errors: list[str] = []
        if self.data.train_size <= 0:
            errors.append(f"data.train_size must be > 0, got {self.data.train_size}")
        if self.data.validation_size is not None and self.data.validation_size <= 0:
            errors.append(
                "data.validation_size must be > 0 when provided, "
                f"got {self.data.validation_size}"
            )
        if not self.data.feature_config:
            errors.append("data.feature_config must be set")
        try:
            TargetType(self.research.target_type)
        except ValueError:
            errors.append(
                f"research.target_type must be one of {[t.value for t in TargetType]}, "
                f"got '{self.research.target_type}'"
            )
        if not self.research.horizons or any(h <= 0 for h in self.research.horizons):
            errors.append(
                f"research.horizons must be a non-empty list of positive ints, got {self.research.horizons}"
            )
        if self.research.vol_window <= 1:
            errors.append(
                f"research.vol_window must be > 1, got {self.research.vol_window}"
            )
        if self.research.top_k <= 0:
            errors.append(f"research.top_k must be > 0, got {self.research.top_k}")
        if self.research.icir_threshold < 0:
            errors.append(
                "research.icir_threshold must be >= 0, "
                f"got {self.research.icir_threshold}"
            )
        if errors:
            raise ValueError(
                "Feature research configuration validation failed:\n  - "
                + "\n  - ".join(errors)
            )

    @classmethod
    def from_yaml(
        cls, yaml_path: str | Path, overrides: list[str] | None = None
    ) -> "FeatureResearchConfig":
        """Load feature research configuration from YAML via OmegaConf."""
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Feature research config file not found: {yaml_path}")

        overrides = overrides or []
        if OmegaConf is not None:
            cfg = OmegaConf.load(str(yaml_path))
            if overrides:
                cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
            config_dict = OmegaConf.to_container(cfg, resolve=True)
        else:
            if overrides:
                raise ImportError(
                    "OmegaConf is required for config overrides. "
                    "Install it or remove overrides."
                )
            with yaml_path.open("r", encoding="utf-8") as handle:
                config_dict = yaml.safe_load(handle)

        config_dict = config_dict or {}
        if not isinstance(config_dict, dict):
            raise ValueError(
                f"Expected top-level config mapping in {yaml_path}, got "
                f"{type(config_dict).__name__}"
            )

        if "experiment_name" not in config_dict:
            config_dict["experiment_name"] = yaml_path.stem
        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "FeatureResearchConfig":
        """Create feature research config from a dictionary."""
        data_defaults = FeatureResearchDataConfig()
        research_defaults = FeatureResearchRunConfig()

        data_dict = {
            field_name: getattr(data_defaults, field_name)
            for field_name in data_defaults.__dataclass_fields__
        }
        for key, value in config_dict.get("data", {}).items():
            if key in data_dict:
                data_dict[key] = value
        # Normalise data_paths: OmegaConf may deserialise it as a ListConfig
        if data_dict.get("data_paths") is not None:
            data_dict["data_paths"] = list(data_dict["data_paths"])

        research_dict = {
            field_name: getattr(research_defaults, field_name)
            for field_name in research_defaults.__dataclass_fields__
        }
        for key, value in config_dict.get("research", {}).items():
            if key in research_dict:
                research_dict[key] = value

        if "target_type" in research_dict:
            research_dict["target_type"] = TargetType(research_dict["target_type"])

        return cls(
            experiment_name=config_dict.get("experiment_name", "feature_research"),
            data=FeatureResearchDataConfig(**data_dict),
            research=FeatureResearchRunConfig(**research_dict),
        )

    @classmethod
    def from_experiment_config(
        cls,
        experiment_config: ExperimentConfig,
        output_dir: str | None = None,
        overrides: list[str] | None = None,
    ) -> "FeatureResearchConfig":
        """Derive a feature research config from an experiment scenario.

        ``overrides`` are OmegaConf dotlist entries (e.g.
        ``["research.target_type=return", "research.horizons=[10,50]"]``)
        applied on top of the derived config so CLI flags reach research
        settings that are not part of the experiment scenario YAML.
        """
        default_output_dir = output_dir or str(
            Path(experiment_config.logging.log_dir) / "feature_research"
        )
        base_dict: dict = {
            "experiment_name": experiment_config.experiment_name,
            "data": {
                "data_path": experiment_config.data.data_path,
                "data_paths": experiment_config.data.data_paths,
                "train_size": experiment_config.data.train_size,
                "validation_size": experiment_config.data.validation_size,
                "feature_config": experiment_config.data.feature_config,
                "feature_cache_dir": getattr(experiment_config.data, "feature_cache_dir", ".cache/feature_transformation"),
            },
            "research": {
                "output_dir": default_output_dir,
            },
        }
        if overrides and OmegaConf is not None:
            cfg = OmegaConf.merge(
                OmegaConf.create(base_dict),
                OmegaConf.from_dotlist(overrides),
            )
            base_dict = OmegaConf.to_container(cfg, resolve=True)
        return cls.from_dict(base_dict)
