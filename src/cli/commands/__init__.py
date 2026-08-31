"""Command classes for CLI interface.

Submodules are imported lazily (PEP 562): accessing ``cli.commands.X`` imports
only the submodule that defines ``X``. This keeps lightweight subcommands
(dashboard, ps, checkpoints, ...) from dragging in the training/evaluation
stack -- torch, torchrl, and the statistical-test registry -- just to print a
table. ``cli.py`` reinforces this by importing command classes inside each
callback rather than at module load.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

# Public name -> submodule (without the leading dot) that defines it.
_EXPORTS = {
    "ArtifactsCommand": "artifacts_command",
    "ArtifactsParams": "artifacts_command",
    "BaseCommand": "base_command",
    "CheckpointsCommand": "checkpoints_command",
    "CheckpointsParams": "checkpoints_command",
    "CollectResultsCommand": "collect_results_command",
    "CollectResultsParams": "collect_results_command",
    "DashboardCommand": "dashboard_command",
    "DashboardParams": "dashboard_command",
    "DataGenerationParams": "data_generator_command",
    "DataGeneratorCommand": "data_generator_command",
    "SineWaveParams": "data_generator_command",
    "UpwardDriftParams": "data_generator_command",
    "EvaluateCommand": "evaluate_command",
    "EvaluateParams": "evaluate_command",
    "ExperimentCommand": "experiment_command",
    "ExperimentParams": "experiment_command",
    "ExperimentsCommand": "experiments_command",
    "ExperimentsParams": "experiments_command",
    "FeatureResearchCommand": "feature_research_command",
    "FeatureResearchParams": "feature_research_command",
    "PeekCommand": "peek_command",
    "PeekParams": "peek_command",
    "AttachCommand": "ps_command",
    "AttachParams": "ps_command",
    "PsCommand": "ps_command",
    "PsParams": "ps_command",
    "ScenariosCommand": "scenarios_command",
    "ScenariosParams": "scenarios_command",
    "TrainingCommand": "training_command",
    "TrainingParams": "training_command",
    "ValidateDataCommand": "validate_data_command",
    "ValidateDataParams": "validate_data_command",
    "ValidationCommand": "validation_command",
    "ValidationParams": "validation_command",
}


def __getattr__(name: str):
    """Import the defining submodule on first access, then cache the object."""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{module_name}", __name__), name)
    globals()[name] = value  # subsequent lookups hit the module dict directly
    return value


def __dir__() -> list[str]:
    return sorted(_EXPORTS)


if TYPE_CHECKING:  # eager imports for type checkers and IDEs only
    from .artifacts_command import ArtifactsCommand, ArtifactsParams
    from .base_command import BaseCommand
    from .checkpoints_command import CheckpointsCommand, CheckpointsParams
    from .collect_results_command import CollectResultsCommand, CollectResultsParams
    from .dashboard_command import DashboardCommand, DashboardParams
    from .data_generator_command import (
        DataGenerationParams,
        DataGeneratorCommand,
        SineWaveParams,
        UpwardDriftParams,
    )
    from .evaluate_command import EvaluateCommand, EvaluateParams
    from .experiment_command import ExperimentCommand, ExperimentParams
    from .experiments_command import ExperimentsCommand, ExperimentsParams
    from .feature_research_command import (
        FeatureResearchCommand,
        FeatureResearchParams,
    )
    from .peek_command import PeekCommand, PeekParams
    from .ps_command import AttachCommand, AttachParams, PsCommand, PsParams
    from .scenarios_command import ScenariosCommand, ScenariosParams
    from .training_command import TrainingCommand, TrainingParams
    from .validate_data_command import ValidateDataCommand, ValidateDataParams
    from .validation_command import ValidationCommand, ValidationParams

__all__ = [
    "ArtifactsCommand",
    "ArtifactsParams",
    "AttachCommand",
    "AttachParams",
    "BaseCommand",
    "CheckpointsCommand",
    "CheckpointsParams",
    "CollectResultsCommand",
    "CollectResultsParams",
    "DashboardCommand",
    "DashboardParams",
    "DataGenerationParams",
    "DataGeneratorCommand",
    "EvaluateCommand",
    "EvaluateParams",
    "ExperimentCommand",
    "ExperimentParams",
    "ExperimentsCommand",
    "ExperimentsParams",
    "FeatureResearchCommand",
    "FeatureResearchParams",
    "PeekCommand",
    "PeekParams",
    "PsCommand",
    "PsParams",
    "ScenariosCommand",
    "ScenariosParams",
    "SineWaveParams",
    "TrainingCommand",
    "TrainingParams",
    "UpwardDriftParams",
    "ValidateDataCommand",
    "ValidateDataParams",
    "ValidationCommand",
    "ValidationParams",
]
