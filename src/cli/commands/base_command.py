"""Base command class with common functionality."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import typer
import yaml
from rich.console import Console

from logger import get_logger


class BaseCommand(ABC):
    """Base class for all CLI commands."""

    def __init__(self, console: Console):
        self.console = console
        self.logger = get_logger(self.__class__.__name__)

    def load_config(self, config_path: Path | None) -> dict[str, Any]:
        """Load configuration from YAML file."""
        if not config_path:
            return {}

        self.logger.info("load config path=%s", config_path)
        with config_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def handle_error(self, error: Exception, context: str) -> None:
        """Standardized error handling."""
        self.console.print(f"[bold red]{context} failed: {error}[/bold red]")
        self.logger.error("command failed context=%s", context, exc_info=error)
        raise typer.Exit(1) from error

    def resolve_seed(self, provided_seed: int | None) -> int:
        """Resolve seed value - use provided or generate random."""
        if provided_seed is not None:
            self.console.print(f"[blue]Using specified seed: {provided_seed}[/blue]")
            return provided_seed

        import random
        generated_seed = random.randint(1, 100000)  # noqa: S311
        self.console.print(f"[yellow]Using random seed: {generated_seed}[/yellow]")
        return generated_seed

    def _resolve_scenario_config_path(
        self, scenario: str, command_file: str = "train.yaml"
    ) -> Path:
        """Resolve a scenario name or path to a config file.

        Resolution order for each candidate location:
        1. If it is a directory: look for <command_file> inside it,
           then fall back to train.yaml.
        2. If it is an existing file: use it directly.

        Candidate locations tried in order:
        - The path as given
        - src/configs/scenarios/<scenario>
        - src/configs/scenarios/<scenario>.yaml  (legacy flat-file fallback)
        """
        def _pick_from_dir(d: Path) -> Path | None:
            specific = d / command_file
            if specific.exists():
                return specific.resolve()
            fallback = d / "train.yaml"
            if fallback.exists():
                return fallback.resolve()
            return None

        candidate = Path(scenario)
        search_paths: list[Path] = [
            candidate,
            Path("src/configs/scenarios") / scenario,
            Path("src/configs/scenarios") / f"{scenario}.yaml",
        ]

        for path in search_paths:
            if path.is_dir():
                resolved = _pick_from_dir(path)
                if resolved is not None:
                    return resolved
            elif path.exists():
                return path.resolve()

        raise ValueError(
            f"Scenario '{scenario}' not found. Provide a valid path or name under "
            "src/configs/scenarios."
        )

    def _load_experiment_config(
        self,
        scenario_or_path: str | Path,
        command: str = "train",
        overrides: list[str] | None = None,
    ) -> Any:
        """Load ExperimentConfig from a scenario directory or a legacy YAML file.

        Prefers the component-file layout (observation.yaml + {command}.yaml) when the
        resolved path is a directory; falls back to a single-file load otherwise.
        """
        from trading_rl import ExperimentConfig

        path = Path(scenario_or_path)

        # Expand short scenario names to their canonical directory under src/configs/scenarios
        if not path.exists():
            candidate = Path("src/configs/scenarios") / scenario_or_path
            if candidate.exists():
                path = candidate

        if path.is_dir():
            command_file = f"{command}.yaml"
            if (path / command_file).exists() or (path / "observation.yaml").exists():
                return ExperimentConfig.from_scenario(path, command=command, overrides=overrides)
            # Directory exists but has no component files — fall through to legacy resolution
            resolved = self._resolve_scenario_config_path(str(scenario_or_path), command_file=command_file)
            return ExperimentConfig.from_yaml(resolved, overrides=overrides)

        # Legacy: single YAML file (absolute or relative)
        if path.is_file():
            return ExperimentConfig.from_yaml(path, overrides=overrides)

        # Last resort: resolve via existing helper (handles flat .yaml fallback)
        resolved = self._resolve_scenario_config_path(
            str(scenario_or_path), command_file=f"{command}.yaml"
        )
        return ExperimentConfig.from_yaml(resolved, overrides=overrides)

    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """Execute the command."""
        pass