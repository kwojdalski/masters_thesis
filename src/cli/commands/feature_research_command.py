"""Offline feature research command implementation."""

from dataclasses import dataclass
from pathlib import Path

import typer

from trading_rl import ExperimentConfig
from trading_rl.feature_research import FeatureResearchConfig, run_feature_research

from .base_command import BaseCommand


@dataclass
class FeatureResearchParams:
    """Parameters for offline feature research."""

    config_file: Path | None = None
    experiment_config_file: Path | None = None
    scenario: str | None = None
    config_overrides: list[str] | None = None


class FeatureResearchCommand(BaseCommand):
    """Run offline feature scoring and shortlist generation."""

    def execute(self, params: FeatureResearchParams) -> None:
        """Execute offline feature research."""
        try:
            config = self._load_config(params)

            self.console.print("[bold blue]Running Offline Feature Research[/bold blue]")
            self.console.print(f"Experiment: [green]{config.experiment_name}[/green]")
            self.console.print(
                f"Feature config: [cyan]{config.data.feature_config}[/cyan]"
            )

            artifacts = run_feature_research(config=config)

            self.console.print(f"[green]Scores:[/green] {artifacts.scores_csv}")
            self.console.print(
                f"[green]Correlations:[/green] {artifacts.correlation_csv}"
            )
            self.console.print(
                f"[green]Suggested feature config:[/green] {artifacts.selected_yaml}"
            )
            self.console.print(f"[green]Summary:[/green] {artifacts.summary_md}")
        except Exception as error:
            self.handle_error(error, "Offline feature research")

    def _load_config(self, params: FeatureResearchParams) -> FeatureResearchConfig:
        """Load feature research configuration."""
        input_count = sum(
            value is not None
            for value in (
                params.config_file,
                params.experiment_config_file,
                params.scenario,
            )
        )
        if input_count > 1:
            raise typer.BadParameter(
                "Use only one of --config, --experiment-config, or --scenario."
            )

        if params.config_file:
            return FeatureResearchConfig.from_yaml(
                params.config_file,
                overrides=params.config_overrides,
            )
        if params.experiment_config_file:
            exp_path = (
                params.experiment_config_file
                if params.experiment_config_file.exists()
                else self._resolve_scenario_config_path(str(params.experiment_config_file))
            )
            experiment_config = ExperimentConfig.from_yaml(
                exp_path,
                overrides=params.config_overrides,
            )
            return FeatureResearchConfig.from_experiment_config(experiment_config)
        if params.scenario:
            scenario_path = self._resolve_scenario_config_path(params.scenario)
            experiment_config = ExperimentConfig.from_yaml(
                scenario_path, overrides=params.config_overrides
            )
            return FeatureResearchConfig.from_experiment_config(experiment_config)
        raise typer.BadParameter(
            "Provide --config, --experiment-config, or --scenario for feature research."
        )

    def _resolve_scenario_config_path(self, scenario: str) -> Path:
        """Resolve scenario name to config file path."""
        candidate_path = Path(scenario)
        if candidate_path.is_dir():
            candidate_path = candidate_path / "config.yaml"

        search_paths = [
            candidate_path,
            Path("src/configs/scenarios") / scenario,
            Path("src/configs/scenarios") / f"{scenario}.yaml",
        ]
        for path in search_paths:
            if path.exists():
                return path.resolve()

        raise typer.BadParameter(
            f"Scenario '{scenario}' not found. Provide a valid path or scenario name."
        )
