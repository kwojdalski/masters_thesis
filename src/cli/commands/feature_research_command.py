"""Offline feature research command implementation."""

from dataclasses import dataclass
from pathlib import Path

import typer

from trading_rl import ExperimentConfig
from trading_rl.feature_research import run_feature_research

from .base_command import BaseCommand


@dataclass
class FeatureResearchParams:
    """Parameters for offline feature research."""

    config_file: Path | None = None
    scenario: str | None = None
    config_overrides: list[str] | None = None
    output_dir: Path | None = None
    horizon: int = 1
    top_k: int = 10
    corr_threshold: float = 0.85


class FeatureResearchCommand(BaseCommand):
    """Run offline feature scoring and shortlist generation."""

    def execute(self, params: FeatureResearchParams) -> None:
        """Execute offline feature research."""
        try:
            config = self._load_config(params)
            output_dir = params.output_dir or self._default_output_dir(config)

            self.console.print("[bold blue]Running Offline Feature Research[/bold blue]")
            self.console.print(f"Experiment: [green]{config.experiment_name}[/green]")
            self.console.print(
                f"Feature config: [cyan]{config.data.feature_config}[/cyan]"
            )

            artifacts = run_feature_research(
                config=config,
                output_dir=output_dir,
                horizon=params.horizon,
                top_k=params.top_k,
                corr_threshold=params.corr_threshold,
            )

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

    def _load_config(self, params: FeatureResearchParams) -> ExperimentConfig:
        """Load experiment configuration."""
        if params.config_file and params.scenario:
            raise typer.BadParameter("Cannot specify both --config and --scenario.")

        if params.config_file:
            return ExperimentConfig.from_yaml(
                params.config_file,
                overrides=params.config_overrides,
            )
        if params.scenario:
            scenario_path = self._resolve_scenario_config_path(params.scenario)
            return ExperimentConfig.from_yaml(
                scenario_path,
                overrides=params.config_overrides,
            )
        raise typer.BadParameter("Provide --config or --scenario for feature research.")

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

    def _default_output_dir(self, config: ExperimentConfig) -> Path:
        """Build default output directory for offline research artifacts."""
        return Path(config.logging.log_dir) / "feature_research"
