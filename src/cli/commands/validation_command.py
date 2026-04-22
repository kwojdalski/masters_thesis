"""Validation command implementation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer
from rich.table import Table

from cli.services import validate_experiment_config

from .base_command import BaseCommand


@dataclass
class ValidationParams:
    """Parameters for config/data validation."""

    config_file: Path | None = None
    scenario: str | None = None
    config_overrides: list[str] | None = None


class ValidationCommand(BaseCommand):
    """Command for validating experiment configuration and data dependencies."""

    def execute(self, params: ValidationParams) -> None:
        try:
            config = self._load_validation_config(params)
            report = validate_experiment_config(config)
            self._render_report(report)
            if report.has_errors:
                raise typer.Exit(1)
        except typer.Exit:
            raise
        except Exception as exc:
            self.handle_error(exc, "Validation")

    def _load_validation_config(self, params: ValidationParams) -> Any:
        from trading_rl import ExperimentConfig

        if params.config_file and params.scenario:
            raise typer.BadParameter("Cannot specify both --config and --scenario.")

        if params.config_file:
            config_path = (
                params.config_file
                if params.config_file.exists()
                else self._resolve_scenario_config_path(str(params.config_file))
            )
            config = ExperimentConfig.from_yaml(
                config_path, overrides=params.config_overrides
            )
            self.console.print(f"[blue]Loaded config from: {config_path}[/blue]")
            return config

        if params.scenario:
            config_file = self._resolve_scenario_config_path(params.scenario)
            config = ExperimentConfig.from_yaml(
                config_file, overrides=params.config_overrides
            )
            self.console.print(
                f"[blue]Loaded config from scenario: {params.scenario} -> {config_file}[/blue]"
            )
            return config

        if params.config_overrides:
            raise typer.BadParameter("--config-override requires --config or --scenario.")

        return ExperimentConfig()

    def _resolve_scenario_config_path(self, scenario: str) -> Path:
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
            f"Scenario '{scenario}' not found. Provide a valid path or name in src/configs/scenarios."
        )

    def _render_report(self, report) -> None:
        table = Table(title="Validation Report")
        table.add_column("Severity")
        table.add_column("Check")
        table.add_column("Code")
        table.add_column("Message")

        if not report.issues:
            self.console.print("[green]No validation issues found.[/green]")
            return

        for issue in report.issues:
            severity_color = "red" if issue.severity == "error" else "yellow"
            table.add_row(
                f"[{severity_color}]{issue.severity.upper()}[/{severity_color}]",
                issue.check,
                issue.code,
                issue.message,
            )
        self.console.print(table)
        self.console.print(
            f"Summary: errors={report.error_count}, warnings={report.warning_count}"
        )
