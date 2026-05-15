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
        if params.config_file and params.scenario:
            raise typer.BadParameter("Cannot specify both --config and --scenario.")

        if params.config_file:
            return self._load_experiment_config(params.config_file, overrides=params.config_overrides)

        if params.scenario:
            return self._load_experiment_config(params.scenario, overrides=params.config_overrides)

        if params.config_overrides:
            raise typer.BadParameter("--config-override requires --config or --scenario.")

        from trading_rl import ExperimentConfig
        return ExperimentConfig()

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
