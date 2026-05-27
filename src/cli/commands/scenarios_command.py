"""Scenarios command — list and delete scenario YAML configs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import typer

from .base_command import BaseCommand


@dataclass
class ScenariosParams:
    delete: str | None = None
    delete_all: bool = False
    force: bool = False
    dry_run: bool = False


class ScenariosCommand(BaseCommand):
    """List available scenario configurations or delete them."""

    def execute(self, params: ScenariosParams) -> None:
        import yaml

        config_dir = Path("src/configs/scenarios")
        if not config_dir.exists():
            self.console.print("[red]Config directory not found: src/configs/scenarios[/red]")
            return

        config_files = sorted(config_dir.glob("**/*.yaml"))

        if not config_files:
            self.console.print("[yellow]No configuration files found[/yellow]")
            return

        if params.delete_all or params.delete:
            pattern = re.compile(params.delete) if params.delete else None
            targets = [
                f
                for f in config_files
                if params.delete_all or (pattern and pattern.search(f.stem))
            ]
            if not targets:
                self.console.print("[yellow]No scenarios matched for deletion.[/yellow]")
                raise typer.Exit(0)
            if params.dry_run:
                self.console.print("[yellow]Dry run: scenarios to delete[/yellow]")
                for path in targets:
                    self.console.print(f"  {path}")
                raise typer.Exit(0)
            if not self._confirm_delete([str(f) for f in targets], params.force):
                self.console.print("[yellow]Deletion cancelled.[/yellow]")
                raise typer.Exit(0)
            for path in targets:
                path.unlink(missing_ok=True)
            self.console.print(f"[green]Deleted {len(targets)} scenarios.[/green]")
            raise typer.Exit(0)

        self.console.print("[bold blue]Available Scenario Configurations:[/bold blue]\n")
        for config_file in config_files:
            try:
                with config_file.open("r", encoding="utf-8") as f:
                    config = yaml.safe_load(f) or {}

                experiment_name = config.get("experiment_name", "N/A")
                algorithm = config.get("training", {}).get("algorithm", "N/A")
                data_path = config.get("data", {}).get("data_path", "N/A")
                pattern_type = config.get("data_generator", {}).get("pattern_type", "N/A")

                self.console.print(f"[green]{config_file.stem}[/green]:")
                self.console.print(f"  Experiment: {experiment_name}")
                self.console.print(f"  Algorithm: {algorithm}")
                self.console.print(f"  Pattern: {pattern_type}")
                self.console.print(
                    f"  Data: {Path(data_path).name if data_path != 'N/A' else 'N/A'}"
                )
                self.console.print()

            except Exception as e:
                self.console.print(f"[red]{config_file.stem}[/red]: Error reading file ({e})")
                self.console.print()

    def _confirm_delete(self, items: list[str], force: bool) -> bool:
        if force:
            return True
        self.console.print("[yellow]Delete the following items?[/yellow]")
        for item in items:
            self.console.print(f"  {item}")
        return typer.confirm("Proceed with deletion?", default=False)
