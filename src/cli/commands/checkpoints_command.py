"""Checkpoints command — list and delete .pt checkpoint files."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import typer
from rich.table import Table

from .base_command import BaseCommand


@dataclass
class CheckpointsParams:
    log_dir: Path = Path("logs")
    delete: str | None = None
    delete_all: bool = False
    force: bool = False
    dry_run: bool = False


def _parse_checkpoint_step(filename: str) -> int | None:
    marker = "_checkpoint_step_"
    if marker not in filename:
        return None
    suffix = filename.split(marker, 1)[1]
    if suffix.endswith(".pt"):
        suffix = suffix[:-3]
    try:
        return int(suffix)
    except ValueError:
        return None


class CheckpointsCommand(BaseCommand):
    """List checkpoints grouped by experiment with size, mtime, and step."""

    def execute(self, params: CheckpointsParams) -> None:
        checkpoints: dict[str, list[dict[str, str]]] = {}
        checkpoint_paths: list[Path] = []

        if not params.log_dir.exists():
            self.console.print(f"[red]Log directory not found: {params.log_dir}[/red]")
            raise typer.Exit(1)

        for root, _dirs, files in os.walk(params.log_dir):
            for name in files:
                if not name.endswith(".pt") or "_checkpoint" not in name:
                    continue
                experiment = name.split("_checkpoint", 1)[0]
                path = Path(root) / name
                checkpoint_paths.append(path)
                stat = path.stat()
                size_kb = f"{stat.st_size / 1024:.1f} KB"
                modified = (
                    datetime.fromtimestamp(stat.st_mtime, tz=UTC)
                    .astimezone()
                    .strftime("%Y-%m-%d %H:%M:%S")
                )
                step = _parse_checkpoint_step(name)
                checkpoints.setdefault(experiment, []).append(
                    {
                        "path": str(path),
                        "size": size_kb,
                        "modified": modified,
                        "step": str(step) if step is not None else "-",
                    }
                )

        if not checkpoint_paths:
            self.console.print("[yellow]No checkpoints found.[/yellow]")
            raise typer.Exit(0)

        if params.delete_all or params.delete:
            pattern = re.compile(params.delete) if params.delete else None
            targets = [
                p
                for p in checkpoint_paths
                if params.delete_all or (pattern and pattern.search(p.name))
            ]
            if not targets:
                self.console.print(
                    "[yellow]No checkpoints matched for deletion.[/yellow]"
                )
                raise typer.Exit(0)
            if params.dry_run:
                self.console.print("[yellow]Dry run: checkpoints to delete[/yellow]")
                for path in targets:
                    self.console.print(f"  {path}")
                raise typer.Exit(0)
            if not self._confirm_delete([str(p) for p in targets], params.force):
                self.console.print("[yellow]Deletion cancelled.[/yellow]")
                raise typer.Exit(0)
            for path in targets:
                path.unlink(missing_ok=True)
            self.console.print(f"[green]Deleted {len(targets)} checkpoints.[/green]")
            raise typer.Exit(0)

        for experiment, items in sorted(checkpoints.items()):
            table = Table(title=f"Experiment: {experiment}")
            table.add_column("Checkpoint")
            table.add_column("Step", justify="right")
            table.add_column("Size", justify="right")
            table.add_column("Modified", justify="right")
            for item in sorted(items, key=lambda x: x["path"]):
                table.add_row(
                    item["path"], item["step"], item["size"], item["modified"]
                )
            self.console.print(table)

    def _confirm_delete(self, items: list[str], force: bool) -> bool:
        if force:
            return True
        self.console.print("[yellow]Delete the following items?[/yellow]")
        for item in items:
            self.console.print(f"  {item}")
        return typer.confirm("Proceed with deletion?", default=False)
