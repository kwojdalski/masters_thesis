"""Experiments command — list and delete MLflow experiments."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import typer

from .base_command import BaseCommand

if TYPE_CHECKING:
    pass


@dataclass
class ExperimentsParams:
    tracking_uri: str | None = None
    delete: str | None = None
    delete_all: bool = False
    purge: bool = False
    force: bool = False
    dry_run: bool = False


class ExperimentsCommand(BaseCommand):
    """List available MLflow experiments or permanently delete them."""

    def execute(self, params: ExperimentsParams) -> None:
        if params.purge:
            db_path = self._resolve_sqlite_path(params.tracking_uri)
            pattern = re.compile(params.delete) if params.delete else None
            self._purge_experiments_sqlite(
                db_path,
                pattern,
                params.delete_all or not params.delete,
                params.dry_run,
                params.force,
            )
            return

        if not params.delete and not params.delete_all:
            self._list_experiments(params.tracking_uri)
            return

        import mlflow

        if params.tracking_uri:
            mlflow.set_tracking_uri(params.tracking_uri)
        experiments_list = mlflow.search_experiments()
        pattern = re.compile(params.delete) if params.delete else None
        targets = [
            exp
            for exp in experiments_list
            if params.delete_all or (pattern and pattern.search(exp.name))
        ]
        if not targets:
            self.console.print("[yellow]No experiments matched for deletion.[/yellow]")
            raise typer.Exit(0)
        if params.dry_run:
            self.console.print("[yellow]Dry run: experiments to soft-delete[/yellow]")
            for exp in targets:
                self.console.print(f"  {exp.name}")
            raise typer.Exit(0)
        if not self._confirm_delete([exp.name for exp in targets], params.force):
            self.console.print("[yellow]Deletion cancelled.[/yellow]")
            raise typer.Exit(0)
        for exp in targets:
            mlflow.delete_experiment(exp.experiment_id)
        db_path = self._resolve_sqlite_path(params.tracking_uri)
        pattern = re.compile(params.delete) if params.delete else None
        self._purge_experiments_sqlite(
            db_path,
            pattern,
            params.delete_all or not params.delete,
            dry_run=False,
            force=True,
        )
        self.console.print(
            f"[green]Permanently deleted {len(targets)} experiment(s).[/green]"
        )

    def _list_experiments(self, tracking_uri: str | None) -> None:
        import mlflow
        from rich.table import Table

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        experiments_list = mlflow.search_experiments()
        if not experiments_list:
            self.console.print("[yellow]No experiments found.[/yellow]")
            return
        table = Table(title="MLflow Experiments")
        table.add_column("ID")
        table.add_column("Name")
        table.add_column("Lifecycle")
        for exp in experiments_list:
            table.add_row(exp.experiment_id, exp.name, exp.lifecycle_stage)
        self.console.print(table)

    def _resolve_sqlite_path(self, tracking_uri: str | None) -> str:
        uri = tracking_uri or "sqlite:///mlflow.db"
        if uri.startswith("sqlite:///"):
            return uri[len("sqlite:///") :]
        raise typer.BadParameter(
            f"--purge only supports sqlite:/// tracking URIs, got: {uri}"
        )

    def _purge_experiments_sqlite(
        self,
        db_path: str,
        pattern: re.Pattern | None,
        delete_all: bool,
        dry_run: bool,
        force: bool,
    ) -> None:
        import sqlite3

        con = sqlite3.connect(db_path)
        rows = con.execute(
            "SELECT experiment_id, name FROM experiments WHERE lifecycle_stage = 'deleted'"
        ).fetchall()

        targets = [
            (exp_id, name)
            for exp_id, name in rows
            if delete_all or (pattern and pattern.search(name))
        ]

        if not targets:
            self.console.print(
                "[yellow]No soft-deleted experiments matched. "
                "Use `experiments` (without --purge) to soft-delete active ones first.[/yellow]"
            )
            con.close()
            raise typer.Exit(0)

        if dry_run:
            self.console.print(
                "[yellow]Dry run: experiments to permanently delete[/yellow]"
            )
            for exp_id, name in targets:
                run_count = con.execute(
                    "SELECT COUNT(*) FROM runs WHERE experiment_id = ?", (exp_id,)
                ).fetchone()[0]
                self.console.print(f"  [{exp_id}] {name}  ({run_count} run(s))")
            con.close()
            raise typer.Exit(0)

        labels = [f"[{exp_id}] {name}" for exp_id, name in targets]
        self.console.print(
            "[bold red]This will PERMANENTLY remove the following experiments "
            "and all their runs from the database:[/bold red]"
        )
        for label in labels:
            self.console.print(f"  {label}")

        if not self._confirm_delete(labels, force):
            self.console.print("[yellow]Purge cancelled.[/yellow]")
            con.close()
            raise typer.Exit(0)

        _RUN_CHILD_TABLES = (
            "params",
            "metrics",
            "latest_metrics",
            "tags",
            "inputs",
            "input_tags",
        )
        n_purged = 0
        for exp_id, _name in targets:
            run_ids = [
                r[0]
                for r in con.execute(
                    "SELECT run_uuid FROM runs WHERE experiment_id = ?", (exp_id,)
                ).fetchall()
            ]
            for run_uuid in run_ids:
                for tbl in _RUN_CHILD_TABLES:
                    try:
                        con.execute(
                            f"DELETE FROM {tbl} WHERE run_uuid = ?",  # noqa: S608
                            (run_uuid,),
                        )
                    except sqlite3.OperationalError:
                        pass  # table may not exist in all schema versions
                con.execute("DELETE FROM runs WHERE run_uuid = ?", (run_uuid,))
            con.execute("DELETE FROM experiments WHERE experiment_id = ?", (exp_id,))
            n_purged += 1

        con.commit()
        con.close()
        self.console.print(
            f"[green]Permanently deleted {n_purged} experiment(s).[/green]"
        )

    def _confirm_delete(self, items: list[str], force: bool) -> bool:
        if force:
            return True
        self.console.print("[yellow]Delete the following items?[/yellow]")
        for item in items:
            self.console.print(f"  {item}")
        return typer.confirm("Proceed with deletion?", default=False)
