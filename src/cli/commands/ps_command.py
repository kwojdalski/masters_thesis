"""ps/attach commands — list and inspect live trainer IPC servers.

Requires the trainer to have been started with `training.ipc_enabled=true`
(see trading_rl/ipc.py); the CLI side is deliberately thin, all
state lives in the trainer process.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import typer
from rich.live import Live
from rich.table import Table

from trading_rl.ipc import IPC_DIR, IpcClient, list_registered

from .base_command import BaseCommand


@dataclass
class PsParams:
    pass


class PsCommand(BaseCommand):
    """List currently registered (IPC-enabled) trainer processes."""

    def execute(self, params: PsParams) -> None:
        entries = list_registered()
        if not entries:
            self.console.print(
                "[yellow]No registered processes.[/yellow] Start one with "
                "[cyan]-o training.ipc_enabled=true[/cyan]."
            )
            raise typer.Exit(0)

        table = Table(title=f"Registered processes ({IPC_DIR})")
        table.add_column("run_id")
        table.add_column("pid", justify="right")
        table.add_column("algorithm")
        table.add_column("label")
        table.add_column("started_at")
        for entry in entries:
            table.add_row(
                entry.get("run_id", "-"),
                str(entry.get("pid", "-")),
                str(entry.get("algorithm", "-")),
                str(entry.get("label", "-")),
                str(entry.get("started_at", "-")),
            )
        self.console.print(table)


@dataclass
class AttachParams:
    run_id: str
    watch: bool = False
    interval: float = 1.0
    path: str | None = None


class AttachCommand(BaseCommand):
    """Print (or live-watch) one registered process's status/getter values."""

    def _resolve_client(self, run_id: str) -> IpcClient:
        sock_path = IPC_DIR / f"{run_id}.sock"
        if not sock_path.exists():
            # run_ids are opaque short hexes (socket-path-length constrained),
            # so also let the user match on the human-readable label from
            # `ps` (e.g. algorithm name or checkpoint prefix).
            matches = [
                e
                for e in list_registered()
                if e["run_id"].startswith(run_id)
                or run_id.lower() in str(e.get("label", "")).lower()
            ]
            if len(matches) == 1:
                sock_path = Path(matches[0]["sock_path"])
            elif len(matches) > 1:
                self.console.print(
                    f"[red]Ambiguous match for {run_id!r}: "
                    f"{[m['run_id'] for m in matches]}[/red]"
                )
                raise typer.Exit(1)
            else:
                self.console.print(
                    f"[red]No registered process matching {run_id!r}[/red]"
                )
                raise typer.Exit(1)
        return IpcClient(sock_path)

    def _render(self, client: IpcClient, path: str | None) -> Table:
        table = Table()
        table.add_column("key")
        table.add_column("value")
        if path:
            table.add_row(path, str(client.get(path)))
        else:
            for key, value in client.status().items():
                table.add_row(key, str(value))
        return table

    def execute(self, params: AttachParams) -> None:
        client = self._resolve_client(params.run_id)
        try:
            if not params.watch:
                self.console.print(self._render(client, params.path))
                return
            with Live(self._render(client, params.path), console=self.console) as live:
                while True:
                    time.sleep(params.interval)
                    try:
                        live.update(self._render(client, params.path))
                    except (ConnectionRefusedError, FileNotFoundError, OSError):
                        self.console.print("[yellow]Process exited.[/yellow]")
                        return
        except (ConnectionRefusedError, FileNotFoundError, OSError) as exc:
            self.handle_error(exc, "attach")
