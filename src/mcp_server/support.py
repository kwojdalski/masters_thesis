"""Shared helpers for exposing CLI command classes as MCP tools.

CLI commands in this repo (see `cli/commands/`) don't return structured results:
they write to a `rich.Console` and use `typer.Exit` / `typer.BadParameter` for
control flow, following the pattern in `BaseCommand.handle_error`. These helpers
adapt that convention into plain dicts an MCP tool can return, and guard against
`typer.confirm()` prompts that would otherwise hang with no interactive stdin.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from typing import Any

import typer
from rich.console import Console


def new_capture_console(width: int = 100) -> tuple[Console, io.StringIO]:
    """Build a Console that renders to an in-memory buffer instead of a terminal."""
    buffer = io.StringIO()
    console = Console(
        file=buffer, force_terminal=False, no_color=True, width=width, highlight=False
    )
    return console, buffer


def run_command(
    console: Console,
    buffer: io.StringIO,
    build: Callable[[Console], Any],
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run a `BaseCommand` and normalize its console/typer.Exit outcome into a dict.

    `build(console)` must return the command instance; its `execute` is then
    called with `*args, **kwargs`.
    """
    command = build(console)
    try:
        result = command.execute(*args, **kwargs)
        return {"ok": True, "output": buffer.getvalue(), "result": _jsonable(result)}
    except typer.Exit as exc:
        return {
            "ok": not exc.exit_code,
            "output": buffer.getvalue(),
            "exit_code": exc.exit_code,
        }
    except Exception as exc:
        return {"ok": False, "output": buffer.getvalue(), "error": str(exc)}


def _jsonable(value: Any) -> Any:
    """Best-effort conversion of a command's return value to JSON-safe data."""
    if value is None or isinstance(value, str | int | float | bool | dict | list):
        return value
    return str(value)


def require_force_for_delete(
    delete: str | None, delete_all: bool, force: bool, dry_run: bool
) -> None:
    """Refuse delete requests that would hit an interactive `typer.confirm()` prompt.

    MCP tool calls run non-interactively, so without this guard a delete request
    made with `force=False` would block on stdin indefinitely.
    """
    if (delete or delete_all) and not force and not dry_run:
        raise ValueError(
            "Deletion requires force=True in a non-interactive MCP session "
            "(or dry_run=True to preview the targets first)."
        )
