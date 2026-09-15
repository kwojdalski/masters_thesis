"""Single entry point for moving artifacts into the thesis snapshot.

    uv run thesis-export              # every stage whose inputs are present
    uv run thesis-export --list       # what exists, and what this machine can run
    uv run thesis-export -s value-macros -s peek
    uv run thesis-export --dry-run

Stages that cannot run here are *skipped*, not failed: ``mlruns/``,
``mlflow.db`` and ``data/`` are gitignored, so CI legitimately has none of
them and must still be able to regenerate the value macros from the committed
snapshot. ``--strict`` turns a skip into an error for the cases where the
caller knows the inputs ought to be present.

See ``registry.py`` for why this is a registry rather than a fixed sequence,
and ``docs/masters_thesis/data_pipeline.md`` for where each artifact ends up.
"""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console

from masters_thesis.export import stages as _stages  # noqa: F401  (registers stages)
from masters_thesis.export.registry import (
    Requirement,
    StageResult,
    ThesisExportRegistry,
)

app = typer.Typer(
    add_completion=False,
    help="Export experiment artifacts into thesis/qmd/results.",
)

_con = Console()
_err = Console(stderr=True)


def _print_listing() -> None:
    _con.print("[bold]Requirements on this machine[/bold]")
    for req in Requirement:
        mark = (
            "[green]present[/green]" if req.satisfied() else "[yellow]absent[/yellow]"
        )
        _con.print(f"  {req.value:<14} {mark}")

    _con.print("\n[bold]Stages[/bold] (in run order)")
    for stage in ThesisExportRegistry.stages():
        missing = stage.missing()
        state = "[yellow]skip[/yellow]" if missing else "[green]run [/green]"
        why = f"  needs {', '.join(r.value for r in missing)}" if missing else ""
        _con.print(f"  {state} {stage.name:<22} {stage.description}{why}")


# Declared as Annotated aliases rather than call-in-default arguments, matching
# experiments.py and keeping ruff's B008 quiet without a suppression.
_StageOpt = Annotated[
    list[str] | None,
    typer.Option(
        "--stage",
        "-s",
        help="Run only these stages (repeatable). Default: every runnable stage.",
    ),
]
_ListOpt = Annotated[
    bool,
    typer.Option(
        "--list", help="Show stages and what this machine can run, then exit."
    ),
]
_DryRunOpt = Annotated[
    bool, typer.Option("--dry-run", help="Report what would run without running it.")
]
_StrictOpt = Annotated[
    bool, typer.Option("--strict", help="Treat a skipped stage as a failure.")
]


@app.callback(invoke_without_command=True)
def main(
    stage: _StageOpt = None,
    list_only: _ListOpt = False,
    dry_run: _DryRunOpt = False,
    strict: _StrictOpt = False,
) -> None:
    if list_only:
        _print_listing()
        raise typer.Exit(0)

    if stage:
        try:
            selected = [ThesisExportRegistry.get(name) for name in stage]
        except ValueError as exc:
            _err.print(f"[red]{exc}[/red]")
            raise typer.Exit(2) from exc
        selected.sort(key=lambda s: (s.order, s.name))
    else:
        selected = ThesisExportRegistry.stages()

    results: list[StageResult] = []
    for item in selected:
        result = item.run(dry_run=dry_run)
        results.append(result)
        if result.status == "ok":
            _con.print(f"[green]ok      [/green] {item.name}")
        elif result.status == "skipped":
            _con.print(f"[yellow]skipped [/yellow] {item.name}  ({result.detail})")
        else:
            _err.print(f"[red]failed  [/red] {item.name}  ({result.detail})")

    failed = [r for r in results if r.status == "failed"]
    skipped = [r for r in results if r.status == "skipped"]
    _con.print(
        f"\n{len(results) - len(failed) - len(skipped)} ok, "
        f"{len(skipped)} skipped, {len(failed)} failed"
    )

    if failed or (strict and skipped):
        raise typer.Exit(1)


__all__ = ["app"]
