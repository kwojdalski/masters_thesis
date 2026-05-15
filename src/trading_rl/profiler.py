"""Pipeline profiler — tracks wall-clock time per named stage and prints a summary table."""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator

from rich.console import Console
from rich.table import Table


@dataclass
class _Record:
    calls: int = 0
    total_s: float = 0.0
    level: int = 1


class PipelineProfiler:
    """Accumulates timing records for named pipeline stages.

    Child stages (level 2) are automatically prefixed with their parent stage name:

        with profiler.stage("runtime_build"):
            with profiler.stage("data_preparation", 2):
                ...
        # recorded as "runtime_build.data_preparation"
    """

    def __init__(self, level: int = 1) -> None:
        self.level = level
        self._records: dict[str, _Record] = defaultdict(_Record)
        self._wall_start: float = time.monotonic()
        self._stack: list[str] = []

    @property
    def enabled(self) -> bool:
        return self.level > 0

    @contextmanager
    def stage(self, name: str, level: int = 1) -> Generator[None, None, None]:
        if self.level < level:
            yield
            return

        full_name = f"{self._stack[-1]}.{name}" if self._stack else name
        self._stack.append(full_name)
        t0 = time.monotonic()
        try:
            yield
        finally:
            elapsed = time.monotonic() - t0
            self._stack.pop()
            rec = self._records[full_name]
            rec.calls += 1
            rec.total_s += elapsed
            rec.level = level

    def print_table(self, console: Console | None = None) -> None:
        if not self.enabled or not self._records:
            return

        console = console or Console()
        wall_total = time.monotonic() - self._wall_start
        total_accounted = sum(r.total_s for r in self._records.values() if r.level == 1)

        table = Table(title="Pipeline Timing", show_header=True, header_style="bold green")
        table.add_column("L", justify="center", style="dim")
        table.add_column("Stage", style="cyan", no_wrap=True)
        table.add_column("Calls", justify="right")
        table.add_column("Total (s)", justify="right")
        table.add_column("Avg (s)", justify="right")
        table.add_column("% of wall", justify="right")

        for name, rec in sorted(self._records.items(), key=lambda kv: -kv[1].total_s):
            pct = 100.0 * rec.total_s / wall_total if wall_total > 0 else 0.0
            avg = rec.total_s / rec.calls if rec.calls else 0.0
            table.add_row(
                str(rec.level),
                name,
                str(rec.calls),
                f"{rec.total_s:.2f}",
                f"{avg:.2f}",
                f"{pct:.1f}%",
            )

        unaccounted = wall_total - total_accounted
        if unaccounted > 0:
            pct = 100.0 * unaccounted / wall_total
            table.add_row("—", "[dim]unaccounted[/dim]", "—", f"{unaccounted:.2f}", "—", f"{pct:.1f}%")

        table.add_section()
        table.add_row("—", "[bold]wall total[/bold]", "—", f"{wall_total:.2f}", "—", "100.0%")

        console.print(table)


# Module-level singleton — initialized once per experiment run.
_profiler: PipelineProfiler = PipelineProfiler(level=0)


def init_profiler(level: int) -> PipelineProfiler:
    """Initialize (or reset) the module-level profiler. Call once at experiment start."""
    global _profiler
    _profiler = PipelineProfiler(level=level)
    return _profiler


def get_profiler() -> PipelineProfiler:
    """Return the current module-level profiler."""
    return _profiler
