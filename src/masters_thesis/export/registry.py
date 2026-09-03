"""Registry of the stages that move artifacts into the thesis snapshot.

Everything the thesis renders is supposed to come from
``thesis/qmd/results/**``, written by a handful of export scripts. Those
scripts accumulated one at a time, so there was no single place recording
what the snapshot needs, which of them had run, or when. Only two were wired
into the Quarto pre-render hook; the rest were manual and could silently go
stale -- the ``peek/*`` artifacts in particular had no freshness signal at
all.

The awkward part, and the reason this is a registry rather than a script that
runs everything in sequence, is that the stages do not share a prerequisite.
Exporting an evaluation snapshot needs the MLflow store; the ``peek``
artifacts need the prepared parquet data; the LaTeX value macros need only
the snapshot the earlier stages produced. ``mlruns/``, ``mlflow.db`` and
``data/`` are all gitignored, so on CI the first two classes cannot run at
all while the third must. A single "export everything" entry point is only
possible if each stage states what it needs and the runner skips the ones
whose inputs are absent instead of failing the build.

Stages therefore declare a :class:`Requirement` set, and
:meth:`ThesisExportRegistry.runnable` filters against what the machine
actually has. ``order`` exists for the one real dependency: stages that read
the snapshot must run after the stages that write it.

Follows the same shape as the reward, trainer and feature registries -- a
module-level dict, a classmethod ``register`` decorator, and lookup helpers.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]


class Requirement(Enum):
    """An input a stage needs that may legitimately be absent.

    Each maps to a gitignored tree, which is exactly why absence has to be a
    skip rather than an error: a fresh clone or a CI runner has none of them.
    """

    MLFLOW = "mlflow"
    PREPARED_DATA = "prepared-data"
    # reports/peek/, written by `cli.py peek dataset --export`. Distinct from
    # PREPARED_DATA: the peek stage copies that scratch output rather than
    # recomputing from the prepared parquet, so having the data is not enough
    # -- the peek command must actually have been run.
    PEEK_SCRATCH = "peek-scratch"
    SNAPSHOT = "snapshot"

    def satisfied(self, repo_root: Path = _REPO_ROOT) -> bool:
        if self is Requirement.MLFLOW:
            return (repo_root / "mlruns").is_dir() and (
                repo_root / "mlflow.db"
            ).exists()
        if self is Requirement.PREPARED_DATA:
            return (repo_root / "data" / "prepared").is_dir()
        if self is Requirement.PEEK_SCRATCH:
            return (repo_root / "reports" / "peek").is_dir()
        return (repo_root / "thesis" / "qmd" / "results").is_dir()


@dataclass
class StageResult:
    """Outcome of one stage: ran and succeeded, ran and failed, or skipped."""

    stage: str
    status: str  # "ok" | "failed" | "skipped"
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status in ("ok", "skipped")


@dataclass(frozen=True)
class Stage:
    """One artifact-producing step.

    ``command`` is run as a subprocess rather than imported because these are
    argparse scripts with their own ``main()``; ``export_all_to_thesis.py``
    already invokes ``export_eval_to_thesis.py`` this way, so the pattern is
    the existing one rather than a new one.
    """

    name: str
    description: str
    command: tuple[str, ...]
    requires: frozenset[Requirement] = field(default_factory=frozenset)
    # Stages that consume the snapshot must follow those that write it.
    # Producers sit below 50, consumers above.
    order: int = 50

    def missing(self, repo_root: Path = _REPO_ROOT) -> list[Requirement]:
        return [
            r
            for r in sorted(self.requires, key=lambda x: x.value)
            if not r.satisfied(repo_root)
        ]

    def run(
        self, repo_root: Path = _REPO_ROOT, *, dry_run: bool = False
    ) -> StageResult:
        missing = self.missing(repo_root)
        if missing:
            names = ", ".join(r.value for r in missing)
            return StageResult(self.name, "skipped", f"missing {names}")
        if dry_run:
            return StageResult(self.name, "skipped", "dry run")
        proc = subprocess.run(  # noqa: S603 — command is a literal built in stages.py, not external input
            self.command, cwd=repo_root, capture_output=True, text=True
        )
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()
            return StageResult(
                self.name, "failed", tail[-1] if tail else f"exit {proc.returncode}"
            )
        return StageResult(self.name, "ok")


_REGISTRY: dict[str, Stage] = {}


class ThesisExportRegistry:
    """Central registry for thesis export stages."""

    @classmethod
    def register(cls, stage: Stage) -> Stage:
        """Add *stage*, rejecting a duplicate name.

        Registration is a plain call rather than a decorator: a stage is data
        (a name, a command, a requirement set), not a function body, so there
        is nothing to decorate.
        """
        if stage.name in _REGISTRY:
            raise ValueError(f"Stage already registered: {stage.name!r}")
        _REGISTRY[stage.name] = stage
        return stage

    @classmethod
    def get(cls, name: str) -> Stage:
        stage = _REGISTRY.get(name)
        if stage is None:
            raise ValueError(
                f"Unknown stage: {name!r}. Registered: {sorted(_REGISTRY)}"
            )
        return stage

    @classmethod
    def stages(cls) -> list[Stage]:
        """Every registered stage, in run order."""
        return sorted(_REGISTRY.values(), key=lambda s: (s.order, s.name))

    @classmethod
    def names(cls) -> list[str]:
        return [s.name for s in cls.stages()]

    @classmethod
    def runnable(cls, repo_root: Path = _REPO_ROOT) -> list[Stage]:
        """Stages whose inputs are all present on this machine."""
        return [s for s in cls.stages() if not s.missing(repo_root)]
