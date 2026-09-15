"""Guards the contract that lets one export command work everywhere.

The registry exists so ``uv run thesis-export`` can run on a machine holding
the MLflow store and the prepared data, and on CI, which has neither. That
only holds if two things stay true: a stage whose inputs are absent is
skipped rather than failed, and the stages that read the snapshot run after
the ones that write it. Both are easy to break by adding a stage without
thinking about where it can run, so they are asserted here.
"""

from __future__ import annotations

from pathlib import Path

from masters_thesis.export import stages as _stages  # noqa: F401  (registers stages)
from masters_thesis.export.registry import (
    Requirement,
    Stage,
    ThesisExportRegistry,
)


def test_stages_are_registered() -> None:
    names = ThesisExportRegistry.names()
    assert "eval" in names
    assert "value-macros" in names


def test_snapshot_consumers_run_after_producers() -> None:
    """A stage reading the snapshot must not run before one writing it."""
    by_name = {s.name: s for s in ThesisExportRegistry.stages()}
    producers = [s for s in by_name.values() if Requirement.SNAPSHOT not in s.requires]
    consumers = [s for s in by_name.values() if Requirement.SNAPSHOT in s.requires]
    assert consumers, "expected at least one snapshot-derived stage"
    assert max(p.order for p in producers) < min(c.order for c in consumers)


def test_absent_requirement_skips_rather_than_fails(tmp_path: Path) -> None:
    """The CI case: no mlruns/, no data/ -- the stage must skip, not fail.

    A failure here would break the published build, since the render's
    pre-render hook calls this runner.
    """
    stage = Stage(
        name="needs-everything",
        description="test double",
        command=("false",),  # would fail if it were ever executed
        requires=frozenset({Requirement.MLFLOW, Requirement.PREPARED_DATA}),
    )
    result = stage.run(repo_root=tmp_path)
    assert result.status == "skipped"
    assert result.ok
    assert "mlflow" in result.detail


def test_satisfied_requirement_runs_the_command(tmp_path: Path) -> None:
    (tmp_path / "thesis" / "qmd" / "results").mkdir(parents=True)
    stage = Stage(
        name="snapshot-only",
        description="test double",
        command=("true",),
        requires=frozenset({Requirement.SNAPSHOT}),
    )
    assert stage.run(repo_root=tmp_path).status == "ok"


def test_failing_command_is_reported_as_failed(tmp_path: Path) -> None:
    (tmp_path / "thesis" / "qmd" / "results").mkdir(parents=True)
    stage = Stage(
        name="broken",
        description="test double",
        command=("false",),
        requires=frozenset({Requirement.SNAPSHOT}),
    )
    result = stage.run(repo_root=tmp_path)
    assert result.status == "failed"
    assert not result.ok


def test_duplicate_registration_is_rejected() -> None:
    existing = ThesisExportRegistry.stages()[0]
    try:
        ThesisExportRegistry.register(existing)
    except ValueError as exc:
        assert existing.name in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected duplicate registration to raise")


def test_peek_runs_before_the_stages_that_recompute_its_files() -> None:
    """peek copies correlations.csv / feature_stats.csv; two stages recompute them.

    export_peek_to_thesis.py copies four files out of the gitignored
    reports/peek/ scratch, two of which -- correlations.csv and
    feature_stats.csv -- are also written, from the memmaps, by the
    feature-correlations and feature-stats stages. Whichever runs last wins.
    The computed pair is authoritative (the checked-in correlations.csv is
    2,370 bytes against the scratch copy's 740), so peek has to run first.

    At equal order the tie breaks alphabetically and "peek" sorts after both
    "feature-correlations" and "feature-stats", which would silently publish
    stale scratch. Hence the explicit lower order, guarded here.
    """
    by_name = {s.name: s for s in ThesisExportRegistry.stages()}
    peek = by_name["peek"]
    for recomputes in ("feature-correlations", "feature-stats"):
        assert peek.order < by_name[recomputes].order, (
            f"peek must run before {recomputes}, or its copy of that file "
            "overwrites the freshly computed one"
        )
