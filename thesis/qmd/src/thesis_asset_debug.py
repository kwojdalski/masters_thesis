"""Debug overlay for pipeline-generated assets in Quarto chapters.

Enable by setting THESIS_DEBUG_ASSETS=1 before rendering:

    THESIS_DEBUG_ASSETS=1 quarto render thesis/qmd/src/masters-thesis.qmd

Each call to ``show_asset_debug(path)`` reads the .meta.json sidecar written
by the pipeline at asset-generation time and displays commit hash, UTC
datetime, and source module beneath the asset in the rendered document.
When the env var is unset (production renders) every call is a no-op.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_IMAGE_REF = re.compile(r"!\[[^]]*\]\(([^)\s]+)")
_IMAGE_SUFFIXES = {".eps", ".jpeg", ".jpg", ".pdf", ".png", ".svg", ".tif", ".tiff"}


def _is_debug() -> bool:
    return os.environ.get("THESIS_DEBUG_ASSETS", "").lower() in {"1", "true", "yes"}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _git_provenance(path: Path, repo_root: Path) -> dict[str, str]:
    """Return the latest committed revision and date for *path*."""
    git_bin = shutil.which("git") or "git"
    try:
        relative = path.relative_to(repo_root)
        result = subprocess.run(  # noqa: S603 -- fixed git arguments
            [git_bin, "log", "-1", "--format=%H%x00%cI", "--", str(relative)],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and "\0" in result.stdout:
            commit, generated = result.stdout.strip().split("\0", 1)
            if commit and generated:
                return {
                    "commit": commit,
                    "datetime": generated,
                    "generator": "git history",
                }
    except (OSError, subprocess.SubprocessError, ValueError):
        pass

    try:
        generated = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat()
    except OSError:
        generated = "unknown"
    return {"commit": "untracked", "datetime": generated, "generator": "file mtime"}


def _load_sidecar(path: Path) -> dict[str, str] | None:
    sidecar = path.with_name(path.name + ".meta.json")
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def collect_asset_inventory(
    source_dir: str | Path | None = None,
) -> list[dict[str, str]]:
    """Collect provenance for images referenced by QMD and generated figures.

    Sidecar metadata is authoritative. Committed static assets fall back to the
    latest Git commit that changed the file; untracked render outputs use their
    filesystem modification time.
    """
    src = Path(source_dir).resolve() if source_dir else Path(__file__).resolve().parent
    repo_root = _repo_root()
    assets: set[Path] = set()

    for qmd in src.glob("*.qmd"):
        try:
            text = qmd.read_text(encoding="utf-8")
        except OSError:
            continue
        for reference in _IMAGE_REF.findall(text):
            if reference.startswith(("http://", "https://", "data:")):
                continue
            path = (qmd.parent / reference).resolve()
            if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES:
                assets.add(path)

    generated_dir = src / "_figures"
    if generated_dir.is_dir():
        assets.update(
            path.resolve()
            for path in generated_dir.iterdir()
            if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
        )

    rows: list[dict[str, str]] = []
    for path in assets:
        meta: dict[str, Any] = _load_sidecar(path) or _git_provenance(path, repo_root)
        commit = str(meta.get("commit", "unknown"))
        if re.fullmatch(r"[0-9a-fA-F]{12,}", commit):
            commit = commit[:8]
        generated = str(meta.get("datetime", "unknown"))
        try:
            parsed = datetime.fromisoformat(generated.replace("Z", "+00:00"))
            generated = parsed.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass
        try:
            label = str(path.relative_to(src))
        except ValueError:
            try:
                label = str(path.relative_to(repo_root))
            except ValueError:
                label = str(path)
        generator = str(meta.get("generator", "unknown"))
        if generator not in {"file mtime", "git history", "unknown"}:
            generator = Path(generator).name
        rows.append(
            {
                "asset": label,
                "datetime": generated,
                "commit": commit,
                "generator": generator,
            }
        )

    return sorted(rows, key=lambda row: (row["datetime"], row["asset"]), reverse=True)


def show_asset_audit_log(source_dir: str | Path | None = None) -> None:
    """Render the audit-only inventory of thesis image assets as Markdown."""
    if not _is_debug():
        return
    from IPython.display import Markdown, display

    rows = collect_asset_inventory(source_dir)
    lines = [
        "# Asset Provenance Audit Log",
        "",
        "This audit-only inventory lists rendered image assets, newest first. "
        "Sidecar metadata takes precedence; static assets use their latest Git revision.",
        "",
        "| Generated/modified (UTC) | Commit | Asset | Provenance source |",
        "|---|---|---|---|",
    ]
    for row in rows:
        values = [row["datetime"], row["commit"], row["asset"], row["generator"]]
        escaped = [value.replace("|", "\\|") for value in values]
        lines.append("| " + " | ".join(escaped) + " |")
    if not rows:
        lines.append("| — | — | No rendered image assets found | — |")
    display(Markdown("\n".join(lines)))


def show_asset_debug(path: str | Path) -> None:
    """Display provenance metadata below *path* when THESIS_DEBUG_ASSETS=1."""
    if not _is_debug():
        return
    from IPython.display import HTML, display

    from trading_rl.evaluation.asset_meta import load_asset_meta

    meta = load_asset_meta(path)
    name = Path(path).name
    if meta is None:
        display(
            HTML(
                f'<div style="font-size:0.7em;color:#aaa;margin-top:2px">'
                f"[debug] no metadata found for <code>{name}</code></div>"
            )
        )
        return

    commit = meta.get("commit", "unknown")[:8]
    dt = meta.get("datetime", "?")
    gen = meta.get("generator", "")
    parts = [
        f"commit: <code>{commit}</code>",
        f"generated: <code>{dt}</code>",
    ]
    if gen:
        parts.append(f"source: <code>{gen}</code>")

    display(
        HTML(
            '<div style="font-size:0.7em;color:#666;border-left:3px solid #ccc;'
            'padding-left:6px;margin-top:3px">' + "  &middot;  ".join(parts) + "</div>"
        )
    )
