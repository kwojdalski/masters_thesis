"""Asset provenance metadata — write and read sidecar .meta.json files.

Every pipeline-generated asset (plot PNG, benchmark JSON/PNG, stats CSV) gets
a companion <filename>.meta.json written alongside it at save time.  The
sidecar records the git commit and UTC datetime so that Quarto chapters can
display it in debug mode.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone, UTC
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def _git_commit() -> str:
    """Return HEAD commit hash, cached for the process lifetime."""
    try:
        repo_root = Path(__file__).resolve().parents[3]
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=repo_root,
        )
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def meta_path(asset_path: str | Path) -> Path:
    """Return the sidecar path for *asset_path* (appends .meta.json to the full name)."""
    p = Path(asset_path)
    return p.with_name(p.name + ".meta.json")


def write_asset_meta(path: str | Path, *, generator: str | None = None) -> None:
    """Write a .meta.json sidecar alongside *path*.

    Silently does nothing if the parent directory does not exist (e.g. temp dirs
    that will be deleted before the sidecar is ever read).
    """
    p = Path(path)
    if not p.parent.exists():
        return
    data: dict[str, str] = {
        "commit": _git_commit(),
        "datetime": datetime.now(UTC).isoformat(),
    }
    if generator:
        data["generator"] = generator
    meta_path(p).write_text(json.dumps(data, indent=2))


def load_asset_meta(path: str | Path) -> dict[str, str] | None:
    """Read the .meta.json sidecar for *path*.  Returns None if absent or unreadable."""
    mp = meta_path(path)
    if not mp.exists():
        return None
    try:
        return json.loads(mp.read_text())
    except Exception:
        return None
