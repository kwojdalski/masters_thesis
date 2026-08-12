#!/usr/bin/env python3
"""Render the thesis PDF, commit it, and push to origin.

Steps
-----
1. ``uv run quarto render … --to pdf``  (optionally with AUDIT_PLOTS=1)
2. ``git add thesis/qmd/src/masters_thesis.pdf``
3. ``git commit`` with an auto-generated timestamped message
4. ``git push --force-with-lease`` (or ``--force`` with --hard-force)

Usage
-----
    uv run python scripts/publish_thesis.py
    uv run python scripts/publish_thesis.py --audit
    uv run python scripts/publish_thesis.py --force
    uv run python scripts/publish_thesis.py --audit --force
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_QMD = _REPO_ROOT / "thesis" / "qmd" / "src" / "masters-thesis.qmd"
_PDF = _REPO_ROOT / "thesis" / "qmd" / "src" / "masters_thesis.pdf"
_GIT = shutil.which("git") or "git"
_UV = shutil.which("uv") or "uv"


def _run(
    cmd: list[str], *, env: dict | None = None, check: bool = True
) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(  # noqa: S603 -- callers only pass fixed, hardcoded command lists
        cmd, cwd=_REPO_ROOT, env=env, check=check
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description="Render thesis PDF, commit, and push.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--audit",
        action="store_true",
        help="Set AUDIT_PLOTS=1 during the Quarto render.",
    )
    p.add_argument(
        "--force",
        "--hard-force",
        dest="force",
        action="store_true",
        help="Use --force instead of --force-with-lease on git push.",
    )
    args = p.parse_args()

    # --- render ---
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    if args.audit:
        env["AUDIT_PLOTS"] = "1"
        print("AUDIT_PLOTS=1 enabled")

    _run(
        [_UV, "run", "quarto", "render", str(_QMD), "--to", "pdf"],
        env=env,
    )

    if not _PDF.exists():
        print(f"ERROR: expected PDF not found at {_PDF}", file=sys.stderr)
        return 1

    print(f"PDF size: {_PDF.stat().st_size:,} bytes")

    # --- commit ---
    _run([_GIT, "add", str(_PDF)])

    # Check whether there is actually something staged.
    diff = subprocess.run(  # noqa: S603 -- resolved via shutil.which, fixed args
        [_GIT, "diff", "--cached", "--quiet"],
        cwd=_REPO_ROOT,
        check=False,
    )
    if diff.returncode == 0:
        print("Nothing changed in the PDF — skipping commit and push.")
        return 0

    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    msg = f"Chore: update rendered thesis PDF ({timestamp})"
    _run([_GIT, "commit", "-m", msg])

    # --- push ---
    push_flag = "--force" if args.force else "--force-with-lease"
    _run([_GIT, "push", push_flag, "origin", "HEAD"])

    print("Done — thesis PDF published.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
