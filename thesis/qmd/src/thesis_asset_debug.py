"""Debug overlay for pipeline-generated assets in Quarto chapters.

Enable by setting THESIS_DEBUG_ASSETS=1 before rendering:

    THESIS_DEBUG_ASSETS=1 quarto render thesis/qmd/src/masters-thesis.qmd

Each call to ``show_asset_debug(path)`` reads the .meta.json sidecar written
by the pipeline at asset-generation time and displays commit hash, UTC
datetime, and source module beneath the asset in the rendered document.
When the env var is unset (production renders) every call is a no-op.
"""

from __future__ import annotations

import os
from pathlib import Path


def _is_debug() -> bool:
    return bool(os.environ.get("THESIS_DEBUG_ASSETS"))


def show_asset_debug(path: str | Path) -> None:
    """Display provenance metadata below *path* when THESIS_DEBUG_ASSETS=1."""
    if not _is_debug():
        return
    from IPython.display import display, HTML
    from trading_rl.evaluation.asset_meta import load_asset_meta

    meta = load_asset_meta(path)
    name = Path(path).name
    if meta is None:
        display(HTML(
            f'<div style="font-size:0.7em;color:#aaa;margin-top:2px">'
            f'[debug] no metadata found for <code>{name}</code></div>'
        ))
        return

    commit = meta.get("commit", "unknown")[:8]
    dt = meta.get("datetime", "?")
    gen = meta.get("generator", "")
    parts = [
        f'commit: <code>{commit}</code>',
        f'generated: <code>{dt}</code>',
    ]
    if gen:
        parts.append(f'source: <code>{gen}</code>')

    display(HTML(
        '<div style="font-size:0.7em;color:#666;border-left:3px solid #ccc;'
        'padding-left:6px;margin-top:3px">'
        + "  &middot;  ".join(parts)
        + "</div>"
    ))
