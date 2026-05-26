"""Shared HTML table rendering helpers for thesis QMD chapters.

All table-building logic lives here so QMD cells only handle data loading
and a single display() call.
"""

from __future__ import annotations

import math

import pandas as pd
from IPython.display import HTML, display


def fmt_val(val: object, fmt: str) -> str:
    """Format a metric value; return '—' for None or non-finite floats."""
    if val is None or (isinstance(val, float) and not math.isfinite(val)):
        return "—"
    try:
        return f"{val:{fmt}}"
    except Exception:
        return str(val)


def fmt_delta(val: object, baseline: object, fmt: str, higher_better: bool = True) -> str:
    """Format the signed delta between val and baseline using fmt."""
    if not isinstance(val, (int, float)) or not isinstance(baseline, (int, float)):
        return ""
    if not math.isfinite(val) or not math.isfinite(baseline):
        return ""
    delta = val - baseline
    try:
        return f"{delta:+{fmt}}"
    except Exception:
        return ""


def simple_html_table(rows: list[dict], index_col: str | None = None) -> HTML:
    """Render a list of dicts as a plain HTML table."""
    df = pd.DataFrame(rows)
    return HTML(df.to_html(index=index_col is not None))


def display_df(frame: pd.DataFrame) -> None:
    """Display a DataFrame as an HTML table without the row index."""
    display(HTML(frame.to_html(index=False)))


# ---------------------------------------------------------------------------
# Two-sub-row table (value row + delta row per group)
# ---------------------------------------------------------------------------

def comparison_table_html(
    rows: list[dict],
    cols: list[tuple[str, str, str]],
    index_col: str = "Feature Set",
) -> HTML:
    """Build an HTML table where each group occupies two rows.

    First row: metric values.
    Second row: deltas (dimmed, smaller font).
    The index_col cell spans both rows via rowspan=2.

    Args:
        rows: list of dicts produced by build_comparison_rows().
        cols: list of (metric_key, display_label, fmt) tuples — column order.
        index_col: name of the column used as the row label (rowspan=2).

    Returns:
        IPython HTML object ready for display().
    """
    def _th(label: str, rowspan: int = 1) -> str:
        rs = f' rowspan="{rowspan}"' if rowspan > 1 else ""
        return f"<th{rs}>{label}</th>"

    def _td(val: str, rowspan: int = 1, style: str = "") -> str:
        rs = f' rowspan="{rowspan}"' if rowspan > 1 else ""
        st = f' style="{style}"' if style else ""
        return f"<td{rs}{st}>{val}</td>"

    hdr = (
        "<thead><tr>"
        + _th(index_col, rowspan=2)
        + "".join(_th(label) for _, label, _ in cols)
        + "</tr></thead>"
    )

    body = "<tbody>"
    for row in rows:
        # Value row
        body += (
            "<tr>"
            + _td(row.get(index_col, ""), rowspan=2)
            + "".join(_td(row.get(label, "—")) for _, label, _ in cols)
            + "</tr>"
        )
        # Delta row (dimmed)
        body += (
            "<tr style='color:#666;font-size:0.9em'>"
            + "".join(_td(row.get(f"Δ {label}", "")) for _, label, _ in cols)
            + "</tr>"
        )
    body += "</tbody>"

    return HTML(f'<table border="1" class="dataframe">{hdr}{body}</table>')


def build_comparison_rows(
    scenarios: list[tuple[str, dict, bool]],
    cols: list[tuple[str, str, str]],
    index_col: str,
    higher_better: dict[str, bool] | None = None,
) -> list[dict]:
    """Build rows for comparison_table_html.

    Args:
        scenarios: list of (label, metrics_dict, is_baseline) tuples.
        cols: list of (metric_key, display_label, fmt).
        index_col: label key in the output dict (e.g. "Feature Set").
        higher_better: mapping from metric_key to bool for delta sign convention.

    Returns:
        list of row dicts suitable for comparison_table_html().
    """
    hb = higher_better or {}
    baseline_m = next((m for _, m, is_base in scenarios if is_base), {})

    rows = []
    for label, m, is_base in scenarios:
        row: dict = {index_col: label}
        for key, col, fmt in cols:
            row[col] = fmt_val(m.get(key), fmt)
            if is_base:
                row[f"Δ {col}"] = "(baseline)"
            elif baseline_m:
                row[f"Δ {col}"] = fmt_delta(m.get(key), baseline_m.get(key), fmt, hb.get(key, True))
        rows.append(row)
    return rows
