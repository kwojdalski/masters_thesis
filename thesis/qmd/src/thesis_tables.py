"""Shared display and rendering helpers for thesis QMD chapters.

All formatting, HTML rendering, and display logic lives here so QMD cells
only handle data loading and a single display() call.

Sections
--------
Value formatters     fmt_val, fmt_delta, fmt_duration, fmt_scientific
Hyperparameter fmts  fmt_network_dims, fmt_reward_type, fmt_loss_fn, wrap_html
Display helpers      display_df, display_image_from_path
Comparison tables    comparison_table_html, build_comparison_rows
"""

from __future__ import annotations

import math
import textwrap
from pathlib import Path

import pandas as pd
from IPython.display import HTML, display


# ---------------------------------------------------------------------------
# Value formatters
# ---------------------------------------------------------------------------

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


def fmt_duration(s: float | None, na: str = "—") -> str:
    """Format a duration in seconds as µs / ms / s depending on magnitude."""
    if s is None or (isinstance(s, float) and (math.isnan(s) or math.isinf(s))):
        return na
    if s < 1e-3:
        return f"{s * 1e6:.1f} µs"
    if s < 1.0:
        return f"{s * 1e3:.2f} ms"
    return f"{s:.3f} s"


def fmt_scientific(v: float) -> str:
    """Format a float as a compact scientific-notation string (e.g. '1.5 x 10^-4')."""
    if v == 0.0:
        return "0.0"
    exp = math.floor(math.log10(abs(v)))
    mantissa = v / (10 ** exp)
    m = f"{mantissa:g}" if abs(mantissa - round(mantissa)) > 1e-9 else str(int(round(mantissa)))
    return f"{m} x 10^{exp}" if m != "1" else f"10^{exp}"


# ---------------------------------------------------------------------------
# Hyperparameter display formatters
# ---------------------------------------------------------------------------

def fmt_network_dims(dims: object) -> str:
    """Format a list of hidden layer widths, e.g. [128, 64] -> '[128, 64]'."""
    if isinstance(dims, list):
        return "[" + ", ".join(str(d) for d in dims) + "]"
    return str(dims)


def fmt_reward_type(rt: object) -> str:
    """Convert a reward_type key to a human-readable label."""
    mapping = {"differential_sharpe": "Differential Sharpe Ratio"}
    return mapping.get(str(rt), str(rt).replace("_", " ").title()) if rt else "—"


def fmt_loss_fn(loss: object) -> str:
    """Convert a loss_function key to a human-readable label."""
    mapping = {"smooth_l1": "Smooth L1 (Huber)"}
    return mapping.get(str(loss), str(loss)) if loss else "—"


def wrap_html(text: object, width: int = 38) -> str:
    """Wrap text at word boundaries and join with <br> for HTML table cells."""
    return "<br>".join(textwrap.wrap(str(text), width=width))


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def display_df(frame: pd.DataFrame) -> None:
    """Display a DataFrame as an HTML table without the row index."""
    display(HTML(frame.to_html(index=False)))


def simple_html_table(rows: list[dict], index_col: str | None = None) -> HTML:
    """Render a list of dicts as a plain HTML table."""
    df = pd.DataFrame(rows)
    return HTML(df.to_html(index=index_col is not None))


def display_image_from_path(
    path: str | Path,
    title: str | None = None,
    figsize: tuple[float, float] = (10, 6),
) -> None:
    """Open an image file and display it via matplotlib.

    Prints a warning instead of raising if the file is missing.
    """
    import matplotlib.pyplot as plt
    from PIL import Image

    p = Path(path)
    if not p.exists():
        label = title or str(path)
        print(f"{label}: artifact not found")
        return
    img = Image.open(p)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img)
    ax.axis("off")
    if title:
        ax.set_title(title)
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Comparison tables (value row + delta row per group)
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
        body += (
            "<tr>"
            + _td(row.get(index_col, ""), rowspan=2)
            + "".join(_td(row.get(label, "—")) for _, label, _ in cols)
            + "</tr>"
        )
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
