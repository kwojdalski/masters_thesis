"""Shared display and rendering helpers for thesis QMD chapters.

All formatting, HTML rendering, and display logic lives here so QMD cells
only handle data loading and a single display() call.

Sections
--------
Value formatters     fmt_val, fmt_delta, fmt_duration, fmt_scientific
Hyperparameter fmts  fmt_network_dims, fmt_reward_type, fmt_loss_fn, wrap_html
Display helpers      display_df, display_image_from_path, table_note
Comparison tables    comparison_table_html, build_comparison_rows
"""

from __future__ import annotations

import math
import textwrap
from pathlib import Path

import pandas as pd
from IPython.display import HTML, Markdown, display

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


def fmt_delta(
    val: object, baseline: object, fmt: str, higher_better: bool = True
) -> str:
    """Format the signed delta between val and baseline using fmt."""
    if not isinstance(val, int | float) or not isinstance(baseline, int | float):
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
    """Format a float as compact HTML scientific notation (e.g. '1.5 × 10<sup>-4</sup>').

    This string is embedded (escape=False) into a pandas-generated HTML table
    that pandoc converts to LaTeX for the PDF build, so the exponent uses an
    HTML <sup> tag rather than a bare '^' (unrendered by HTML/LaTeX alike) or
    Unicode superscript digits (e.g. '⁻⁶'), which are missing from the
    thesis's default LaTeX font and render as blank boxes. A raw LaTeX
    '$...$' math string also does not work here: this cell's content is
    parsed as HTML, not markdown, so '$...$' passes through as literal text.
    The '×' multiplication sign is a plain character and renders fine
    directly in both HTML and the PDF's default font.
    """
    if v == 0.0:
        return "0.0"
    exp = math.floor(math.log10(abs(v)))
    mantissa = v / (10**exp)
    m = (
        f"{mantissa:g}"
        if abs(mantissa - round(mantissa)) > 1e-9
        else str(round(mantissa))
    )
    exp_str = f"<sup>{exp}</sup>"
    return f"{m} × 10{exp_str}" if m != "1" else f"10{exp_str}"


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


def table_note(
    *,
    source: str | None = None,
    legend: str | None = None,
    note: str | None = None,
) -> None:
    """Display a standardised Source / Legend / Note block below a table.

    Emits a Markdown ``:::{.table-note}`` fenced div that the ``tablenote.lua``
    Pandoc filter transforms into appropriately styled output for both HTML
    and PDF/LaTeX renders.  All three arguments are optional; pass at least one.

    The text supports plain Markdown (bold, italic, inline math) but NOT
    Pandoc ``[@citation]`` keys — use author-year text for in-note citations.

    Usage::

        table_note(
            source="Author's own synthesis based on Fujimoto et al. (2018).",
            legend="yes = included in model; — = not used.",
            note="First 500 events skipped for rolling-window warm-up.",
        )
    """
    parts: list[str] = []
    if source:
        parts.append(f"**Source:** {source}")
    if legend:
        parts.append(f"**Legend:** {legend}")
    if note:
        parts.append(f"**Note:** {note}")
    if not parts:
        return
    # Two hard-space-separated lines inside the fenced div so the Lua
    # filter sees them as a single block (one Para per label).
    inner = "  \n".join(parts)
    display(Markdown(f"\n::: {{.table-note}}\n{inner}\n:::\n"))


def missing_data_notice(message: str) -> None:
    """Display a missing-data fallback notice as flowing Markdown text.

    Used in place of a bare ``print(...)`` so the message renders as normal
    word-wrapped prose instead of a verbatim code-cell output block. Verbatim
    blocks do not wrap long lines, and a long "<file> not found — run: <cmd>"
    message can overflow past the page margin in PDF output.
    """
    display(Markdown(f"*{message}*"))


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
    In debug mode (THESIS_DEBUG_ASSETS=1) also renders provenance metadata.
    """
    import matplotlib.pyplot as plt
    from PIL import Image

    p = Path(path) if path else None
    if not p or not p.is_file():
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

    try:
        from thesis_asset_debug import show_asset_debug

        show_asset_debug(p)
    except Exception as exc:
        print(f"show_asset_debug skipped: {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Feature distributional statistics table (landscape PDF / HTML)
# ---------------------------------------------------------------------------


def feature_stats_table(raw_df: pd.DataFrame, *, obs_clip: float = 5.0) -> None:
    """Emit feature distributional statistics as a landscape table.

    Requires ``#| output: asis`` on the calling cell.

    HTML: standard table wrapped in a ``#tbl-feature-stats`` cross-ref div.
    PDF: landscape LaTeX via ``\\begin{landscape}`` (pdflscape), placed on its
    own page with ``\\clearpage`` guards.

    Min/Max values whose magnitude exceeds *obs_clip* are flagged with ``*``
    (HTML) or ``$^{*}$`` (LaTeX) to indicate that the RL environment clips
    them before passing observations to the policy network.
    """
    import math

    df = raw_df.copy()

    # Clean feature names
    df["feature"] = (
        df["feature"]
        .str.replace("feature_hft_", "", regex=False)
        .str.replace("feature_", "", regex=False)
    )

    # Determine clipped rows BEFORE string formatting
    clip_min = df["min"].apply(
        lambda x: math.isfinite(float(x)) and float(x) < -obs_clip
    )
    clip_max = df["max"].apply(
        lambda x: math.isfinite(float(x)) and float(x) > obs_clip
    )
    any_clipped = bool(clip_min.any() or clip_max.any())

    # Format numeric columns
    for col in ("mean", "std", "min", "max"):
        df[col] = df[col].apply(lambda x: f"{float(x):.4f}")
    for col in ("skew", "kurt"):
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{float(x):.3f}")
    if "q2" in df.columns:
        df["q2"] = df["q2"].apply(lambda x: f"{float(x):.4f}")

    keep = ["feature", "mean", "std", "skew", "kurt", "q2", "min", "max"]
    df = df[[c for c in keep if c in df.columns]]
    df = df.rename(
        columns={
            "feature": "Feature",
            "mean": "Mean",
            "std": "Std",
            "skew": "Skew",
            "kurt": "Kurt",
            "q2": "Median",
            "min": "Min",
            "max": "Max",
        }
    )

    caption = "Distributional statistics of engineered microstructure features."
    note_base = (
        "Computed on the training split; the first 500 events are skipped to allow "
        "rolling-window features to reach steady state. "
        "Skew and Kurt are the Fisher skewness and excess kurtosis. "
        "Median is the 50th percentile."
    )
    clip_suffix_html = (
        (
            f" * the RL environment clips observations to \\u00b1{obs_clip:.0f};"
            " starred values exceed this bound."
        )
        if any_clipped
        else ""
    )
    note_html = note_base + clip_suffix_html.replace("\\u00b1", "±")

    # ── HTML version ──────────────────────────────────────────────────
    html_df = df.copy()
    html_df["Min"] = [
        v + "*" if c else v for v, c in zip(html_df["Min"], clip_min, strict=False)
    ]
    html_df["Max"] = [
        v + "*" if c else v for v, c in zip(html_df["Max"], clip_max, strict=False)
    ]

    html_table = html_df.to_html(index=False, classes="dataframe")
    note_p = (
        f'<p style="font-size:0.85em"><em><strong>Note:</strong> {note_html}</em></p>'
    )

    html_block = (
        f"::: {{#tbl-feature-stats}}\n\n{html_table}\n\n{note_p}\n\n{caption}\n\n:::"
    )

    # ── LaTeX version ─────────────────────────────────────────────────
    def _esc(s: str) -> str:
        return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")

    cols = list(df.columns)
    col_spec = "l " + " ".join(["r"] * (len(cols) - 1))
    header_cells = " & ".join(f"\\textbf{{{_esc(c)}}}" for c in cols)

    rows_latex: list[str] = []
    for i, (_, row) in enumerate(df.iterrows()):
        cells = []
        for col, val in zip(cols, row, strict=False):
            s = _esc(str(val))
            if col == "Min" and clip_min.iloc[i]:
                s += r"$^{*}$"
            elif col == "Max" and clip_max.iloc[i]:
                s += r"$^{*}$"
            cells.append(s)
        rows_latex.append(" & ".join(cells) + r" \\")

    note_latex = _esc(note_base)
    if any_clipped:
        note_latex += (
            f" $^{{*}}$~the RL environment clips observations to"
            f" $\\pm{obs_clip:.0f}$; starred values exceed this bound."
        )

    latex_block = "\n".join(
        [
            r"\clearpage",
            r"\begin{landscape}",
            r"\begin{table}[htbp]",
            f"\\caption{{{_esc(caption)}}}",
            r"\label{tbl-feature-stats}",
            r"\centering",
            r"\footnotesize",
            f"\\begin{{tabular}}{{{col_spec}}}",
            r"\toprule",
            header_cells + r" \\",
            r"\midrule",
            *rows_latex,
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{4pt}",
            r"\begin{minipage}{\linewidth}",
            f"\\footnotesize\\textit{{{note_latex}}}",
            r"\end{minipage}",
            r"\end{table}",
            r"\end{landscape}",
            r"\clearpage",
        ]
    )

    # ── Emit conditional blocks ────────────────────────────────────────
    content = "\n".join(
        [
            '::: {.content-visible when-format="html"}',
            "",
            html_block,
            "",
            ":::",
            "",
            '::: {.content-visible when-format="pdf"}',
            "",
            "```{=latex}",
            latex_block,
            "```",
            "",
            ":::",
        ]
    )
    display(Markdown(content))


# ---------------------------------------------------------------------------
# Feature–return correlation table
# ---------------------------------------------------------------------------


def feature_correlation_table(raw_df: pd.DataFrame) -> None:
    """Emit feature Pearson and Spearman correlations with next-step log return.

    Requires ``#| output: asis`` on the calling cell.

    HTML: standard table wrapped in a ``#tbl-feature-correlations`` cross-ref div.
    PDF: plain booktabs LaTeX table.
    """
    df = raw_df.copy()
    df["feature"] = (
        df["feature"]
        .str.replace("feature_hft_", "", regex=False)
        .str.replace("feature_", "", regex=False)
    )
    df["pearson"] = df["pearson"].apply(lambda x: f"{float(x):+.4f}")
    df["spearman"] = df["spearman"].apply(lambda x: f"{float(x):+.4f}")
    df = df.rename(
        columns={"feature": "Feature", "pearson": "Pearson", "spearman": "Spearman"}
    )

    caption = (
        "Pearson and Spearman rank correlations between each engineered feature "
        "and the next-step log return, computed on the training split."
    )
    note = (
        "Correlations computed against one-step-ahead log mid-price returns on the training split "
        "after skipping the first 500 events. All |r| < 0.005 across both measures, "
        "indicating negligible linear and monotone dependence between individual features "
        "and the prediction target. This motivates a non-linear function approximator "
        "(the neural network policy) rather than a linear model."
    )

    # ── HTML version ──────────────────────────────────────────────────
    html_table = df.to_html(index=False, classes="dataframe")
    note_p = f'<p style="font-size:0.85em"><em><strong>Note:</strong> {note}</em></p>'
    html_block = (
        f"::: {{#tbl-feature-correlations}}\n\n"
        f"{html_table}\n\n"
        f"{note_p}\n\n"
        f"{caption}\n\n"
        f":::"
    )

    # ── LaTeX version ─────────────────────────────────────────────────
    def _esc(s: str) -> str:
        return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")

    rows_latex = [
        " & ".join(_esc(str(v)) for v in row) + r" \\" for _, row in df.iterrows()
    ]
    latex_block = "\n".join(
        [
            r"\begin{table}[htbp]",
            f"\\caption{{{_esc(caption)}}}",
            r"\label{tbl-feature-correlations}",
            r"\centering",
            r"\small",
            r"\begin{tabular}{l r r}",
            r"\toprule",
            r"\textbf{Feature} & \textbf{Pearson} & \textbf{Spearman} \\",
            r"\midrule",
            *rows_latex,
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{4pt}",
            r"\begin{minipage}{0.95\linewidth}",
            f"\\footnotesize\\textit{{{_esc(note)}}}",
            r"\end{minipage}",
            r"\end{table}",
        ]
    )

    content = "\n".join(
        [
            '::: {.content-visible when-format="html"}',
            "",
            html_block,
            "",
            ":::",
            "",
            '::: {.content-visible when-format="pdf"}',
            "",
            "```{=latex}",
            latex_block,
            "```",
            "",
            ":::",
        ]
    )
    display(Markdown(content))


# ---------------------------------------------------------------------------
# LOB events sample table
# ---------------------------------------------------------------------------


def lob_events_table(df: pd.DataFrame) -> None:
    """Select a 12-event window with maximum price activity and display as HTML.

    Picks the 12 consecutive rows with the most bid/ask price changes so the
    illustration captures genuine LOB dynamics rather than a quiet period.
    Events become rows; raw order-book columns on the left, z-score-normalized
    features on the right, separated by a vertical rule.

    Calls display() and table_note() internally — no return value.
    """
    best_start, best_score = 0, 0
    for i in range(0, len(df) - 12, 5):
        w = df.iloc[i : i + 12]
        score = (w["ask_px_00"].diff().abs() > 0).sum() + (
            w["bid_px_00"].diff().abs() > 0
        ).sum()
        if score > best_score:
            best_score, best_start = score, i

    win = df.iloc[best_start : best_start + 12].copy()

    RAW_COLS = ["Time (UTC)", "Best Bid ($)", "Best Ask ($)", "Bid Size", "Ask Size"]
    FEAT_COLS = ["Book Pressure", "Order Imbalance", "Microprice Dev.", "OFI"]
    FIRST_FEAT_COL = FEAT_COLS[0]

    tbl = pd.DataFrame(
        {
            "Time (UTC)": win["ts_event"].dt.strftime("%H:%M:%S.%f").str[:-3].values,
            "Best Bid ($)": win["bid_px_00"].round(2).values,
            "Best Ask ($)": win["ask_px_00"].round(2).values,
            "Bid Size": win["bid_sz_00"].astype(int).values,
            "Ask Size": win["ask_sz_00"].astype(int).values,
            "Book Pressure": win["feature_hft_book_pressure_l0"].round(3).values,
            "Order Imbalance": win["feature_hft_order_book_imbalance_3l"]
            .round(3)
            .values,
            "Microprice Dev.": win["feature_hft_microprice_divergence"].round(3).values,
            "OFI": win["feature_hft_ofi"].round(3).values,
        },
        index=[f"E{i}" for i in range(1, 13)],
    )

    header_cells = ""
    for col in tbl.columns:
        bold = "font-weight:bold;" if col in RAW_COLS else ""
        border = "border-left:2px solid #555;" if col == FIRST_FEAT_COL else ""
        header_cells += f'<th style="text-align:right;{bold}{border}">{col}</th>'
    header = f"<thead><tr>{header_cells}</tr></thead>"

    html_rows = []
    for _event, series in tbl.iterrows():
        cells = ""
        for col, val in series.items():
            border = "border-left:2px solid #555;" if col == FIRST_FEAT_COL else ""
            cells += f'<td style="text-align:right;{border}">{val}</td>'
        html_rows.append(f"<tr>{cells}</tr>")

    body = "<tbody>" + "".join(html_rows) + "</tbody>"
    table_html = (
        '<div style="overflow-x:auto">'
        '<table style="border-collapse:collapse;font-size:0.82em;width:100%">'
        f"{header}{body}"
        "</table></div>"
    )

    display(HTML(table_html))
    table_note(
        source="DataBento Nasdaq MBP-10 feed, AAPL, 2 March 2026.",
        note=(
            "Twelve consecutive order-book events selected from the test split "
            "to include multiple bid/ask price changes. "
            "Book Pressure, Order Imbalance, Microprice Dev., and OFI are "
            "z-score normalized using causal running statistics. "
            "A vertical rule separates the raw order-book state (left columns) "
            "from the normalized features (right columns)."
        ),
    )


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
                row[f"Δ {col}"] = fmt_delta(
                    m.get(key), baseline_m.get(key), fmt, hb.get(key, True)
                )
        rows.append(row)
    return rows
