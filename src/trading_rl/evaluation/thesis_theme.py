"""Shared plotnine theme for thesis figures.

All evaluation plots should use ``thesis_theme()`` so visual style is
controlled in one place.  Colours are from the Wong (2011) colorblind-safe
eight-colour palette.
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path

from plotnine import element_blank, element_line, element_text, theme, theme_classic

# Wong (2011) colorblind-safe palette — ordered by intended use
PALETTE: dict[str, str] = {
    "Deterministic": "#CC0000",       # red
    "Buy-and-Hold":  "#E69F00",       # orange
    "Random":        "#999999",       # grey
    "Max Profit (Unleveraged)": "#009E73",  # green
    "TWAP":          "#56B4E9",       # sky blue
    "VWAP":          "#D55E00",       # vermillion
    "Short-and-Hold": "#CC79A7",      # pink
}

# Line-type mapping for equity-curve plots.
# Agent runs use solid lines; benchmarks use dash patterns so the plot stays
# readable in greyscale and when printed.  Matplotlib supports only four
# named patterns (solid, dashed, dotted, dashdot), so some benchmarks share
# a pattern — they remain distinguishable by colour.
LINETYPE: dict[str, str] = {
    "Deterministic":            "solid",
    "Random":                   "solid",
    "Buy-and-Hold":             "dashed",
    "Short-and-Hold":           "dotted",
    "Max Profit (Unleveraged)": "dashdot",
    "TWAP":                     "dashdot",
    "VWAP":                     "dashed",
}

# Single figure width that fits inside A4 / US-Letter thesis margins
FIGURE_WIDTH = 6.0   # inches
FIGURE_HEIGHT = 3.5  # inches — 16:9-ish aspect, comfortable for line plots
PLOT_DPI = 225       # pixels per inch — higher DPI without changing physical size

# Latin Modern Roman matches the LaTeX Computer Modern body font exactly.
# The four OTF files are copied from TeX Live into ~/Library/Fonts so that
# matplotlib can discover them (see thesis_theme.py setup notes).
FONT_FAMILY = "Latin Modern Roman"


def _get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=2,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def add_debug_watermark(fig, *, base_size: int = 11) -> None:
    """Stamp build metadata in the bottom-right corner of a matplotlib figure.

    Includes git commit hash, build date/time (UTC), figure size in inches,
    and the thesis font family and base size.
    """
    width, height = fig.get_size_inches()
    commit = _get_git_commit()
    build_date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    text = (
        f"build:  {commit}\n"
        f"date:   {build_date}\n"
        f"size:   {width:.1f} × {height:.1f} in\n"
        f"font:   {FONT_FAMILY}  {base_size}pt"
    )
    fig.text(
        0.995, 0.005,
        text,
        ha="right", va="bottom",
        fontsize=6,
        color="#bbbbbb",
        family="monospace",
        transform=fig.transFigure,
        linespacing=1.4,
    )


def save_plot(
    plot_obj,
    path: str | Path,
    *,
    width: float,
    height: float,
    dpi: int = PLOT_DPI,
    debug: bool = False,
    base_size: int = 11,
) -> None:
    """Save a plotnine or matplotlib figure, optionally stamping debug metadata.

    When ``debug=True`` and the plot object is a plotnine ``ggplot``, the figure
    is drawn to a matplotlib ``Figure`` first so the watermark text can be added
    before the final ``savefig`` call.  For matplotlib figures the watermark is
    applied directly.  When ``debug=False`` the fast plotnine ``save()`` path is
    used without an intermediate draw step.
    """
    import matplotlib.pyplot as plt

    path = str(Path(path))

    if debug and hasattr(plot_obj, "draw"):
        # plotnine: override figure size via theme, draw, watermark, save
        from plotnine import theme as _p9theme
        fig = (plot_obj + _p9theme(figure_size=(width, height))).draw()
        add_debug_watermark(fig, base_size=base_size)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    elif debug and hasattr(plot_obj, "savefig"):
        # raw matplotlib figure
        add_debug_watermark(plot_obj, base_size=base_size)
        plot_obj.savefig(path, dpi=dpi, bbox_inches="tight")
    elif hasattr(plot_obj, "save"):
        plot_obj.save(path, width=width, height=height, dpi=dpi, verbose=False)
    elif hasattr(plot_obj, "savefig"):
        plot_obj.savefig(path, dpi=dpi)
    else:
        raise RuntimeError(f"Unsupported plot object type: {type(plot_obj)}")


def thesis_theme(
    base_size: int = 22,
    figure_size: tuple[float, float] = (FIGURE_WIDTH, FIGURE_HEIGHT),
) -> theme:
    """Return a thesis-ready plotnine theme based on theme_classic.

    Args:
        base_size: Base font size in points.  All other text elements scale
            relative to this.
        figure_size: (width, height) in inches.  Default fits a single-column
            thesis layout on A4/US-Letter.
    """
    return (
        theme_classic(base_size=base_size)
        + theme(
            figure_size=figure_size,
            # Font — matches LaTeX Computer Modern body text
            text=element_text(family=FONT_FAMILY),
            # Axes
            axis_title=element_text(size=base_size, family=FONT_FAMILY),
            axis_text=element_text(size=base_size - 1, family=FONT_FAMILY),
            axis_ticks=element_line(color="#444444", size=0.4),
            # Legend
            legend_position="bottom",
            legend_title=element_text(size=base_size - 1, face="bold", family=FONT_FAMILY),
            legend_text=element_text(size=base_size - 1, family=FONT_FAMILY),
            legend_background=element_blank(),
            legend_key=element_blank(),
            # Title and caption
            plot_title=element_text(size=base_size + 1, face="bold", family=FONT_FAMILY),
            plot_caption=element_text(size=round(base_size * 0.65), ha="left", color="#000000", linespacing=1.40, family=FONT_FAMILY),
            # Panel
            panel_border=element_blank(),
        )
    )
