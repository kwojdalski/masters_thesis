"""Shared plotnine theme for thesis figures.

All evaluation plots should use ``thesis_theme()`` so visual style is
controlled in one place.  Colours are from the Wong (2011) colorblind-safe
eight-colour palette.
"""

from __future__ import annotations

from plotnine import element_blank, element_line, element_text, theme, theme_classic

# Wong (2011) colorblind-safe palette — ordered by intended use
PALETTE: dict[str, str] = {
    "Deterministic": "#0072B2",       # blue
    "Buy-and-Hold":  "#E69F00",       # orange
    "Random":        "#999999",       # grey
    "Max Profit (Unleveraged)": "#009E73",  # green
    "accent":        "#CC79A7",       # pink — spare accent colour
}

# Single figure width that fits inside A4 / US-Letter thesis margins
FIGURE_WIDTH = 6.0   # inches
FIGURE_HEIGHT = 3.5  # inches — 16:9-ish aspect, comfortable for line plots

# Latin Modern Roman matches the LaTeX Computer Modern body font exactly.
# The four OTF files are copied from TeX Live into ~/Library/Fonts so that
# matplotlib can discover them (see thesis_theme.py setup notes).
FONT_FAMILY = "Latin Modern Roman"


def thesis_theme(
    base_size: int = 11,
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
            plot_caption=element_text(size=base_size - 2, ha="left", color="#555555", family=FONT_FAMILY),
            # Panel
            panel_border=element_blank(),
        )
    )
