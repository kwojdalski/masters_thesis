"""Presentation views of the committed thesis results; no metric recomputation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from plotnine import (
    aes,
    coord_flip,
    element_blank,
    element_line,
    element_text,
    geom_col,
    geom_hline,
    geom_text,
    ggplot,
    labs,
    scale_fill_identity,
    scale_y_continuous,
    theme,
    theme_minimal,
)

RESULTS = Path(__file__).resolve().parents[1] / "qmd" / "results"
H1 = "pooled_td3_hft_lob_state_space_pooled_streaming_selected_dsr"
BURGUNDY = "#AA273D"


def read_result(relative_path: str):
    """Require a committed snapshot rather than silently using a live run."""
    return json.loads((RESULTS / relative_path).read_text())


def load_h1() -> dict[str, dict]:
    reports = {}
    for algorithm in ("TD3", "DDPG", "PPO", "Random"):
        name = H1.replace("pooled_td3_", f"pooled_{algorithm.lower()}_")
        report = read_result(f"{name}/latest_finished/evaluation_report.json")
        if report["__source_split__"] != "test":
            raise ValueError(f"{algorithm} is not a test-split report")
        reports[algorithm] = report
    # A visually plausible chart must not silently mix evaluation windows.
    windows = [
        {symbol: result["n_steps"] for symbol, result in r["__per_symbol__"].items()}
        for r in reports.values()
    ]
    if any(window != windows[0] for window in windows[1:]):
        raise ValueError("The H1 algorithms have different test windows")
    return reports


def bar_chart(
    labels: list[str],
    values: list[float],
    *,
    colours: list[str],
    axis: str,
    limit: float,
    breaks: list[float],
    digits: int,
    suffix: str = "",
):
    """A directly labelled chart with a visible unit-one reference line."""
    data = pd.DataFrame({"label": labels, "value": values, "colour": colours})
    data["label"] = pd.Categorical(data["label"], categories=labels[::-1], ordered=True)
    data["printed"] = [f"{value:.{digits}f}{suffix}" for value in values]
    return (
        ggplot(data, aes("label", "value", fill="colour"))
        + geom_col(width=0.53)
        + geom_hline(yintercept=1, linetype="dashed", color="#686868", size=0.6)
        + geom_text(
            aes(label="printed"),
            ha="left",
            nudge_y=limit * 0.025,
            size=15,
            color="#242424",
        )
        + scale_fill_identity()
        + scale_y_continuous(limits=(0, limit), breaks=breaks, expand=(0, 0))
        + coord_flip()
        + labs(x=None, y=axis)
        + theme_minimal(base_size=15, base_family="DejaVu Sans")
        + theme(
            axis_title_y=element_blank(),
            panel_grid_major_y=element_blank(),
            panel_grid_minor=element_blank(),
            panel_grid_major_x=element_line(color="#ECE8E8", size=0.4),
            axis_text_y=element_text(size=16, color="#242424", weight="bold"),
            axis_text_x=element_text(size=12, color="#686868"),
            axis_title_x=element_text(size=12, color="#686868", margin={"t": 12}),
            plot_margin=0.025,
        )
    )
