"""Regression tests for the equity progression plot's point budget (#513).

The plot draws one line per training checkpoint and the periodic-eval hook
re-renders the whole history on every firing, so an unbounded point count
makes the total rendering work quadratic over a run. These tests pin the
figure-wide bound that keeps each render flat.
"""

from __future__ import annotations

import numpy as np
import pytest

from trading_rl.evaluation.plots import create_equity_progression_plot
from trading_rl.evaluation.returns import ReturnKind, ReturnSeries

# Matches training.temp_eval.max_steps on the pooled thesis scenarios.
_SERIES_LEN = 50_000


def _history(n_checkpoints: int) -> list[tuple[int, ReturnSeries]]:
    rng = np.random.default_rng(0)
    return [
        (
            k * 100_000,
            ReturnSeries(rng.normal(0, 1e-4, _SERIES_LEN), ReturnKind.SIMPLE),
        )
        for k in range(1, n_checkpoints + 1)
    ]


@pytest.mark.parametrize("n_checkpoints", [2, 10, 30])
def test_plotted_points_stay_within_budget_as_checkpoints_accumulate(
    n_checkpoints: int,
) -> None:
    """Total plotted rows must not grow with checkpoint count."""
    plot = create_equity_progression_plot(_history(n_checkpoints))

    assert plot is not None
    # Without the budget this is n_checkpoints * _SERIES_LEN (1.5M at 30).
    assert len(plot.data) <= 60_000
    assert plot.data["Training_Step"].nunique() == n_checkpoints


def test_explicit_max_plot_points_further_caps_each_series() -> None:
    """An explicit per-series cap tightens, never loosens, the figure budget."""
    plot = create_equity_progression_plot(_history(30), max_plot_points=100)

    assert plot is not None
    # +1 per series: to_equity() prepends the pre-first-step baseline, so the
    # exact count depends on ceil-vs-floor striding. The figure-wide budget
    # above is what guards #513; this only pins that the explicit cap applies.
    assert len(plot.data) <= 30 * (100 + 1)


def test_returns_none_below_two_checkpoints() -> None:
    assert create_equity_progression_plot(_history(1)) is None
