"""Win/loss direction of the H1 and H2 report indicators.

max_drawdown is the only reported metric stored on a signed-negative scale
(``np.min(equity / running_max - 1)``), so a shallower drawdown is the *larger*
value. Both report scripts previously declared it lower-is-better and compared
the raw signed values, inverting the verdict in both directions (#498).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_script_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def h1():
    return _load_script_module(
        "scripts/h1_performance_report.py", "h1_performance_report"
    )


@pytest.fixture(scope="module")
def h2():
    return _load_script_module(
        "scripts/h2_feature_sensitivity_report.py", "h2_feature_sensitivity_report"
    )


def _max_dd_cell(h1_module, agent_dd: float, bench_dd: float) -> str:
    """Render the benchmark row through build_agent_table and return its Max DD cell.

    Goes through the real table builder rather than re-deriving the direction
    map, so the assertion covers the mapping build_agent_table actually uses.
    """
    agent_data = {
        "metrics": {"max_drawdown": agent_dd, "sharpe_ratio": 1.0},
        "benchmarks": {
            "buy_and_hold": {
                "benchmark_metrics": {"max_drawdown": bench_dd, "sharpe_ratio": 0.5},
                "relative_metrics": {},
            }
        },
    }
    table = h1_module.build_agent_table(
        "Agent", agent_data, [{"name": "buy_and_hold", "label": "Buy & Hold"}]
    )
    col = next(c for c in table.columns if c.header == "Max DD")
    # cell 0 is the agent row's placeholder-free own value; cell 1 is the benchmark
    return str(col._cells[1])


def test_h1_deeper_drawdown_loses(h1) -> None:
    """Agent at -20% must not be shown as beating a benchmark at -5%."""
    assert "red" in _max_dd_cell(h1, -0.20, -0.05)


def test_h1_shallower_drawdown_wins(h1) -> None:
    """Agent at -2% must be shown as beating a benchmark at -10%."""
    assert "green" in _max_dd_cell(h1, -0.02, -0.10)


def test_h1_higher_is_better_metrics_unaffected(h1) -> None:
    assert "green" in h1.beats(1.4, 0.9, True)
    assert "red" in h1.beats(0.9, 1.4, True)


def test_h2_shallower_drawdown_delta_is_green(h2) -> None:
    """A variant at -2% against a -10% baseline is an improvement."""
    assert "green" in h2.fmt_delta("max_drawdown", -0.02, -0.10, ".2%")


def test_h2_deeper_drawdown_delta_is_red(h2) -> None:
    assert "red" in h2.fmt_delta("max_drawdown", -0.20, -0.05, ".2%")


def test_h2_sharpe_delta_direction_unaffected(h2) -> None:
    assert "green" in h2.fmt_delta("sharpe_ratio", 1.4, 0.9, ".3f")
    assert "red" in h2.fmt_delta("sharpe_ratio", 0.9, 1.4, ".3f")
