"""Tests for extended benchmark baselines in statistical testing."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from trading_rl.evaluation.benchmarks import BenchmarkEngine
from trading_rl.evaluation.statistical_benchmarks import (
    build_benchmark_comparison_table,
    compute_buy_and_hold_returns,
    compute_short_and_hold_returns,
    compute_twap_returns,
    compute_vwap_returns,
)
from trading_rl.evaluation.statistical_tests import run_all_statistical_tests


def _make_test_config() -> SimpleNamespace:
    return SimpleNamespace(
        # Statistical testing params
        enabled=True,
        tests=["t_test"],
        n_bootstrap_samples=100,
        n_permutations=100,
        confidence_level=0.95,
        # Benchmark params (new field names — no compare_to_ prefix)
        buy_and_hold=True,
        short_and_hold=True,
        twap=True,
        vwap=True,
        random=False,
        n_random_trials=5,
        random_seed=42,
    )


def test_short_and_hold_is_inverse_of_buy_and_hold() -> None:
    prices = pd.Series([100.0, 102.0, 101.0, 103.0])
    bh = compute_buy_and_hold_returns(prices, max_steps=4)
    sh = compute_short_and_hold_returns(prices, max_steps=4)
    assert np.allclose(sh, -bh)


def test_buy_and_hold_uses_max_steps_price_transitions() -> None:
    prices = pd.Series([100.0, 101.0, 102.0, 103.0])

    returns = compute_buy_and_hold_returns(prices, max_steps=3)

    assert len(returns) == 3
    assert np.isclose(np.prod(1.0 + returns), 1.03)


def test_twap_returns_are_finite() -> None:
    prices = pd.Series(np.linspace(100.0, 110.0, 21))
    twap = compute_twap_returns(prices, max_steps=20)
    assert len(twap) == 20
    assert np.isfinite(twap).all()


def test_vwap_returns_use_volume_schedule() -> None:
    prices = pd.Series(np.linspace(100.0, 110.0, 11))
    volumes = pd.Series([1000, 900, 800, 700, 600, 500, 400, 300, 200, 100])
    vwap = compute_vwap_returns(prices, volumes, max_steps=10)
    twap = compute_twap_returns(prices, max_steps=10)
    assert len(vwap) == 10
    assert np.isfinite(vwap).all()
    assert not np.allclose(vwap, twap)


def test_benchmark_table_captures_initial_drawdown() -> None:
    table = build_benchmark_comparison_table(
        strategy_returns=np.array([-0.05, 0.10]),
        benchmark_returns={},
        periods_per_year=252,
    )

    assert np.isclose(table[0]["max_drawdown"], -0.05)


def test_run_all_statistical_tests_includes_extended_benchmark_table() -> None:
    prices = pd.Series(np.linspace(100.0, 105.0, 30))
    strategy_returns = np.full(30, 0.0002)
    market_data = pd.DataFrame(
        {
            "close": prices,
            "bid_sz_00": np.linspace(200.0, 400.0, 30),
            "ask_sz_00": np.linspace(250.0, 450.0, 30),
        }
    )
    config = _make_test_config()

    benchmarks, _ = BenchmarkEngine.build(market_data, config, price_column="close")
    results = run_all_statistical_tests(
        strategy_returns=strategy_returns,
        benchmarks=benchmarks,
        max_steps=30,
        config=config,
        periods_per_year=252,
    )

    baselines = {entry.get("baseline") for entry in results["baselines"]}
    assert "buy_and_hold" in baselines
    assert "short_and_hold" in baselines
    assert "twap" in baselines
    assert "vwap" in baselines

    table = results.get("benchmark_comparison_table", [])
    table_names = {row["strategy"] for row in table}
    assert "agent" in table_names
    assert "buy_and_hold" in table_names
    assert "short_and_hold" in table_names
    assert "twap" in table_names
    assert "vwap" in table_names
