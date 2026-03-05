"""Tests for extended benchmark baselines in statistical testing."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from trading_rl.evaluation.statistical_tests import (
    compute_buy_and_hold_returns,
    compute_short_and_hold_returns,
    compute_twap_returns,
    compute_vwap_returns,
    run_all_statistical_tests,
)


def _make_test_config() -> SimpleNamespace:
    return SimpleNamespace(
        enabled=True,
        tests=["t_test"],
        compare_to_buy_and_hold=True,
        compare_to_short_and_hold=True,
        compare_to_twap=True,
        compare_to_vwap=True,
        compare_to_random=False,
        n_bootstrap_samples=100,
        n_permutations=100,
        confidence_level=0.95,
        random_seed=42,
        n_random_trials=5,
    )


def test_short_and_hold_is_inverse_of_buy_and_hold() -> None:
    prices = pd.Series([100.0, 102.0, 101.0, 103.0])
    bh = compute_buy_and_hold_returns(prices, max_steps=4)
    sh = compute_short_and_hold_returns(prices, max_steps=4)
    assert np.allclose(sh, -bh)


def test_twap_returns_are_finite() -> None:
    prices = pd.Series(np.linspace(100.0, 110.0, 20))
    twap = compute_twap_returns(prices, max_steps=20)
    assert len(twap) == 20
    assert np.isfinite(twap).all()


def test_vwap_returns_use_volume_schedule() -> None:
    prices = pd.Series(np.linspace(100.0, 110.0, 10))
    volumes = pd.Series([1000, 900, 800, 700, 600, 500, 400, 300, 200, 100])
    vwap = compute_vwap_returns(prices, volumes, max_steps=10)
    twap = compute_twap_returns(prices, max_steps=10)
    assert len(vwap) == 10
    assert np.isfinite(vwap).all()
    assert not np.allclose(vwap, twap)


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

    results = run_all_statistical_tests(
        strategy_returns=strategy_returns,
        prices=prices,
        env=None,
        max_steps=30,
        config=config,
        market_data=market_data,
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
