"""Tests for extended benchmark baselines in statistical testing."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from trading_rl.config import StatisticalTestingConfig
from trading_rl.constants import BenchmarkName
from trading_rl.evaluation import statistical_tests as statistical_tests_module
from trading_rl.evaluation.benchmarks import BenchmarkEngine, BenchmarkSpec
from trading_rl.evaluation.metrics import build_metric_report
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


def test_vwap_returns_warns_and_degrades_to_twap_on_zero_volume() -> None:
    prices = pd.Series(np.linspace(100.0, 110.0, 11))
    volumes = pd.Series(np.zeros(10))

    from loguru import logger as loguru_logger

    messages: list[str] = []
    sink_id = loguru_logger.add(
        lambda msg: messages.append(msg.record["message"]), level="WARNING"
    )
    try:
        vwap = compute_vwap_returns(prices, volumes, max_steps=10)
    finally:
        loguru_logger.remove(sink_id)
    twap = compute_twap_returns(prices, max_steps=10)

    assert np.allclose(vwap, twap)
    assert any("zero" in m.lower() for m in messages)


def test_benchmark_metric_report_fills_direction_percentages() -> None:
    prices = pd.Series(np.linspace(100.0, 105.0, 8))
    volumes = pd.Series(np.arange(1.0, 9.0))
    config = SimpleNamespace(
        buy_and_hold=True,
        short_and_hold=False,
        twap=True,
        vwap=True,
    )

    benchmarks, _ = BenchmarkEngine.build(
        pd.DataFrame({"close": prices, "volume": volumes}),
        config,
        price_column="close",
    )
    metrics_by_name = {
        spec.name: build_metric_report(
            strategy_simple_returns=spec.compute_returns(7),
            benchmark_simple_returns=None,
            actions=None,
            periods_per_year=252,
        )
        for spec in benchmarks
    }

    assert metrics_by_name["buy_and_hold"]["pct_long"] == 1.0
    assert metrics_by_name["buy_and_hold"]["pct_short"] == 0.0
    assert metrics_by_name["twap"]["pct_long"] == 1.0
    assert metrics_by_name["twap"]["pct_short"] == 0.0
    assert metrics_by_name["vwap"]["pct_long"] == 1.0
    assert metrics_by_name["vwap"]["pct_short"] == 0.0


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
    assert "twap" in baselines
    assert "vwap" in baselines

    table = results.get("benchmark_comparison_table", [])
    table_names = {row["strategy"] for row in table}
    assert "agent" in table_names
    assert "buy_and_hold" in table_names
    assert "twap" in table_names
    assert "vwap" in table_names


def test_statistical_tests_run_with_real_config_without_random_seed() -> None:
    prices = pd.Series(np.linspace(100.0, 105.0, 30))
    strategy_returns = np.full(29, 0.0002)
    market_data = pd.DataFrame({"close": prices})
    config = StatisticalTestingConfig(
        enabled=True,
        tests=["t_test"],
    )
    benchmarks, _ = BenchmarkEngine.build(
        market_data,
        SimpleNamespace(
            buy_and_hold=True, short_and_hold=False, twap=False, vwap=False
        ),
        price_column="close",
    )

    results = run_all_statistical_tests(
        strategy_returns=strategy_returns,
        benchmarks=benchmarks,
        max_steps=29,
        config=config,
        periods_per_year=252,
    )

    baseline = results["baselines"][0]
    assert "error" not in baseline["t_test"]
    assert "t_statistic" in baseline["t_test"]


def test_run_all_statistical_tests_returns_disabled_payload_without_status() -> None:
    status_messages: list[str] = []

    results = run_all_statistical_tests(
        strategy_returns=np.array([0.01, -0.02]),
        benchmarks=[],
        max_steps=2,
        config=SimpleNamespace(enabled=False),
        status_fn=status_messages.append,
    )

    assert results == {"enabled": False}
    assert status_messages == []


def test_run_all_statistical_tests_records_benchmark_compute_errors() -> None:
    def failing_returns(_max_steps: int) -> np.ndarray:
        raise RuntimeError("benchmark failed")

    results = run_all_statistical_tests(
        strategy_returns=np.array([0.01, 0.02]),
        benchmarks=[BenchmarkSpec("broken", failing_returns)],
        max_steps=2,
        config=_make_test_config(),
    )

    assert results["baselines"] == [{"baseline": "broken", "error": "benchmark failed"}]
    assert [row["strategy"] for row in results["benchmark_comparison_table"]] == [
        "agent"
    ]


def test_run_all_statistical_tests_truncates_uneven_random_baseline_trials(
    monkeypatch,
) -> None:
    captured: dict[str, np.ndarray] = {}

    def fake_run_statistical_tests(
        strategy_returns, baseline_returns, baseline_name, config
    ):
        captured["strategy"] = strategy_returns
        captured["baseline"] = baseline_returns
        return {"baseline": baseline_name}

    monkeypatch.setattr(
        statistical_tests_module,
        "run_statistical_tests",
        fake_run_statistical_tests,
    )
    trials = [
        np.array([0.01, 0.03, 0.99]),
        np.array([0.05, 0.07]),
    ]

    results = run_all_statistical_tests(
        strategy_returns=np.array([0.02, 0.04, 0.06]),
        benchmarks=[],
        max_steps=3,
        config=_make_test_config(),
        random_baseline_trials=trials,
    )

    assert results["baselines"][0]["baseline"] == BenchmarkName.RANDOM_ACTIONS
    assert captured["baseline"] == pytest.approx(np.array([0.03, 0.05]))
    assert captured["strategy"] == pytest.approx(np.array([0.02, 0.04]))
    table = {row["strategy"]: row for row in results["benchmark_comparison_table"]}
    assert set(table) == {"agent", BenchmarkName.RANDOM_ACTIONS}
