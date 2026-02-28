"""Test actual returns plot with benchmarks."""

import numpy as np
import pandas as pd
import pytest
import torch
from tensordict import TensorDict

from trading_rl.utils import create_actual_returns_plot


def test_actual_returns_plot_includes_benchmarks():
    """Test that benchmarks are added to actual returns plot when df_prices is provided."""
    # Create sample price data
    n_steps = 50
    prices = 100 + np.random.randn(n_steps).cumsum()
    df_prices = pd.DataFrame({
        "close": prices,
        "open": prices * 0.99,
        "high": prices * 1.01,
        "low": prices * 0.98,
    })

    # Create mock rollouts
    rollouts = []
    for _ in range(2):  # Deterministic and Random
        rollout = TensorDict({
            "action": torch.randn(n_steps, 1),
            "next": TensorDict({
                "reward": torch.randn(n_steps),
            }, batch_size=[n_steps]),
        }, batch_size=[n_steps])
        rollouts.append(rollout)

    # Create plot with benchmarks
    plot = create_actual_returns_plot(
        rollouts=rollouts,
        n_obs=n_steps,
        df_prices=df_prices,
        env=None,
        actual_returns_list=None,
    )

    # Check that plot data includes benchmarks
    plot_data = plot.data
    unique_runs = plot_data["Run"].unique()

    # Should have 4 series: Deterministic, Random, Buy-and-Hold, Max Profit
    assert len(unique_runs) == 4
    assert "Deterministic" in unique_runs
    assert "Random" in unique_runs
    assert "Buy-and-Hold" in unique_runs
    assert "Max Profit (Unleveraged)" in unique_runs

    # Check that each series has the correct number of points
    for run_name in unique_runs:
        run_data = plot_data[plot_data["Run"] == run_name]
        assert len(run_data) == n_steps, f"{run_name} should have {n_steps} points"
        assert "Portfolio_Value" in run_data.columns


def test_actual_returns_plot_without_benchmarks():
    """Test that plot works without df_prices (no benchmarks)."""
    n_steps = 50

    # Create mock rollouts
    rollouts = []
    for _ in range(2):
        rollout = TensorDict({
            "action": torch.randn(n_steps, 1),
            "next": TensorDict({
                "reward": torch.randn(n_steps),
            }, batch_size=[n_steps]),
        }, batch_size=[n_steps])
        rollouts.append(rollout)

    # Create plot without benchmarks (df_prices=None)
    plot = create_actual_returns_plot(
        rollouts=rollouts,
        n_obs=n_steps,
        df_prices=None,  # No price data
        env=None,
        actual_returns_list=None,
    )

    # Check that plot data only includes agent runs
    plot_data = plot.data
    unique_runs = plot_data["Run"].unique()

    # Should only have 2 series: Deterministic, Random
    assert len(unique_runs) == 2
    assert "Deterministic" in unique_runs
    assert "Random" in unique_runs


def test_benchmark_calculations():
    """Test that benchmarks are calculated correctly."""
    # Create simple upward trend
    n_steps = 10
    prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0])
    df_prices = pd.DataFrame({"close": prices})

    # Create mock rollouts
    rollouts = []
    for _ in range(2):
        rollout = TensorDict({
            "action": torch.randn(n_steps, 1),
            "next": TensorDict({
                "reward": torch.zeros(n_steps),  # Zero rewards for this test
            }, batch_size=[n_steps]),
        }, batch_size=[n_steps])
        rollouts.append(rollout)

    # Create plot
    plot = create_actual_returns_plot(
        rollouts=rollouts,
        n_obs=n_steps,
        df_prices=df_prices,
        env=None,
        actual_returns_list=None,
    )

    # Extract benchmark data
    plot_data = plot.data
    buy_hold_data = plot_data[plot_data["Run"] == "Buy-and-Hold"]
    max_profit_data = plot_data[plot_data["Run"] == "Max Profit (Unleveraged)"]

    # Buy-and-Hold should reflect portfolio value from initial capital
    # From 100 to 109 => +9%, so 10,000 -> 10,900
    final_buy_hold = buy_hold_data.iloc[-1]["Portfolio_Value"]
    expected_buy_hold = 10000.0 * (109.0 / 100.0)
    assert abs(final_buy_hold - expected_buy_hold) < 0.001

    # Max Profit should be positive and >= Buy-and-Hold
    final_max_profit = max_profit_data.iloc[-1]["Portfolio_Value"]
    assert final_max_profit > 10000.0
    assert final_max_profit >= final_buy_hold


def test_actual_returns_plot_uses_custom_initial_capital():
    """Test that a custom initial capital scales the plotted values."""
    n_steps = 5
    initial_capital = 25000.0

    rollouts = []
    for _ in range(2):
        rollout = TensorDict({
            "action": torch.randn(n_steps, 1),
            "next": TensorDict({
                "reward": torch.zeros(n_steps),  # 0 log-return => flat equity
            }, batch_size=[n_steps]),
        }, batch_size=[n_steps])
        rollouts.append(rollout)

    plot = create_actual_returns_plot(
        rollouts=rollouts,
        n_obs=n_steps,
        df_prices=None,
        env=None,
        actual_returns_list=None,
        initial_capital=initial_capital,
    )

    plot_data = plot.data
    deterministic_values = plot_data[plot_data["Run"] == "Deterministic"][
        "Portfolio_Value"
    ].to_numpy()
    assert np.allclose(deterministic_values, initial_capital)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
