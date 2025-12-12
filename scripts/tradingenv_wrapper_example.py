#!/usr/bin/env python3
"""Test script for TradingEnv wrapper integration.

This script tests the TradingEnvXYFactory to ensure it works properly with TorchRL.
Run this to verify the tradingenv library integration is functioning correctly.
"""

# %%
import numpy as np
import pandas as pd

from src.trading_rl.envs import TradingEnvXYFactory


# %%
def create_sample_data(n_days=200, n_assets=3):
    """Create realistic sample price and feature data for testing.

    Args:
        n_days: Number of days of data to generate
        n_assets: Number of assets to simulate

    Returns:
        DataFrame with price and feature columns
    """
    print(f"Generating {n_days} days of data for {n_assets} assets...")

    dates = pd.date_range(start="2020-01-01", periods=n_days, freq="D")

    # Create correlated price data
    np.random.seed(42)
    returns = np.random.randn(n_days, n_assets) * 0.02
    # Add some correlation
    market_factor = np.random.randn(n_days) * 0.01
    for i in range(n_assets):
        returns[:, i] += market_factor * 0.5

    # Generate prices from returns
    prices = {}
    for i in range(n_assets):
        prices[f"asset_{i}"] = 100 * np.exp(returns[:, i].cumsum())

    df = pd.DataFrame(prices, index=dates)

    # Add technical features
    print("  Adding technical features...")
    for col in df.columns:
        # Moving averages
        df[f"{col}_ma_5"] = df[col].rolling(window=5).mean().fillna(df[col])
        df[f"{col}_ma_20"] = df[col].rolling(window=20).mean().fillna(df[col])
        # Returns
        df[f"{col}_returns"] = df[col].pct_change().fillna(0)
        # Volatility
        df[f"{col}_vol"] = df[col].pct_change().rolling(window=10).std().fillna(0)

    print(f"  Created DataFrame with shape {df.shape}")
    return df


def test_basic_functionality():
    """Test basic environment creation and interaction."""
    print("\n" + "=" * 60)
    print("TEST 1: Basic Functionality")
    print("=" * 60)

    df = create_sample_data(n_days=100, n_assets=2)

    # Get price and feature columns
    price_columns = [
        col
        for col in df.columns
        if not ("_ma_" in col or "_returns" in col or "_vol" in col)
    ]
    feature_columns = df.columns.tolist()

    print(f"\nPrice columns ({len(price_columns)}): {price_columns}")
    print(f"Feature columns: {len(feature_columns)} total")

    # Create factory
    factory = TradingEnvXYFactory()

    # Create environment
    print("\nCreating environment...")
    env = factory.make(
        df=df,
        price_columns=price_columns,
        feature_columns=feature_columns,
        cash=10000,
        fee=0.001,  # 0.1% transaction fee
    )

    print("✓ Environment created successfully")
    print(f"  Action spec: {env.action_spec}")
    print(f"  Observation spec: {env.observation_spec}")

    # Test reset
    print("\nTesting reset...")
    obs_dict = env.reset()
    print("✓ Reset successful")
    print(f"  Observation keys: {list(obs_dict.keys())}")
    print(f"  Observation shape: {obs_dict['CustomFeature'].shape}")

    # Test step
    print("\nTesting step...")
    action_dict = env.rand_step(obs_dict)
    obs_dict = env.step(action_dict)
    print("✓ Step successful")
    print(f"  Done: {obs_dict['done'].item()}")
    print(f"  Terminated: {obs_dict['terminated'].item()}")

    print("\n✓ Basic functionality test PASSED\n")


def test_episode_rollout():
    """Test running a complete episode."""
    print("\n" + "=" * 60)
    print("TEST 2: Episode Rollout")
    print("=" * 60)

    df = create_sample_data(n_days=50, n_assets=3)

    price_columns = [
        col
        for col in df.columns
        if not ("_ma_" in col or "_returns" in col or "_vol" in col)
    ]
    feature_columns = df.columns.tolist()

    factory = TradingEnvXYFactory()
    env = factory.make(
        df=df,
        price_columns=price_columns,
        feature_columns=feature_columns,
        cash=10000,
        fee=0.001,
    )

    print("\nRunning episode with random actions...")
    obs_dict = env.reset()

    step_count = 0
    max_steps = 100

    while step_count < max_steps:
        action_dict = env.rand_step(obs_dict)
        obs_dict = env.step(action_dict)

        step_count += 1

        if obs_dict["done"].item() or obs_dict["terminated"].item():
            print(f"  Episode ended at step {step_count}")
            break

    print(f"✓ Completed {step_count} steps")
    print(f"  Final step_count from env: {obs_dict['step_count'].item()}")

    print("\n✓ Episode rollout test PASSED\n")


def test_with_config():
    """Test environment creation with config object."""
    print("\n" + "=" * 60)
    print("TEST 3: Config Integration")
    print("=" * 60)

    from dataclasses import dataclass

    @dataclass
    class EnvConfig:
        cash: float = 5000
        trading_fees: float = 0.002

    @dataclass
    class ExperimentConfig:
        env: EnvConfig

    config = ExperimentConfig(env=EnvConfig())

    df = create_sample_data(n_days=50, n_assets=2)
    price_columns = [
        col
        for col in df.columns
        if not ("_ma_" in col or "_returns" in col or "_vol" in col)
    ]

    print("\nCreating environment with config:")
    print(f"  Initial cash: {config.env.cash}")
    print(f"  Trading fees: {config.env.trading_fees}")

    factory = TradingEnvXYFactory(config=config)
    env = factory.make(
        df=df,
        price_columns=price_columns,
    )

    obs_dict = env.reset()
    print("✓ Environment created with config")
    print(f"  Observation shape: {obs_dict['CustomFeature'].shape}")

    # Run a few steps
    for _ in range(5):
        action_dict = env.rand_step(obs_dict)
        obs_dict = env.step(action_dict)

    print("✓ Ran 5 steps successfully")

    print("\n✓ Config integration test PASSED\n")


def test_different_asset_counts():
    """Test with different numbers of assets."""
    print("\n" + "=" * 60)
    print("TEST 4: Different Asset Counts")
    print("=" * 60)

    for n_assets in [1, 2, 5, 10]:
        print(f"\nTesting with {n_assets} asset(s)...")

        df = create_sample_data(n_days=50, n_assets=n_assets)
        price_columns = [
            col
            for col in df.columns
            if not ("_ma_" in col or "_returns" in col or "_vol" in col)
        ]
        feature_columns = df.columns.tolist()

        factory = TradingEnvXYFactory()
        env = factory.make(
            df=df,
            price_columns=price_columns,
            feature_columns=feature_columns,
            cash=10000,
            fee=0.001,
        )

        obs_dict = env.reset()
        action_dict = env.rand_step(obs_dict)
        obs_dict = env.step(action_dict)

        print(f"  ✓ {n_assets} asset(s) works")
        print(f"    Action space shape: {env.action_spec.shape}")
        print(f"    Observation shape: {obs_dict['CustomFeature'].shape}")

    print("\n✓ Different asset counts test PASSED\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("TradingEnv Wrapper Integration Test Suite")
    print("=" * 60)

    try:
        test_basic_functionality()
        test_episode_rollout()
        test_with_config()
        test_different_asset_counts()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        print("\nThe TradingEnv wrapper is working correctly and ready to use.")
        print("You can now integrate it into your training pipelines.\n")

        return 0

    except Exception as e:
        print("\n" + "=" * 60)
        print("TEST FAILED ✗")
        print("=" * 60)
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

# %%
