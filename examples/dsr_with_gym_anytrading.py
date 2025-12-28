"""Example: Using DSR reward with gym_anytrading environments.

This demonstrates how to use the StatefulRewardWrapper to automatically
manage DSR state resets with gym_anytrading (forex-v0, stocks-v0).
"""

import gymnasium as gym
import pandas as pd
from torchrl.envs import GymWrapper, TransformedEnv
from torchrl.envs.transforms import StepCounter

from trading_rl.rewards.dsr_wrapper import (
    DifferentialSharpeRatioAnyTrading,
    StatefulRewardWrapper,
)


def create_dsr_forex_env(df: pd.DataFrame, eta: float = 0.01) -> TransformedEnv:
    """Create forex-v0 environment with DSR reward.

    Args:
        df: Price data with columns: open, high, low, close, volume
        eta: DSR learning rate

    Returns:
        TorchRL-wrapped environment with automatic DSR reset
    """
    # 1. Create DSR reward instance
    dsr_reward = DifferentialSharpeRatioAnyTrading(eta=eta)

    # 2. Rename columns for gym_anytrading
    df_capitalized = df.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )

    # 3. Create base gym_anytrading environment
    base_env = gym.make("forex-v0", df=df_capitalized, window_size=10, frame_bound=(10, len(df_capitalized)))

    # 4. Wrap with StatefulRewardWrapper (automatically resets DSR on env.reset())
    base_env = StatefulRewardWrapper(base_env, reward_fn=dsr_reward)

    # 5. Wrap for TorchRL
    env = GymWrapper(base_env)
    env = TransformedEnv(env, StepCounter())

    return env


def create_dsr_stocks_env(df: pd.DataFrame, eta: float = 0.01) -> TransformedEnv:
    """Create stocks-v0 environment with DSR reward.

    Args:
        df: Price data with columns: open, high, low, close, volume
        eta: DSR learning rate

    Returns:
        TorchRL-wrapped environment with automatic DSR reset
    """
    # Same pattern as forex
    dsr_reward = DifferentialSharpeRatioAnyTrading(eta=eta)

    df_capitalized = df.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )

    base_env = gym.make("stocks-v0", df=df_capitalized, window_size=10, frame_bound=(10, len(df_capitalized)))
    base_env = StatefulRewardWrapper(base_env, reward_fn=dsr_reward)

    env = GymWrapper(base_env)
    env = TransformedEnv(env, StepCounter())

    return env


if __name__ == "__main__":
    # Example usage
    import numpy as np

    # Create sample price data
    np.random.seed(42)
    n_steps = 100
    df = pd.DataFrame(
        {
            "open": 100 + np.random.randn(n_steps).cumsum(),
            "high": 101 + np.random.randn(n_steps).cumsum(),
            "low": 99 + np.random.randn(n_steps).cumsum(),
            "close": 100 + np.random.randn(n_steps).cumsum(),
            "volume": np.random.randint(1000, 10000, n_steps),
        }
    )

    # Create environment with DSR
    env = create_dsr_forex_env(df, eta=0.01)

    # Test automatic reset
    print("Testing DSR with gym_anytrading...")
    obs = env.reset()
    print(f"Initial observation shape: {obs['observation'].shape}")

    # Run a few steps
    for i in range(10):
        action = env.action_spec.sample()  # Random action
        obs = env.step(action)
        reward = obs["next", "reward"].item()
        print(f"Step {i+1}: DSR reward = {reward:.6f}")

    # Reset should automatically reset DSR state
    print("\nResetting environment (DSR state should reset automatically)...")
    obs = env.reset()

    # Run a few more steps - should produce same sequence with same actions
    print("After reset:")
    for i in range(3):
        action = env.action_spec.sample()
        obs = env.step(action)
        reward = obs["next", "reward"].item()
        print(f"Step {i+1}: DSR reward = {reward:.6f}")

    print("\nDSR successfully integrated with gym_anytrading!")
    print("The StatefulRewardWrapper automatically resets DSR state on env.reset()")
