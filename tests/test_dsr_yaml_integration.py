"""Test DSR integration with YAML configuration workflow."""

import numpy as np
import pandas as pd
import pytest

from trading_rl.config import EnvConfig, ExperimentConfig
from trading_rl.envs.trading_envs import create_environment


def create_sample_dataframe(n_steps: int = 100) -> pd.DataFrame:
    """Create sample price data for testing."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "open": 100 + np.random.randn(n_steps).cumsum(),
            "high": 101 + np.random.randn(n_steps).cumsum(),
            "low": 99 + np.random.randn(n_steps).cumsum(),
            "close": 100 + np.random.randn(n_steps).cumsum(),
            "volume": np.random.randint(1000, 10000, n_steps),
        }
    )


class TestDSRYamlIntegration:
    """Test DSR reward configuration via YAML workflow."""

    def test_forex_with_dsr_from_config(self):
        """Test forex environment with DSR configured via ExperimentConfig."""
        # Create config with DSR
        env_config = EnvConfig(
            name="forex-test",
            backend="gym_anytrading.forex",
            positions=[0, 1],
            reward_type="differential_sharpe",
            reward_eta=0.02,
        )

        config = ExperimentConfig(env=env_config)
        df = create_sample_dataframe()

        # Create environment via factory
        env = create_environment(df, config=config)

        # Check environment was created
        assert env is not None

        # Use rollout to test
        rollout = env.rollout(max_steps=10)
        rewards = rollout["next", "reward"]

        # DSR rewards should be finite
        assert all(np.isfinite(r.item()) for r in rewards)
        # First reward should be 0 (no previous NLV)
        assert rewards[0].item() == 0.0

    def test_stocks_with_dsr_from_config(self):
        """Test stocks environment with DSR configured via ExperimentConfig."""
        # Create config with DSR
        env_config = EnvConfig(
            name="stocks-test",
            backend="gym_anytrading.stocks",
            positions=[0, 1],
            reward_type="differential_sharpe",
            reward_eta=0.05,
        )

        config = ExperimentConfig(env=env_config)
        df = create_sample_dataframe()

        # Create environment via factory
        env = create_environment(df, config=config)

        # Check environment was created
        assert env is not None

        # Use rollout to test
        rollout = env.rollout(max_steps=10)
        rewards = rollout["next", "reward"]

        # DSR rewards should be finite
        assert all(np.isfinite(r.item()) for r in rewards)
        # First reward should be 0
        assert rewards[0].item() == 0.0

    def test_forex_with_log_return_default(self):
        """Test forex environment defaults to log_return when reward_type not specified."""
        # Create config WITHOUT reward_type (should default to log_return)
        env_config = EnvConfig(
            name="forex-default",
            backend="gym_anytrading.forex",
            positions=[0, 1],
            # reward_type defaults to "log_return"
        )

        config = ExperimentConfig(env=env_config)
        df = create_sample_dataframe()

        # Create environment via factory
        env = create_environment(df, config=config)

        # Use rollout to test
        rollout = env.rollout(max_steps=10)
        rewards = rollout["next", "reward"]

        # Rewards should be finite
        assert all(np.isfinite(r.item()) for r in rewards)

    def test_dsr_state_resets_between_episodes(self):
        """Test DSR state automatically resets on env.reset()."""
        env_config = EnvConfig(
            name="forex-reset-test",
            backend="gym_anytrading.forex",
            positions=[0, 1],
            reward_type="differential_sharpe",
            reward_eta=0.1,
        )

        config = ExperimentConfig(env=env_config)
        df = create_sample_dataframe()
        env = create_environment(df, config=config)

        # Episode 1
        rollout1 = env.rollout(max_steps=10)
        rewards1 = rollout1["next", "reward"]

        # Episode 2 (environment auto-resets after rollout completes)
        rollout2 = env.rollout(max_steps=10)
        rewards2 = rollout2["next", "reward"]

        # Both episodes should have finite rewards
        assert all(np.isfinite(r.item()) for r in rewards1)
        assert all(np.isfinite(r.item()) for r in rewards2)

        # First reward in each episode should be 0 (DSR reset)
        assert rewards1[0].item() == 0.0
        assert rewards2[0].item() == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
