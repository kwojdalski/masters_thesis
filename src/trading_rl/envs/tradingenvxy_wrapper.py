"""TradingEnv wrapper integrating the tradingenv library with TorchRL.

This module provides a wrapper around the tradingenv.TradingEnv environment
from XAI Asset Management, making it compatible with TorchRL training pipelines.
"""

from typing import Any

import gymnasium as gym
import pandas as pd
from torchrl.envs import GymWrapper, TransformedEnv
from torchrl.envs.transforms import RenameTransform
from tradingenv import TradingEnv
from tradingenv.contracts import Stock
from tradingenv.features import Feature
from tradingenv.rewards import LogReturn
from tradingenv.spaces import BoxPortfolio

from logger import get_logger
from trading_rl.config import ExperimentConfig
from trading_rl.envs.trading_envs import BaseTradingEnvironmentFactory

logger = get_logger(__name__)


class GymnasiumTradingEnvWrapper(gym.Env):
    """Gymnasium-compatible wrapper for TradingEnv (old Gym API).

    TradingEnv uses the old Gym API where reset() returns only observation.
    This wrapper converts it to the new Gymnasium API where reset() returns (observation, info).
    """

    def __init__(self, trading_env: TradingEnv):
        self._env = trading_env
        self.observation_space = trading_env.observation_space
        self.action_space = trading_env.action_space

    def reset(self, *, seed=None, options=None):
        """Reset and return (observation, info) tuple."""
        obs = self._env.reset()
        info = {}
        return obs, info

    def step(self, action):
        """Step and return (observation, reward, terminated, truncated, info) tuple."""
        obs, reward, done, info = self._env.step(action)
        terminated = done
        truncated = False
        return obs, reward, terminated, truncated, info

    def render(self):
        """Render the environment."""
        return self._env.render()

    def close(self):
        """Close the environment."""
        pass


class CustomFeature(Feature):
    """Custom feature that extracts specific columns from the dataframe."""

    def __init__(self, columns: list[str], data: pd.DataFrame):
        import gymnasium as gym
        import numpy as np

        self.columns = columns
        self._data = data

        # Create space for the feature
        n_features = len(columns)
        space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32
        )

        # Call parent __init__
        super().__init__(space=space, name="CustomFeature")

    def parse(self) -> Any:
        """Parse the current observation.

        This method is called by the Feature base class to get the current observation.
        """
        import numpy as np

        now = self._now()
        if now is None or now not in self._data.index:
            return np.zeros(len(self.columns), dtype=np.float32)
        return self._data.loc[now, self.columns].values.astype(np.float32)


class TradingEnvXYFactory(BaseTradingEnvironmentFactory):
    """Factory for creating TradingEnv environments compatible with TorchRL.

    This factory creates environments using the tradingenv library and wraps them
    for use with TorchRL training pipelines.

    The environment uses:
    - Stock contracts for defining tradable assets
    - BoxPortfolio for continuous portfolio allocation actions
    - Custom features for observations
    - LogReturn rewards for RL training
    """

    def __init__(self, config: ExperimentConfig | None = None):
        self.config = config

    def make(
        self,
        df: pd.DataFrame,
        config: ExperimentConfig | None = None,
        feature_columns: list[str] | None = None,
        price_columns: list[str] | None = None,
        **kwargs: Any,
    ) -> TransformedEnv:
        """Create a TradingEnv environment wrapped for TorchRL.

        Args:
            df: DataFrame with both features and prices
            config: Optional experiment configuration
            feature_columns: List of columns to use as features (observations)
            price_columns: List of columns to use as asset prices
            **kwargs: Additional arguments for TradingEnv

        Returns:
            TransformedEnv: TorchRL-compatible environment
        """
        config = config or self.config

        # Default to all columns as both features and prices if not specified
        if feature_columns is None:
            feature_columns = df.columns.tolist()
        if price_columns is None:
            price_columns = df.columns.tolist()

        # Extract environment parameters from config if available
        initial_cash = 10000
        fee = 0.0
        if config is not None:
            env_config = getattr(config, "env", None)
            if env_config is not None:
                initial_cash = getattr(env_config, "cash", 10000)
                fee = getattr(env_config, "trading_fees", 0.0)

        # Override with explicit kwargs
        initial_cash = kwargs.pop("cash", initial_cash)
        fee = kwargs.pop("fee", fee)

        logger.info(
            "Creating TradingEnv environment",
            extra={
                "df_shape": df.shape,
                "feature_columns": feature_columns,
                "price_columns": price_columns,
                "initial_cash": initial_cash,
                "fee": fee,
            },
        )

        # Create Stock contracts for each price column
        stocks = [Stock(col) for col in price_columns]

        # Create BoxPortfolio action space for continuous allocations
        action_space = BoxPortfolio(stocks)

        # Create custom features for observations
        features = [CustomFeature(feature_columns, df[feature_columns])]

        # Prepare price DataFrame with Contract objects as columns
        prices = df[price_columns].copy()
        # Rename columns to use Contract objects instead of strings
        prices.columns = stocks

        # Create TradingEnv
        from tradingenv.broker.fees import BrokerFees

        broker_fees = BrokerFees(
            proportional=fee,
            fixed=0.0,
        )

        env = TradingEnv(
            action_space=action_space,
            state=features,
            reward=LogReturn(),
            prices=prices,
            initial_cash=initial_cash,
            broker_fees=broker_fees,
            **kwargs,
        )

        # Wrap with Gymnasium compatibility layer
        gym_env = GymnasiumTradingEnvWrapper(env)

        # Wrap for TorchRL
        wrapped_env = GymWrapper(gym_env)

        # Rename observation key to match expected format
        # TradingEnv creates observations with key "CustomFeature"
        # but training script expects "observation"
        wrapped_env = TransformedEnv(
            wrapped_env,
            RenameTransform(in_keys=["CustomFeature"], out_keys=["observation"]),
        )

        # Add step counter
        wrapped_env = self._wrap_with_step_counter(wrapped_env)

        logger.info("Created TradingEnv environment successfully")
        return wrapped_env


# Alias for backward compatibility
TradingEnvXYWrapper = TradingEnvXYFactory
