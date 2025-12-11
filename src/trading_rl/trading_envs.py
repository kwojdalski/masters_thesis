"""Environment creation helpers for trading experiments."""

import warnings
from typing import Literal

import gym_anytrading  # noqa: F401  # registers envs if available
import gym_trading_env  # noqa: F401
import gymnasium as gym
import pandas as pd
from torchrl.envs import GymWrapper, TransformedEnv
from torchrl.envs.transforms import StepCounter

from logger import get_logger
from trading_rl.config import ExperimentConfig
from trading_rl.continuous_action_wrapper import ContinuousToDiscreteAction
from trading_rl.data_utils import reward_function

logger = get_logger(__name__)

# Type for backend selection
Backend = Literal[
    "gym_trading_env.discrete",
    "gym_trading_env.continuous",
    "gym_anytrading.forex",
    "gym_anytrading.stocks",
]

# Supported backend values for validation
SUPPORTED_BACKENDS = [
    "gym_trading_env.discrete",
    "gym_trading_env.continuous",
    "gym_anytrading.forex",
    "gym_anytrading.stocks",
]


def validate_backend(backend: str, log_backend: bool = False) -> None:
    """Validate that backend is one of the supported values.

    Args:
        backend: Backend string to validate
        log_backend: Whether to log the backend being used

    Raises:
        ValueError: If backend is not supported
    """
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Invalid backend '{backend}'. Supported backends are: {SUPPORTED_BACKENDS}"
        )
    if log_backend:
        logger.warning(f"Using backend: {backend}")



class DiscreteActionWrapper(gym.ActionWrapper):
    """Wrapper to ensure actions are discrete scalars for gym-anytrading.
    
    Handles various input formats from RL agents:
    - Scalar tensors/arrays -> int
    - One-hot vectors -> argmax -> int
    - Probability distributions -> argmax -> int
    """

    def action(self, action):
        # Handle Tensor or NumPy array
        if hasattr(action, "flatten"):
            # Don't flatten immediately if we want to check shape for one-hot
            pass
            
        # Check if it's a vector (length > 1) which implies OneHot or Probs
        if hasattr(action, "shape") and len(action.shape) >= 1 and action.shape[-1] > 1:
            # Assume it's a one-hot or probability vector, take argmax
            if hasattr(action, "argmax"):
                action = action.argmax(dim=-1)
            else:
                import numpy as np
                action = np.argmax(action, axis=-1)
                
        # Now handle scalar/single-element conversion
        if hasattr(action, "flatten"):
            action = action.flatten()
        elif hasattr(action, "reshape") and hasattr(action, "shape"):
             if len(action.shape) > 0:
                action = action.reshape(-1)

        # If array-like with elements, take first
        if hasattr(action, "__len__"):
            try:
                if len(action) > 0:
                    action = action[0]
            except TypeError:
                pass

        # Convert to int
        if hasattr(action, "item"):
            return int(action.item())
        return int(action)


class BaseTradingEnvironmentFactory:
    """Base class for all trading environment factories."""

    def _wrap_with_step_counter(self, env: GymWrapper) -> TransformedEnv:
        """Add step counter transform to environment."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*auto_unwrap_transformed_env.*")
            return TransformedEnv(env, StepCounter())

    def create_environment(self, *args, **kwargs) -> TransformedEnv:
        """Abstract method to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement create_environment")


class CustomTradingEnvironmentFactory(BaseTradingEnvironmentFactory):
    """Factory for custom TradingEnv environments with config-based setup."""

    def __init__(self, config: ExperimentConfig | None = None):
        self.config = config

    def _create_base_environment(
        self, df: pd.DataFrame, config: ExperimentConfig
    ) -> gym.Env:
        """Create the base trading environment."""
        return gym.make(
            "TradingEnv",
            name=config.env.name,
            df=df[: config.data.train_size],
            positions=config.env.positions,
            trading_fees=config.env.trading_fees,
            borrow_interest_rate=config.env.borrow_interest_rate,
            reward_function=reward_function,
        )

    def create_continuous_trading_environment(
        self, df: pd.DataFrame, config: ExperimentConfig
    ) -> TransformedEnv:
        """Create a continuous-action trading environment (TD3/DDPG)."""
        positions = config.env.positions
        base_env = self._create_base_environment(df, config)
        logger.info(f"Created discrete base environment with positions: {positions}")

        env = GymWrapper(base_env)

        continuous_wrapper = ContinuousToDiscreteAction(
            discrete_actions=positions,
            thresholds=[-0.33, 0.33],
            device=getattr(config.training, "device", "cpu"),
        )
        env = TransformedEnv(env, continuous_wrapper)
        env = self._wrap_with_step_counter(env)

        logger.info("Continuous trading environment created successfully")
        return env

    def create_discrete_trading_environment(
        self, df: pd.DataFrame, config: ExperimentConfig
    ) -> TransformedEnv:
        """Create a discrete-action trading environment."""
        base_env = self._create_base_environment(df, config)
        env = GymWrapper(base_env)
        return self._wrap_with_step_counter(env)

    def create_environment(
        self, df: pd.DataFrame, config: ExperimentConfig | None = None
    ) -> TransformedEnv:
        """Create and configure trading environment based on algorithm and backend."""
        config = config or self.config
        if config is None:
            raise ValueError(
                "Config must be provided either in constructor or method call"
            )

        # Check backend setting first
        backend = getattr(config.env, "backend", "gym_trading_env.discrete")
        validate_backend(backend, log_backend=False)

        if backend == "gym_trading_env.continuous":
            logger.info(f"Backend {backend} - creating continuous action environment")
            return self.create_continuous_trading_environment(df, config)
        elif backend == "gym_trading_env.discrete":
            logger.info(f"Backend {backend} - creating discrete action environment")
            return self.create_discrete_trading_environment(df, config)
        else:
            # This should not happen after validation, but keeping for safety
            raise ValueError(f"Unsupported backend for CustomTradingEnvironmentFactory: {backend}")


class ForexEnvironmentFactory(BaseTradingEnvironmentFactory):
    """Factory for forex-v0 environments."""

    def create_environment(
        self, df: pd.DataFrame | None = None, **kwargs
    ) -> TransformedEnv:
        """Create forex trading environment."""
        env_kwargs = kwargs.copy()
        if df is not None:
            # Map lowercase columns to Capitalized for gym-anytrading
            rename_map = {
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
            df = df.rename(columns=rename_map)
            env_kwargs["df"] = df

        base_env = gym.make("forex-v0", **env_kwargs)
        base_env = DiscreteActionWrapper(base_env)
        env = GymWrapper(base_env)
        env = self._wrap_with_step_counter(env)

        logger.info("Created forex-v0 environment")
        return env


class StocksEnvironmentFactory(BaseTradingEnvironmentFactory):
    """Factory for stocks-v0 environments."""

    def create_environment(
        self, df: pd.DataFrame | None = None, **kwargs
    ) -> TransformedEnv:
        """Create stocks trading environment."""
        env_kwargs = kwargs.copy()
        if df is not None:
            # Map lowercase columns to Capitalized for gym-anytrading
            rename_map = {
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
            df = df.rename(columns=rename_map)
            env_kwargs["df"] = df

        base_env = gym.make("stocks-v0", **env_kwargs)
        base_env = DiscreteActionWrapper(base_env)
        env = GymWrapper(base_env)
        env = self._wrap_with_step_counter(env)

        logger.info("Created stocks-v0 environment")
        return env


def get_environment_factory(
    backend: Backend, **kwargs
) -> BaseTradingEnvironmentFactory:
    """Factory function to get the appropriate environment factory based on backend."""
    validate_backend(backend, log_backend=False)

    if backend in ["gym_trading_env.discrete", "gym_trading_env.continuous"]:
        config = kwargs.get("config")
        return CustomTradingEnvironmentFactory(config)
    elif backend == "gym_anytrading.forex":
        return ForexEnvironmentFactory()
    elif backend == "gym_anytrading.stocks":
        return StocksEnvironmentFactory()
    else:
        # This should not happen after validation, but keeping for safety
        raise ValueError(f"Unsupported backend: {backend}")


# Convenience functions that maintain backward compatibility
_factory = CustomTradingEnvironmentFactory()


def create_continuous_trading_environment(
    df: pd.DataFrame, config: ExperimentConfig
) -> TransformedEnv:
    """Create a continuous-action trading environment (TD3/DDPG).

    Builds the discrete TradingEnv then wraps it with a transform that maps
    continuous actions in [-1, 1] to discrete positions.
    """
    return _factory.create_continuous_trading_environment(df, config)


def create_environment(
    df: pd.DataFrame,
    config: ExperimentConfig | None = None,
    backend: Backend | None = None,
    **kwargs,
) -> TransformedEnv:
    """Create and configure trading environment based on backend and algorithm.

    Args:
        df: DataFrame with trading data
        config: Configuration for gym_trading_env backends (required for gym_trading_env backends)
        backend: Backend type. If None, uses config.env.backend if available, else defaults to "gym_trading_env.discrete"
        **kwargs: Additional arguments passed to the environment factory
    """
    # Determine backend: explicit parameter > config.env.backend > default
    if backend is None:
        if config is not None and hasattr(config.env, "backend"):
            backend = config.env.backend
            logger.warning(f"Backend determined from config: {backend}")
        else:
            backend = "gym_trading_env.discrete"
            logger.warning(f"Using default backend: {backend}")
    else:
        logger.warning(f"Backend explicitly specified: {backend}")

    # Validate backend early (without additional logging)
    validate_backend(backend, log_backend=False)

    if backend in ["gym_trading_env.discrete", "gym_trading_env.continuous"]:
        if config is None:
            raise ValueError("config is required when using gym_trading_env backends")
        factory = get_environment_factory(backend, config=config)

        # Handle discrete vs continuous logic
        if backend == "gym_trading_env.continuous":
            return factory.create_continuous_trading_environment(df, config)
        else:
            return factory.create_discrete_trading_environment(df, config)
    else:
        factory = get_environment_factory(backend, **kwargs)
        return factory.create_environment(df, **kwargs)
