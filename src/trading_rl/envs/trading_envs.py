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
    "tradingenv",
]

# Supported backend values for validation
SUPPORTED_BACKENDS = [
    "gym_trading_env.discrete",
    "gym_trading_env.continuous",
    "gym_anytrading.forex",
    "gym_anytrading.stocks",
    "tradingenv",
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


def validate_actions(backend: Backend, positions: list[int] | None) -> None:
    """Validate action set for a given backend.

    For gym-anytrading (forex/stocks), only two actions are supported: 0 (short), 1 (long).
    """
    if positions is None:
        return

    if backend in {"gym_anytrading.forex", "gym_anytrading.stocks"}:
        allowed = [0, 1]
        if positions != allowed:
            raise ValueError(
                f"{backend} supports only two actions {allowed} (short/long), "
                f"but positions={positions}. Please set env.positions to {allowed}."
            )
        logger.debug("Validated actions for %s backend: %s", backend, positions)


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

    def make(self, *args, **kwargs) -> TransformedEnv:
        """Abstract method to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement make")


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

    def _build_env(
        self, df: pd.DataFrame, config: ExperimentConfig, *, continuous: bool
    ) -> TransformedEnv:
        """Create base env, wrap for TorchRL, optionally add continuous mapping."""
        base_env = self._create_base_environment(df, config)
        env = GymWrapper(base_env)

        if continuous:
            env = TransformedEnv(
                env,
                ContinuousToDiscreteAction(
                    discrete_actions=config.env.positions,
                    thresholds=[-0.33, 0.33],
                    device=getattr(config.training, "device", "cpu"),
                ),
            )

        return self._wrap_with_step_counter(env)

    def make(
        self,
        df: pd.DataFrame,
        config: ExperimentConfig | None = None,
        *,
        backend: Backend | None = None,
    ) -> TransformedEnv:
        """Create trading environment; uses backend to toggle continuous mapping."""
        config = config or self.config
        if config is None:
            raise ValueError(
                "Config must be provided either in constructor or method call"
            )

        backend = backend or getattr(config.env, "backend", "gym_trading_env.discrete")
        validate_backend(backend, log_backend=False)

        continuous = backend == "gym_trading_env.continuous"
        logger.info(
            "Creating trading environment",
            extra={"backend": backend, "continuous": continuous},
        )
        logger.debug(
            "Env settings",
            extra={
                "positions": config.env.positions,
                "trading_fees": config.env.trading_fees,
                "borrow_interest_rate": config.env.borrow_interest_rate,
            },
        )

        return self._build_env(df, config, continuous=continuous)


class ForexEnvironmentFactory(BaseTradingEnvironmentFactory):
    """Factory for forex-v0 environments."""

    def __init__(self, config: ExperimentConfig | None = None):
        self.config = config

    def make(self, df: pd.DataFrame | None = None, **kwargs) -> TransformedEnv:
        """Create forex trading environment with optional DSR reward."""
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

        # Apply DSR wrapper if configured
        if self.config is not None:
            reward_type = getattr(self.config.env, "reward_type", "log_return")
            if reward_type == "differential_sharpe":
                from trading_rl.rewards.dsr_wrapper import (
                    DifferentialSharpeRatioAnyTrading,
                    StatefulRewardWrapper,
                )

                reward_eta = getattr(self.config.env, "reward_eta", 0.01)
                dsr = DifferentialSharpeRatioAnyTrading(eta=reward_eta)
                base_env = StatefulRewardWrapper(base_env, reward_fn=dsr)
                logger.info(
                    f"Applied DSR reward to forex-v0 environment (eta={reward_eta})"
                )
            elif reward_type != "log_return":
                raise ValueError(
                    f"Unknown reward type: {reward_type}. "
                    "Supported types: 'log_return', 'differential_sharpe'"
                )

        base_env = DiscreteActionWrapper(base_env)
        env = GymWrapper(base_env)
        env = self._wrap_with_step_counter(env)

        logger.info("Created forex-v0 environment")
        return env


class StocksEnvironmentFactory(BaseTradingEnvironmentFactory):
    """Factory for stocks-v0 environments."""

    def __init__(self, config: ExperimentConfig | None = None):
        self.config = config

    def make(self, df: pd.DataFrame | None = None, **kwargs) -> TransformedEnv:
        """Create stocks trading environment with optional DSR reward."""
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

        # Apply DSR wrapper if configured
        if self.config is not None:
            reward_type = getattr(self.config.env, "reward_type", "log_return")
            if reward_type == "differential_sharpe":
                from trading_rl.rewards.dsr_wrapper import (
                    DifferentialSharpeRatioAnyTrading,
                    StatefulRewardWrapper,
                )

                reward_eta = getattr(self.config.env, "reward_eta", 0.01)
                dsr = DifferentialSharpeRatioAnyTrading(eta=reward_eta)
                base_env = StatefulRewardWrapper(base_env, reward_fn=dsr)
                logger.info(
                    f"Applied DSR reward to stocks-v0 environment (eta={reward_eta})"
                )
            elif reward_type != "log_return":
                raise ValueError(
                    f"Unknown reward type: {reward_type}. "
                    "Supported types: 'log_return', 'differential_sharpe'"
                )

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

    config = kwargs.get("config")

    if backend in ["gym_trading_env.discrete", "gym_trading_env.continuous"]:
        return CustomTradingEnvironmentFactory(config)
    elif backend == "gym_anytrading.forex":
        return ForexEnvironmentFactory(config)
    elif backend == "gym_anytrading.stocks":
        return StocksEnvironmentFactory(config)
    elif backend == "tradingenv":
        # Import here to avoid circular dependency and keep it optional
        from trading_rl.envs.tradingenvxy_wrapper import TradingEnvXYFactory

        return TradingEnvXYFactory(config)
    else:
        # This should not happen after validation, but keeping for safety
        raise ValueError(f"Unsupported backend: {backend}")


# Convenience functions that maintain backward compatibility
_factory = CustomTradingEnvironmentFactory()


def create_continuous_trading_environment(
    df: pd.DataFrame, config: ExperimentConfig
) -> TransformedEnv:
    """Create a continuous-action trading environment (TD3/DDPG)."""
    return create_environment(df, config=config, backend="gym_trading_env.continuous")


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
    positions = (
        getattr(config.env, "positions", None)
        if config is not None
        else kwargs.get("positions")
    )
    validate_actions(backend, positions)
    logger.debug(
        "Environment validation passed",
        extra={"backend": backend, "positions": positions},
    )

    if backend in ["gym_trading_env.discrete", "gym_trading_env.continuous"]:
        if config is None:
            raise ValueError("config is required when using gym_trading_env backends")
        factory = get_environment_factory(backend, config=config)

        # Handle discrete vs continuous logic
        if backend == "gym_trading_env.continuous":
            return factory.make(df, config, backend=backend)
        else:
            return factory.make(df, config, backend=backend)
    elif backend == "tradingenv":
        # TradingEnv backend - supports continuous portfolio allocation
        if config is None:
            raise ValueError("config is required when using tradingenv backend")
        factory = get_environment_factory(backend, config=config)

        # Extract column specifications from config if available
        price_columns = getattr(config.env, "price_columns", None)
        feature_columns = getattr(config.env, "feature_columns", None)

        # Pass to factory with column specifications
        return factory.make(
            df=df,
            config=config,
            price_columns=price_columns,
            feature_columns=feature_columns,
            **kwargs,
        )
    else:
        # gym_anytrading backends (forex, stocks)
        factory = get_environment_factory(backend, config=config, **kwargs)
        return factory.make(df, **kwargs)
