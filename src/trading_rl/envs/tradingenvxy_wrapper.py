"""TradingEnv wrapper integrating the tradingenv library with TorchRL.

This module provides a wrapper around the tradingenv.TradingEnv environment
from XAI Asset Management, making it compatible with TorchRL training pipelines.
"""

from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from torchrl.envs import GymWrapper, TransformedEnv
from torchrl.envs.transforms import RenameTransform
from tradingenv import TradingEnv
from tradingenv.broker.fees import BrokerFees
from tradingenv.contracts import Stock
from tradingenv.features import Feature
from tradingenv.rewards import LogReturn
from tradingenv.spaces import BoxPortfolio

from logger import get_logger
from trading_rl.config import DEFAULT_INITIAL_PORTFOLIO_VALUE, ExperimentConfig
from trading_rl.data_loading import MemmapPaths
from trading_rl.envs.trading_envs import BaseTradingEnvironmentFactory
from trading_rl.rewards import DifferentialSharpeRatio

logger = get_logger(__name__)

RUNTIME_POSITION_FEATURE = "feature_position"
SUPPORTED_RUNTIME_FEATURES = {RUNTIME_POSITION_FEATURE}


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

    def __init__(
        self,
        columns: list[str],
        data: pd.DataFrame,
        runtime_features: list[str] | None = None,
        traded_contracts: list[Stock] | None = None,
    ):
        import gymnasium as gym
        import numpy as np

        self.columns = columns
        self._data = data
        self.runtime_features = runtime_features or []
        self.traded_contracts = traded_contracts or []

        # Create space for the feature
        n_features = len(columns) + len(self.runtime_features)
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
            static_values = np.zeros(len(self.columns), dtype=np.float32)
        else:
            static_values = self._data.loc[now, self.columns].values.astype(np.float32)

        if not self.runtime_features:
            return static_values

        runtime_values = [
            self._parse_runtime_feature(feature_name)
            for feature_name in self.runtime_features
        ]
        runtime_array = np.asarray(runtime_values, dtype=np.float32)
        return np.concatenate([static_values, runtime_array]).astype(np.float32)

    def _parse_runtime_feature(self, feature_name: str) -> float:
        if feature_name != RUNTIME_POSITION_FEATURE:
            raise ValueError(f"Unsupported runtime feature requested: {feature_name}")

        if self.broker is None or not self.traded_contracts:
            return 0.0

        holdings = self.broker.holdings_weights()
        contract = self.traded_contracts[0]
        return float(holdings.get(contract, 0.0))


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
        price_column: str | None = None,
        **kwargs: Any,
    ) -> TransformedEnv:
        """Create a TradingEnv environment wrapped for TorchRL.

        Args:
            df: DataFrame with both features and prices
            config: Optional experiment configuration
            feature_columns: List of columns to use as features (observations)
            price_column: Column to use as asset price
            **kwargs: Additional arguments for TradingEnv

        Returns:
            TransformedEnv: TorchRL-compatible environment
        """
        config = config or self.config

        # Observations should come only from feature pipeline outputs.
        if feature_columns is None:
            feature_columns = [
                col for col in df.columns.tolist() if str(col).startswith("feature_")
            ]
        # Backward compatibility with legacy list-style parameter.
        legacy_price_columns = kwargs.pop("price_columns", None)
        if price_column is None and legacy_price_columns:
            if isinstance(legacy_price_columns, list) and legacy_price_columns:
                price_column = str(legacy_price_columns[0])
            elif isinstance(legacy_price_columns, str):
                price_column = legacy_price_columns

        if price_column is None and config is not None and hasattr(config, "env"):
            price_column = getattr(config.env, "price_column", None)

        if price_column is None:
            if "close" in df.columns:
                price_column = "close"
            elif "price" in df.columns:
                price_column = "price"
            else:
                raise ValueError(
                    "TradingEnv requires env.price_column (single string) or "
                    "a dataframe containing 'close'/'price' for fallback."
                )

        if price_column not in df.columns:
            raise ValueError(
                f"Price column '{price_column}' not found in dataframe columns."
            )
        price_columns = [price_column]

        if not feature_columns:
            raise ValueError(
                "No observation feature columns provided/found. "
                "Configure env.feature_columns with feature_* columns or include "
                "feature pipeline outputs in the dataframe."
            )

        include_position_feature = False
        if config is not None:
            env_config = getattr(config, "env", None)
            if env_config is not None:
                include_position_feature = bool(
                    getattr(env_config, "include_position_feature", False)
                )

        if include_position_feature and RUNTIME_POSITION_FEATURE not in feature_columns:
            feature_columns = [*feature_columns, RUNTIME_POSITION_FEATURE]

        non_feature_columns = [
            col for col in feature_columns if not str(col).startswith("feature_")
        ]
        if non_feature_columns:
            raise ValueError(
                "TradingEnv observations must use only feature_* columns. "
                f"Found non-feature columns in env.feature_columns: {non_feature_columns}"
            )

        runtime_feature_columns = [
            col for col in feature_columns if col in SUPPORTED_RUNTIME_FEATURES
        ]
        static_feature_columns = [
            col for col in feature_columns if col not in SUPPORTED_RUNTIME_FEATURES
        ]

        missing_feature_columns = sorted(set(static_feature_columns) - set(df.columns))
        if missing_feature_columns:
            raise ValueError(
                "Observation feature columns missing from dataframe: "
                f"{missing_feature_columns}"
            )

        # Extract environment parameters from config if available
        initial_cash = DEFAULT_INITIAL_PORTFOLIO_VALUE
        fee = 0.0
        reward_type = "log_return"
        reward_eta = 0.01
        if config is not None:
            env_config = getattr(config, "env", None)
            if env_config is not None:
                initial_cash = getattr(env_config, "initial_portfolio_value", DEFAULT_INITIAL_PORTFOLIO_VALUE)
                fee = getattr(env_config, "trading_fees", 0.0)
                reward_type = getattr(env_config, "reward_type", "log_return")
                reward_eta = getattr(env_config, "reward_eta", 0.01)

        # Override with explicit kwargs
        initial_cash = kwargs.pop("cash", initial_cash)
        fee = kwargs.pop("fee", fee)
        reward_type = kwargs.pop("reward_type", reward_type)
        reward_eta = kwargs.pop("reward_eta", reward_eta)

        # Create reward function based on configuration
        if reward_type == "differential_sharpe":
            reward = DifferentialSharpeRatio(eta=reward_eta)
            logger.info("reward differential_sharpe eta=%s", reward_eta)
        elif reward_type == "log_return":
            reward = LogReturn()
            logger.info("using log_return reward")
        else:
            raise ValueError(
                f"Unknown reward type: {reward_type}. "
                "Supported types: 'log_return', 'differential_sharpe'"
            )

        logger.info(
            "Creating TradingEnv environment",
            extra={
                "df_shape": df.shape,
                "feature_columns": feature_columns,
                "runtime_feature_columns": runtime_feature_columns,
                "price_column": price_column,
                "initial_cash": initial_cash,
                "fee": fee,
                "reward_type": reward_type,
            },
        )

        # Create Stock contracts for each price column
        stocks = [Stock(col) for col in price_columns]

        # Create BoxPortfolio action space for continuous allocations
        # Allow short selling by setting low to -1.0
        action_space = BoxPortfolio(stocks, low=-1.0, high=1.0)

        # Create custom features for observations
        features = [
            CustomFeature(
                static_feature_columns,
                df[static_feature_columns],
                runtime_features=runtime_feature_columns,
                traded_contracts=stocks,
            )
        ]

        # Prepare price DataFrame with Contract objects as columns
        prices = df[price_columns].copy()
        # Rename columns to use Contract objects instead of strings
        prices.columns = stocks

        # Create TradingEnv
        broker_fees = BrokerFees(
            proportional=fee,
            fixed=0.0,
        )

        env = TradingEnv(
            action_space=action_space,
            state=features,
            reward=reward,
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

        logger.info("create TradingEnv environment done")
        return wrapped_env


class StreamingTradingEnvXY(gym.Env):
    """Streaming tradingenv environment that loads episode windows from numpy memmaps.

    Rebuilds the inner ``tradingenv.TradingEnv`` on every ``reset()`` using a
    randomly sampled window so peak memory stays proportional to
    ``episode_length``, not the full dataset.

    Presents stable ``observation_space`` and ``action_space`` to TorchRL
    (determined by feature/price column count, not window size).

    Args:
        memmap_paths: Per-symbol memmap metadata from
            :func:`~trading_rl.data_loading.save_symbol_memmap`.
        episode_length: Rows per episode window.
        feature_columns: Static ``feature_*`` columns used as observations.
        price_column: Column used as the asset price.
        initial_cash: Starting portfolio value.
        fee: Proportional broker fee.
        reward_type: ``"log_return"`` or ``"differential_sharpe"``.
        reward_eta: DSR learning rate (only used when
            ``reward_type="differential_sharpe"``).
        runtime_feature_columns: Runtime-computed feature names (e.g.
            ``"feature_position"``).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        memmap_paths: list[MemmapPaths],
        episode_length: int,
        feature_columns: list[str],
        price_column: str,
        initial_cash: float = DEFAULT_INITIAL_PORTFOLIO_VALUE,
        fee: float = 0.0,
        reward_type: str = "log_return",
        reward_eta: float = 0.01,
        runtime_feature_columns: list[str] | None = None,
    ) -> None:
        if not memmap_paths:
            raise ValueError("memmap_paths must contain at least one entry")

        self._memmap_paths = memmap_paths
        self._episode_length = episode_length
        self._feature_columns = feature_columns
        self._price_column = price_column
        self._initial_cash = initial_cash
        self._fee = fee
        self._reward_type = reward_type
        self._reward_eta = reward_eta
        self._runtime_feature_columns = runtime_feature_columns or []

        stocks = [Stock(price_column)]
        self.action_space = BoxPortfolio(stocks, low=-1.0, high=1.0)
        n_obs = len(feature_columns) + len(self._runtime_feature_columns)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32
        )
        self._inner_env: TradingEnv | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_reward(self):
        if self._reward_type == "differential_sharpe":
            return DifferentialSharpeRatio(eta=self._reward_eta)
        if self._reward_type == "log_return":
            return LogReturn()
        raise ValueError(
            f"Unknown reward type: {self._reward_type!r}. "
            "Supported: 'log_return', 'differential_sharpe'"
        )

    def _load_window(self, file_idx: int, start: int) -> pd.DataFrame:
        mp = self._memmap_paths[file_idx]
        end = start + self._episode_length
        data_mm = np.load(mp.data_path, mmap_mode="r")
        index_mm = np.load(mp.index_path, mmap_mode="r")
        window_data = np.array(data_mm[start:end], dtype=np.float32)
        window_index = np.array(index_mm[start:end])
        try:
            index = pd.DatetimeIndex(window_index)
        except Exception:
            index = pd.RangeIndex(len(window_data))
        return pd.DataFrame(window_data, columns=mp.columns, index=index)

    def _build_inner_env(self, window_df: pd.DataFrame) -> TradingEnv:
        stocks = [Stock(self._price_column)]
        prices = window_df[[self._price_column]].copy()
        prices.columns = pd.Index(stocks)
        features = [
            CustomFeature(
                self._feature_columns,
                window_df[self._feature_columns],
                runtime_features=self._runtime_feature_columns,
                traded_contracts=stocks,
            )
        ]
        return TradingEnv(
            action_space=BoxPortfolio(stocks, low=-1.0, high=1.0),
            state=features,
            reward=self._make_reward(),
            prices=prices,
            initial_cash=self._initial_cash,
            broker_fees=BrokerFees(proportional=self._fee, fixed=0.0),
        )

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        file_idx = np.random.randint(0, len(self._memmap_paths))
        mp = self._memmap_paths[file_idx]
        max_start = mp.n_rows - self._episode_length
        start = np.random.randint(0, max(1, max_start))
        window_df = self._load_window(file_idx, start)
        self._inner_env = self._build_inner_env(window_df)
        obs = self._inner_env.reset()
        return obs, {}

    def step(self, action):
        obs, reward, done, info = self._inner_env.step(action)
        return obs, reward, bool(done), False, info

    def render(self):
        if self._inner_env is not None:
            return self._inner_env.render()

    def close(self):
        pass


# Alias for backward compatibility
TradingEnvXYWrapper = TradingEnvXYFactory
