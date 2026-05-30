"""TradingEnv wrapper integrating the tradingenv library with TorchRL.

This module provides a wrapper around the tradingenv.TradingEnv environment
from XAI Asset Management, making it compatible with TorchRL training pipelines.
"""

from pathlib import Path
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
import pandas as pd
from torchrl.envs import GymWrapper, TransformedEnv
from torchrl.envs.transforms import RenameTransform
from tradingenv import TradingEnv
from tradingenv.broker.broker import EndOfEpisodeError
from tradingenv.broker.fees import BrokerFees
from tradingenv.contracts import Stock
from tradingenv.features import Feature
from tradingenv.spaces import BoxPortfolio

from logger import get_logger
from trading_rl.config import DEFAULT_INITIAL_PORTFOLIO_VALUE, ExperimentConfig
from trading_rl.constants import ActionPenaltyType, RewardType
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

    def __init__(self, trading_env: TradingEnv, obs_clip: float | None = None):
        self._env = trading_env
        self.observation_space = trading_env.observation_space
        self.action_space = trading_env.action_space
        self._obs_clip = obs_clip

    def _clip_obs(self, obs):
        if self._obs_clip is None:
            return obs
        if isinstance(obs, dict):
            return {k: np.clip(v, -self._obs_clip, self._obs_clip) for k, v in obs.items()}
        return np.clip(obs, -self._obs_clip, self._obs_clip)

    def reset(self, *, seed=None, options=None):
        """Reset and return (observation, info) tuple."""
        obs = self._env.reset()
        return self._clip_obs(obs), {}

    def step(self, action):
        """Step and return (observation, reward, terminated, truncated, info) tuple."""
        try:
            obs, reward, done, info = self._env.step(action)
        except EndOfEpisodeError:
            # Portfolio went bankrupt — end the episode with a penalty reward.
            # TradingEnv observation_space is a Dict (shape=None); reconstruct a
            # zero observation in the same format that normal steps return.
            if self.observation_space.shape is not None:
                obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            else:
                obs = {
                    k: np.zeros(v.shape, dtype=np.float32)
                    for k, v in self.observation_space.spaces.items()
                }
            return obs, -1.0, True, False, {"bankrupt": True}
        # Distinguish bankruptcy (terminated) from data exhaustion (truncated).
        # When done because the transmitter ran out of data, the state is not
        # truly terminal — value should bootstrap from the final observation.
        # Bankruptcy is signaled by EndOfEpisodeError (caught above) or by
        # the broker's net liquidation value dropping to zero or below.
        if done:
            broker = getattr(self._env, "broker", None)
            is_bankrupt = broker is not None and broker.net_liquidation_value() <= 0
            terminated = is_bankrupt
            truncated = not is_bankrupt
        else:
            terminated = False
            truncated = False
        return self._clip_obs(obs), reward, terminated, truncated, info

    def render(self):
        """Render the environment."""
        return self._env.render()

    def close(self):
        if hasattr(self._env, "close"):
            self._env.close()


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
        # Pop explicit override kwargs before they leak into TradingEnv(**kwargs).
        include_position_feature_override = kwargs.pop("include_position_feature", None)
        obs_clip_override = kwargs.pop("obs_clip", None)

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
        if include_position_feature_override is not None:
            include_position_feature = bool(include_position_feature_override)

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
        reward_type = RewardType.LOG_RETURN
        reward_eta = 0.01
        reward_scale = 1.0
        if config is not None:
            env_config = getattr(config, "env", None)
            if env_config is not None:
                initial_cash = getattr(env_config, "initial_portfolio_value", DEFAULT_INITIAL_PORTFOLIO_VALUE)
                fee = getattr(env_config, "trading_fees", 0.0)
                reward_type = getattr(env_config, "reward_type", RewardType.LOG_RETURN)
                reward_eta = getattr(env_config, "reward_eta", 0.01)
                reward_scale = getattr(env_config, "reward_scale", 1.0)

        # Override with explicit kwargs
        initial_cash = kwargs.pop("cash", initial_cash)
        fee = kwargs.pop("fee", fee)
        reward_type = kwargs.pop("reward_type", reward_type)
        reward_eta = kwargs.pop("reward_eta", reward_eta)
        reward_scale = kwargs.pop("reward_scale", reward_scale)

        from trading_rl.rewards.registry import RewardRegistry
        reward = RewardRegistry.create(reward_type, eta=reward_eta, scale=reward_scale)
        logger.info("reward reward_type={} eta={} scale={}", reward_type, reward_eta, reward_scale)

        logger.info(
            "creating TradingEnv environment n_rows={} n_static_cols={} n_runtime_cols={} price_column={} fee={}",
            len(df),
            len(static_feature_columns),
            len(runtime_feature_columns) if runtime_feature_columns else 0,
            price_column,
            fee,
        )

        logger.trace("building stock contracts n_price_columns={}", len(price_columns))
        stocks = [Stock(col) for col in price_columns]
        action_space = BoxPortfolio(stocks, low=-1.0, high=1.0)
        logger.trace("action space built low=-1.0 high=1.0")

        logger.trace("building CustomFeature n_static={}", len(static_feature_columns))
        features = [
            CustomFeature(
                static_feature_columns,
                df[static_feature_columns],
                runtime_features=runtime_feature_columns,
                traded_contracts=stocks,
            )
        ]
        logger.trace("CustomFeature built")

        logger.trace("preparing price dataframe n_rows={} n_cols={}", len(df), len(price_columns))
        prices = df[price_columns].copy()
        prices.columns = stocks
        logger.trace("price dataframe ready")

        broker_fees = BrokerFees(proportional=fee, fixed=0.0)

        logger.trace("constructing TradingEnv initial_cash={}", initial_cash)
        env = TradingEnv(
            action_space=action_space,
            state=features,
            reward=reward,
            prices=prices,
            initial_cash=initial_cash,
            broker_fees=broker_fees,
            **kwargs,
        )
        logger.trace("TradingEnv constructed")

        obs_clip = None
        if config is not None:
            env_cfg = getattr(config, "env", None)
            if env_cfg is not None:
                obs_clip = getattr(env_cfg, "obs_clip", None)
        if obs_clip_override is not None:
            obs_clip = obs_clip_override

        logger.trace("wrapping with GymnasiumTradingEnvWrapper obs_clip={}", obs_clip)
        gym_env = GymnasiumTradingEnvWrapper(env, obs_clip=obs_clip)

        logger.trace("wrapping with GymWrapper and transforms")
        wrapped_env = GymWrapper(gym_env)
        wrapped_env = TransformedEnv(
            wrapped_env,
            RenameTransform(in_keys=["CustomFeature"], out_keys=["observation"]),
        )
        wrapped_env = self._wrap_with_step_counter(wrapped_env)
        logger.trace("env wrapping done")

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

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}

    def __init__(
        self,
        memmap_paths: list[MemmapPaths],
        episode_length: int,
        feature_columns: list[str],
        price_column: str,
        initial_cash: float = DEFAULT_INITIAL_PORTFOLIO_VALUE,
        fee: float = 0.0,
        reward_type: str = RewardType.LOG_RETURN,
        reward_eta: float = 0.01,
        reward_scale: float = 1.0,
        runtime_feature_columns: list[str] | None = None,
        obs_clip: float | None = None,
        seed: int | None = None,
        dsr_persist_across_symbols: bool = False,
        action_penalty_lambda: float = 0.0,
        action_penalty_type: str | ActionPenaltyType = ActionPenaltyType.QUADRATIC,
    ) -> None:
        if not memmap_paths:
            raise ValueError("memmap_paths must contain at least one entry")

        self._memmap_paths = memmap_paths
        self._episode_length = episode_length
        self._symbol_queue: list[int] = []
        self._symbol_rng = np.random.default_rng(seed)
        self._feature_columns = feature_columns
        self._price_column = price_column
        self._initial_cash = initial_cash
        self._fee = fee
        self._reward_type = reward_type
        self._reward_eta = reward_eta
        self._reward_scale = reward_scale
        self._runtime_feature_columns = runtime_feature_columns or []
        self._obs_clip = obs_clip
        self._dsr_persist_across_symbols = dsr_persist_across_symbols

        self._action_penalty_lambda = float(action_penalty_lambda)
        self._action_penalty_type = ActionPenaltyType(action_penalty_type)
        self._prev_action: float = 0.0

        stocks = [Stock(price_column)]
        self.action_space = BoxPortfolio(stocks, low=-1.0, high=1.0)
        n_obs = len(feature_columns) + len(self._runtime_feature_columns)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32
        )
        self._inner_env: TradingEnv | None = None
        # NLV at the end of the most recently completed episode, snapshotted
        # before reset() replaces _inner_env.  Used by _log_episode_stats to
        # report per-episode portfolio value without reading a partial broker.
        self._last_episode_final_nlv: float | None = None
        self._last_episode_steps: int | None = None
        # Symbol name and timestamp range of the episode that just started.
        # Snapshotted on reset() so _log_episode_stats can report them.
        self._current_episode_symbol: str | None = None
        self._current_episode_start_ts: str | None = None
        self._current_episode_end_ts: str | None = None

        # DSR state management.
        # dsr_persist_across_symbols=False (default): one DifferentialSharpeRatio
        #   per symbol so A_t/B_t only accumulate within a single asset's episodes.
        # dsr_persist_across_symbols=True (legacy): a single shared object whose
        #   moments carry over even when the symbol changes between episodes.
        _is_dsr = RewardType(reward_type) == RewardType.DIFFERENTIAL_SHARPE
        if _is_dsr and dsr_persist_across_symbols:
            self._persistent_dsr: DifferentialSharpeRatio | None = (
                DifferentialSharpeRatio(eta=reward_eta, scale=reward_scale)
            )
            self._dsr_per_symbol: dict[str, DifferentialSharpeRatio] = {}
        elif _is_dsr:
            self._persistent_dsr = None
            self._dsr_per_symbol = {}
        else:
            self._persistent_dsr = None
            self._dsr_per_symbol = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_reward(self, symbol: str = ""):
        if self._reward_type == RewardType.DIFFERENTIAL_SHARPE:
            if self._dsr_persist_across_symbols:
                # Legacy: single shared object, moments persist across all symbols.
                return self._persistent_dsr
            # Default: per-symbol object so A_t/B_t only reflect that asset.
            # Created on first encounter; moments cleared only at _prev_nlv reset.
            if symbol not in self._dsr_per_symbol:
                self._dsr_per_symbol[symbol] = DifferentialSharpeRatio(
                    eta=self._reward_eta, scale=self._reward_scale
                )
            return self._dsr_per_symbol[symbol]
        from trading_rl.rewards.registry import RewardRegistry
        return RewardRegistry.create(
            self._reward_type, eta=self._reward_eta, scale=self._reward_scale
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
        except (ValueError, OverflowError) as e:
            logger.warning(
                "Could not parse timestamps dtype={} error={}, using RangeIndex",
                window_index.dtype, e,
            )
            index = pd.RangeIndex(len(window_data))
        return pd.DataFrame(window_data, columns=mp.columns, index=index)

    def _build_inner_env(self, window_df: pd.DataFrame, symbol: str = "", reward: Any = None) -> TradingEnv:
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
            reward=reward or self._make_reward(symbol),
            prices=prices,
            initial_cash=self._initial_cash,
            broker_fees=BrokerFees(proportional=self._fee, fixed=0.0),
        )

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def _next_symbol_idx(self) -> int:
        if not self._symbol_queue:
            order = list(range(len(self._memmap_paths)))
            self._symbol_rng.shuffle(order)
            self._symbol_queue = order
        return self._symbol_queue.pop()

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._symbol_rng = np.random.default_rng(seed)
            self._symbol_queue = []
        # Snapshot the final NLV of the episode that just ended before we
        # replace _inner_env.  _log_episode_stats reads this attribute so it
        # always sees the completed episode's value, not the new episode's.
        if self._inner_env is not None:
            broker = self._inner_env.broker
            if broker is not None and hasattr(broker, "track_record") and broker.track_record:
                last_record = broker.track_record[-1]
                if hasattr(last_record, "context_post") and hasattr(last_record.context_post, "nlv"):
                    self._last_episode_final_nlv = float(last_record.context_post.nlv)
                    self._last_episode_steps = len(broker.track_record)
        if self._dsr_persist_across_symbols and self._persistent_dsr is not None:
            # Legacy mode: shared object — clear only _prev_nlv.
            self._persistent_dsr.reset(persist_moments=True)
        attempts = 0
        while True:
            file_idx = self._next_symbol_idx()
            mp = self._memmap_paths[file_idx]
            if mp.n_rows >= self._episode_length:
                break
            logger.warning(
                "StreamingTradingEnvXY: skipping symbol {} ({} rows < episode_length {})",
                file_idx,
                mp.n_rows,
                self._episode_length,
            )
            attempts += 1
            if attempts >= len(self._memmap_paths):
                raise RuntimeError(
                    f"No symbol files have enough rows for episode_length={self._episode_length}. "
                    "Reduce episode_length or provide longer data files."
                )
        max_start = mp.n_rows - self._episode_length
        # Use max_start + 1 to include final valid start position (numpy.integers is exclusive)
        start = int(self._symbol_rng.integers(0, max_start + 1))
        window_df = self._load_window(file_idx, start)
        # Snapshot symbol and date range for the new episode.
        self._current_episode_symbol = mp.symbol or Path(mp.data_path).stem.split("_")[0]
        if isinstance(window_df.index, pd.DatetimeIndex) and len(window_df) > 0:
            fmt = "%Y-%m-%dT%H:%M:%S"
            self._current_episode_start_ts = window_df.index[0].strftime(fmt)
            self._current_episode_end_ts = window_df.index[-1].strftime(fmt)
        else:
            self._current_episode_start_ts = None
            self._current_episode_end_ts = None
        # Per-symbol mode: clear _prev_nlv on the symbol's DSR object so the
        # new episode starts with a fresh NLV reference while moments persist.
        _dsr_to_restore: object | None = None
        _saved_moments: tuple[float, float] | None = None
        if not self._dsr_persist_across_symbols and self._reward_type == RewardType.DIFFERENTIAL_SHARPE:
            sym = self._current_episode_symbol
            if sym in self._dsr_per_symbol:
                dsr = self._dsr_per_symbol[sym]
                _saved_moments = (dsr.A_t, dsr.B_t)
                dsr.reset(persist_moments=True)  # clear only _prev_nlv
                _dsr_to_restore = dsr
        elif self._dsr_persist_across_symbols and self._persistent_dsr is not None:
            dsr = self._persistent_dsr
            _saved_moments = (dsr.A_t, dsr.B_t)
            dsr.reset(persist_moments=True)  # clear only _prev_nlv
            _dsr_to_restore = dsr
        self._inner_env = self._build_inner_env(window_df, symbol=self._current_episode_symbol, reward=_dsr_to_restore)
        obs = self._inner_env.reset()
        # TradingEnv.reset() calls self._reward.reset() with no arguments, which
        # resets A_t/B_t to 0 regardless of persist_moments. Restore the saved
        # moments explicitly so cross-episode EMA continuity is preserved.
        if _dsr_to_restore is not None and _saved_moments is not None:
            _dsr_to_restore.A_t, _dsr_to_restore.B_t = _saved_moments
        self._prev_action = 0.0
        return self._extract_obs(obs), {}

    def _compute_action_penalty(self, action: float) -> float:
        lam = self._action_penalty_lambda
        if lam == 0.0:
            return 0.0
        if self._action_penalty_type == ActionPenaltyType.QUADRATIC:
            return lam * float(action) ** 2
        if self._action_penalty_type == ActionPenaltyType.ABSOLUTE:
            return lam * abs(float(action))
        # CHANGE_QUADRATIC
        return lam * (float(action) - self._prev_action) ** 2

    def step(self, action):
        try:
            obs, reward, done, info = self._inner_env.step(action)
        except EndOfEpisodeError:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            self._prev_action = 0.0
            return obs, -1.0, True, False, {"bankrupt": True}
        # BoxPortfolio.contains() checks shape, so pass the array to the inner
        # env as-is; extract a scalar only for penalty/tracking.
        action_val = float(np.asarray(action).flat[0])
        penalty = self._compute_action_penalty(action_val)
        reward = float(reward) - penalty
        self._prev_action = action_val
        # Distinguish bankruptcy (terminated) from data exhaustion (truncated).
        if done:
            broker = getattr(self._inner_env, "broker", None)
            is_bankrupt = broker is not None and broker.net_liquidation_value() <= 0
            terminated = is_bankrupt
            truncated = not is_bankrupt
        else:
            terminated = False
            truncated = False
        return self._extract_obs(obs), reward, terminated, truncated, info

    def _extract_obs(self, obs) -> np.ndarray:
        # TradingEnv returns {"CustomFeature": array}; unwrap to flat array
        # to match the Box observation_space declared in __init__.
        if isinstance(obs, dict):
            obs = obs["CustomFeature"]
        if self._obs_clip is not None:
            obs = np.clip(obs, -self._obs_clip, self._obs_clip)
        return obs

    @property
    def broker(self):
        if self._inner_env is None:
            return None
        return self._inner_env.broker

    def render(self):
        if self._inner_env is not None:
            return self._inner_env.render()

    def close(self):
        if self._inner_env is not None:
            self._inner_env.close()


# Alias for backward compatibility
TradingEnvXYWrapper = TradingEnvXYFactory
