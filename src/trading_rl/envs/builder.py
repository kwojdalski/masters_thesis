"""Environment builder abstractions used by training scripts."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gymnasium as gym
import pandas as pd
from torchrl.envs import GymWrapper, TransformedEnv
from torchrl.envs.transforms import StepCounter

from logger import get_logger
from trading_rl.config import DEFAULT_INITIAL_PORTFOLIO_VALUE, ExperimentConfig
from trading_rl.constants import Algorithm, EnvBackend, RewardType
from trading_rl.data_loading import MemmapPaths, load_memmap_paths


@dataclass(frozen=True)
class CommonEnvParams:
    """Environment parameters shared by all backends."""

    env_name: str
    positions: list[int]
    trading_fees: float
    borrow_interest_rate: float
    reward_type: str
    reward_eta: float
    reward_scale: float
    seed: int | None = None
    backend: str | None = None


@dataclass(frozen=True)
class TradingEnvParams:
    """Parameters specific to TradingEnv (TradingEnvXY and StreamingTradingEnvXY)."""

    feature_columns: list[str] | None = None
    price_column: str = "close"
    initial_portfolio_value: float = DEFAULT_INITIAL_PORTFOLIO_VALUE
    include_position_feature: bool = False
    obs_clip: float | None = 5.0
    action_penalty_lambda: float = 0.0
    action_penalty_type: str = "quadratic"
    bid_column: str | None = None
    ask_column: str | None = None


@dataclass(frozen=True)
class GymTradingEnvParams:
    """Parameters specific to gym_trading_env backends (discrete and continuous)."""

    continuous_action_thresholds: list[float] = field(default_factory=lambda: [-0.33, 0.33])
    device: str = "cpu"


@dataclass(frozen=True)
class StreamingEnvParams:
    """Parameters specific to streaming (memmap) environments."""

    memmap_dir: str | None = None
    streaming_episode_length: int = 10_000


@dataclass(frozen=True)
class EnvBuildParams:
    """Composite parameter object for environment construction.

    Composed of focused parameter objects by backend. Decouples the builder
    from ExperimentConfig while keeping related parameters grouped by their
    usage domain. Construct via ``EnvBuildParams.from_config(config)`` or
    supply fields directly in tests.
    """

    common: CommonEnvParams
    algorithm: str
    trading_env: TradingEnvParams = field(default_factory=TradingEnvParams)
    gym_trading: GymTradingEnvParams = field(default_factory=GymTradingEnvParams)
    streaming: StreamingEnvParams = field(default_factory=StreamingEnvParams)

    @classmethod
    def from_config(cls, config: ExperimentConfig) -> "EnvBuildParams":
        """Extract environment build parameters from a full ExperimentConfig."""
        env = config.env
        common = CommonEnvParams(
            env_name=env.name,
            positions=env.positions,
            trading_fees=env.trading_fees,
            borrow_interest_rate=env.borrow_interest_rate,
            reward_type=getattr(env, "reward_type", RewardType.LOG_RETURN),
            reward_eta=getattr(env, "reward_eta", 0.01),
            reward_scale=getattr(env, "reward_scale", 1.0),
            seed=getattr(config, "seed", None),
            backend=getattr(env, "backend", None),
        )
        trading_env = TradingEnvParams(
            feature_columns=getattr(env, "feature_columns", None),
            price_column=getattr(env, "price_column", None) or "close",
            initial_portfolio_value=getattr(env, "initial_portfolio_value", DEFAULT_INITIAL_PORTFOLIO_VALUE),
            include_position_feature=getattr(env, "include_position_feature", False),
            obs_clip=getattr(env, "obs_clip", 5.0),
            action_penalty_lambda=getattr(env, "action_penalty_lambda", 0.0),
            action_penalty_type=getattr(env, "action_penalty_type", "quadratic"),
            bid_column=getattr(env, "bid_column", None),
            ask_column=getattr(env, "ask_column", None),
        )
        gym_trading = GymTradingEnvParams(
            continuous_action_thresholds=getattr(env, "continuous_action_thresholds", [-0.33, 0.33]),
            device=getattr(config.training, "device", "cpu"),
        )
        streaming = StreamingEnvParams(
            memmap_dir=getattr(getattr(config, "data", None), "memmap_dir", None),
            streaming_episode_length=getattr(env, "streaming_episode_length", 10_000),
        )
        return cls(
            common=common,
            algorithm=getattr(config.training, "algorithm", Algorithm.PPO),
            trading_env=trading_env,
            gym_trading=gym_trading,
            streaming=streaming,
        )


@dataclass
class BaseEnvironmentBuilder(ABC):
    """Base class for environment builders."""

    logger: Any = field(
        default_factory=lambda: get_logger(__name__), init=False, repr=False
    )

    @abstractmethod
    def create(self, df: pd.DataFrame, params: EnvBuildParams, *, use_memmap: bool = True) -> TransformedEnv:
        """Create environment instance for given data and params."""


class AlgorithmicEnvironmentBuilder(BaseEnvironmentBuilder):
    """Backend-aware environment builder that also respects algorithm defaults."""

    def __init__(
        self,
        default_backend: str = EnvBackend.GYM_TRADING_DISCRETE,
    ):
        super().__init__()
        self.default_backend = EnvBackend(default_backend)

    def _resolve_backend(self, params: EnvBuildParams) -> str:
        """Determine backend from params, falling back to algorithm defaults."""
        explicit_backend = (
            EnvBackend(params.common.backend) if params.common.backend else None
        )
        algorithm_display = params.algorithm  # preserve original case for error messages
        algorithm = str(params.algorithm).upper()

        if algorithm in {Algorithm.TD3, Algorithm.DDPG}:
            if explicit_backend and explicit_backend not in {
                EnvBackend.GYM_TRADING_CONTINUOUS,
                EnvBackend.TRADINGENV,
            }:
                raise ValueError(
                    f"{algorithm_display} requires a continuous backend ('gym_trading_env.continuous' or 'tradingenv'), "
                    f"but params.backend is '{explicit_backend}'. "
                    "Please set env.backend to 'gym_trading_env.continuous' or 'tradingenv', or switch algorithm."
                )
            algo_backend: str = EnvBackend.GYM_TRADING_CONTINUOUS
        else:
            algo_backend = EnvBackend.GYM_TRADING_DISCRETE

        backend = explicit_backend or algo_backend or self.default_backend
        self.logger.debug(
            "resolved backend={} explicit_backend={} algorithm={}",
            backend, explicit_backend, algorithm,
        )
        return backend

    def create(
        self,
        df: pd.DataFrame,
        params: EnvBuildParams,
        *,
        use_memmap: bool = True,
    ) -> TransformedEnv:
        """Create environment using resolved backend and provided params."""
        memmap_paths = self._resolve_memmap_paths(params) if use_memmap else None
        if memmap_paths:
            env = self._create_streaming_env(memmap_paths, params)
            self.logger.info(
                "created StreamingTradingEnv n_symbols={} episode_length={}",
                len(memmap_paths),
                params.streaming.streaming_episode_length,
            )
            return env

        backend = self._resolve_backend(params)
        env = self._create_non_streaming_env(df, params, backend)
        self.logger.info(
            "created environment backend={} positions={} trading_fees={}",
            backend, params.common.positions, params.common.trading_fees,
        )
        return env

    def _resolve_memmap_paths(self, params: EnvBuildParams) -> list[MemmapPaths] | None:
        """Return per-symbol MemmapPaths if memmap_dir is configured and populated."""
        if not params.streaming.memmap_dir:
            return None
        p = Path(params.streaming.memmap_dir)
        if not p.exists():
            return None
        paths = load_memmap_paths(p)
        return paths if paths else None

    def _create_non_streaming_env(
        self,
        df: pd.DataFrame,
        params: EnvBuildParams,
        backend: str,
    ) -> TransformedEnv:
        """Create a non-streaming (in-memory DataFrame) environment from params."""
        from trading_rl.continuous_action_wrapper import ContinuousToDiscreteAction
        from trading_rl.rewards import reward_function

        if backend == EnvBackend.TRADINGENV:
            from trading_rl.envs.tradingenvxy_wrapper import TradingEnvXYFactory
            factory = TradingEnvXYFactory()
            return factory.make(
                df=df,
                feature_columns=params.trading_env.feature_columns,
                price_column=params.trading_env.price_column,
                cash=params.trading_env.initial_portfolio_value,
                fee=params.common.trading_fees,
                reward_type=params.common.reward_type,
                reward_eta=params.common.reward_eta,
                reward_scale=params.common.reward_scale,
                include_position_feature=params.trading_env.include_position_feature,
                obs_clip=params.trading_env.obs_clip,
            )

        _ANYTRADING_ENV_IDS = {
            EnvBackend.GYM_ANYTRADING_FOREX: "forex-v0",
            EnvBackend.GYM_ANYTRADING_STOCKS: "stocks-v0",
        }
        if backend in _ANYTRADING_ENV_IDS:
            return self._create_anytrading_env(df, params, _ANYTRADING_ENV_IDS[backend])

        _GYM_TRADING_BACKENDS = {
            EnvBackend.GYM_TRADING_DISCRETE,
            EnvBackend.GYM_TRADING_CONTINUOUS,
        }
        if backend not in _GYM_TRADING_BACKENDS:
            raise ValueError(f"Unsupported backend: '{backend}'")

        continuous = backend == EnvBackend.GYM_TRADING_CONTINUOUS
        base_env = gym.make(
            "TradingEnv",
            name=params.common.env_name,
            df=df,
            positions=params.common.positions,
            trading_fees=params.common.trading_fees,
            borrow_interest_rate=params.common.borrow_interest_rate,
            reward_function=reward_function,
        )
        env = GymWrapper(base_env)

        if continuous:
            env = TransformedEnv(
                env,
                ContinuousToDiscreteAction(
                    discrete_actions=params.common.positions,
                    thresholds=params.gym_trading.continuous_action_thresholds,
                    device=params.gym_trading.device,
                ),
            )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*auto_unwrap_transformed_env.*")
            return TransformedEnv(env, StepCounter())

    def _create_anytrading_env(
        self,
        df: pd.DataFrame,
        params: EnvBuildParams,
        env_id: str,
    ) -> TransformedEnv:
        """Create a gym-anytrading (forex-v0 / stocks-v0) environment from params."""
        from trading_rl.envs.trading_envs import DiscreteActionWrapper

        rename_map = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        base_env = gym.make(env_id, df=df)

        reward_type = str(params.common.reward_type)
        from trading_rl.constants import RewardType
        if reward_type != RewardType.LOG_RETURN:
            from trading_rl.rewards.dsr_wrapper import DifferentialSharpeRatioAnyTrading, StatefulRewardWrapper
            dsr = DifferentialSharpeRatioAnyTrading(eta=params.common.reward_eta, scale=params.common.reward_scale)
            base_env = StatefulRewardWrapper(base_env, reward_fn=dsr)

        base_env = DiscreteActionWrapper(base_env)
        env = GymWrapper(base_env)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*auto_unwrap_transformed_env.*")
            return TransformedEnv(env, StepCounter())

    def _create_streaming_env(
        self,
        memmap_paths: list[MemmapPaths],
        params: EnvBuildParams,
    ) -> TransformedEnv:
        from trading_rl.continuous_action_wrapper import ContinuousToDiscreteAction
        from trading_rl.rewards import reward_function
        from trading_rl.envs.streaming_env import StreamingTradingEnv

        backend = self._resolve_backend(params)
        episode_length = params.streaming.streaming_episode_length

        if backend == EnvBackend.TRADINGENV:
            return self._create_streaming_tradingenv(memmap_paths, episode_length, params)

        _GYM_TRADING_BACKENDS = {
            EnvBackend.GYM_TRADING_DISCRETE,
            EnvBackend.GYM_TRADING_CONTINUOUS,
        }
        if backend not in _GYM_TRADING_BACKENDS:
            raise ValueError(
                f"memmap streaming is not supported for backend '{backend}'. "
                "Supported: gym_trading_env.discrete, gym_trading_env.continuous, tradingenv."
            )

        continuous = backend == EnvBackend.GYM_TRADING_CONTINUOUS

        base_env = StreamingTradingEnv(
            memmap_paths=memmap_paths,
            episode_length=episode_length,
            seed=params.common.seed,
            name=params.common.env_name,
            positions=params.common.positions,
            trading_fees=params.common.trading_fees,
            borrow_interest_rate=params.common.borrow_interest_rate,
            reward_function=reward_function,
        )
        env = GymWrapper(base_env)

        if continuous:
            env = TransformedEnv(
                env,
                ContinuousToDiscreteAction(
                    discrete_actions=params.common.positions,
                    thresholds=params.gym_trading.continuous_action_thresholds,
                    device=params.gym_trading.device,
                ),
            )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*auto_unwrap_transformed_env.*")
            return TransformedEnv(env, StepCounter())

    def _create_streaming_tradingenv(
        self,
        memmap_paths: list[MemmapPaths],
        episode_length: int,
        params: EnvBuildParams,
    ) -> TransformedEnv:
        from trading_rl.envs.tradingenvxy_wrapper import StreamingTradingEnvXY

        feature_columns = params.trading_env.feature_columns
        if not feature_columns:
            feature_columns = [c for c in memmap_paths[0].columns if c.startswith("feature_")]

        runtime_cols = ["feature_position"] if params.trading_env.include_position_feature else []
        # feature_position lives in the env at runtime, not in the memmap data
        static_feature_columns = [c for c in feature_columns if c not in runtime_cols]

        base_env = StreamingTradingEnvXY(
            memmap_paths=memmap_paths,
            episode_length=episode_length,
            feature_columns=static_feature_columns,
            price_column=params.trading_env.price_column,
            initial_cash=params.trading_env.initial_portfolio_value,
            fee=params.common.trading_fees,
            reward_type=params.common.reward_type,
            reward_eta=params.common.reward_eta,
            reward_scale=params.common.reward_scale,
            runtime_feature_columns=runtime_cols,
            obs_clip=params.trading_env.obs_clip,
            seed=params.common.seed,
            action_penalty_lambda=params.trading_env.action_penalty_lambda,
            action_penalty_type=params.trading_env.action_penalty_type,
            bid_column=params.trading_env.bid_column,
            ask_column=params.trading_env.ask_column,
        )
        env = GymWrapper(base_env)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*auto_unwrap_transformed_env.*")
            return TransformedEnv(env, StepCounter())
