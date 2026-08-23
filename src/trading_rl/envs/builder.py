"""Environment builder abstractions used by training scripts."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
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
    execution_price: str = "mid"
    bid_column: str = "bid_px_00"
    ask_column: str = "ask_px_00"


@dataclass(frozen=True)
class GymTradingEnvParams:
    """Parameters specific to gym_trading_env backends (discrete and continuous)."""

    continuous_action_thresholds: list[float] = field(
        default_factory=lambda: [-0.33, 0.33]
    )
    device: str = "cpu"


@dataclass(frozen=True)
class StreamingEnvParams:
    """Parameters specific to streaming (memmap) environments."""

    memmap_dir: str | None = None
    streaming_episode_length: int = 10_000
    obs_latency_ticks: int = 0
    exec_latency_ticks: int = 0
    obs_latency_us: float = 0.0
    exec_latency_us: float = 0.0


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
    def from_config(cls, config: ExperimentConfig) -> EnvBuildParams:
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
            initial_portfolio_value=getattr(
                env, "initial_portfolio_value", DEFAULT_INITIAL_PORTFOLIO_VALUE
            ),
            include_position_feature=getattr(env, "include_position_feature", False),
            obs_clip=getattr(env, "obs_clip", 5.0),
            action_penalty_lambda=getattr(env, "action_penalty_lambda", 0.0),
            action_penalty_type=getattr(env, "action_penalty_type", "quadratic"),
            execution_price=getattr(env, "execution_price", "mid"),
            bid_column=getattr(env, "bid_column", "bid_px_00"),
            ask_column=getattr(env, "ask_column", "ask_px_00"),
        )
        gym_trading = GymTradingEnvParams(
            continuous_action_thresholds=getattr(
                env, "continuous_action_thresholds", [-0.33, 0.33]
            ),
            device=getattr(config.training, "device", "cpu"),
        )
        streaming = StreamingEnvParams(
            memmap_dir=getattr(getattr(config, "data", None), "memmap_dir", None),
            streaming_episode_length=getattr(env, "streaming_episode_length", 10_000),
            obs_latency_ticks=getattr(env, "obs_latency_ticks", 0),
            exec_latency_ticks=getattr(env, "exec_latency_ticks", 0),
            obs_latency_us=getattr(env, "obs_latency_us", 0.0),
            exec_latency_us=getattr(env, "exec_latency_us", 0.0),
        )
        return cls(
            common=common,
            algorithm=getattr(config.training, "algorithm", Algorithm.PPO),
            trading_env=trading_env,
            gym_trading=gym_trading,
            streaming=streaming,
        )


@dataclass(frozen=True)
class BackendResolutionPolicy:
    """Resolve environment backend defaults and algorithm backend constraints."""

    default_backend: str = EnvBackend.GYM_TRADING_DISCRETE
    algorithm_defaults: Mapping[str, str] = field(
        default_factory=lambda: {
            Algorithm.TD3: EnvBackend.GYM_TRADING_CONTINUOUS,
            Algorithm.DDPG: EnvBackend.GYM_TRADING_CONTINUOUS,
        }
    )
    allowed_backends: Mapping[str, frozenset[str]] = field(
        default_factory=lambda: {
            Algorithm.TD3: frozenset(
                {EnvBackend.GYM_TRADING_CONTINUOUS, EnvBackend.TRADINGENV}
            ),
            Algorithm.DDPG: frozenset(
                {EnvBackend.GYM_TRADING_CONTINUOUS, EnvBackend.TRADINGENV}
            ),
        }
    )

    def resolve(self, params: EnvBuildParams) -> str:
        """Return the backend selected for params."""
        explicit_backend = (
            EnvBackend(params.common.backend) if params.common.backend else None
        )
        algorithm = str(params.algorithm).upper()
        allowed_backends = self.allowed_backends.get(algorithm)

        if explicit_backend is not None:
            if allowed_backends and explicit_backend not in allowed_backends:
                allowed = ", ".join(
                    sorted(str(backend) for backend in allowed_backends)
                )
                raise ValueError(
                    f"{params.algorithm} requires one of these backends: {allowed}; "
                    f"got '{explicit_backend}'."
                )
            return explicit_backend

        backend = self.algorithm_defaults.get(algorithm, self.default_backend)
        return EnvBackend(backend)


@dataclass
class BaseEnvironmentBuilder(ABC):
    """Base class for environment builders."""

    logger: Any = field(
        default_factory=lambda: get_logger(__name__), init=False, repr=False
    )

    @abstractmethod
    def create(
        self, df: pd.DataFrame, params: EnvBuildParams, *, use_memmap: bool = True
    ) -> TransformedEnv:
        """Create environment instance for given data and params."""


class AlgorithmicEnvironmentBuilder(BaseEnvironmentBuilder):
    """Backend-aware environment builder."""

    def __init__(
        self,
        default_backend: str = EnvBackend.GYM_TRADING_DISCRETE,
        backend_policy: BackendResolutionPolicy | None = None,
    ):
        super().__init__()
        self.default_backend = EnvBackend(default_backend)
        self.backend_policy = backend_policy or BackendResolutionPolicy(
            default_backend=self.default_backend
        )

    def _resolve_backend(self, params: EnvBuildParams) -> str:
        """Determine backend from explicit params and the configured policy."""
        explicit_backend = params.common.backend
        backend = self.backend_policy.resolve(params)
        self.logger.debug(
            "resolved backend={} explicit_backend={} algorithm={}",
            backend,
            explicit_backend,
            str(params.algorithm).upper(),
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
            backend,
            params.common.positions,
            params.common.trading_fees,
        )
        return env

    def _resolve_history_reward_function(
        self, params: EnvBuildParams
    ) -> Callable[[Any], float]:
        """Resolve the ``(history) -> float`` reward callable for gym_trading_env backends.

        Single source of reward_type dispatch for this backend family, replacing
        two previously-duplicated hardcoded ``reward_function`` imports (one in
        ``_create_non_streaming_env``, one in ``_create_streaming_env``) that
        silently ignored ``reward_type`` and always used log_return.

        Only log_return is supported here: differential_sharpe cannot reuse the
        ``StatefulRewardWrapper`` pattern used for gym_anytrading in
        ``_create_anytrading_env`` -- ``gym_trading_env.TradingEnv`` already
        invokes ``reward_function(history)`` internally once per step, so
        wrapping it again would call the stateful DSR object twice per step,
        double-updating its EMA state. Use the tradingenv backend for DSR.
        """
        reward_type = str(params.common.reward_type)
        if reward_type == RewardType.LOG_RETURN:
            from trading_rl.rewards import reward_function

            return reward_function
        raise ValueError(
            f"reward_type={reward_type!r} is not supported for gym_trading_env "
            f"backends (only {RewardType.LOG_RETURN!r} is). Use the tradingenv "
            "backend for differential_sharpe support."
        )

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
                action_penalty_lambda=params.trading_env.action_penalty_lambda,
                action_penalty_type=params.trading_env.action_penalty_type,
                execution_price=params.trading_env.execution_price,
                bid_column=params.trading_env.bid_column,
                ask_column=params.trading_env.ask_column,
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
            # Flat/neutral start; see streaming branch note on 'random' default.
            initial_position=params.common.positions[0],
            trading_fees=params.common.trading_fees,
            borrow_interest_rate=params.common.borrow_interest_rate,
            reward_function=self._resolve_history_reward_function(params),
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

        rename_map = {
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        base_env = gym.make(env_id, df=df)

        reward_type = str(params.common.reward_type)
        from trading_rl.constants import RewardType

        if reward_type != RewardType.LOG_RETURN:
            from trading_rl.rewards.dsr_wrapper import (
                DifferentialSharpeRatioAnyTrading,
                StatefulRewardWrapper,
            )

            dsr = DifferentialSharpeRatioAnyTrading(
                eta=params.common.reward_eta, scale=params.common.reward_scale
            )
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
        from trading_rl.envs.streaming_env import StreamingTradingEnv

        backend = self._resolve_backend(params)
        episode_length = params.streaming.streaming_episode_length

        if backend == EnvBackend.TRADINGENV:
            return self._create_streaming_tradingenv(
                memmap_paths, episode_length, params
            )

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
            # gym_trading_env defaults to 'random', drawn from *global* np.random
            # (not the seeded self.np_random), giving un-chosen opening exposure
            # and breaking seed reproducibility. Start flat/neutral instead.
            initial_position=params.common.positions[0],
            trading_fees=params.common.trading_fees,
            borrow_interest_rate=params.common.borrow_interest_rate,
            reward_function=self._resolve_history_reward_function(params),
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
        from trading_rl.envs.latency import make_latency_model
        from trading_rl.envs.tradingenvxy_wrapper import StreamingTradingEnvXY

        feature_columns = params.trading_env.feature_columns
        if not feature_columns:
            feature_columns = [
                c for c in memmap_paths[0].columns if c.startswith("feature_")
            ]

        runtime_cols = (
            ["feature_position"] if params.trading_env.include_position_feature else []
        )
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
            execution_price=params.trading_env.execution_price,
            bid_column=params.trading_env.bid_column,
            ask_column=params.trading_env.ask_column,
            obs_latency=make_latency_model(
                params.streaming.obs_latency_ticks, params.streaming.obs_latency_us
            ),
            exec_latency=make_latency_model(
                params.streaming.exec_latency_ticks, params.streaming.exec_latency_us
            ),
        )
        env = GymWrapper(base_env)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*auto_unwrap_transformed_env.*")
            return TransformedEnv(env, StepCounter())
