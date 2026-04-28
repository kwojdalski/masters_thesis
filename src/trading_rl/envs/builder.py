"""Environment builder abstractions used by training scripts."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from torchrl.envs import GymWrapper, TransformedEnv

from logger import get_logger
from trading_rl.config import DEFAULT_INITIAL_PORTFOLIO_VALUE, ExperimentConfig
from trading_rl.data_loading import MemmapPaths, load_memmap_paths
from trading_rl.envs.trading_envs import Backend, create_environment as build_backend_env


@dataclass
class BaseEnvironmentBuilder:
    """Base class for environment builders."""

    logger: logging.Logger = field(
        default_factory=lambda: get_logger(__name__), init=False, repr=False
    )

    def create(self, df: pd.DataFrame, config: ExperimentConfig) -> TransformedEnv:
        """Create environment instance for given data and config."""
        raise NotImplementedError


class AlgorithmicEnvironmentBuilder(BaseEnvironmentBuilder):
    """Backend-aware environment builder that also respects algorithm defaults."""

    def __init__(self, default_backend: Backend = "gym_trading_env.discrete"):
        super().__init__()
        self.default_backend = default_backend

    def _resolve_backend(self, config: ExperimentConfig) -> Backend:
        """Determine backend from config, falling back to algorithm defaults."""
        explicit_backend = getattr(getattr(config, "env", None), "backend", None)
        algorithm = getattr(getattr(config, "training", None), "algorithm", "PPO")
        algo_backend: Backend | None

        # TD3/DDPG require continuous action space; enforce compatible backend
        if str(algorithm).upper() in {"TD3", "DDPG"}:
            if explicit_backend and explicit_backend not in {"gym_trading_env.continuous", "tradingenv"}:
                raise ValueError(
                    f"{algorithm} requires a continuous backend ('gym_trading_env.continuous' or 'tradingenv'), "
                    f"but config.env.backend is '{explicit_backend}'. "
                    "Please set env.backend to 'gym_trading_env.continuous' or 'tradingenv', or switch algorithm."
                )
            algo_backend = "gym_trading_env.continuous"
        else:
            algo_backend = "gym_trading_env.discrete"

        backend: Backend = explicit_backend or algo_backend or self.default_backend
        self.logger.debug(
            "Resolved backend",
            extra={
                "backend": backend,
                "explicit_backend": explicit_backend,
                "algorithm": algorithm,
            },
        )
        return backend

    def create(self, df: pd.DataFrame, config: ExperimentConfig) -> TransformedEnv:
        """Create environment using resolved backend and provided config."""
        memmap_paths = self._resolve_memmap_paths(config)
        if memmap_paths:
            env = self._create_streaming_env(memmap_paths, config)
            self.logger.info(
                "Created StreamingTradingEnv",
                extra={
                    "n_symbols": len(memmap_paths),
                    "episode_length": getattr(config.env, "streaming_episode_length", 10_000),
                },
            )
            return env

        backend = self._resolve_backend(config)
        env = build_backend_env(df=df, config=config, backend=backend)
        self.logger.info(
            "Created environment",
            extra={
                "backend": backend,
                "positions": config.env.positions,
                "trading_fees": config.env.trading_fees,
            },
        )
        return env

    def _resolve_memmap_paths(self, config: ExperimentConfig) -> list[MemmapPaths] | None:
        """Return per-symbol MemmapPaths if memmap_dir is configured and populated."""
        memmap_dir = getattr(getattr(config, "data", None), "memmap_dir", None)
        if not memmap_dir:
            return None
        p = Path(memmap_dir)
        if not p.exists():
            return None
        paths = load_memmap_paths(p)
        return paths if paths else None

    def _create_streaming_env(
        self,
        memmap_paths: list[MemmapPaths],
        config: ExperimentConfig,
    ) -> TransformedEnv:
        import warnings
        from trading_rl.continuous_action_wrapper import ContinuousToDiscreteAction
        from trading_rl.rewards import reward_function
        from trading_rl.envs.streaming_env import StreamingTradingEnv
        from torchrl.envs.transforms import StepCounter

        backend = self._resolve_backend(config)
        episode_length = getattr(config.env, "streaming_episode_length", 10_000)

        if backend == "tradingenv":
            return self._create_streaming_tradingenv(memmap_paths, episode_length, config)

        _GYM_TRADING_BACKENDS = {"gym_trading_env.discrete", "gym_trading_env.continuous"}
        if backend not in _GYM_TRADING_BACKENDS:
            raise ValueError(
                f"memmap streaming is not supported for backend '{backend}'. "
                "Supported: gym_trading_env.discrete, gym_trading_env.continuous, tradingenv."
            )

        continuous = backend == "gym_trading_env.continuous"

        base_env = StreamingTradingEnv(
            memmap_paths=memmap_paths,
            episode_length=episode_length,
            name=config.env.name,
            positions=config.env.positions,
            trading_fees=config.env.trading_fees,
            borrow_interest_rate=config.env.borrow_interest_rate,
            reward_function=reward_function,
        )
        env = GymWrapper(base_env)

        if continuous:
            env = TransformedEnv(
                env,
                ContinuousToDiscreteAction(
                    discrete_actions=config.env.positions,
                    thresholds=getattr(config.env, "continuous_action_thresholds", [-0.33, 0.33]),
                    device=getattr(config.training, "device", "cpu"),
                ),
            )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*auto_unwrap_transformed_env.*")
            return TransformedEnv(env, StepCounter())

    def _create_streaming_tradingenv(
        self,
        memmap_paths: list[MemmapPaths],
        episode_length: int,
        config: ExperimentConfig,
    ) -> TransformedEnv:
        import warnings
        from trading_rl.envs.tradingenvxy_wrapper import StreamingTradingEnvXY
        from torchrl.envs.transforms import RenameTransform, StepCounter

        feature_columns = getattr(config.env, "feature_columns", None)
        if not feature_columns:
            feature_columns = [c for c in memmap_paths[0].columns if c.startswith("feature_")]

        price_column = getattr(config.env, "price_column", None) or "close"
        initial_cash = getattr(config.env, "initial_portfolio_value", DEFAULT_INITIAL_PORTFOLIO_VALUE)
        fee = getattr(config.env, "trading_fees", 0.0)
        reward_type = getattr(config.env, "reward_type", "log_return")
        reward_eta = getattr(config.env, "reward_eta", 0.01)
        include_position = getattr(config.env, "include_position_feature", False)
        runtime_cols = ["feature_position"] if include_position else []

        base_env = StreamingTradingEnvXY(
            memmap_paths=memmap_paths,
            episode_length=episode_length,
            feature_columns=feature_columns,
            price_column=price_column,
            initial_cash=initial_cash,
            fee=fee,
            reward_type=reward_type,
            reward_eta=reward_eta,
            runtime_feature_columns=runtime_cols,
        )
        env = GymWrapper(base_env)
        env = TransformedEnv(
            env,
            RenameTransform(in_keys=["CustomFeature"], out_keys=["observation"]),
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*auto_unwrap_transformed_env.*")
            return TransformedEnv(env, StepCounter())
