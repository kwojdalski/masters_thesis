"""Environment builder abstractions used by training scripts."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd
from torchrl.envs import TransformedEnv

from logger import get_logger
from trading_rl.config import ExperimentConfig
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
            if explicit_backend and explicit_backend not in {"gym_trading_env.continuous"}:
                raise ValueError(
                    f"{algorithm} requires a continuous backend ('gym_trading_env.continuous'), "
                    f"but config.env.backend is '{explicit_backend}'. "
                    "Please set env.backend to 'gym_trading_env.continuous' or switch algorithm."
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
