"""Explicit trainer catalog for algorithm-to-class dispatch."""

from __future__ import annotations

from types import MappingProxyType
from typing import ClassVar

from trading_rl.trainers.ddpg import DDPGTrainer
from trading_rl.trainers.ppo import PPOTrainer, PPOTrainerContinuous
from trading_rl.trainers.random_trainer import RandomTrainer
from trading_rl.trainers.recurrent_ppo import RecurrentPPOTrainer
from trading_rl.trainers.sac import SACTrainer
from trading_rl.trainers.td3 import TD3Trainer


class TrainerRegistry:
    """Maps algorithm names to trainer classes.

    Separate dicts for discrete and continuous variants so PPO can register
    both without collision.  ``get`` prefers the variant that matches
    ``is_continuous``; when only one variant is registered it is returned
    regardless of the flag.
    """

    _discrete: ClassVar[MappingProxyType[str, type]] = MappingProxyType(
        {
            "PPO": PPOTrainer,
            "RANDOM": RandomTrainer,
        }
    )
    _continuous: ClassVar[MappingProxyType[str, type]] = MappingProxyType(
        {
            "DDPG": DDPGTrainer,
            "PPO": PPOTrainerContinuous,
            "RECURRENT_PPO": RecurrentPPOTrainer,
            "SAC": SACTrainer,
            "TD3": TD3Trainer,
        }
    )

    @classmethod
    def get(cls, algorithm: str, is_continuous: bool = False) -> type:
        """Return the trainer class for *algorithm*.

        Args:
            algorithm: Algorithm name (case-insensitive).
            is_continuous: Prefer the continuous-action variant when available.

        Raises:
            ValueError: If *algorithm* is not registered.
        """
        key = algorithm.upper()
        if is_continuous:
            trainer = cls._continuous.get(key) or cls._discrete.get(key)
        else:
            trainer = cls._discrete.get(key) or cls._continuous.get(key)
        if trainer is None:
            available = sorted(set(cls._discrete) | set(cls._continuous))
            raise ValueError(
                f"Unknown algorithm: {algorithm!r}. Registered: {available}"
            )
        return trainer

    @classmethod
    def list_algorithms(cls) -> list[str]:
        """Return all registered algorithm names."""
        return sorted(set(cls._discrete) | set(cls._continuous))
