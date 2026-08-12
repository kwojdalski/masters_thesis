"""Trainer registry for algorithm→class dispatch."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    pass


class TrainerRegistry:
    """Maps algorithm names to trainer classes.

    Separate dicts for discrete and continuous variants so PPO can register
    both without collision.  ``get`` prefers the variant that matches
    ``is_continuous``; when only one variant is registered it is returned
    regardless of the flag.
    """

    _discrete: ClassVar[dict[str, type]] = {}
    _continuous: ClassVar[dict[str, type]] = {}

    @classmethod
    def register(cls, algorithm: str, *, continuous: bool = False):
        """Register a trainer class under *algorithm*.

        Args:
            algorithm: Upper-cased algorithm name (e.g. ``"PPO"``).
            continuous: If True, register as the continuous-action variant.
        """

        def decorator(trainer_cls: type) -> type:
            key = algorithm.upper()
            if continuous:
                cls._continuous[key] = trainer_cls
            else:
                cls._discrete[key] = trainer_cls
            return trainer_cls

        return decorator

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


def register_trainer(algorithm: str, *, continuous: bool = False):
    """Decorator to register a trainer class in :class:`TrainerRegistry`.

    Args:
        algorithm: Algorithm name used for lookup (e.g. ``"TD3"``).
        continuous: Set True for the continuous-action variant (e.g.
            ``PPOTrainerContinuous``).

    Example::

        @register_trainer("TD3")
        class TD3Trainer(BaseTrainer): ...
    """
    return TrainerRegistry.register(algorithm, continuous=continuous)
