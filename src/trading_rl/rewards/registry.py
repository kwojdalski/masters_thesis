"""Reward registry: maps RewardType strings to reward objects / wrappers.

Each entry is a callable that accepts ``(eta, scale)`` kwargs and returns
a reward object compatible with the environment it is registered for.
The registry is keyed by the ``RewardType`` string value so YAML configs
work without conversion.
"""
from __future__ import annotations

from collections.abc import Callable

RewardFactory = Callable[..., object]

_REGISTRY: dict[str, RewardFactory] = {}


class RewardRegistry:
    """Central registry for reward factories."""

    @classmethod
    def register(cls, reward_type: str):
        """Decorator to register a factory under *reward_type*.

        The factory signature must be ``(*, eta: float, scale: float) -> reward``.

        Example::

            @RewardRegistry.register("log_return")
            def _make_log_return(*, eta: float, scale: float):
                from tradingenv.rewards import LogReturn
                return LogReturn(scale=scale)
        """
        def decorator(fn: RewardFactory) -> RewardFactory:
            _REGISTRY[reward_type] = fn
            return fn
        return decorator

    @classmethod
    def create(cls, reward_type: str, *, eta: float = 0.01, scale: float = 1.0) -> object:
        """Instantiate a reward object for *reward_type*.

        Args:
            reward_type: One of the registered type strings (e.g. ``"log_return"``).
            eta: Adaptation rate used by DSR-style rewards (ignored for log_return).
            scale: Reward scaling factor.

        Raises:
            ValueError: If *reward_type* is not registered.
        """
        factory = _REGISTRY.get(reward_type)
        if factory is None:
            raise ValueError(
                f"Unknown reward type: {reward_type!r}. "
                f"Registered: {list(_REGISTRY)}"
            )
        return factory(eta=eta, scale=scale)

    @classmethod
    def list_reward_types(cls) -> list[str]:
        """Return all registered reward type strings."""
        return list(_REGISTRY)

    @classmethod
    def is_registered(cls, reward_type: str) -> bool:
        return reward_type in _REGISTRY


def register_reward(reward_type: str):
    """Module-level shorthand for :meth:`RewardRegistry.register`."""
    return RewardRegistry.register(reward_type)


# ---------------------------------------------------------------------------
# Built-in registrations
# ---------------------------------------------------------------------------

@register_reward("log_return")
def _make_log_return(*, eta: float, scale: float) -> object:
    from tradingenv.rewards import LogReturn
    return LogReturn(scale=scale)


@register_reward("differential_sharpe")
def _make_differential_sharpe(*, eta: float, scale: float) -> object:
    from trading_rl.rewards.differential_sharpe import DifferentialSharpeRatio
    return DifferentialSharpeRatio(eta=eta, scale=scale)
