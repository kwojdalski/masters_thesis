"""Tests for AlgorithmicEnvironmentBuilder._resolve_backend().

Verifies that TD3/DDPG force a continuous backend and that explicit backend
overrides are applied correctly.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from trading_rl.envs.builder import AlgorithmicEnvironmentBuilder


def _cfg(algorithm: str = "PPO", backend: str | None = None) -> SimpleNamespace:
    """Minimal config with only the fields _resolve_backend inspects."""
    return SimpleNamespace(
        env=SimpleNamespace(backend=backend),
        training=SimpleNamespace(algorithm=algorithm),
    )


class TestResolveBackend:
    def test_ppo_defaults_to_discrete(self):
        backend = AlgorithmicEnvironmentBuilder()._resolve_backend(_cfg("PPO"))
        assert backend == "gym_trading_env.discrete"

    def test_td3_defaults_to_continuous(self):
        backend = AlgorithmicEnvironmentBuilder()._resolve_backend(_cfg("TD3"))
        assert backend == "gym_trading_env.continuous"

    def test_ddpg_defaults_to_continuous(self):
        backend = AlgorithmicEnvironmentBuilder()._resolve_backend(_cfg("DDPG"))
        assert backend == "gym_trading_env.continuous"

    def test_td3_with_discrete_backend_raises(self):
        with pytest.raises(ValueError, match="TD3"):
            AlgorithmicEnvironmentBuilder()._resolve_backend(
                _cfg("TD3", "gym_trading_env.discrete")
            )

    def test_ddpg_with_discrete_backend_raises(self):
        with pytest.raises(ValueError, match="DDPG"):
            AlgorithmicEnvironmentBuilder()._resolve_backend(
                _cfg("DDPG", "gym_trading_env.discrete")
            )

    def test_td3_with_continuous_backend_accepted(self):
        backend = AlgorithmicEnvironmentBuilder()._resolve_backend(
            _cfg("TD3", "gym_trading_env.continuous")
        )
        assert backend == "gym_trading_env.continuous"

    def test_td3_with_tradingenv_backend_accepted(self):
        backend = AlgorithmicEnvironmentBuilder()._resolve_backend(
            _cfg("TD3", "tradingenv")
        )
        assert backend == "tradingenv"

    def test_ppo_explicit_backend_overrides_default(self):
        backend = AlgorithmicEnvironmentBuilder()._resolve_backend(
            _cfg("PPO", "tradingenv")
        )
        assert backend == "tradingenv"

    def test_lowercase_td3_enforced(self):
        """Algorithm matching must be case-insensitive."""
        backend = AlgorithmicEnvironmentBuilder()._resolve_backend(_cfg("td3"))
        assert backend == "gym_trading_env.continuous"

    def test_lowercase_ddpg_raises_for_discrete(self):
        with pytest.raises(ValueError, match="ddpg"):
            AlgorithmicEnvironmentBuilder()._resolve_backend(
                _cfg("ddpg", "gym_trading_env.discrete")
            )

    def test_no_explicit_backend_ppo_uses_default_builder_backend(self):
        """When no explicit backend, algo_backend takes precedence over default_backend."""
        builder = AlgorithmicEnvironmentBuilder(default_backend="tradingenv")
        backend = builder._resolve_backend(_cfg("PPO"))
        # PPO → algo_backend = "gym_trading_env.discrete", which wins over default_backend
        assert backend == "gym_trading_env.discrete"
