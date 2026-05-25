"""Tests for AlgorithmicEnvironmentBuilder._resolve_backend().

Verifies that TD3/DDPG force a continuous backend and that explicit backend
overrides are applied correctly.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from trading_rl.envs import builder as builder_module
from trading_rl.envs.builder import AlgorithmicEnvironmentBuilder


def _cfg(algorithm: str = "PPO", backend: str | None = None) -> SimpleNamespace:
    """Minimal config with the fields the builder inspects/logs."""
    return SimpleNamespace(
        env=SimpleNamespace(
            backend=backend,
            positions=[-1, 0, 1],
            trading_fees=0.0,
            name="test-env",
        ),
        training=SimpleNamespace(algorithm=algorithm),
        data=SimpleNamespace(memmap_dir=None),
    )


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [1_000.0, 1_100.0, 1_200.0],
        }
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


class TestCreate:
    def test_create_uses_backend_environment_when_memmap_disabled(self, monkeypatch):
        builder = AlgorithmicEnvironmentBuilder()
        config = _cfg("PPO", "tradingenv")
        df = _df()
        calls = []

        def fail_resolve_memmap_paths(_config):
            raise AssertionError("use_memmap=False should not inspect memmap paths")

        def fake_build_backend_env(**kwargs):
            calls.append(kwargs)
            return "batch-env"

        monkeypatch.setattr(builder, "_resolve_memmap_paths", fail_resolve_memmap_paths)
        monkeypatch.setattr(builder_module, "build_backend_env", fake_build_backend_env)

        env = builder.create(df, config, use_memmap=False)

        assert env == "batch-env"
        assert calls == [
            {
                "df": df,
                "config": config,
                "backend": "tradingenv",
            }
        ]

    def test_create_uses_streaming_environment_when_memmaps_exist(self, monkeypatch):
        builder = AlgorithmicEnvironmentBuilder()
        config = _cfg("TD3", "gym_trading_env.continuous")
        df = _df()
        memmap_paths = [object()]
        streaming_calls = []

        def fail_build_backend_env(**_kwargs):
            raise AssertionError("memmap paths should route to streaming env")

        def fake_create_streaming_env(paths, cfg):
            streaming_calls.append((paths, cfg))
            return "streaming-env"

        monkeypatch.setattr(builder, "_resolve_memmap_paths", lambda _config: memmap_paths)
        monkeypatch.setattr(builder, "_create_streaming_env", fake_create_streaming_env)
        monkeypatch.setattr(builder_module, "build_backend_env", fail_build_backend_env)

        env = builder.create(df, config)

        assert env == "streaming-env"
        assert streaming_calls == [(memmap_paths, config)]
