"""Tests for AlgorithmicEnvironmentBuilder._resolve_backend().

Verifies that TD3/DDPG force a continuous backend and that explicit backend
overrides are applied correctly.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from trading_rl.constants import EnvBackend, RewardType
from trading_rl.envs import builder as builder_module
from trading_rl.envs.builder import AlgorithmicEnvironmentBuilder
from trading_rl.rewards import reward_function


def _cfg(algorithm: str = "PPO", backend: str | None = None) -> SimpleNamespace:
    """Minimal config with the fields the builder inspects/logs."""
    return SimpleNamespace(
        env=SimpleNamespace(
            backend=backend,
            positions=[-1, 0, 1],
            trading_fees=0.0,
            borrow_interest_rate=0.0,
            name="test-env",
            streaming_episode_length=12,
            continuous_action_thresholds=[-0.25, 0.25],
        ),
        training=SimpleNamespace(algorithm=algorithm, device="cpu"),
        data=SimpleNamespace(memmap_dir=None),
        seed=123,
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


class TestCreateStreamingEnv:
    def test_create_streaming_env_builds_gym_streaming_env(self, monkeypatch):
        import torchrl.envs.transforms as transforms_module

        import trading_rl.envs.streaming_env as streaming_module

        calls = {"streaming": [], "gym": [], "transformed": [], "step_counter": 0}

        class FakeStreamingTradingEnv:
            def __init__(self, **kwargs):
                calls["streaming"].append(kwargs)

        class FakeStepCounter:
            def __init__(self):
                calls["step_counter"] += 1

        def fake_gym_wrapper(env):
            calls["gym"].append(env)
            return ("gym", env)

        def fake_transformed_env(env, transform):
            calls["transformed"].append((env, transform))
            return ("transformed", env, transform)

        monkeypatch.setattr(streaming_module, "StreamingTradingEnv", FakeStreamingTradingEnv)
        monkeypatch.setattr(transforms_module, "StepCounter", FakeStepCounter)
        monkeypatch.setattr(builder_module, "GymWrapper", fake_gym_wrapper)
        monkeypatch.setattr(builder_module, "TransformedEnv", fake_transformed_env)

        config = _cfg("PPO", EnvBackend.GYM_TRADING_DISCRETE)
        result = AlgorithmicEnvironmentBuilder()._create_streaming_env(["memmap"], config)

        assert result[0] == "transformed"
        assert calls["streaming"] == [
            {
                "memmap_paths": ["memmap"],
                "episode_length": 12,
                "seed": 123,
                "name": "test-env",
                "positions": [-1, 0, 1],
                "trading_fees": 0.0,
                "borrow_interest_rate": 0.0,
                "reward_function": reward_function,
            }
        ]
        assert len(calls["gym"]) == 1
        assert calls["step_counter"] == 1
        assert len(calls["transformed"]) == 1

    def test_create_streaming_env_adds_continuous_action_mapping(self, monkeypatch):
        import torchrl.envs.transforms as transforms_module

        import trading_rl.continuous_action_wrapper as continuous_module
        import trading_rl.envs.streaming_env as streaming_module

        transforms = []
        action_maps = []

        class FakeStreamingTradingEnv:
            def __init__(self, **_kwargs):
                pass

        class FakeStepCounter:
            pass

        class FakeContinuousToDiscreteAction:
            def __init__(self, **kwargs):
                action_maps.append(kwargs)

        def fake_transformed_env(env, transform):
            transforms.append(transform)
            return ("transformed", env, transform)

        monkeypatch.setattr(streaming_module, "StreamingTradingEnv", FakeStreamingTradingEnv)
        monkeypatch.setattr(transforms_module, "StepCounter", FakeStepCounter)
        monkeypatch.setattr(
            continuous_module,
            "ContinuousToDiscreteAction",
            FakeContinuousToDiscreteAction,
        )
        monkeypatch.setattr(builder_module, "GymWrapper", lambda env: ("gym", env))
        monkeypatch.setattr(builder_module, "TransformedEnv", fake_transformed_env)

        config = _cfg("TD3", EnvBackend.GYM_TRADING_CONTINUOUS)
        AlgorithmicEnvironmentBuilder()._create_streaming_env(["memmap"], config)

        assert action_maps == [
            {
                "discrete_actions": [-1, 0, 1],
                "thresholds": [-0.25, 0.25],
                "device": "cpu",
            }
        ]
        assert isinstance(transforms[0], FakeContinuousToDiscreteAction)
        assert isinstance(transforms[1], FakeStepCounter)

    def test_create_streaming_env_delegates_tradingenv_backend(self, monkeypatch):
        builder = AlgorithmicEnvironmentBuilder()
        config = _cfg("TD3", EnvBackend.TRADINGENV)
        calls = []

        def fake_create_streaming_tradingenv(paths, episode_length, cfg):
            calls.append((paths, episode_length, cfg))
            return "streaming-tradingenv"

        monkeypatch.setattr(
            builder,
            "_create_streaming_tradingenv",
            fake_create_streaming_tradingenv,
        )

        result = builder._create_streaming_env(["memmap"], config)

        assert result == "streaming-tradingenv"
        assert calls == [(["memmap"], 12, config)]

    def test_create_streaming_env_rejects_unsupported_backend(self):
        config = _cfg("PPO", EnvBackend.GYM_ANYTRADING_FOREX)

        with pytest.raises(ValueError, match="memmap streaming is not supported"):
            AlgorithmicEnvironmentBuilder()._create_streaming_env(["memmap"], config)

    def test_create_streaming_tradingenv_forwards_runtime_and_reward_config(
        self, monkeypatch
    ):
        import torchrl.envs.transforms as transforms_module

        import trading_rl.envs.tradingenvxy_wrapper as xy_module

        calls = {"xy": [], "gym": [], "transformed": []}

        class FakeStreamingTradingEnvXY:
            def __init__(self, **kwargs):
                calls["xy"].append(kwargs)

        class FakeStepCounter:
            pass

        def fake_gym_wrapper(env):
            calls["gym"].append(env)
            return ("gym", env)

        def fake_transformed_env(env, transform):
            calls["transformed"].append((env, transform))
            return ("transformed", env, transform)

        monkeypatch.setattr(
            xy_module,
            "StreamingTradingEnvXY",
            FakeStreamingTradingEnvXY,
        )
        monkeypatch.setattr(transforms_module, "StepCounter", FakeStepCounter)
        monkeypatch.setattr(builder_module, "GymWrapper", fake_gym_wrapper)
        monkeypatch.setattr(builder_module, "TransformedEnv", fake_transformed_env)

        memmap = SimpleNamespace(
            columns=["close", "feature_signal", "feature_position"]
        )
        config = _cfg("TD3", EnvBackend.TRADINGENV)
        config.env.feature_columns = ["feature_signal", "feature_position"]
        config.env.price_column = "mid_price"
        config.env.initial_portfolio_value = 25_000.0
        config.env.trading_fees = 0.0002
        config.env.reward_type = RewardType.DIFFERENTIAL_SHARPE
        config.env.reward_eta = 0.07
        config.env.reward_scale = 2.5
        config.env.include_position_feature = True
        config.env.obs_clip = 3.0

        result = AlgorithmicEnvironmentBuilder()._create_streaming_tradingenv(
            [memmap],
            episode_length=12,
            config=config,
        )

        assert result[0] == "transformed"
        assert calls["xy"] == [
            {
                "memmap_paths": [memmap],
                "episode_length": 12,
                "feature_columns": ["feature_signal"],
                "price_column": "mid_price",
                "initial_cash": 25_000.0,
                "fee": 0.0002,
                "reward_type": RewardType.DIFFERENTIAL_SHARPE,
                "reward_eta": 0.07,
                "reward_scale": 2.5,
                "runtime_feature_columns": ["feature_position"],
                "obs_clip": 3.0,
                "seed": 123,
                "action_penalty_lambda": 0.0,
                "action_penalty_type": "quadratic",
            }
        ]
        assert isinstance(calls["gym"][0], FakeStreamingTradingEnvXY)
        assert isinstance(calls["transformed"][0][1], FakeStepCounter)
