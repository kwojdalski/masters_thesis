"""Tests for AlgorithmicEnvironmentBuilder backend resolution."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from trading_rl.constants import EnvBackend, RewardType
from trading_rl.envs import builder as builder_module
from trading_rl.envs.builder import (
    AlgorithmicEnvironmentBuilder,
    BackendResolutionPolicy,
    CommonEnvParams,
    EnvBuildParams,
    GymTradingEnvParams,
    StreamingEnvParams,
    TradingEnvParams,
)
from trading_rl.rewards import reward_function


def _params(algorithm: str = "PPO", backend: str | None = None) -> EnvBuildParams:
    """Minimal EnvBuildParams for tests."""
    return EnvBuildParams(
        common=CommonEnvParams(
            env_name="test-env",
            positions=[-1, 0, 1],
            trading_fees=0.0,
            borrow_interest_rate=0.0,
            reward_type="log_return",
            reward_eta=0.01,
            reward_scale=1.0,
            backend=backend,
            seed=123,
        ),
        algorithm=algorithm,
        trading_env=TradingEnvParams(
            feature_columns=None,
            price_column="close",
            initial_portfolio_value=100_000.0,
            include_position_feature=False,
            obs_clip=5.0,
            action_penalty_lambda=0.0,
            action_penalty_type="quadratic",
        ),
        gym_trading=GymTradingEnvParams(
            continuous_action_thresholds=[-0.25, 0.25],
            device="cpu",
        ),
        streaming=StreamingEnvParams(
            memmap_dir=None,
            streaming_episode_length=12,
        ),
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
        backend = AlgorithmicEnvironmentBuilder()._resolve_backend(_params("PPO"))
        assert backend == "gym_trading_env.discrete"

    def test_td3_defaults_to_continuous(self):
        backend = AlgorithmicEnvironmentBuilder()._resolve_backend(_params("TD3"))
        assert backend == "gym_trading_env.continuous"

    def test_ddpg_defaults_to_continuous(self):
        backend = AlgorithmicEnvironmentBuilder()._resolve_backend(_params("DDPG"))
        assert backend == "gym_trading_env.continuous"

    def test_td3_with_discrete_backend_raises(self):
        with pytest.raises(ValueError, match="TD3"):
            AlgorithmicEnvironmentBuilder()._resolve_backend(
                _params("TD3", "gym_trading_env.discrete")
            )

    def test_ddpg_with_discrete_backend_raises(self):
        with pytest.raises(ValueError, match="DDPG"):
            AlgorithmicEnvironmentBuilder()._resolve_backend(
                _params("DDPG", "gym_trading_env.discrete")
            )

    def test_td3_with_continuous_backend_accepted(self):
        backend = AlgorithmicEnvironmentBuilder()._resolve_backend(
            _params("TD3", "gym_trading_env.continuous")
        )
        assert backend == "gym_trading_env.continuous"

    def test_td3_with_tradingenv_backend_accepted(self):
        backend = AlgorithmicEnvironmentBuilder()._resolve_backend(
            _params("TD3", "tradingenv")
        )
        assert backend == "tradingenv"

    def test_ppo_explicit_backend_overrides_default(self):
        backend = AlgorithmicEnvironmentBuilder()._resolve_backend(
            _params("PPO", "tradingenv")
        )
        assert backend == "tradingenv"

    def test_lowercase_td3_enforced(self):
        """Algorithm matching must be case-insensitive."""
        backend = AlgorithmicEnvironmentBuilder()._resolve_backend(_params("td3"))
        assert backend == "gym_trading_env.continuous"

    def test_lowercase_ddpg_raises_for_discrete(self):
        with pytest.raises(ValueError, match="ddpg"):
            AlgorithmicEnvironmentBuilder()._resolve_backend(
                _params("ddpg", "gym_trading_env.discrete")
            )

    def test_no_explicit_backend_ppo_uses_default_builder_backend(self):
        builder = AlgorithmicEnvironmentBuilder(default_backend="tradingenv")
        backend = builder._resolve_backend(_params("PPO"))
        assert backend == "tradingenv"

    def test_custom_policy_can_add_algorithm_default_without_builder_change(self):
        policy = BackendResolutionPolicy(
            algorithm_defaults={"MODEL_BASED": EnvBackend.TRADINGENV},
        )
        builder = AlgorithmicEnvironmentBuilder(backend_policy=policy)

        backend = builder._resolve_backend(_params("MODEL_BASED"))

        assert backend == "tradingenv"

    def test_custom_policy_can_constrain_algorithm_backends(self):
        policy = BackendResolutionPolicy(
            allowed_backends={"MODEL_BASED": frozenset({EnvBackend.TRADINGENV})},
        )
        builder = AlgorithmicEnvironmentBuilder(backend_policy=policy)

        with pytest.raises(ValueError, match="MODEL_BASED"):
            builder._resolve_backend(
                _params("MODEL_BASED", EnvBackend.GYM_TRADING_CONTINUOUS)
            )


class TestCreate:
    def test_create_uses_backend_environment_when_memmap_disabled(self, monkeypatch):
        builder = AlgorithmicEnvironmentBuilder()
        params = _params("PPO", "tradingenv")
        df = _df()
        calls = []

        def fail_resolve_memmap_paths(_params):
            raise AssertionError("use_memmap=False should not inspect memmap paths")

        def fake_create_non_streaming_env(d, p, backend):
            calls.append({"df": d, "params": p, "backend": backend})
            return "batch-env"

        monkeypatch.setattr(builder, "_resolve_memmap_paths", fail_resolve_memmap_paths)
        monkeypatch.setattr(builder, "_create_non_streaming_env", fake_create_non_streaming_env)

        env = builder.create(df, params, use_memmap=False)

        assert env == "batch-env"
        assert calls == [{"df": df, "params": params, "backend": "tradingenv"}]

    def test_create_uses_streaming_environment_when_memmaps_exist(self, monkeypatch):
        builder = AlgorithmicEnvironmentBuilder()
        params = _params("TD3", "gym_trading_env.continuous")
        df = _df()
        memmap_paths = [object()]
        streaming_calls = []

        def fake_create_non_streaming_env(d, p, backend):
            raise AssertionError("memmap paths should route to streaming env")

        def fake_create_streaming_env(paths, p):
            streaming_calls.append((paths, p))
            return "streaming-env"

        monkeypatch.setattr(builder, "_resolve_memmap_paths", lambda _params: memmap_paths)
        monkeypatch.setattr(builder, "_create_streaming_env", fake_create_streaming_env)
        monkeypatch.setattr(builder, "_create_non_streaming_env", fake_create_non_streaming_env)

        env = builder.create(df, params)

        assert env == "streaming-env"
        assert streaming_calls == [(memmap_paths, params)]


class TestCreateStreamingEnv:
    def test_create_streaming_env_builds_gym_streaming_env(self, monkeypatch):
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
        monkeypatch.setattr(builder_module, "StepCounter", FakeStepCounter)
        monkeypatch.setattr(builder_module, "GymWrapper", fake_gym_wrapper)
        monkeypatch.setattr(builder_module, "TransformedEnv", fake_transformed_env)

        params = _params("PPO", EnvBackend.GYM_TRADING_DISCRETE)
        result = AlgorithmicEnvironmentBuilder()._create_streaming_env(["memmap"], params)

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
        monkeypatch.setattr(builder_module, "StepCounter", FakeStepCounter)
        monkeypatch.setattr(
            continuous_module,
            "ContinuousToDiscreteAction",
            FakeContinuousToDiscreteAction,
        )
        monkeypatch.setattr(builder_module, "GymWrapper", lambda env: ("gym", env))
        monkeypatch.setattr(builder_module, "TransformedEnv", fake_transformed_env)

        params = _params("TD3", EnvBackend.GYM_TRADING_CONTINUOUS)
        AlgorithmicEnvironmentBuilder()._create_streaming_env(["memmap"], params)

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
        params = _params("TD3", EnvBackend.TRADINGENV)
        calls = []

        def fake_create_streaming_tradingenv(paths, episode_length, p):
            calls.append((paths, episode_length, p))
            return "streaming-tradingenv"

        monkeypatch.setattr(
            builder,
            "_create_streaming_tradingenv",
            fake_create_streaming_tradingenv,
        )

        result = builder._create_streaming_env(["memmap"], params)

        assert result == "streaming-tradingenv"
        assert calls == [(["memmap"], 12, params)]

    def test_create_streaming_env_rejects_unsupported_backend(self):
        params = _params("PPO", EnvBackend.GYM_ANYTRADING_FOREX)

        with pytest.raises(ValueError, match="memmap streaming is not supported"):
            AlgorithmicEnvironmentBuilder()._create_streaming_env(["memmap"], params)

    def test_create_streaming_tradingenv_forwards_runtime_and_reward_config(
        self, monkeypatch
    ):
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
        monkeypatch.setattr(builder_module, "StepCounter", FakeStepCounter)
        monkeypatch.setattr(builder_module, "GymWrapper", fake_gym_wrapper)
        monkeypatch.setattr(builder_module, "TransformedEnv", fake_transformed_env)

        memmap = SimpleNamespace(
            columns=["close", "feature_signal", "feature_position"]
        )
        params = EnvBuildParams(
            common=CommonEnvParams(
                env_name="test-env",
                positions=[-1, 0, 1],
                trading_fees=0.0002,
                borrow_interest_rate=0.0,
                reward_type=RewardType.DIFFERENTIAL_SHARPE,
                reward_eta=0.07,
                reward_scale=2.5,
                backend=EnvBackend.TRADINGENV,
                seed=123,
            ),
            algorithm="TD3",
            trading_env=TradingEnvParams(
                feature_columns=["feature_signal", "feature_position"],
                price_column="mid_price",
                initial_portfolio_value=25_000.0,
                include_position_feature=True,
                obs_clip=3.0,
                action_penalty_lambda=0.0,
                action_penalty_type="quadratic",
            ),
            gym_trading=GymTradingEnvParams(
                continuous_action_thresholds=[-0.25, 0.25],
                device="cpu",
            ),
            streaming=StreamingEnvParams(
                memmap_dir=None,
                streaming_episode_length=12,
            ),
        )

        result = AlgorithmicEnvironmentBuilder()._create_streaming_tradingenv(
            [memmap],
            episode_length=12,
            params=params,
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
                "execution_price": "mid",
                "bid_column": "bid_px_00",
                "ask_column": "ask_px_00",
                "obs_latency": None,
                "exec_latency": None,
            }
        ]
        assert isinstance(calls["gym"][0], FakeStreamingTradingEnvXY)
        assert isinstance(calls["transformed"][0][1], FakeStepCounter)
