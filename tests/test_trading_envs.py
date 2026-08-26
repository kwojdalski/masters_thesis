from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from trading_rl.constants import EnvBackend, RewardType
from trading_rl.envs import trading_envs
from trading_rl.envs.trading_envs import (
    CustomTradingEnvironmentFactory,
    DiscreteActionWrapper,
    ForexEnvironmentFactory,
    create_environment,
    validate_actions,
    validate_backend,
)
from trading_rl.rewards import reward_function
from trading_rl.rewards.dsr_wrapper import (
    DifferentialSharpeRatioAnyTrading,
    StatefulRewardWrapper,
)


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "volume": [1000.0, 1100.0],
        }
    )


def _config(backend: str, positions: list[int] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        env=SimpleNamespace(
            backend=backend,
            positions=positions,
            name="unit-test-env",
            trading_fees=0.001,
            borrow_interest_rate=0.0001,
            price_column="close",
            feature_columns=["feature_a", "feature_b"],
            reward_type=RewardType.LOG_RETURN,
            reward_eta=0.01,
        )
    )


class _RecordingFactory:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def make(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        return "env"


def test_validate_backend_accepts_supported_string_and_enum() -> None:
    validate_backend("gym_trading_env.discrete")
    validate_backend(EnvBackend.TRADINGENV)


def test_validate_backend_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="Invalid backend 'unknown'"):
        validate_backend("unknown")


def test_validate_actions_rejects_custom_positions_for_anytrading() -> None:
    with pytest.raises(ValueError, match="supports only two actions"):
        validate_actions(EnvBackend.GYM_ANYTRADING_FOREX, [-1, 0, 1])


def test_validate_actions_allows_none_and_non_anytrading_positions() -> None:
    validate_actions(EnvBackend.GYM_ANYTRADING_FOREX, None)
    validate_actions(EnvBackend.GYM_TRADING_DISCRETE, [-1, 0, 1])


@pytest.mark.parametrize(
    ("raw_action", "expected"),
    [
        (torch.tensor(1), 1),
        (torch.tensor([0.1, 0.9]), 1),
        (np.array([0.7, 0.3]), 0),
        (np.array([[0.2, 0.8]]), 1),
        ([1], 1),
        (0, 0),
    ],
)
def test_discrete_action_wrapper_coerces_agent_outputs_to_scalar_int(
    raw_action, expected: int
) -> None:
    wrapper = object.__new__(DiscreteActionWrapper)

    assert wrapper.action(raw_action) == expected


def test_create_environment_requires_config_for_gym_trading_backends() -> None:
    with pytest.raises(ValueError, match="config is required"):
        create_environment(_df(), backend=EnvBackend.GYM_TRADING_DISCRETE)


def test_create_environment_routes_gym_trading_backend_with_config(monkeypatch) -> None:
    factory = _RecordingFactory()
    df = _df()

    def fake_get_environment_factory(backend, **kwargs):
        assert backend == EnvBackend.GYM_TRADING_CONTINUOUS
        assert kwargs == {"config": config}
        return factory

    config = _config("gym_trading_env.continuous", positions=[-1, 0, 1])
    monkeypatch.setattr(
        trading_envs, "get_environment_factory", fake_get_environment_factory
    )

    env = create_environment(df, config=config)

    assert env == "env"
    assert len(factory.calls) == 1
    assert factory.calls[0]["args"] == (df, config)
    assert factory.calls[0]["kwargs"] == {"backend": EnvBackend.GYM_TRADING_CONTINUOUS}


def test_create_environment_passes_tradingenv_column_specs(monkeypatch) -> None:
    factory = _RecordingFactory()
    df = _df()

    def fake_get_environment_factory(backend, **kwargs):
        assert backend == EnvBackend.TRADINGENV
        assert kwargs == {"config": config}
        return factory

    config = _config("tradingenv", positions=[-1, 0, 1])
    monkeypatch.setattr(
        trading_envs, "get_environment_factory", fake_get_environment_factory
    )

    env = create_environment(df, config=config)

    assert env == "env"
    assert factory.calls[0]["args"] == ()
    assert factory.calls[0]["kwargs"]["df"] is df
    assert factory.calls[0]["kwargs"]["config"] is config
    assert factory.calls[0]["kwargs"]["price_column"] == "close"
    assert factory.calls[0]["kwargs"]["feature_columns"] == ["feature_a", "feature_b"]


def test_create_environment_rejects_invalid_anytrading_positions_before_factory(
    monkeypatch,
) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("factory should not be requested for invalid actions")

    config = _config("gym_anytrading.stocks", positions=[-1, 0, 1])
    monkeypatch.setattr(trading_envs, "get_environment_factory", fail_if_called)

    with pytest.raises(ValueError, match="supports only two actions"):
        create_environment(_df(), config=config)


def test_create_environment_routes_anytrading_backend_with_kwargs(monkeypatch) -> None:
    factory = _RecordingFactory()
    df = _df()

    def fake_get_environment_factory(backend, **kwargs):
        assert backend == EnvBackend.GYM_ANYTRADING_FOREX
        assert kwargs == {"config": config, "window_size": 12}
        return factory

    config = _config("gym_anytrading.forex", positions=[0, 1])
    monkeypatch.setattr(
        trading_envs, "get_environment_factory", fake_get_environment_factory
    )

    env = create_environment(df, config=config, window_size=12)

    assert env == "env"
    assert len(factory.calls) == 1
    assert factory.calls[0]["args"] == (df,)
    assert factory.calls[0]["kwargs"] == {"window_size": 12}


def test_custom_factory_passes_reward_function_to_gym_trading_env(monkeypatch) -> None:
    calls: list[dict] = []
    df = _df()
    config = _config("gym_trading_env.discrete", positions=[-1, 0, 1])

    def fake_make(env_id, **kwargs):
        calls.append({"env_id": env_id, "kwargs": kwargs})
        return "base-env"

    monkeypatch.setattr(trading_envs.gym, "make", fake_make)

    env = CustomTradingEnvironmentFactory(config)._create_base_environment(df, config)

    assert env == "base-env"
    assert calls == [
        {
            "env_id": "TradingEnv",
            "kwargs": {
                "name": "unit-test-env",
                "df": df,
                "positions": [-1, 0, 1],
                "initial_position": -1,
                "trading_fees": 0.001,
                "borrow_interest_rate": 0.0001,
                "reward_function": reward_function,
            },
        }
    ]


def test_anytrading_dsr_reward_wiring_wraps_base_env(monkeypatch) -> None:
    import gymnasium as gym

    class FakeAnyTradingEnv(gym.Env):
        action_space = gym.spaces.Discrete(2)
        observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

        @property
        def unwrapped(self):
            return self

    calls: list[dict] = []
    base_env = FakeAnyTradingEnv()

    def fake_make(env_id, **kwargs):
        calls.append({"env_id": env_id, "kwargs": kwargs})
        return base_env

    monkeypatch.setattr(trading_envs.gym, "make", fake_make)
    monkeypatch.setattr(trading_envs, "GymWrapper", lambda env: env)
    monkeypatch.setattr(
        trading_envs.ForexEnvironmentFactory,
        "_wrap_with_step_counter",
        lambda self, env: env,
    )

    config = _config("gym_anytrading.forex", positions=[0, 1])
    config.env.reward_type = RewardType.DIFFERENTIAL_SHARPE
    config.env.reward_eta = 0.123

    env = ForexEnvironmentFactory(config).make(_df(), window_size=1, frame_bound=(1, 2))

    assert isinstance(env, DiscreteActionWrapper)
    assert isinstance(env.env, StatefulRewardWrapper)
    assert env.env.env is base_env
    assert isinstance(env.env.reward_fn, DifferentialSharpeRatioAnyTrading)
    assert env.env.reward_fn.eta == pytest.approx(0.123)
    assert calls[0]["env_id"] == "forex-v0"
    assert list(calls[0]["kwargs"]["df"].columns) == [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]


def test_custom_factory_rejects_non_log_return_reward_type() -> None:
    """Regression test for the silent reward_type override (#312).

    gym_trading_env backends must not silently fall back to log_return when a
    different reward_type is configured.
    """
    config = _config("gym_trading_env.discrete", positions=[-1, 0, 1])
    config.env.reward_type = RewardType.DIFFERENTIAL_SHARPE

    with pytest.raises(ValueError, match="differential_sharpe.*not supported"):
        CustomTradingEnvironmentFactory(config)._create_base_environment(_df(), config)


def test_custom_factory_defaults_to_log_return_when_reward_type_unset() -> None:
    """Configs without a reward_type attribute keep the log_return default."""
    config = _config("gym_trading_env.discrete", positions=[-1, 0, 1])
    del config.env.reward_type

    calls: list[dict] = []

    def fake_make(env_id, **kwargs):
        calls.append(kwargs)
        return "base-env"

    import gymnasium as gym

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(gym, "make", fake_make)
        env = CustomTradingEnvironmentFactory(config)._create_base_environment(
            _df(), config
        )

    assert env == "base-env"
    assert calls[0]["reward_function"] is reward_function
