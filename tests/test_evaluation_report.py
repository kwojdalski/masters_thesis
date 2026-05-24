from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from trading_rl.constants import RewardType
from trading_rl.evaluation.report import (
    _extract_action_array,
    _periods_per_year_from_index,
    build_evaluation_report_for_trainer,
    periods_per_year_from_timeframe,
)


class _FakeRollout:
    def __init__(self, rewards: list[float], actions: list[float]) -> None:
        self._rewards = torch.tensor(rewards, dtype=torch.float32)
        self._actions = torch.tensor(actions, dtype=torch.float32)

    def __getitem__(self, key):
        if key == ("next", "reward"):
            return self._rewards
        if key == "action":
            return self._actions
        raise KeyError(key)

    def get(self, key, default=None):
        if key == "action":
            return self._actions
        return default


class _FakeEnv:
    def __init__(self, rollout: _FakeRollout, broker=None) -> None:
        self._rollout = rollout
        self.broker = broker

    def rollout(self, *, max_steps: int, policy):
        return self._rollout


def _broker_from_nlv(values: list[float]):
    records = []
    for i in range(len(values) - 1):
        records.append(
            SimpleNamespace(
                context_pre=SimpleNamespace(nlv=values[i]),
                context_post=SimpleNamespace(nlv=values[i + 1]),
            )
        )
    return SimpleNamespace(track_record=records)


def _trainer(env):
    return SimpleNamespace(env=env, actor=object())


def _config(*, reward_type: str, backend: str = "tradingenv"):
    return SimpleNamespace(
        env=SimpleNamespace(
            reward_type=reward_type,
            backend=backend,
            price_column="close",
            positions=[-1, 0, 1],
        ),
        data=SimpleNamespace(timeframe="1d"),
    )


def test_periods_per_year_from_timeframe_uses_equity_calendar() -> None:
    assert periods_per_year_from_timeframe("1m") == 252 * 390
    assert periods_per_year_from_timeframe("1h") == 252 * 6.5
    assert periods_per_year_from_timeframe("unknown") == 252


def test_periods_per_year_from_index_uses_observed_event_rate() -> None:
    df = pd.DataFrame(
        {"close": np.arange(10, dtype=float)},
        index=pd.date_range("2024-01-01 09:30:00", periods=10, freq="1s"),
    )

    periods = _periods_per_year_from_index(df)

    assert periods == 252 * 6.5 * 3600


def test_periods_per_year_from_index_handles_business_day_bars() -> None:
    df = pd.DataFrame(
        {"close": np.arange(252, dtype=float)},
        index=pd.bdate_range("2024-01-01", periods=252),
    )

    assert _periods_per_year_from_index(df) == 252


def test_periods_per_year_from_index_returns_none_for_non_datetime_index() -> None:
    df = pd.DataFrame({"close": [100.0, 101.0]}, index=[0, 1])

    assert _periods_per_year_from_index(df) is None


def test_report_uses_broker_nlv_not_shaped_rewards_for_dsr() -> None:
    rollout = _FakeRollout(rewards=[999.0, 999.0], actions=[1.0, 1.0])
    env = _FakeEnv(rollout, broker=_broker_from_nlv([100.0, 100.0, 110.0]))
    prices = pd.DataFrame({"close": [100.0, 100.0, 100.0]})

    report = build_evaluation_report_for_trainer(
        trainer=_trainer(env),
        df_prices=prices,
        max_steps=2,
        config=_config(reward_type=RewardType.DIFFERENTIAL_SHARPE),
        eval_env=env,
    )

    assert report["total_return"] == pytest.approx(0.10)
    assert report["expectancy_per_period"] == pytest.approx(0.05)


def test_report_converts_log_return_rewards_when_no_broker_returns_exist() -> None:
    rollout = _FakeRollout(
        rewards=[float(np.log1p(0.10)), float(np.log1p(-0.05))],
        actions=[1.0, 1.0],
    )
    env = _FakeEnv(rollout)
    prices = pd.DataFrame({"close": [100.0, 110.0, 104.5]})

    report = build_evaluation_report_for_trainer(
        trainer=_trainer(env),
        df_prices=prices,
        max_steps=2,
        config=_config(
            reward_type=RewardType.LOG_RETURN,
            backend="gym_trading_env.continuous",
        ),
        eval_env=env,
    )

    assert report["total_return"] == pytest.approx(1.10 * 0.95 - 1.0)


def test_report_reuses_supplied_rollout_without_rerunning_env() -> None:
    rollout = _FakeRollout(
        rewards=[float(np.log1p(0.10)), float(np.log1p(-0.05))],
        actions=[1.0, 1.0],
    )

    class NoRolloutEnv:
        def rollout(self, *, max_steps: int, policy):
            raise AssertionError("report should reuse the rollout from trainer.evaluate")

    env = NoRolloutEnv()
    prices = pd.DataFrame({"close": [100.0, 110.0, 104.5]})

    report = build_evaluation_report_for_trainer(
        trainer=_trainer(env),
        df_prices=prices,
        max_steps=2,
        config=_config(
            reward_type=RewardType.LOG_RETURN,
            backend="gym_trading_env.continuous",
        ),
        eval_env=env,
        rollout=rollout,
    )

    assert report["total_return"] == pytest.approx(1.10 * 0.95 - 1.0)


def test_report_prefers_configured_timeframe_for_annualization() -> None:
    returns = np.array([0.01, -0.02, 0.015, -0.005, 0.02], dtype=float)
    rollout = _FakeRollout(rewards=[0.0] * len(returns), actions=[0.0] * len(returns))
    env = _FakeEnv(rollout)
    prices = pd.DataFrame(
        {"close": np.arange(len(returns) + 1, dtype=float) + 100.0},
        index=pd.date_range(
            "2024-01-01 09:30:00",
            periods=len(returns) + 1,
            freq="1s",
        ),
    )

    report = build_evaluation_report_for_trainer(
        trainer=_trainer(env),
        df_prices=prices,
        max_steps=len(returns),
        config=_config(
            reward_type=RewardType.LOG_RETURN,
            backend="gym_trading_env.continuous",
        ),
        eval_env=env,
        rollout=rollout,
        strategy_simple_returns=returns,
    )

    expected_vol = float(np.std(returns, ddof=1) * np.sqrt(252))
    assert report["annualized_volatility"] == pytest.approx(expected_vol)


def test_extract_action_array_maps_one_hot_actions_to_positions() -> None:
    rollout = _FakeRollout(
        rewards=[0.0, 0.0, 0.0],
        actions=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
    )

    actions = _extract_action_array(
        rollout,
        is_portfolio=False,
        positions=[-1, 0, 1],
    )

    np.testing.assert_array_equal(actions, np.array([-1.0, 0.0, 1.0]))

    single_action = _extract_action_array(
        _FakeRollout(rewards=[0.0], actions=[[0.0, 0.0, 1.0]]),
        is_portfolio=False,
        positions=[-1, 0, 1],
    )

    np.testing.assert_array_equal(single_action, np.array([1.0]))
