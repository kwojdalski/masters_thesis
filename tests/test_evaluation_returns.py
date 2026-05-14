from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from trading_rl.constants import RewardType
from trading_rl.evaluation.returns import (
    ReturnKind,
    ReturnSeries,
    RewardSeries,
    extract_tradingenv_return_series,
    extract_tradingenv_returns,
)


def test_return_series_converts_simple_returns_to_equity() -> None:
    series = ReturnSeries(np.array([0.10, -0.05]), ReturnKind.SIMPLE)

    equity = series.to_equity(initial_value=100.0)

    assert equity.kind == ReturnKind.EQUITY
    assert equity.includes_initial is True
    np.testing.assert_allclose(equity.values, np.array([100.0, 110.0, 104.5]))


def test_return_series_converts_equity_to_simple_returns() -> None:
    series = ReturnSeries(
        np.array([100.0, 110.0, 104.5]),
        ReturnKind.EQUITY,
        includes_initial=True,
    )

    simple = series.to_simple()

    assert simple.kind == ReturnKind.SIMPLE
    np.testing.assert_allclose(simple.values, np.array([0.10, -0.05]))


def test_return_series_rejects_equity_returns_without_initial_value() -> None:
    series = ReturnSeries(
        np.array([110.0, 104.5]),
        ReturnKind.EQUITY,
        includes_initial=False,
    )

    with pytest.raises(ValueError, match="initial value"):
        series.to_simple()


def test_return_series_converts_cumulative_log_with_initial_to_simple_returns() -> None:
    cumulative = np.array([
        0.0,
        np.log1p(0.10),
        np.log1p(0.10) + np.log1p(-0.05),
    ])
    series = ReturnSeries(
        cumulative,
        ReturnKind.CUMULATIVE_LOG,
        includes_initial=True,
    )

    simple = series.to_simple()

    np.testing.assert_allclose(simple.values, np.array([0.10, -0.05]))


def test_reward_series_rejects_shaped_rewards_as_returns() -> None:
    rewards = RewardSeries(
        np.array([0.1, 0.2]),
        RewardType.DIFFERENTIAL_SHARPE,
    )

    with pytest.raises(ValueError, match="shaped reward"):
        rewards.to_return_series()


def test_extract_tradingenv_returns_includes_initial_pre_step_nlv() -> None:
    broker = SimpleNamespace(
        track_record=[
            SimpleNamespace(
                context_pre=SimpleNamespace(nlv=100.0),
                context_post=SimpleNamespace(nlv=101.0),
            ),
            SimpleNamespace(
                context_pre=SimpleNamespace(nlv=101.0),
                context_post=SimpleNamespace(nlv=102.0),
            ),
        ]
    )
    env = SimpleNamespace(broker=broker)

    series = extract_tradingenv_return_series(env, n_steps=2)
    cumulative = extract_tradingenv_returns(env, n_steps=2)

    assert series is not None
    assert series.kind == ReturnKind.EQUITY
    assert series.includes_initial is True
    np.testing.assert_allclose(series.values, np.array([100.0, 101.0, 102.0]))

    expected = np.cumsum([0.0, np.log(101.0 / 100.0), np.log(102.0 / 101.0)])
    np.testing.assert_allclose(cumulative, expected)
