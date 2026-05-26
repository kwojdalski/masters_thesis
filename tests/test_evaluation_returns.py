from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from trading_rl.constants import RewardType
from trading_rl.evaluation.returns import (
    ReturnKind,
    ReturnSeries,
    RewardSeries,
    cross_validate_nlv,
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


def test_return_series_converts_cumulative_log_without_initial_to_simple_returns() -> None:
    cumulative = np.array([
        np.log1p(0.10),
        np.log1p(0.10) + np.log1p(-0.05),
    ])
    series = ReturnSeries(
        cumulative,
        ReturnKind.CUMULATIVE_LOG,
        includes_initial=False,
    )

    simple = series.to_simple()

    np.testing.assert_allclose(simple.values, np.array([0.10, -0.05]))


def test_return_series_simple_to_cumulative_log_can_include_initial_zero() -> None:
    series = ReturnSeries(np.array([0.10, -0.05]), ReturnKind.SIMPLE)

    cumulative = series.to_cumulative_log(include_initial=True)

    assert cumulative.includes_initial is True
    expected = np.array([
        0.0,
        np.log1p(0.10),
        np.log1p(0.10) + np.log1p(-0.05),
    ])
    np.testing.assert_allclose(cumulative.values, expected)


def test_return_series_rejects_non_positive_initial_equity() -> None:
    series = ReturnSeries(np.array([0.10]), ReturnKind.SIMPLE)

    with pytest.raises(ValueError, match="initial_value"):
        series.to_equity(initial_value=0.0)


def test_reward_series_converts_log_rewards_to_simple_returns() -> None:
    rewards = RewardSeries(
        np.array([np.log1p(0.10), np.log1p(-0.05)]),
        RewardType.LOG_RETURN,
    )

    simple = rewards.to_return_series().to_simple()

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


def test_extract_tradingenv_return_series_caps_to_requested_steps() -> None:
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
            SimpleNamespace(
                context_pre=SimpleNamespace(nlv=102.0),
                context_post=SimpleNamespace(nlv=103.0),
            ),
        ]
    )
    env = SimpleNamespace(broker=broker)

    series = extract_tradingenv_return_series(env, n_steps=2)

    assert series is not None
    np.testing.assert_allclose(series.values, np.array([100.0, 101.0, 102.0]))


def test_extract_tradingenv_return_series_finds_nested_broker() -> None:
    broker = SimpleNamespace(
        track_record=[
            SimpleNamespace(
                context_pre=SimpleNamespace(nlv=100.0),
                context_post=SimpleNamespace(nlv=105.0),
            )
        ]
    )
    env = SimpleNamespace(_env=SimpleNamespace(env=SimpleNamespace(broker=broker)))

    series = extract_tradingenv_return_series(env, n_steps=1)

    assert series is not None
    np.testing.assert_allclose(series.values, np.array([100.0, 105.0]))


def test_extract_tradingenv_return_series_rejects_missing_post_nlv() -> None:
    broker = SimpleNamespace(
        track_record=[
            SimpleNamespace(context_pre=SimpleNamespace(nlv=100.0)),
        ]
    )
    env = SimpleNamespace(broker=broker)

    assert extract_tradingenv_return_series(env, n_steps=1) is None


def test_extract_tradingenv_return_series_rejects_non_positive_nlv() -> None:
    broker = SimpleNamespace(
        track_record=[
            SimpleNamespace(
                context_pre=SimpleNamespace(nlv=100.0),
                context_post=SimpleNamespace(nlv=0.0),
            ),
        ]
    )
    env = SimpleNamespace(broker=broker)

    assert extract_tradingenv_return_series(env, n_steps=1) is None


def test_cross_validate_nlv_recomputes_weighted_price_path() -> None:
    class Rollout:
        def get(self, key, default=None):
            if key == "action":
                return torch.tensor([1.0, -0.5, 0.0])
            return default

    prices = np.array([100.0, 110.0, 99.0, 120.0])
    # Correct NLV: actions[t] applied to prices[t+1]/prices[t] - 1
    #   step 0: 100 * (1 + 1.0 * 0.10) = 110.0
    #   step 1: 110 * (1 + -0.5 * -0.10) = 115.5
    #   step 2: 115.5 * (1 + 0.0 * ...) = 115.5
    broker_nlv = np.array([100.0, 110.0, 115.5, 115.5])

    max_err, recomputed = cross_validate_nlv(Rollout(), broker_nlv, prices)

    assert max_err == pytest.approx(0.0)
    np.testing.assert_allclose(recomputed, broker_nlv)
