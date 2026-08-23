"""Tests for trading_rl.rewards.reward_function's invalid-valuation penalty."""

from __future__ import annotations

import numpy as np
import pytest
from gym_trading_env.utils.history import History

from trading_rl.rewards import _INVALID_VALUATION_PENALTY, reward_function


def _history(valuations: list[float]) -> History:
    """Build a History object stepping through the given portfolio valuations.

    Mirrors gym_trading_env.environments.TradingEnv.step(): each row is
    appended with a reward=0 placeholder, reward_function is called, and
    the returned reward is written back to history["reward", -1] --
    exactly like the real env loop, so history["reward", -2] behaves the
    same way it would in production.
    """
    h = History(max_size=len(valuations))
    h.set(portfolio_valuation=valuations[0], reward=0)
    for v in valuations[1:]:
        h.add(portfolio_valuation=v, reward=0)
        reward = reward_function(h)
        h["reward", -1] = reward
    return h


def test_valid_sequence_returns_log_return() -> None:
    h = _history([100.0, 110.0])
    assert h["reward", -1] == pytest.approx(np.log(1.1))


def test_single_invalid_step_returns_penalty() -> None:
    h = _history([100.0, float("nan")])
    assert h["reward", -1] == _INVALID_VALUATION_PENALTY


def test_streak_of_invalid_steps_penalizes_only_once() -> None:
    h = _history([100.0, float("nan"), float("nan"), float("nan")])

    rewards = list(h["reward"])
    # index 0: reset placeholder (0), index 1: first invalid step (penalty),
    # indices 2-3: latched at 0.0 for the rest of the streak.
    assert rewards[1] == _INVALID_VALUATION_PENALTY
    assert rewards[2] == 0.0
    assert rewards[3] == 0.0


def test_recovery_after_streak_resumes_normal_reward() -> None:
    h = _history([100.0, float("nan"), float("nan"), 100.0, 105.0])

    rewards = list(h["reward"])
    assert rewards[1] == _INVALID_VALUATION_PENALTY
    assert rewards[2] == 0.0
    # recovering: prev is still invalid (prev=nan), so no log-return is
    # computable -- skip silently rather than re-penalize.
    assert rewards[3] == 0.0
    assert rewards[4] == pytest.approx(np.log(105.0 / 100.0))


def test_new_streak_after_valid_step_is_penalized_again() -> None:
    h = _history([100.0, float("nan"), 100.0, 101.0, float("nan"), float("nan")])

    rewards = list(h["reward"])
    assert rewards[1] == _INVALID_VALUATION_PENALTY
    assert rewards[2] == 0.0  # recovering from step 1's invalid prev
    assert rewards[3] == pytest.approx(np.log(101.0 / 100.0))
    assert rewards[4] == _INVALID_VALUATION_PENALTY  # new streak, penalized again
    assert rewards[5] == 0.0  # latched
