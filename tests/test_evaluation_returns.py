from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from trading_rl.evaluation.returns import extract_tradingenv_returns


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

    cumulative = extract_tradingenv_returns(env, n_steps=2)

    expected = np.cumsum([0.0, np.log(101.0 / 100.0), np.log(102.0 / 101.0)])
    np.testing.assert_allclose(cumulative, expected)
