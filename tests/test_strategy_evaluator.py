from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from tensordict import TensorDict

from trading_rl.evaluation.evaluator import (
    EvaluatorEnvConfig,
    StrategyEvaluatorConfig,
    StrategyEvaluator,
)
from trading_rl.evaluation.returns import ReturnKind


class _RolloutEnv:
    def __init__(self) -> None:
        self.rollout_calls = 0

    def rollout(self, *, max_steps: int, policy: object, callback=None) -> TensorDict:
        self.rollout_calls += 1
        return TensorDict(
            {
                "action": torch.zeros(max_steps, 1),
                "next": TensorDict(
                    {"reward": torch.zeros(max_steps, 1)},
                    batch_size=[max_steps],
                ),
            },
            batch_size=[max_steps],
        )


def test_evaluate_split_uses_provided_environment_without_rebuilding() -> None:
    def forbidden_factory(_df: pd.DataFrame, _config: StrategyEvaluatorConfig) -> object:
        raise AssertionError("evaluation should use the already-built split env")

    env = _RolloutEnv()
    evaluator = StrategyEvaluator(
        env_factory=forbidden_factory,
        policy=object(),
        config=StrategyEvaluatorConfig(
            backend="tradingenv",
            max_steps=2,
            enable_plots=False,
            enable_metrics=False,
        ),
    )

    result = evaluator.evaluate_split(
        "test",
        pd.DataFrame({"close": [100.0, 101.0, 102.0]}),
        env=env,
    )

    assert env.rollout_calls == 1
    assert result.rollout is not None
    assert result.return_series is not None
    assert result.return_series.kind == ReturnKind.LOG
    np.testing.assert_allclose(result.simple_returns, np.zeros(2))


def test_shaped_rewards_without_broker_do_not_become_returns() -> None:
    def forbidden_factory(_df: pd.DataFrame, _config: StrategyEvaluatorConfig) -> object:
        raise AssertionError("evaluation should use the already-built split env")

    env = _RolloutEnv()
    evaluator = StrategyEvaluator(
        env_factory=forbidden_factory,
        policy=object(),
        config=StrategyEvaluatorConfig(
            reward_type="differential_sharpe",
            backend="tradingenv",
            max_steps=2,
            enable_plots=False,
            enable_metrics=False,
        ),
    )

    result = evaluator.evaluate_split(
        "test",
        pd.DataFrame({"close": [100.0, 101.0, 102.0]}),
        env=env,
    )

    assert env.rollout_calls == 1
    assert result.simple_returns.size == 0


def test_extract_last_positions_maps_one_hot_actions_to_configured_positions() -> None:
    evaluator = StrategyEvaluator(
        env_factory=lambda _df, _config: object(),
        policy=object(),
        config=StrategyEvaluatorConfig(
            backend="gym_trading_env.discrete",
            env=EvaluatorEnvConfig(positions=[-1, 0, 1]),
        ),
    )
    actions = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    positions = evaluator._extract_last_positions(actions, max_steps=3)

    assert positions == [-1, 0, 1]

    single_position = evaluator._extract_last_positions(
        torch.tensor([[0.0, 0.0, 1.0]]),
        max_steps=1,
    )

    assert single_position == [1]
