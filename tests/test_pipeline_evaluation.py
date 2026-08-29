from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd

import trading_rl.pipeline.evaluation as evaluation_module
from trading_rl.evaluation import EvaluationContext


class _NoRolloutEnv:
    def rollout(self, *, max_steps: int, policy):
        raise AssertionError(
            "statistical tests should reuse precomputed strategy returns"
        )


def test_statistical_tests_reuse_precomputed_strategy_returns(
    monkeypatch,
) -> None:
    captured: dict[str, np.ndarray] = {}
    split_ctx = EvaluationContext(
        split="test",
        df=pd.DataFrame({"close": [100.0, 101.0, 102.0]}),
        env=_NoRolloutEnv(),
        max_steps=2,
    )
    config = SimpleNamespace(
        env=SimpleNamespace(price_column="close", reward_type="log_return"),
        benchmarks=SimpleNamespace(is_random=False),
        data=SimpleNamespace(timeframe="1d"),
        statistical_testing=SimpleNamespace(
            log_to_research_artifacts=False,
            research_artifact_subdir="statistics",
        ),
    )
    expected_returns = np.array([0.01, -0.02])

    monkeypatch.setattr(
        evaluation_module.BenchmarkEngine,
        "build",
        staticmethod(
            lambda df, benchmarks_config, price_column: (
                {"buy_hold": np.array([0.0, 0.0])},
                {},
            )
        ),
    )

    def fake_run_all_statistical_tests(**kwargs):
        captured["strategy_returns"] = kwargs["strategy_returns"]
        return {}

    monkeypatch.setattr(
        evaluation_module,
        "run_all_statistical_tests",
        fake_run_all_statistical_tests,
    )
    monkeypatch.setattr(
        evaluation_module.MLflowTrainingCallback,
        "log_statistical_tests",
        staticmethod(lambda *args, **kwargs: None),
    )

    evaluation_module.run_statistical_tests_for_split(
        trainer=SimpleNamespace(actor=object()),
        split_ctx=split_ctx,
        config=config,
        logger=logging.getLogger(__name__),
        strategy_simple_returns=expected_returns,
    )

    np.testing.assert_array_equal(captured["strategy_returns"], expected_returns)
