from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from trading_rl.constants import RewardType
from trading_rl.evaluation.returns import ReturnKind, ReturnSeries
from trading_rl.trainers.base import _run_evaluation
from trading_rl.trainers.ddpg import DDPGTrainer
from trading_rl.trainers.ppo import PPOTrainer
from trading_rl.trainers.td3 import TD3Trainer


class _EvalConfig:
    eval_steps = 2

    def __init__(self, resolved_steps: int = 3) -> None:
        self.resolved_steps = resolved_steps
        self.requested_lengths: list[int] = []

    def resolve_eval_steps(self, data_len: int) -> int:
        self.requested_lengths.append(data_len)
        return self.resolved_steps


class _RolloutEnv:
    def __init__(self, rewards: list[float], step_counts: list[int]) -> None:
        self.rewards = rewards
        self.step_counts = step_counts
        self.calls: list[tuple[int, Any]] = []

    def rollout(self, n_eval: int, actor: Any) -> TensorDict:
        self.calls.append((n_eval, actor))
        n = len(self.rewards)
        return TensorDict(
            {
                "step_count": torch.tensor(self.step_counts, dtype=torch.int64),
                "next": TensorDict(
                    {
                        "reward": torch.tensor(self.rewards, dtype=torch.float32),
                    },
                    batch_size=[n],
                ),
            },
            batch_size=[n],
        )


def _minimal_experiment_config() -> SimpleNamespace:
    return SimpleNamespace(
        env=SimpleNamespace(
            reward_type=RewardType.LOG_RETURN,
            backend="tradingenv",
            price_column="close",
            name="eval-env",
            positions=[-1, 0, 1],
            mode="mft",
            trading_fees=0.0,
            borrow_interest_rate=0.0,
            initial_portfolio_value=1234.0,
        ),
        training=SimpleNamespace(
            max_plot_points=7,
            show_allocation_ma=False,
            allocation_ma_window=4,
        ),
        evaluation=SimpleNamespace(eval_plots=("rewards", "positions", "portfolio_value")),
        benchmarks=None,
        data=SimpleNamespace(data_paths=["a.csv", "b.csv"]),
    )


def test_run_evaluation_returns_expected_plots_and_records_result(monkeypatch) -> None:
    calls: dict[str, Any] = {}

    class FakeEvaluator:
        def __init__(self, env_factory, policy, config):
            calls["env_factory_env"] = env_factory(None, None)
            calls["policy"] = policy
            calls["config"] = config

        def evaluate_split(self, split, df, env):
            calls["evaluate_split"] = (split, df, env)
            return SimpleNamespace(
                plots={"reward_plot": "reward-plot", "action_plot": "action-plot"},
                final_reward=1.25,
                last_positions=[-1.0, 0.0, 1.0],
                simple_returns=np.array([0.01, -0.02]),
                return_series=ReturnSeries(
                    np.array([0.01, -0.02]), ReturnKind.SIMPLE
                ),
            )

    def fake_equity_curve_plot(*args, **kwargs):
        calls["equity_kwargs"] = kwargs
        return "equity-plot"

    def fake_merged_plot(reward_plot, action_plot, equity_plot):
        calls["merged_args"] = (reward_plot, action_plot, equity_plot)
        return "merged-plot"

    import trading_rl.evaluation.evaluator as evaluator_module
    import trading_rl.utils as utils_module

    monkeypatch.setattr(evaluator_module, "StrategyEvaluator", FakeEvaluator)
    monkeypatch.setattr(utils_module, "create_equity_curve_plot", fake_equity_curve_plot)
    monkeypatch.setattr(utils_module, "create_merged_comparison_plot", fake_merged_plot)

    trainer = SimpleNamespace(
        actor="actor",
        env="train-env",
        total_count=11,
        total_episodes=3,
        _last_evaluation_result=None,
    )
    df = SimpleNamespace(name="prices")

    result = _run_evaluation(
        trainer,
        df,
        max_steps=4,
        config=_minimal_experiment_config(),
        algorithm="PPO",
        eval_env="dedicated-eval-env",
    )

    assert result == (
        "reward-plot",
        "action-plot",
        None,
        pytest.approx(1.25),
        [-1.0, 0.0, 1.0],
        "equity-plot",
        "merged-plot",
    )
    assert trainer._last_evaluation_result.final_reward == pytest.approx(1.25)
    assert calls["env_factory_env"] == "dedicated-eval-env"
    assert calls["policy"] == "actor"
    assert calls["evaluate_split"] == ("eval", df, "dedicated-eval-env")
    assert calls["equity_kwargs"]["max_plot_points"] == 7
    assert calls["equity_kwargs"]["initial_portfolio_value"] == pytest.approx(1234.0)
    assert calls["equity_kwargs"]["training_steps"] == 11
    assert calls["equity_kwargs"]["training_episodes"] == 3
    assert calls["equity_kwargs"]["n_total_symbols"] == 2
    assert calls["merged_args"] == ("reward-plot", "action-plot", "equity-plot")


def test_ppo_evaluate_falls_back_after_mode_rollout_error() -> None:
    class FallbackEnv(_RolloutEnv):
        def rollout(self, n_eval: int, actor: Any) -> TensorDict:
            self.calls.append((n_eval, actor))
            if len(self.calls) == 1:
                raise RuntimeError("mode unavailable")
            n = len(self.rewards)
            return TensorDict(
                {
                    "step_count": torch.tensor(self.step_counts, dtype=torch.int64),
                    "next": TensorDict(
                        {
                            "reward": torch.tensor(self.rewards, dtype=torch.float32),
                        },
                        batch_size=[n],
                    ),
                },
                batch_size=[n],
            )

    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.eval_config = _EvalConfig(resolved_steps=5)
    trainer._eval_data_len = 17
    trainer._eval_env = FallbackEnv([1.0, 3.0], [1, 2])
    trainer.env = _RolloutEnv([9.0], [1])
    trainer.actor = "actor"
    trainer.logs = defaultdict(list)

    PPOTrainer._evaluate(trainer)

    assert trainer.eval_config.requested_lengths == [17]
    assert trainer._eval_env.calls == [(5, "actor"), (5, "actor")]
    assert trainer.env.calls == []
    assert trainer.logs["eval_reward_mean"] == pytest.approx([2.0])
    assert trainer.logs["eval_reward_sum"] == pytest.approx([4.0])
    assert trainer.logs["eval_step_count"] == [2]


def test_td3_evaluate_uses_dedicated_eval_env_and_logs_rollout_stats() -> None:
    trainer = TD3Trainer.__new__(TD3Trainer)
    trainer.eval_config = _EvalConfig(resolved_steps=4)
    trainer._eval_data_len = 23
    trainer._eval_env = _RolloutEnv([0.5, 1.5, -0.5], [1, 2, 3])
    trainer.env = _RolloutEnv([9.0], [1])
    trainer.actor = "td3-actor"
    trainer.logs = defaultdict(list)

    TD3Trainer._evaluate(trainer)

    assert trainer.eval_config.requested_lengths == [23]
    assert trainer._eval_env.calls == [(4, "td3-actor")]
    assert trainer.env.calls == []
    assert trainer.logs["eval_reward_mean"] == pytest.approx([0.5])
    assert trainer.logs["eval_reward_sum"] == pytest.approx([1.5])
    assert trainer.logs["eval_step_count"] == [3]


def test_ddpg_evaluate_uses_dedicated_eval_env_and_logs_rollout_stats() -> None:
    trainer = DDPGTrainer.__new__(DDPGTrainer)
    trainer.eval_config = _EvalConfig(resolved_steps=6)
    trainer._eval_data_len = 31
    trainer._eval_env = _RolloutEnv([2.0, -1.0, 4.0], [2, 4, 6])
    trainer.env = _RolloutEnv([9.0], [1])
    trainer.actor = "ddpg-actor"
    trainer.logs = defaultdict(list)

    DDPGTrainer._evaluate(trainer)

    assert trainer.eval_config.requested_lengths == [31]
    assert trainer._eval_env.calls == [(6, "ddpg-actor")]
    assert trainer.env.calls == []
    assert trainer.logs["eval_reward_mean"] == pytest.approx([5.0 / 3.0])
    assert trainer.logs["eval_reward_sum"] == pytest.approx([5.0])
    assert trainer.logs["eval_step_count"] == [6]
