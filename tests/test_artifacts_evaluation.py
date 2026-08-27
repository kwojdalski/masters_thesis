"""Regression tests for artifacts_evaluation.py bugfinder fixes (#458, #462)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from trading_rl.callbacks import artifacts_evaluation as ae


class TestAvgPositionChangeRatioTransitionNormalization:
    def test_ratio_normalizes_by_transitions_not_raw_step_count(self, monkeypatch):
        # Two episodes of length 4 (n=4, n-1=3 transitions each), each with
        # 3 recorded position changes -- i.e. every possible transition
        # changed position, so the ratio must be able to reach 1.0.
        logged_metrics: dict[str, float] = {}
        monkeypatch.setattr(
            ae.mlflow,
            "log_metric",
            lambda name, value, **_: logged_metrics.__setitem__(name, value),
        )
        monkeypatch.setattr(ae.mlflow, "log_artifact", MagicMock())

        training_callback = SimpleNamespace(
            get_training_curves=lambda: {
                "episode_rewards": [1.0, 1.0],
                "portfolio_valuations": [100.0, 100.0],
                "position_change_counts": [3, 3],
            },
            training_stats={"actions_taken": [0.0] * 8},
        )

        ae.log_final_metrics(
            {}, {"final_reward": 1.0}, training_callback=training_callback
        )

        assert logged_metrics["episode_avg_position_change_ratio"] == 1.0

    def test_single_step_episodes_fall_back_to_one_without_zero_division(
        self, monkeypatch
    ):
        logged_metrics: dict[str, float] = {}
        monkeypatch.setattr(
            ae.mlflow,
            "log_metric",
            lambda name, value, **_: logged_metrics.__setitem__(name, value),
        )
        monkeypatch.setattr(ae.mlflow, "log_artifact", MagicMock())

        training_callback = SimpleNamespace(
            get_training_curves=lambda: {
                "episode_rewards": [1.0],
                "portfolio_valuations": [100.0],
                "position_change_counts": [0],
            },
            training_stats={"actions_taken": [0.0]},
        )

        ae.log_final_metrics(
            {}, {"final_reward": 1.0}, training_callback=training_callback
        )

        assert np.isfinite(logged_metrics["episode_avg_position_change_ratio"])


class TestLogEvaluationReportLogsMetrics:
    def test_finite_fields_are_logged_as_metrics(self, monkeypatch):
        logged_metrics: dict[str, float] = {}
        monkeypatch.setattr(ae.mlflow, "active_run", lambda: True)
        monkeypatch.setattr(
            ae.mlflow,
            "log_metric",
            lambda name, value, **_: logged_metrics.__setitem__(name, value),
        )
        monkeypatch.setattr(ae.mlflow, "log_artifact", MagicMock())

        ae.log_evaluation_report(
            {"sharpe": 1.5, "max_drawdown": -0.2}, split_prefix="val"
        )

        assert logged_metrics["val_sharpe"] == 1.5
        assert logged_metrics["val_max_drawdown"] == -0.2

    def test_non_finite_fields_are_not_logged_as_metrics(self, monkeypatch):
        logged_metrics: dict[str, float] = {}
        monkeypatch.setattr(ae.mlflow, "active_run", lambda: True)
        monkeypatch.setattr(
            ae.mlflow,
            "log_metric",
            lambda name, value, **_: logged_metrics.__setitem__(name, value),
        )
        monkeypatch.setattr(ae.mlflow, "log_artifact", MagicMock())

        ae.log_evaluation_report({"sharpe": float("nan"), "sortino": 2.0})

        assert "sharpe" not in logged_metrics
        assert logged_metrics["sortino"] == 2.0
