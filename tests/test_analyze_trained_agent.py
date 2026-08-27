"""Regression test: analyze_trained_agent must roll out with the trained
agent's actor, not TorchRL's default random policy (issue #466)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import torch

from trading_rl.evaluation import explainability as ex


class TestAnalyzeTrainedAgentUsesTrainedPolicy:
    def test_rollout_is_driven_by_trainer_actor(self, monkeypatch):
        actor = object()
        trainer = SimpleNamespace(actor=actor)

        env = MagicMock()
        env.rollout.return_value = {"observation": torch.randn(10, 3)}

        analyzer = MagicMock()
        analyzer.compute_global_importance.return_value = pd.DataFrame(
            {"feature": ["f0"], "importance": [1.0]}
        )
        analyzer.plot_importance.return_value = object()
        monkeypatch.setattr(ex, "RLInterpretabilityAnalyzer", lambda *a, **k: analyzer)

        ex.analyze_trained_agent(
            trainer, env, feature_names=["f0", "f1", "f2"], n_steps=10
        )

        env.rollout.assert_called_once_with(max_steps=10, policy=actor)
