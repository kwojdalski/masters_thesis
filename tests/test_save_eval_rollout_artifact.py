"""Regression test: multi-asset weight vectors must serialize to one column
per asset, not a single misaligned "action" column (issue #473)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_rl.callbacks.artifacts_evaluation import save_eval_rollout_artifact


class TestSaveEvalRolloutArtifactMultiAsset:
    def test_multi_asset_actions_get_one_column_per_asset(self, tmp_path):
        last_positions = [[0.8, -0.1], [-0.5, 0.2], [0.1, 0.9]]
        simple_returns = np.array([0.01, -0.02, 0.03])
        df_index = pd.RangeIndex(3)

        out_path = save_eval_rollout_artifact(
            split="test",
            last_positions=last_positions,
            simple_returns=simple_returns,
            cumulative_returns=None,
            df_index=df_index,
            output_dir=tmp_path,
        )

        df = pd.read_parquet(out_path)
        assert "action_0" in df.columns
        assert "action_1" in df.columns
        assert "action" not in df.columns
        np.testing.assert_allclose(df["action_0"], [0.8, -0.5, 0.1], atol=1e-6)
        np.testing.assert_allclose(df["action_1"], [-0.1, 0.2, 0.9], atol=1e-6)

    def test_single_asset_actions_stay_one_action_column(self, tmp_path):
        last_positions = [0.8, -0.5, 0.1]
        simple_returns = np.array([0.01, -0.02, 0.03])
        df_index = pd.RangeIndex(3)

        out_path = save_eval_rollout_artifact(
            split="test",
            last_positions=last_positions,
            simple_returns=simple_returns,
            cumulative_returns=None,
            df_index=df_index,
            output_dir=tmp_path,
        )

        df = pd.read_parquet(out_path)
        assert list(df.columns) == ["action", "simple_return"]
        np.testing.assert_allclose(df["action"], [0.8, -0.5, 0.1], atol=1e-6)
