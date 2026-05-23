"""Tests for evaluation observation sample artifacts."""

import pandas as pd

from trading_rl.callbacks.artifacts import save_observation_sample_artifact


def test_save_observation_sample_artifact_preserves_index_and_row_cap(tmp_path) -> None:
    df = pd.DataFrame(
        {
            "close": [100.0, 100.1, 100.2],
            "feature_x": [0.1, 0.2, 0.3],
        },
        index=pd.date_range("2026-03-02 14:30:00", periods=3, freq="s"),
    )

    out_path = save_observation_sample_artifact(
        split="val/AAPL",
        df=df,
        output_dir=tmp_path,
        max_rows=2,
    )

    saved = pd.read_parquet(out_path)

    assert out_path.name == "val_AAPL_observations_head_2.parquet"
    assert len(saved) == 2
    pd.testing.assert_index_equal(saved.index, df.head(2).index)
    pd.testing.assert_frame_equal(saved, df.head(2), check_freq=False)
