"""Regression tests for session-aware feature normalization."""

import pandas as pd
import pytest

from trading_rl.features.base import FeatureConfig
from trading_rl.features.column_features import ColumnValueFeature


def test_running_normalization_resets_and_updates_within_session():
    idx = pd.to_datetime(
        [
            "2024-01-02 09:30:00",
            "2024-01-02 09:31:00",
            "2024-01-02 09:32:00",
            "2024-01-03 09:30:00",
            "2024-01-03 09:31:00",
            "2024-01-03 09:32:00",
        ]
    )
    df = pd.DataFrame({"x": [10.0, 11.0, 12.0, 110.0, 111.0, 112.0]}, index=idx)
    feature = ColumnValueFeature(
        FeatureConfig(
            name="x",
            feature_type="column_value",
            params={"column": "x"},
            normalize=True,
            normalization_method="running",
            reset_on_session_break=True,
            session_break_threshold_hours=1.0,
        )
    )

    feature.fit(df)
    result = feature.transform(df)

    assert not result.equals(df["x"])
    assert result.iloc[0] == pytest.approx(0.0)
    assert result.iloc[1] != pytest.approx(0.0)
    assert result.iloc[2] != pytest.approx(0.0)
    assert result.iloc[3] == pytest.approx(0.0)
    assert result.iloc[4] != pytest.approx(0.0)
    assert result.iloc[5] != pytest.approx(0.0)
