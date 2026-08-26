"""Regression tests for session-aware feature normalization."""

import numpy as np
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


@pytest.mark.parametrize(
    ("reset_on_session_break", "expected_session_open"),
    [(True, 0.0), (False, 99.5 / (0.5001**0.5))],
)
def test_rolling_normalization_honors_session_reset(
    reset_on_session_break, expected_session_open
):
    idx = pd.to_datetime(
        [
            "2024-01-02 09:30:00",
            "2024-01-02 09:31:00",
            "2024-01-03 09:30:00",
        ]
    )
    df = pd.DataFrame({"x": [10.0, 11.0, 110.0]}, index=idx)
    feature = ColumnValueFeature(
        FeatureConfig(
            name="x",
            feature_type="column_value",
            params={"column": "x"},
            normalize=True,
            normalization_method="rolling",
            rolling_window=1000,
            reset_on_session_break=reset_on_session_break,
            session_break_threshold_hours=1.0,
        )
    )

    feature.fit(df)
    result = feature.transform(df)

    assert result.iloc[2] == pytest.approx(expected_session_open)


def test_time_weighted_running_normalization_rejects_infinite_values():
    """An inf value must not poison the time-weighted running scaler's state
    for the rest of the session — only the offending row should read 0.0."""
    idx = pd.to_datetime(
        [
            "2024-01-02 09:30:00",
            "2024-01-02 09:31:00",
            "2024-01-02 09:32:00",
        ]
    )
    df = pd.DataFrame({"x": [float("inf"), 5.0, 6.0]}, index=idx)
    feature = ColumnValueFeature(
        FeatureConfig(
            name="x",
            feature_type="column_value",
            params={"column": "x"},
            normalize=True,
            normalization_method="running",
            use_time_weights=True,
            reset_on_session_break=True,
            session_break_threshold_hours=1.0,
        )
    )

    feature.fit(df)
    result = feature.transform(df)

    assert result.iloc[0] == pytest.approx(0.0)
    # Real signal after the inf must survive, not be silently zeroed by a
    # poisoned (inf/NaN) scaler state.
    assert result.iloc[2] != pytest.approx(0.0)
    assert np.isfinite(result.iloc[2])
