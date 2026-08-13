from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_rl.features.base import FeatureConfig
from trading_rl.features.price_features import (
    ReturnLagFeature,
    RSIFeature,
    TrendFeature,
)
from trading_rl.features.temporal_features import (
    DayOfWeekCosFeature,
    DayOfWeekSinFeature,
    HourCosFeature,
    HourSinFeature,
    MinuteOfHourCosFeature,
    MinuteOfHourSinFeature,
)
from trading_rl.features.volatility_features import (
    RealizedVolatilityFeature,
    VolatilityRatioFeature,
)
from trading_rl.features.volume_features import (
    AmihudIlliquidityFeature,
    LogVolumeFeature,
    RelativeVolumeFeature,
    VolumeChangeFeature,
    VolumeMAFeature,
)


def _cfg(feature_type: str, **params) -> FeatureConfig:
    return FeatureConfig(name=feature_type, feature_type=feature_type, params=params)


def _priced_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "close": [100.0, 110.0, 99.0, 108.9],
            "volume": [0.0, 10.0, 20.0, 40.0],
        },
        index=pd.date_range("2024-01-01 00:00:00", periods=4, freq="15min"),
    )


def test_hour_sin_is_one_at_6am() -> None:
    df = pd.DataFrame(index=pd.DatetimeIndex(["2024-01-01 06:00:00"]))

    result = HourSinFeature(_cfg("hour_sin")).compute(df)

    assert result.iloc[0] == pytest.approx(1.0)


def test_hour_cos_is_minus_one_at_noon() -> None:
    df = pd.DataFrame(index=pd.DatetimeIndex(["2024-01-01 12:00:00"]))

    result = HourCosFeature(_cfg("hour_cos")).compute(df)

    assert result.iloc[0] == pytest.approx(-1.0)


def test_day_of_week_encoding_starts_at_monday() -> None:
    df = pd.DataFrame(index=pd.DatetimeIndex(["2024-01-01"]))  # Monday

    sin_value = DayOfWeekSinFeature(_cfg("day_of_week_sin")).compute(df).iloc[0]
    cos_value = DayOfWeekCosFeature(_cfg("day_of_week_cos")).compute(df).iloc[0]

    assert sin_value == pytest.approx(0.0)
    assert cos_value == pytest.approx(1.0)


def test_minute_encoding_wraps_within_hour() -> None:
    df = pd.DataFrame(
        index=pd.DatetimeIndex(["2024-01-01 09:15:00", "2024-01-01 09:30:00"])
    )

    sin_values = MinuteOfHourSinFeature(_cfg("minute_of_hour_sin")).compute(df)
    cos_values = MinuteOfHourCosFeature(_cfg("minute_of_hour_cos")).compute(df)

    assert sin_values.iloc[0] == pytest.approx(1.0)
    assert cos_values.iloc[1] == pytest.approx(-1.0)


def test_temporal_features_reject_non_datetime_index() -> None:
    df = pd.DataFrame(index=[0, 1])

    with pytest.raises(ValueError, match="DatetimeIndex"):
        HourSinFeature(_cfg("hour_sin")).compute(df)


def test_log_volume_uses_log1p() -> None:
    result = LogVolumeFeature(_cfg("log_volume")).compute(_priced_frame())

    np.testing.assert_allclose(result.to_numpy(), np.log1p([0.0, 10.0, 20.0, 40.0]))


def test_volume_change_handles_previous_zero_with_unit_denominator() -> None:
    result = VolumeChangeFeature(_cfg("volume_change")).compute(_priced_frame())

    np.testing.assert_allclose(result.to_numpy(), [0.0, 9.0, 1.0, 1.0])


def test_volume_ma_ratio_uses_causal_rolling_average() -> None:
    result = VolumeMAFeature(_cfg("volume_ma_ratio", window=2)).compute(_priced_frame())

    np.testing.assert_allclose(result.iloc[1:], [1.0, 1.0 / 3.0, 1.0 / 3.0], rtol=1e-7)


def test_relative_volume_uses_rolling_average() -> None:
    result = RelativeVolumeFeature(_cfg("relative_volume", window=2)).compute(
        _priced_frame()
    )

    np.testing.assert_allclose(result.iloc[1:], [2.0, 4.0 / 3.0, 4.0 / 3.0], rtol=1e-7)


def test_amihud_illiquidity_uses_absolute_log_return_over_volume() -> None:
    result = AmihudIlliquidityFeature(_cfg("amihud_illiquidity")).compute(
        _priced_frame()
    )

    expected_second = abs(np.log(110.0 / 100.0)) / 10.0
    assert result.iloc[0] == pytest.approx(0.0)
    assert result.iloc[1] == pytest.approx(expected_second)


def test_realized_volatility_matches_rolling_log_return_std() -> None:
    df = _priced_frame()
    log_returns = np.log(df["close"] / df["close"].shift(1)).fillna(0.0)
    expected = log_returns.rolling(window=2, min_periods=1).std().fillna(0.0)

    result = RealizedVolatilityFeature(_cfg("realized_volatility", window=2)).compute(
        df
    )

    np.testing.assert_allclose(result, expected)


def test_volatility_ratio_rejects_invalid_windows() -> None:
    feature = VolatilityRatioFeature(
        _cfg("volatility_ratio", short_window=5, long_window=5)
    )

    with pytest.raises(ValueError, match="short_window"):
        feature.compute(_priced_frame())


def test_volatility_ratio_matches_short_over_long_rolling_volatility() -> None:
    df = pd.DataFrame({"close": [100.0, 102.0, 101.0, 104.0, 108.0]})
    log_returns = np.log(df["close"] / df["close"].shift(1)).fillna(0.0)
    rv_short = log_returns.rolling(window=2, min_periods=1).std()
    rv_long = log_returns.rolling(window=4, min_periods=1).std()
    expected = (rv_short / (rv_long + 1e-8)).fillna(0.0)

    result = VolatilityRatioFeature(
        _cfg("volatility_ratio", short_window=2, long_window=4)
    ).compute(df)

    np.testing.assert_allclose(result.to_numpy(), expected.to_numpy(), rtol=1e-10)


def test_trend_is_relative_to_first_close() -> None:
    result = TrendFeature(_cfg("trend")).compute(_priced_frame())

    np.testing.assert_allclose(result.to_numpy(), [1.0, 1.1, 0.99, 1.089])


def test_trend_resets_at_session_boundary_instead_of_using_first_row_of_whole_frame() -> (
    None
):
    """Regression test for #279: prepare_data() calls compute() once on a
    concatenated train+val+test frame for caching. Without a session-aware
    reset, every row -- including val/test -- would be relative to the very
    first row of the whole frame (i.e. the start of the training split)."""
    df = pd.DataFrame(
        {"close": [100.0, 110.0, 120.0, 132.0]},
        index=pd.DatetimeIndex(
            [
                "2024-01-01 09:30:00",  # session 1 (e.g. "train")
                "2024-01-01 09:45:00",  # session 1
                "2024-01-02 09:30:00",  # session 2 (e.g. "val"), >1h gap from prior row
                "2024-01-02 09:45:00",  # session 2
            ]
        ),
    )

    result = TrendFeature(_cfg("trend")).compute(df)

    # session 1: relative to its own first close (100.0)
    # session 2: relative to its own first close (120.0), NOT 100.0
    np.testing.assert_allclose(result.to_numpy(), [1.0, 1.1, 1.0, 1.1])


def test_trend_session_threshold_is_configurable() -> None:
    df = pd.DataFrame(
        {"close": [100.0, 150.0]},
        index=pd.DatetimeIndex(["2024-01-01 09:00:00", "2024-01-01 10:30:00"]),
    )

    # 1.5h gap: below a 2h threshold (one session) vs above a 1h threshold (two sessions)
    one_session_cfg = FeatureConfig(
        name="trend", feature_type="trend", session_break_threshold_hours=2.0
    )
    two_sessions_cfg = FeatureConfig(
        name="trend", feature_type="trend", session_break_threshold_hours=1.0
    )
    one_session = TrendFeature(one_session_cfg).compute(df)
    two_sessions = TrendFeature(two_sessions_cfg).compute(df)

    np.testing.assert_allclose(one_session.to_numpy(), [1.0, 1.5])
    np.testing.assert_allclose(two_sessions.to_numpy(), [1.0, 1.0])


def test_return_lag_uses_past_return_not_current_return() -> None:
    result = ReturnLagFeature(_cfg("return_lag", lag=1)).compute(_priced_frame())

    np.testing.assert_allclose(result.to_numpy(), [0.0, 0.0, 0.10, -0.10])


def test_rsi_is_normalized_to_minus_one_one_range() -> None:
    result = RSIFeature(_cfg("rsi", period=2)).compute(_priced_frame())

    assert np.all(result >= -1.0)
    assert np.all(result <= 1.0)


def test_rsi_matches_wilder_ewm_formula_exactly() -> None:
    df = pd.DataFrame({"close": [100.0, 103.0, 101.0, 105.0, 104.0]})
    period = 3
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    expected = (100 - (100 / (1 + rs)) - 50) / 50

    result = RSIFeature(_cfg("rsi", period=period)).compute(df)

    np.testing.assert_allclose(result.to_numpy(), expected.to_numpy(), rtol=1e-10)
    assert result.iloc[1] == pytest.approx(1.0, abs=3e-8)
    assert result.iloc[2] < result.iloc[1]
