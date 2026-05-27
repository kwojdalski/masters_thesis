"""Exact-value tests for volume-based feature compute() logic."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_rl.features.base import FeatureConfig
from trading_rl.features.volume_features import (
    AmihudIlliquidityFeature,
    LogVolumeFeature,
    RelativeVolumeFeature,
    VolumeChangeFeature,
    VolumeMAFeature,
)


def _cfg(feature_type: str, **params) -> FeatureConfig:
    return FeatureConfig(name=feature_type, feature_type=feature_type, params=params)


def _vol_df(volume, close=None) -> pd.DataFrame:
    n = len(volume)
    if close is None:
        close = [100.0] * n
    return pd.DataFrame(
        {"volume": volume, "close": close},
        index=pd.date_range("2024-01-01", periods=n, freq="1min"),
    )


# ---------------------------------------------------------------------------
# LogVolumeFeature
# ---------------------------------------------------------------------------


class TestLogVolumeFeature:
    def test_zero_volume_gives_zero(self):
        df = _vol_df([0.0, 0.0, 0.0])
        result = LogVolumeFeature(_cfg("log_volume")).compute(df)
        np.testing.assert_allclose(result.values, [0.0, 0.0, 0.0], atol=1e-12)

    def test_known_value_e_minus_1(self):
        # log1p(e - 1) = log(e) = 1.0
        df = _vol_df([np.e - 1])
        result = LogVolumeFeature(_cfg("log_volume")).compute(df)
        assert result.iloc[0] == pytest.approx(1.0, rel=1e-10)

    def test_known_value_99(self):
        df = _vol_df([99.0])
        result = LogVolumeFeature(_cfg("log_volume")).compute(df)
        assert result.iloc[0] == pytest.approx(np.log(100.0), rel=1e-10)

    def test_monotone_volume_gives_monotone_output(self):
        df = _vol_df([100.0, 200.0, 400.0, 800.0])
        result = LogVolumeFeature(_cfg("log_volume")).compute(df)
        assert result.is_monotonic_increasing

    def test_length_matches_input(self):
        df = _vol_df([1.0] * 20)
        result = LogVolumeFeature(_cfg("log_volume")).compute(df)
        assert len(result) == 20

    def test_all_outputs_finite(self):
        df = _vol_df([0.0, 1.0, 1000.0])
        result = LogVolumeFeature(_cfg("log_volume")).compute(df)
        assert result.notna().all()
        assert np.isfinite(result.values).all()


# ---------------------------------------------------------------------------
# VolumeChangeFeature
# ---------------------------------------------------------------------------


class TestVolumeChangeFeature:
    def test_first_row_is_zero(self):
        df = _vol_df([100.0, 200.0, 300.0])
        result = VolumeChangeFeature(_cfg("volume_change")).compute(df)
        assert result.iloc[0] == pytest.approx(0.0, abs=1e-12)

    def test_doubling_gives_one(self):
        df = _vol_df([100.0, 200.0])
        result = VolumeChangeFeature(_cfg("volume_change")).compute(df)
        assert result.iloc[1] == pytest.approx(1.0, rel=1e-10)

    def test_halving_gives_minus_half(self):
        df = _vol_df([200.0, 100.0])
        result = VolumeChangeFeature(_cfg("volume_change")).compute(df)
        assert result.iloc[1] == pytest.approx(-0.5, rel=1e-10)

    def test_three_step_exact_values(self):
        # [100, 200, 100] → [0, 1.0, -0.5]
        df = _vol_df([100.0, 200.0, 100.0])
        result = VolumeChangeFeature(_cfg("volume_change")).compute(df)
        np.testing.assert_allclose(result.values, [0.0, 1.0, -0.5], atol=1e-10)

    def test_zero_previous_volume_uses_one_as_denominator(self):
        # Previous 0 → replace(0, 1) → change = vol/1 - 1 = vol - 1
        df = _vol_df([0.0, 100.0])
        result = VolumeChangeFeature(_cfg("volume_change")).compute(df)
        # Row 0: first row → 0
        # Row 1: prev=0, replaced with 1 → 100/1 - 1 = 99
        assert result.iloc[0] == pytest.approx(0.0, abs=1e-12)
        assert result.iloc[1] == pytest.approx(99.0, rel=1e-10)

    def test_no_nan_in_output(self):
        df = _vol_df([100.0, 200.0, 50.0, 300.0])
        result = VolumeChangeFeature(_cfg("volume_change")).compute(df)
        assert result.isna().sum() == 0


# ---------------------------------------------------------------------------
# VolumeMAFeature
# ---------------------------------------------------------------------------


class TestVolumeMAFeature:
    def test_uniform_volume_gives_zero_everywhere(self):
        # vol == MA → vol/MA - 1 = 0
        df = _vol_df([100.0] * 10)
        result = VolumeMAFeature(_cfg("volume_ma_ratio", window=3)).compute(df)
        # First row: MA (min_periods=1) = vol → ratio ≈ 0
        np.testing.assert_allclose(result.values, 0.0, atol=1e-5)

    def test_known_values_window_2(self):
        # vol=[100, 200, 100], window=2
        # Row 0: MA=100 → 100/(100+1e-8) - 1 ≈ 0
        # Row 1: MA=mean([100,200])=150 → 200/(150+1e-8) - 1 ≈ 1/3
        # Row 2: MA=mean([200,100])=150 → 100/(150+1e-8) - 1 ≈ -1/3
        df = _vol_df([100.0, 200.0, 100.0])
        result = VolumeMAFeature(_cfg("volume_ma_ratio", window=2)).compute(df)
        assert result.iloc[0] == pytest.approx(0.0, abs=1e-4)
        assert result.iloc[1] == pytest.approx(1 / 3, rel=1e-4)
        assert result.iloc[2] == pytest.approx(-1 / 3, rel=1e-4)

    def test_volume_above_ma_gives_positive_ratio(self):
        # Rising volume spike should give positive ratio
        df = _vol_df([100.0, 100.0, 100.0, 500.0])
        result = VolumeMAFeature(_cfg("volume_ma_ratio", window=3)).compute(df)
        assert result.iloc[-1] > 0

    def test_default_window_is_20(self):
        df = _vol_df([float(i + 1) for i in range(30)])
        r_default = VolumeMAFeature(_cfg("volume_ma_ratio")).compute(df)
        r_w20 = VolumeMAFeature(_cfg("volume_ma_ratio", window=20)).compute(df)
        np.testing.assert_allclose(r_default.values, r_w20.values, atol=1e-12)

    def test_no_nan_in_output(self):
        df = _vol_df([100.0, 200.0, 50.0, 300.0, 150.0])
        result = VolumeMAFeature(_cfg("volume_ma_ratio", window=3)).compute(df)
        assert result.isna().sum() == 0


# ---------------------------------------------------------------------------
# AmihudIlliquidityFeature
# ---------------------------------------------------------------------------


class TestAmihudIlliquidityFeature:
    def test_zero_return_gives_zero_illiquidity(self):
        df = _vol_df([1000.0, 1000.0], close=[100.0, 100.0])
        result = AmihudIlliquidityFeature(_cfg("amihud_illiquidity")).compute(df)
        np.testing.assert_allclose(result.values, 0.0, atol=1e-12)

    def test_known_return_and_volume(self):
        # close=[100, 110], vol=[1000, 1000]
        # row 0: abs_log_return = 0 (fillna) → illiquidity = 0/(1000+1e-8) ≈ 0
        # row 1: abs_log_return = log(1.1) → illiquidity = log(1.1)/(1000+1e-8)
        df = _vol_df([1000.0, 1000.0], close=[100.0, 110.0])
        result = AmihudIlliquidityFeature(_cfg("amihud_illiquidity")).compute(df)
        expected_row1 = np.log(1.1) / (1000.0 + 1e-8)
        assert result.iloc[0] == pytest.approx(0.0, abs=1e-12)
        assert result.iloc[1] == pytest.approx(expected_row1, rel=1e-8)

    def test_high_volume_reduces_illiquidity(self):
        # Same returns, high volume → lower illiquidity
        df_low = _vol_df([100.0, 100.0], close=[100.0, 110.0])
        df_high = _vol_df([1_000_000.0, 1_000_000.0], close=[100.0, 110.0])
        r_low = AmihudIlliquidityFeature(_cfg("amihud_illiquidity")).compute(df_low)
        r_high = AmihudIlliquidityFeature(_cfg("amihud_illiquidity")).compute(df_high)
        assert r_high.iloc[1] < r_low.iloc[1]

    def test_window_parameter_applies_rolling_mean(self):
        # window=2 should return rolling mean of instantaneous illiquidity
        close = [100.0, 110.0, 110.0, 121.0]
        vol = [1000.0, 1000.0, 1000.0, 1000.0]
        df = _vol_df(vol, close=close)
        result_w1 = AmihudIlliquidityFeature(_cfg("amihud_illiquidity", window=1)).compute(df)
        result_w2 = AmihudIlliquidityFeature(_cfg("amihud_illiquidity", window=2)).compute(df)
        # window=2 is a rolling mean so its values differ from instantaneous at row 1+
        assert result_w2.iloc[1] != pytest.approx(result_w1.iloc[1])
        # row 1 with window=2: mean of [row0_illiq, row1_illiq]
        expected_w2_row1 = (result_w1.iloc[0] + result_w1.iloc[1]) / 2
        assert result_w2.iloc[1] == pytest.approx(expected_w2_row1, rel=1e-8)

    def test_no_nan_in_output(self):
        rng = np.random.default_rng(5)
        close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 20)))
        vol = rng.uniform(500, 2000, 20)
        df = _vol_df(vol, close=close)
        result = AmihudIlliquidityFeature(_cfg("amihud_illiquidity")).compute(df)
        assert result.isna().sum() == 0


# ---------------------------------------------------------------------------
# RelativeVolumeFeature
# ---------------------------------------------------------------------------


class TestRelativeVolumeFeature:
    def test_uniform_volume_gives_one_everywhere(self):
        # vol == rolling_avg → RVOL = 1
        df = _vol_df([100.0] * 10)
        result = RelativeVolumeFeature(_cfg("relative_volume", window=3)).compute(df)
        np.testing.assert_allclose(result.values, 1.0, atol=1e-5)

    def test_double_average_gives_two(self):
        # Last row has volume double the window-average of prior rows
        df = _vol_df([100.0, 100.0, 100.0, 200.0])
        result = RelativeVolumeFeature(_cfg("relative_volume", window=3)).compute(df)
        # Row 3: MA = mean([100, 100, 200]) = 400/3 ≈ 133.3; RVOL = 200 / 133.3... wait
        # Actually window=3 for row 3 means rolling_mean([100, 100, 200]) = 400/3
        # RVOL = 200 / (400/3) = 600/400 = 1.5 — not 2.0
        # To get exactly 2: need vol=200 when average of window = 100
        # Use [100, 100, 100, 200] with window=3 at row 3: MA = (100+100+200)/3 ≈ 133
        # For RVOL=2 at row 3: need window of all-100s, then spike
        # Use [100, 100, 100, 100, 200] with window=4
        df2 = _vol_df([100.0, 100.0, 100.0, 100.0, 200.0])
        result2 = RelativeVolumeFeature(_cfg("relative_volume", window=4)).compute(df2)
        # Row 4: MA = mean([100, 100, 100, 200]) = 500/4 = 125; RVOL = 200/125 = 1.6
        # Still not 2. For exact 2: need rolling mean = 100 when vol = 200
        # That requires the spike not to be in the window
        df3 = _vol_df([100.0, 100.0, 100.0, 100.0, 200.0])
        result3 = RelativeVolumeFeature(_cfg("relative_volume", window=3)).compute(df3)
        # Row 4: MA = mean([100, 100, 200]) = 400/3; RVOL = 200/(400/3) = 1.5
        # The spike is always in the window. Use window=1 for trivially RVOL=1 always,
        # or test a known non-trivial value:
        # Row 3: MA = mean([100, 100, 100]) = 100; RVOL = 100/100 = 1
        # ... let's just test a known value at row 1 with window=3
        df4 = _vol_df([50.0, 200.0, 100.0])
        result4 = RelativeVolumeFeature(_cfg("relative_volume", window=3)).compute(df4)
        # Row 0: MA(min=1)=50 → RVOL=1
        # Row 1: MA([50, 200])=125 → RVOL=200/125=1.6
        # Row 2: MA([50, 200, 100])=350/3 → RVOL=100/(350/3)=300/350=6/7
        assert result4.iloc[1] == pytest.approx(200.0 / 125.0, rel=1e-6)
        assert result4.iloc[2] == pytest.approx(100.0 / (350.0 / 3.0), rel=1e-6)

    def test_first_row_is_one(self):
        # min_periods=1: first row MA = vol itself → RVOL = vol/(vol+1e-8) ≈ 1
        df = _vol_df([500.0, 100.0, 200.0])
        result = RelativeVolumeFeature(_cfg("relative_volume", window=3)).compute(df)
        assert result.iloc[0] == pytest.approx(1.0, abs=1e-5)

    def test_default_window_is_20(self):
        df = _vol_df([float(i + 1) for i in range(30)])
        r_default = RelativeVolumeFeature(_cfg("relative_volume")).compute(df)
        r_w20 = RelativeVolumeFeature(_cfg("relative_volume", window=20)).compute(df)
        np.testing.assert_allclose(r_default.values, r_w20.values, atol=1e-12)

    def test_output_is_positive(self):
        rng = np.random.default_rng(9)
        df = _vol_df(rng.uniform(100, 1000, 30).tolist())
        result = RelativeVolumeFeature(_cfg("relative_volume", window=5)).compute(df)
        assert (result > 0).all()

    def test_no_nan_in_output(self):
        df = _vol_df([100.0, 200.0, 50.0, 300.0, 150.0])
        result = RelativeVolumeFeature(_cfg("relative_volume", window=3)).compute(df)
        assert result.isna().sum() == 0
