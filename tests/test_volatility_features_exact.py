"""Exact-value tests for RealizedVolatilityFeature and VolatilityRatioFeature."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_rl.features.base import FeatureConfig
from trading_rl.features.volatility_features import (
    RealizedVolatilityFeature,
    VolatilityRatioFeature,
)


def _cfg(feature_type: str, **params) -> FeatureConfig:
    return FeatureConfig(name=feature_type, feature_type=feature_type, params=params)


def _close_df(close) -> pd.DataFrame:
    return pd.DataFrame(
        {"close": close},
        index=pd.date_range("2024-01-01", periods=len(close), freq="1min"),
    )


# ---------------------------------------------------------------------------
# RealizedVolatilityFeature
# ---------------------------------------------------------------------------


class TestRealizedVolatilityFeature:
    def test_constant_price_gives_zero_everywhere(self):
        df = _close_df([100.0, 100.0, 100.0, 100.0])
        result = RealizedVolatilityFeature(
            _cfg("realized_volatility", window=3)
        ).compute(df)
        np.testing.assert_allclose(result.values, [0.0, 0.0, 0.0, 0.0], atol=1e-12)

    def test_first_row_is_zero(self):
        # First row log-return is NaN (no prior price), fillna(0) → std of [0] → NaN → fillna → 0
        df = _close_df([100.0, 110.0, 120.0])
        result = RealizedVolatilityFeature(
            _cfg("realized_volatility", window=3)
        ).compute(df)
        assert result.iloc[0] == pytest.approx(0.0, abs=1e-12)

    def test_known_returns_window_3(self):
        # Construct close prices so that log-returns are 0, 0.1, 0.2, 0.3
        log_rets = np.array([0.0, 0.1, 0.2, 0.3])
        close = np.exp(
            np.cumsum(log_rets)
        )  # cumulative so pairwise diff = each log_ret
        # After fillna(0), log_returns series = [0, 0.1, 0.2, 0.3]
        # window=3, min_periods=1, ddof=1:
        #   row 0: std([0]) = NaN → 0
        #   row 1: std([0, 0.1]) = 0.1 / sqrt(2)
        #   row 2: std([0, 0.1, 0.2]) — mean=0.1, deviations=[-0.1,0,+0.1], var=0.02/2=0.01, std=0.1
        #   row 3: std([0.1, 0.2, 0.3]) — mean=0.2, same spread → std=0.1
        df = _close_df(close)
        result = RealizedVolatilityFeature(
            _cfg("realized_volatility", window=3)
        ).compute(df)

        assert result.iloc[0] == pytest.approx(0.0, abs=1e-12)
        assert result.iloc[1] == pytest.approx(0.1 / np.sqrt(2), rel=1e-9)
        assert result.iloc[2] == pytest.approx(0.1, rel=1e-9)
        assert result.iloc[3] == pytest.approx(0.1, rel=1e-9)

    def test_window_parameter_limits_lookback(self):
        # Same data, window=2 vs window=4 should differ on early rows
        log_rets = [0.0, 0.1, 0.2, 0.3, 0.4]
        close = np.exp(np.cumsum(log_rets))
        df = _close_df(close)

        rv2 = RealizedVolatilityFeature(_cfg("realized_volatility", window=2)).compute(
            df
        )
        rv4 = RealizedVolatilityFeature(_cfg("realized_volatility", window=4)).compute(
            df
        )

        # At row 4 (index 4): window=2 sees [0.3, 0.4]; window=4 sees [0.1,0.2,0.3,0.4]
        # std([0.3,0.4]) = 0.1/sqrt(2); std([0.1,0.2,0.3,0.4]) ≠ 0.1/sqrt(2)
        assert rv2.iloc[4] != pytest.approx(rv4.iloc[4])

    def test_default_window_is_20(self):
        n = 25
        close = 100.0 * np.exp(np.cumsum(np.full(n, 0.01)))
        df = _close_df(close)

        rv_default = RealizedVolatilityFeature(_cfg("realized_volatility")).compute(df)
        rv_w20 = RealizedVolatilityFeature(
            _cfg("realized_volatility", window=20)
        ).compute(df)

        np.testing.assert_allclose(rv_default.values, rv_w20.values, atol=1e-12)

    def test_output_is_non_negative(self):
        rng = np.random.default_rng(42)
        close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 50)))
        df = _close_df(close)

        result = RealizedVolatilityFeature(
            _cfg("realized_volatility", window=10)
        ).compute(df)
        assert (result >= 0).all()

    def test_length_matches_input(self):
        df = _close_df([100.0] * 30)
        result = RealizedVolatilityFeature(
            _cfg("realized_volatility", window=5)
        ).compute(df)
        assert len(result) == 30

    def test_no_nan_in_output(self):
        rng = np.random.default_rng(7)
        close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 40)))
        df = _close_df(close)

        result = RealizedVolatilityFeature(
            _cfg("realized_volatility", window=5)
        ).compute(df)
        assert result.isna().sum() == 0


# ---------------------------------------------------------------------------
# VolatilityRatioFeature
# ---------------------------------------------------------------------------


class TestVolatilityRatioFeature:
    def test_raises_when_short_window_equals_long_window(self):
        with pytest.raises(ValueError, match="short_window must be < long_window"):
            VolatilityRatioFeature(
                _cfg("volatility_ratio", short_window=10, long_window=10)
            ).compute(_close_df([100.0] * 20))

    def test_raises_when_short_window_greater_than_long_window(self):
        with pytest.raises(ValueError, match="short_window must be < long_window"):
            VolatilityRatioFeature(
                _cfg("volatility_ratio", short_window=20, long_window=10)
            ).compute(_close_df([100.0] * 25))

    def test_raises_when_short_window_zero(self):
        with pytest.raises(ValueError, match="must be > 0"):
            VolatilityRatioFeature(
                _cfg("volatility_ratio", short_window=0, long_window=10)
            ).compute(_close_df([100.0] * 15))

    def test_raises_when_long_window_zero(self):
        with pytest.raises(ValueError, match="must be > 0"):
            VolatilityRatioFeature(
                _cfg("volatility_ratio", short_window=2, long_window=0)
            ).compute(_close_df([100.0] * 10))

    def test_constant_price_gives_zero_ratio(self):
        # All log returns are 0 → both short and long vols are 0 → ratio = 0/(0+ε) = 0
        df = _close_df([100.0] * 20)
        result = VolatilityRatioFeature(
            _cfg("volatility_ratio", short_window=2, long_window=4)
        ).compute(df)
        np.testing.assert_allclose(result.values, 0.0, atol=1e-6)

    def test_known_ratio_exact_value(self):
        # close = [1, 1, 1, 1, 1, e]  → log_returns = [0, 0, 0, 0, 0, 1] (spike at index 5)
        # At index 5 with short_window=2, long_window=4:
        #   short_std = std([0, 1]) = 1/sqrt(2)
        #   long_std  = std([0, 0, 0, 1]) — mean=0.25, var=0.25, std=0.5
        #   ratio = (1/sqrt(2)) / (0.5 + 1e-8) ≈ sqrt(2)
        close = np.concatenate([[1.0], np.exp(np.cumsum([0.0, 0.0, 0.0, 0.0, 1.0]))])
        df = _close_df(close)
        result = VolatilityRatioFeature(
            _cfg("volatility_ratio", short_window=2, long_window=4)
        ).compute(df)
        assert result.iloc[5] == pytest.approx(np.sqrt(2) / 2 / (0.5 + 1e-8), rel=1e-5)

    def test_short_vol_spike_gives_ratio_greater_than_one(self):
        # Recent volatile period → short vol > long vol → ratio > 1
        rng = np.random.default_rng(0)
        n = 100
        # Calm baseline then volatile period at the end
        returns = np.concatenate(
            [
                rng.normal(0, 0.001, n - 10),
                rng.normal(0, 0.1, 10),  # spike
            ]
        )
        close = 100.0 * np.exp(np.cumsum(returns))
        df = _close_df(close)

        result = VolatilityRatioFeature(
            _cfg("volatility_ratio", short_window=5, long_window=20)
        ).compute(df)
        # At the end of the spike, short_vol should dominate
        assert result.iloc[-1] > 1.0

    def test_no_nan_in_output(self):
        rng = np.random.default_rng(3)
        close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 50)))
        df = _close_df(close)

        result = VolatilityRatioFeature(
            _cfg("volatility_ratio", short_window=3, long_window=10)
        ).compute(df)
        assert result.isna().sum() == 0

    def test_length_matches_input(self):
        df = _close_df([100.0 + i for i in range(30)])
        result = VolatilityRatioFeature(
            _cfg("volatility_ratio", short_window=3, long_window=10)
        ).compute(df)
        assert len(result) == 30
