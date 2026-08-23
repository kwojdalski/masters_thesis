"""Exact-value tests for LogReturnFeature, HighFeature, LowFeature, SimpleReturnFeature."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_rl.features.base import FeatureConfig
from trading_rl.features.price_features import (
    HighFeature,
    LogReturnFeature,
    LowFeature,
    SimpleReturnFeature,
)


def _cfg(feature_type: str, **params) -> FeatureConfig:
    return FeatureConfig(name=feature_type, feature_type=feature_type, params=params)


def _ohlcv(close, high=None, low=None) -> pd.DataFrame:
    n = len(close)
    return pd.DataFrame(
        {
            "close": close,
            "high": high if high is not None else [c * 1.02 for c in close],
            "low": low if low is not None else [c * 0.98 for c in close],
            "volume": [1000.0] * n,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="1min"),
    )


# ---------------------------------------------------------------------------
# LogReturnFeature
# ---------------------------------------------------------------------------


class TestLogReturnFeature:
    def test_first_row_is_zero(self):
        df = _ohlcv([100.0, 110.0, 99.0])
        result = LogReturnFeature(_cfg("log_return")).compute(df)
        assert result.iloc[0] == pytest.approx(0.0, abs=1e-12)

    def test_10pct_gain_equals_log_1pt1(self):
        df = _ohlcv([100.0, 110.0])
        result = LogReturnFeature(_cfg("log_return")).compute(df)
        assert result.iloc[1] == pytest.approx(np.log(1.1), rel=1e-10)

    def test_10pct_loss_equals_log_0pt9(self):
        df = _ohlcv([100.0, 90.0])
        result = LogReturnFeature(_cfg("log_return")).compute(df)
        assert result.iloc[1] == pytest.approx(np.log(0.9), rel=1e-10)

    def test_same_price_gives_zero(self):
        df = _ohlcv([50.0, 50.0, 50.0])
        result = LogReturnFeature(_cfg("log_return")).compute(df)
        np.testing.assert_allclose(result.values, [0.0, 0.0, 0.0], atol=1e-12)

    def test_length_matches_input(self):
        df = _ohlcv([100.0, 101.0, 102.0, 103.0])
        result = LogReturnFeature(_cfg("log_return")).compute(df)
        assert len(result) == 4

    def test_three_step_sequence(self):
        df = _ohlcv([100.0, 110.0, 99.0])
        result = LogReturnFeature(_cfg("log_return")).compute(df)
        np.testing.assert_allclose(
            result.values,
            [0.0, np.log(110.0 / 100.0), np.log(99.0 / 110.0)],
            rtol=1e-10,
        )

    def test_zero_close_raises_instead_of_silently_zeroing(self):
        df = _ohlcv([100.0, 0.0, 101.0])
        with pytest.raises(ValueError, match="non-finite"):
            LogReturnFeature(_cfg("log_return")).compute(df)


# ---------------------------------------------------------------------------
# HighFeature
# ---------------------------------------------------------------------------


class TestHighFeature:
    def test_exact_value_high_above_close(self):
        df = _ohlcv([100.0], high=[105.0])
        result = HighFeature(_cfg("high")).compute(df)
        assert result.iloc[0] == pytest.approx(0.05, rel=1e-9)

    def test_high_equals_close_gives_zero(self):
        df = _ohlcv([100.0], high=[100.0])
        result = HighFeature(_cfg("high")).compute(df)
        assert result.iloc[0] == pytest.approx(0.0, abs=1e-12)

    def test_values_are_non_negative(self):
        closes = [100.0, 101.0, 99.0, 98.0]
        highs = [c * 1.01 for c in closes]
        df = _ohlcv(closes, high=highs)
        result = HighFeature(_cfg("high")).compute(df)
        assert (result.values >= 0.0).all()

    def test_three_step_exact(self):
        df = pd.DataFrame(
            {"close": [100.0, 200.0, 50.0], "high": [110.0, 210.0, 55.0]},
            index=pd.date_range("2024-01-01", periods=3, freq="1min"),
        )
        result = HighFeature(_cfg("high")).compute(df)
        np.testing.assert_allclose(result.values, [0.10, 0.05, 0.10], rtol=1e-9)


# ---------------------------------------------------------------------------
# LowFeature
# ---------------------------------------------------------------------------


class TestLowFeature:
    def test_exact_value_low_below_close(self):
        df = _ohlcv([100.0], low=[95.0])
        result = LowFeature(_cfg("low")).compute(df)
        assert result.iloc[0] == pytest.approx(-0.05, rel=1e-9)

    def test_low_equals_close_gives_zero(self):
        df = _ohlcv([100.0], low=[100.0])
        result = LowFeature(_cfg("low")).compute(df)
        assert result.iloc[0] == pytest.approx(0.0, abs=1e-12)

    def test_values_are_non_positive(self):
        closes = [100.0, 101.0, 99.0, 98.0]
        lows = [c * 0.99 for c in closes]
        df = _ohlcv(closes, low=lows)
        result = LowFeature(_cfg("low")).compute(df)
        assert (result.values <= 0.0).all()

    def test_low_is_negative_of_high_when_symmetric(self):
        closes = [100.0, 200.0]
        df = pd.DataFrame(
            {
                "close": closes,
                "high": [c * 1.05 for c in closes],
                "low": [c * 0.95 for c in closes],
            },
            index=pd.date_range("2024-01-01", periods=2, freq="1min"),
        )
        high = HighFeature(_cfg("high")).compute(df)
        low = LowFeature(_cfg("low")).compute(df)
        np.testing.assert_allclose(low.values, -high.values, rtol=1e-9)


# ---------------------------------------------------------------------------
# SimpleReturnFeature
# ---------------------------------------------------------------------------


class TestSimpleReturnFeature:
    def test_first_row_is_zero(self):
        df = _ohlcv([100.0, 110.0])
        result = SimpleReturnFeature(_cfg("simple_return")).compute(df)
        assert result.iloc[0] == pytest.approx(0.0, abs=1e-12)

    def test_10pct_gain(self):
        df = _ohlcv([100.0, 110.0])
        result = SimpleReturnFeature(_cfg("simple_return")).compute(df)
        assert result.iloc[1] == pytest.approx(0.10, rel=1e-9)

    def test_10pct_loss(self):
        df = _ohlcv([100.0, 90.0])
        result = SimpleReturnFeature(_cfg("simple_return")).compute(df)
        assert result.iloc[1] == pytest.approx(-0.10, rel=1e-9)

    def test_simple_vs_log_return_diverge_for_large_move(self):
        df = _ohlcv([100.0, 200.0])
        simple = SimpleReturnFeature(_cfg("simple_return")).compute(df).iloc[1]
        log = LogReturnFeature(_cfg("log_return")).compute(df).iloc[1]
        assert simple == pytest.approx(1.0, rel=1e-9)
        assert log == pytest.approx(np.log(2.0), rel=1e-9)
        assert abs(simple - log) > 0.1

    def test_three_step_sequence(self):
        df = _ohlcv([100.0, 110.0, 99.0])
        result = SimpleReturnFeature(_cfg("simple_return")).compute(df)
        np.testing.assert_allclose(result.values, [0.0, 0.10, -0.10], rtol=1e-9)
