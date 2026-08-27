"""Tests for TA-Lib-derived feature compute() logic."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

talib = pytest.importorskip("talib")

from trading_rl.features.base import FeatureConfig  # noqa: E402
from trading_rl.features.talib_features import ADFeature, OBVFeature  # noqa: E402


def _obv_df(n: int = 6) -> pd.DataFrame:
    close = pd.Series(np.arange(n, dtype=float) + 100.0)
    volume = pd.Series(np.full(n, 10.0))
    return pd.DataFrame(
        {"close": close, "high": close + 0.5, "low": close - 0.5, "volume": volume}
    )


class TestOBVFeature:
    """OBV is cumulative and signed; pct_change() is not a valid stationary
    transform for it (issue #456)."""

    def test_stays_finite_across_exact_zero_crossing(self, monkeypatch):
        # Synthetic OBV that ticks through exactly zero at index 4->5, the
        # concrete failure mode from the issue: pct_change().fillna(0) does
        # not catch inf (fillna only replaces NaN).
        synthetic_obv = np.array([100.0, -50.0, 30.0, 30.0, 0.0, 40.0])
        monkeypatch.setattr(
            "trading_rl.features.talib_features.talib.OBV",
            lambda close, volume: synthetic_obv,
        )
        feat = OBVFeature(FeatureConfig(name="obv", feature_type="obv", params={}))

        result = feat.compute(_obv_df())

        assert np.all(np.isfinite(result.values))

    def test_preserves_sign_when_obv_crosses_from_negative_to_positive(
        self, monkeypatch
    ):
        # OBV rising from -50 to 30 is a real increase (bullish); the old
        # pct_change formula reports this as negative because it divides by
        # the negative prior value: (30 - -50) / -50 = -1.6.
        synthetic_obv = np.array([100.0, -50.0, 30.0, 30.0, 0.0, 40.0])
        monkeypatch.setattr(
            "trading_rl.features.talib_features.talib.OBV",
            lambda close, volume: synthetic_obv,
        )
        feat = OBVFeature(FeatureConfig(name="obv", feature_type="obv", params={}))

        result = feat.compute(_obv_df())

        assert result.iloc[2] > 0

    def test_all_finite_on_real_talib_computation(self):
        close = pd.Series([100.0] * 5 + [99.0] * 5 + [101.0] * 20)
        volume = pd.Series(np.random.default_rng(0).uniform(50, 150, len(close)))
        df = pd.DataFrame({"close": close, "volume": volume})
        feat = OBVFeature(FeatureConfig(name="obv", feature_type="obv", params={}))

        result = feat.compute(df)

        assert np.all(np.isfinite(result.values))


class TestADFeature:
    """AD is cumulative and signed, same failure mode as OBV (issue #456)."""

    def test_stays_finite_across_exact_zero_crossing(self, monkeypatch):
        synthetic_ad = np.array([100.0, -50.0, 30.0, 30.0, 0.0, 40.0])
        monkeypatch.setattr(
            "trading_rl.features.talib_features.talib.AD",
            lambda high, low, close, volume: synthetic_ad,
        )
        feat = ADFeature(FeatureConfig(name="ad", feature_type="ad", params={}))

        result = feat.compute(_obv_df())

        assert np.all(np.isfinite(result.values))

    def test_all_finite_on_real_talib_computation(self):
        close = pd.Series([100.0] * 5 + [99.0] * 5 + [101.0] * 20)
        volume = pd.Series(np.random.default_rng(0).uniform(50, 150, len(close)))
        df = pd.DataFrame(
            {"close": close, "high": close + 0.5, "low": close - 0.5, "volume": volume}
        )
        feat = ADFeature(FeatureConfig(name="ad", feature_type="ad", params={}))

        result = feat.compute(df)

        assert np.all(np.isfinite(result.values))
