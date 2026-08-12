"""Tests for BenchmarkEngine, BenchmarkSpec, and BenchmarkReturnArray."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from trading_rl.constants import BenchmarkName
from trading_rl.evaluation.benchmarks import (
    BenchmarkEngine,
    BenchmarkReturnArray,
    BenchmarkSpec,
)


def _prices(n: int = 10, start: float = 100.0, step: float = 1.0) -> pd.Series:
    return pd.Series([start + i * step for i in range(n)])


def _volumes(n: int = 10) -> pd.Series:
    return pd.Series([1000.0 - i * 50 for i in range(n)])


# ---------------------------------------------------------------------------
# BenchmarkReturnArray
# ---------------------------------------------------------------------------


class TestBenchmarkReturnArray:
    def test_carries_position_side_metadata(self):
        arr = BenchmarkReturnArray(
            np.array([0.01, -0.02, 0.03]), benchmark_position_side=1.0
        )
        assert arr.benchmark_position_side == pytest.approx(1.0)

    def test_negative_position_side_for_short(self):
        arr = BenchmarkReturnArray(np.array([0.01, 0.02]), benchmark_position_side=-1.0)
        assert arr.benchmark_position_side == pytest.approx(-1.0)

    def test_none_position_side_accepted(self):
        arr = BenchmarkReturnArray(np.zeros(5), benchmark_position_side=None)
        assert arr.benchmark_position_side is None

    def test_behaves_as_numpy_array(self):
        arr = BenchmarkReturnArray(
            np.array([0.01, 0.02, 0.03]), benchmark_position_side=1.0
        )
        assert arr.shape == (3,)
        assert arr.sum() == pytest.approx(0.06)

    def test_metadata_preserved_after_arithmetic(self):
        arr = BenchmarkReturnArray(
            np.array([1.0, 2.0, 3.0]), benchmark_position_side=1.0
        )
        doubled = arr * 2
        assert isinstance(doubled, BenchmarkReturnArray)
        assert doubled.benchmark_position_side == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# BenchmarkEngine.buy_and_hold
# ---------------------------------------------------------------------------


class TestBuyAndHold:
    def test_returns_benchmark_spec(self):
        spec = BenchmarkEngine.buy_and_hold(_prices())
        assert isinstance(spec, BenchmarkSpec)

    def test_name_is_buy_and_hold(self):
        spec = BenchmarkEngine.buy_and_hold(_prices())
        assert spec.name == BenchmarkName.BUY_AND_HOLD

    def test_length_equals_max_steps(self):
        spec = BenchmarkEngine.buy_and_hold(_prices(20))
        returns = spec.compute_returns(10)
        assert len(returns) == 10

    def test_returns_are_positive_for_rising_prices(self):
        prices = _prices(10, start=100.0, step=1.0)
        spec = BenchmarkEngine.buy_and_hold(prices)
        returns = spec.compute_returns(9)
        assert np.all(np.asarray(returns) > 0.0)

    def test_returns_are_negative_for_falling_prices(self):
        prices = pd.Series([100.0 - i for i in range(10)])
        spec = BenchmarkEngine.buy_and_hold(prices)
        returns = spec.compute_returns(9)
        assert np.all(np.asarray(returns) < 0.0)

    def test_position_side_is_long(self):
        spec = BenchmarkEngine.buy_and_hold(_prices())
        arr = spec.compute_returns(5)
        assert arr.benchmark_position_side == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# BenchmarkEngine.short_and_hold
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# BenchmarkEngine.twap
# ---------------------------------------------------------------------------


class TestTwap:
    def test_returns_benchmark_spec(self):
        spec = BenchmarkEngine.twap(_prices())
        assert isinstance(spec, BenchmarkSpec)

    def test_name_is_twap(self):
        spec = BenchmarkEngine.twap(_prices())
        assert spec.name == BenchmarkName.TWAP

    def test_returns_are_finite(self):
        spec = BenchmarkEngine.twap(_prices(20))
        returns = spec.compute_returns(19)
        assert np.all(np.isfinite(np.asarray(returns)))

    def test_length_equals_max_steps(self):
        spec = BenchmarkEngine.twap(_prices(15))
        returns = spec.compute_returns(10)
        assert len(returns) == 10


# ---------------------------------------------------------------------------
# BenchmarkEngine.vwap
# ---------------------------------------------------------------------------


class TestVwap:
    def test_returns_benchmark_spec(self):
        spec = BenchmarkEngine.vwap(_prices(10), _volumes(10))
        assert isinstance(spec, BenchmarkSpec)

    def test_name_is_vwap(self):
        spec = BenchmarkEngine.vwap(_prices(10), _volumes(10))
        assert spec.name == BenchmarkName.VWAP

    def test_returns_are_finite(self):
        spec = BenchmarkEngine.vwap(_prices(10), _volumes(10))
        returns = spec.compute_returns(9)
        assert np.all(np.isfinite(np.asarray(returns)))

    def test_vwap_differs_from_twap_with_skewed_volume(self):
        prices = _prices(11, step=1.0)
        # Heavy volume at start — VWAP entry favours early, TWAP uniform
        volumes = pd.Series([10000.0 - i * 1000 for i in range(10)])
        twap_r = BenchmarkEngine.twap(prices).compute_returns(10)
        vwap_r = BenchmarkEngine.vwap(prices, volumes).compute_returns(10)
        assert not np.allclose(twap_r, vwap_r)

    def test_volume_source_stored_in_metadata(self):
        spec = BenchmarkEngine.vwap(
            _prices(10), _volumes(10), volume_source="bid_volume"
        )
        assert spec.metadata.get("volume_source") == "bid_volume"


# ---------------------------------------------------------------------------
# BenchmarkEngine.build
# ---------------------------------------------------------------------------


class TestBenchmarkEngineBuild:
    def _market_data(self, n: int = 10) -> pd.DataFrame:
        prices = [100.0 + i for i in range(n)]
        volumes = [1000.0] * n
        return pd.DataFrame({"close": prices, "volume": volumes})

    def test_no_config_flags_returns_empty(self):
        cfg = SimpleNamespace(
            buy_and_hold=False, short_and_hold=False, twap=False, vwap=False
        )
        specs, _meta = BenchmarkEngine.build(self._market_data(), cfg)
        assert specs == []

    def test_buy_and_hold_flag_adds_one_spec(self):
        cfg = SimpleNamespace(
            buy_and_hold=True, short_and_hold=False, twap=False, vwap=False
        )
        specs, _ = BenchmarkEngine.build(self._market_data(), cfg)
        assert len(specs) == 1
        assert specs[0].name == BenchmarkName.BUY_AND_HOLD

    def test_missing_price_column_returns_empty(self):
        df = pd.DataFrame({"volume": [1000.0] * 5})
        cfg = SimpleNamespace(
            buy_and_hold=True, short_and_hold=False, twap=False, vwap=False
        )
        specs, _ = BenchmarkEngine.build(df, cfg, price_column="close")
        assert specs == []

    def test_all_flags_returns_at_least_two_specs(self):
        cfg = SimpleNamespace(
            buy_and_hold=True, short_and_hold=False, twap=True, vwap=False
        )
        specs, _ = BenchmarkEngine.build(self._market_data(), cfg)
        assert len(specs) == 2
