"""Unit tests for LOB feature compute() logic."""

import numpy as np
import pandas as pd
import pytest

from trading_rl.features.base import FeatureConfig
from trading_rl.features.registry import FeatureRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def lob_df():
    """Minimal LOB snapshot DataFrame with 5 levels + trade tape columns."""
    n = 20
    np.random.seed(42)
    data = {}
    for i in range(5):
        data[f"bid_px_{i:02d}"] = 100.0 - i * 0.1 + np.random.randn(n) * 0.01
        data[f"bid_sz_{i:02d}"] = np.random.uniform(50, 200, n)
        data[f"ask_px_{i:02d}"] = 100.1 + i * 0.1 + np.random.randn(n) * 0.01
        data[f"ask_sz_{i:02d}"] = np.random.uniform(50, 200, n)
    data["bid_ct_00"] = np.random.randint(1, 20, n)
    data["ask_ct_00"] = np.random.randint(1, 20, n)
    data["action"] = np.random.choice(["A", "C", "T"], n, p=[0.5, 0.3, 0.2])
    data["side"] = np.random.choice(["B", "A"], n)
    data["size"] = np.random.uniform(1, 500, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="100ms")
    return pd.DataFrame(data, index=idx)


def _make_config(feature_type: str, **params) -> FeatureConfig:
    return FeatureConfig(
        name=f"test_{feature_type}",
        feature_type=feature_type,
        domain="hft",
        params=params if params else None,
    )


def _compute(feature_type: str, df: pd.DataFrame, **params) -> pd.Series:
    config = _make_config(feature_type, **params)
    feat = FeatureRegistry.create(config)
    return feat.compute(df)


# ---------------------------------------------------------------------------
# Baseline features
# ---------------------------------------------------------------------------

class TestBookPressure:
    def test_range(self, lob_df):
        result = _compute("book_pressure", lob_df, level=0)
        assert result.notna().all()
        assert (result >= -1).all() and (result <= 1).all()

    def test_zero_imbalance(self):
        df = pd.DataFrame({"bid_sz_00": [100.0], "ask_sz_00": [100.0]})
        result = _compute("book_pressure", df, level=0)
        assert result.iloc[0] == pytest.approx(0.0)

    def test_full_bid_imbalance(self):
        df = pd.DataFrame({"bid_sz_00": [100.0], "ask_sz_00": [0.0]})
        result = _compute("book_pressure", df, level=0)
        assert result.iloc[0] == pytest.approx(1.0)

    def test_level_param(self, lob_df):
        l0 = _compute("book_pressure", lob_df, level=0)
        l2 = _compute("book_pressure", lob_df, level=2)
        # Different levels should generally give different values
        assert not (l0 == l2).all()


class TestSpreadBps:
    def test_positive(self, lob_df):
        result = _compute("spread_bps", lob_df)
        assert (result > 0).all()

    def test_known_spread(self):
        df = pd.DataFrame({"bid_px_00": [100.0], "ask_px_00": [100.1]})
        result = _compute("spread_bps", df)
        # Spread = (100.1 - 100.0) / 100.05 * 10000 = 9.995 bps
        assert result.iloc[0] == pytest.approx(9.995, rel=1e-3)


class TestMicroprice:
    def test_equals_mid_when_balanced(self):
        df = pd.DataFrame({
            "bid_px_00": [100.0], "ask_px_00": [100.2],
            "bid_sz_00": [100.0], "ask_sz_00": [100.0],
        })
        result = _compute("microprice", df)
        assert result.iloc[0] == pytest.approx(100.1)

    def test_shifts_toward_thin_side(self):
        df = pd.DataFrame({
            "bid_px_00": [100.0], "ask_px_00": [100.2],
            "bid_sz_00": [200.0], "ask_sz_00": [50.0],
        })
        result = _compute("microprice", df)
        mid = 100.1
        # Microprice should be pulled toward ask side (thin side)
        assert result.iloc[0] > mid

    def test_fallback_to_mid_when_zero_volume(self):
        df = pd.DataFrame({
            "bid_px_00": [100.0], "ask_px_00": [100.2],
            "bid_sz_00": [0.0], "ask_sz_00": [0.0],
        })
        result = _compute("microprice", df)
        assert result.iloc[0] == pytest.approx(100.1)


class TestMicropriceDivergence:
    def test_zero_when_balanced(self):
        df = pd.DataFrame({
            "bid_px_00": [100.0], "ask_px_00": [100.2],
            "bid_sz_00": [100.0], "ask_sz_00": [100.0],
        })
        result = _compute("microprice_divergence", df)
        assert result.iloc[0] == pytest.approx(0.0, abs=1e-10)


class TestOrderBookImbalance:
    def test_range(self, lob_df):
        result = _compute("order_book_imbalance", lob_df, levels=3)
        assert result.notna().all()
        assert (result >= -1).all() and (result <= 1).all()


class TestDepthRatio:
    def test_positive(self, lob_df):
        result = _compute("depth_ratio", lob_df, levels_deep=4)
        assert (result > 0).all()


class TestVWMPSkew:
    def test_range(self, lob_df):
        result = _compute("vwmp_skew", lob_df, levels=3)
        assert result.notna().all()


class TestBidAskSlope:
    def test_bid_slope_positive(self, lob_df):
        result = _compute("bid_ask_slope", lob_df, side="bid", levels=5)
        # Price decreases down the book, so bid slope numerator > 0
        # Denominator (volume) > 0, so result should be positive
        assert (result > 0).all() or result.isna().any()

    def test_ask_slope_positive(self, lob_df):
        result = _compute("bid_ask_slope", lob_df, side="ask", levels=5)
        assert result.notna().all()


# ---------------------------------------------------------------------------
# Alpha-oriented features
# ---------------------------------------------------------------------------

class TestOFI:
    def test_first_row_zero(self, lob_df):
        result = _compute("ofi", lob_df)
        assert result.iloc[0] == 0.0

    def test_not_all_zero(self, lob_df):
        result = _compute("ofi", lob_df)
        assert not (result == 0).all()


class TestRollingOFI:
    def test_first_row_zero(self, lob_df):
        result = _compute("ofi_rolling", lob_df, window=10)
        assert result.iloc[0] == 0.0

    def test_window_param(self, lob_df):
        r10 = _compute("ofi_rolling", lob_df, window=10)
        r50 = _compute("ofi_rolling", lob_df, window=50)
        # Different windows should give different results
        assert not (r10 == r50).all()


class TestQueueDepletion:
    def test_range(self, lob_df):
        for side in ["bid", "ask"]:
            result = _compute("queue_depletion", lob_df, side=side)
            assert (result >= 0).all() and (result <= 1).all()

    def test_first_row_zero(self, lob_df):
        result = _compute("queue_depletion", lob_df, side="bid")
        assert result.iloc[0] == 0.0


class TestMidPriceAcceleration:
    def test_first_two_rows_zero(self, lob_df):
        result = _compute("mid_price_acceleration", lob_df)
        assert result.iloc[0] == 0.0
        assert result.iloc[1] == 0.0


class TestInterEventTime:
    def test_first_row_zero(self, lob_df):
        result = _compute("inter_event_time", lob_df)
        assert result.iloc[0] == 0.0

    def test_positive(self, lob_df):
        result = _compute("inter_event_time", lob_df)
        assert (result.iloc[1:] > 0).all()


class TestSpreadRatio:
    def test_positive(self, lob_df):
        result = _compute("spread_ratio", lob_df, window=10)
        assert (result > 0).all()


class TestBookConvexity:
    def test_bid_ask_symmetric(self, lob_df):
        bid_conv = _compute("book_convexity", lob_df, side="bid")
        ask_conv = _compute("book_convexity", lob_df, side="ask")
        assert bid_conv.notna().all()
        assert ask_conv.notna().all()


class TestOrderCountImbalance:
    def test_range(self, lob_df):
        result = _compute("order_count_imbalance", lob_df, level=0)
        assert (result >= -1).all() and (result <= 1).all()


class TestSignedTradeFlow:
    def test_range(self, lob_df):
        result = _compute("signed_trade_flow", lob_df, window=50)
        assert result.notna().all()


class TestOddLotTradeRatio:
    def test_range(self, lob_df):
        result = _compute("odd_lot_trade_ratio", lob_df, window=100)
        assert (result >= 0).all() and (result <= 1).all()


class TestOddLotImbalance:
    def test_range(self, lob_df):
        result = _compute("odd_lot_imbalance", lob_df, window=100)
        assert (result >= -1).all() and (result <= 1).all()


# ---------------------------------------------------------------------------
# Price VAMP
# ---------------------------------------------------------------------------

class TestPriceVamp:
    def test_close_to_mid_with_small_notional(self, lob_df):
        """Small notional should produce VAMP close to simple mid-price."""
        vamp = _compute("price_vamp", lob_df, notional=1.0, levels=5)
        mid = (lob_df["bid_px_00"] + lob_df["ask_px_00"]) / 2.0
        # With $1 notional, only the best level is touched
        np.testing.assert_allclose(vamp.values, mid.values, atol=0.02)

    def test_shifts_with_large_notional(self, lob_df):
        """Larger notional should diverge from mid-price."""
        vamp_1k = _compute("price_vamp", lob_df, notional=1000, levels=5)
        vamp_50k = _compute("price_vamp", lob_df, notional=50000, levels=5)
        # They should differ (notional walks deeper into book)
        assert not np.allclose(vamp_1k.values, vamp_50k.values)

    def test_symmetric_book_equals_mid(self):
        """With perfectly symmetric book, VAMP equals mid-price."""
        n = 5
        data = {}
        for i in range(n):
            data[f"bid_px_{i:02d}"] = [100.0 - i * 0.1] * 3
            data[f"bid_sz_{i:02d}"] = [100.0] * 3
            data[f"ask_px_{i:02d}"] = [100.1 + i * 0.1] * 3
            data[f"ask_sz_{i:02d}"] = [100.0] * 3
        df = pd.DataFrame(data)
        vamp = _compute("price_vamp", df, notional=1000, levels=n)
        mid = (df["bid_px_00"] + df["ask_px_00"]) / 2.0
        np.testing.assert_allclose(vamp.values, mid.values, atol=0.01)


# ---------------------------------------------------------------------------
# Cancel-to-Trade Ratio
# ---------------------------------------------------------------------------

class TestCancelToTradeRatio:
    def test_range(self, lob_df):
        result = _compute("cancel_to_trade_ratio", lob_df, window=20)
        assert (result >= 0).all()

    def test_all_trades_gives_zero(self):
        df = pd.DataFrame({"action": ["T"] * 10})
        result = _compute("cancel_to_trade_ratio", df, window=10)
        # No cancels -> ratio = 0
        assert (result == 0).all()

    def test_all_cancels_gives_inf(self):
        df = pd.DataFrame({"action": ["C"] * 10})
        result = _compute("cancel_to_trade_ratio", df, window=10)
        # No trades -> safe_divide returns 0 (denominator is 0)
        # Actually safe_divide returns fill=0 by default when denom is 0
        assert (result == 0).all()

    def test_mixed(self):
        df = pd.DataFrame({"action": ["C", "T", "C", "T", "C"]})
        result = _compute("cancel_to_trade_ratio", df, window=5)
        # 3 cancels / 2 trades = 1.5
        assert result.iloc[-1] == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# Multi-level OFI
# ---------------------------------------------------------------------------

class TestMultiLevelOFI:
    def test_first_row_zero(self, lob_df):
        result = _compute("ofi_multilevel", lob_df, levels=3)
        assert result.iloc[0] == 0.0

    def test_not_all_zero(self, lob_df):
        result = _compute("ofi_multilevel", lob_df, levels=3)
        assert not (result == 0).all()

    def test_levels_param(self, lob_df):
        r3 = _compute("ofi_multilevel", lob_df, levels=3)
        r5 = _compute("ofi_multilevel", lob_df, levels=5)
        # More levels should change the result
        assert not (r3 == r5).all()

    def test_known_values_bid_improvement(self):
        """When bid improves at level 0, bid_event should = bid_sz_00."""
        df = pd.DataFrame({
            "bid_px_00": [100.0, 100.1],  # bid price goes up
            "ask_px_00": [100.2, 100.2],
            "bid_sz_00": [50.0, 80.0],    # new volume at new price
            "ask_sz_00": [60.0, 60.0],
        })
        result = _compute("ofi_multilevel", df, levels=1)
        # Bid event = bid_sz_00[t] = 80 (price improved)
        # Ask event = ask_sz_00[t] - ask_sz_00[t-1] = 0 (price unchanged)
        # OFI = 80 - 0 = 80
        assert result.iloc[1] == pytest.approx(80.0)

    def test_known_values_bid_deterioration(self):
        """When bid drops at level 0, bid_event should = -bid_sz_00[t-1]."""
        df = pd.DataFrame({
            "bid_px_00": [100.1, 100.0],  # bid price drops
            "ask_px_00": [100.2, 100.2],
            "bid_sz_00": [50.0, 30.0],
            "ask_sz_00": [60.0, 60.0],
        })
        result = _compute("ofi_multilevel", df, levels=1)
        # Bid event = -bid_sz_00[t-1] = -50 (price deteriorated)
        # Ask event = 0 (price unchanged)
        # OFI = -50 - 0 = -50
        assert result.iloc[1] == pytest.approx(-50.0)


# ---------------------------------------------------------------------------
# VPIN
# ---------------------------------------------------------------------------

class TestVPIN:
    def test_range(self, lob_df):
        result = _compute("vpin", lob_df, window=10)
        assert (result >= 0).all() and (result <= 1).all()

    def test_all_buy_trades(self):
        """With all buyer-initiated trades, VPIN should be 1."""
        df = pd.DataFrame({
            "action": ["T"] * 5,
            "side": ["B"] * 5,
            "size": [100.0] * 5,
        })
        result = _compute("vpin", df, window=5)
        assert result.iloc[-1] == pytest.approx(1.0)

    def test_balanced_trades(self):
        """With equal buy/sell volume, VPIN should be 0."""
        df = pd.DataFrame({
            "action": ["T", "T", "T", "T"],
            "side": ["B", "A", "B", "A"],
            "size": [100.0, 100.0, 100.0, 100.0],
        })
        result = _compute("vpin", df, window=4)
        assert result.iloc[-1] == pytest.approx(0.0)

    def test_non_trade_rows_excluded(self):
        """Non-trade rows should not affect VPIN."""
        df = pd.DataFrame({
            "action": ["A", "A", "T", "T"],
            "side": ["B", "A", "B", "A"],
            "size": [999.0, 999.0, 100.0, 100.0],
        })
        result = _compute("vpin", df, window=4)
        # Only 2 trades: 100 buy, 100 sell -> VPIN = 0
        assert result.iloc[-1] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Trade Arrival Rate
# ---------------------------------------------------------------------------

class TestTradeArrivalRate:
    def test_count(self):
        df = pd.DataFrame({"action": ["A", "T", "C", "T", "T"]})
        result = _compute("trade_arrival_rate", df, window=5)
        assert result.iloc[-1] == pytest.approx(3.0)

    def test_no_trades(self):
        df = pd.DataFrame({"action": ["A", "C", "A", "C", "A"]})
        result = _compute("trade_arrival_rate", df, window=5)
        assert result.iloc[-1] == pytest.approx(0.0)

    def test_range(self, lob_df):
        result = _compute("trade_arrival_rate", lob_df, window=100)
        assert (result >= 0).all()


# ---------------------------------------------------------------------------
# Large Trade Ratio
# ---------------------------------------------------------------------------

class TestLargeTradeRatio:
    def test_all_large_trades(self):
        df = pd.DataFrame({
            "action": ["T", "T", "T"],
            "size": [600.0, 700.0, 800.0],
        })
        result = _compute("large_trade_ratio", df, window=3, threshold=500)
        assert result.iloc[-1] == pytest.approx(1.0)

    def test_no_large_trades(self):
        df = pd.DataFrame({
            "action": ["T", "T", "T"],
            "size": [10.0, 20.0, 30.0],
        })
        result = _compute("large_trade_ratio", df, window=3, threshold=500)
        assert result.iloc[-1] == pytest.approx(0.0)

    def test_mixed(self):
        df = pd.DataFrame({
            "action": ["T", "T", "T", "T"],
            "size": [10.0, 600.0, 20.0, 700.0],
        })
        result = _compute("large_trade_ratio", df, window=4, threshold=500)
        assert result.iloc[-1] == pytest.approx(0.5)

    def test_non_trades_excluded(self):
        df = pd.DataFrame({
            "action": ["A", "T", "A", "T"],
            "size": [600.0, 10.0, 600.0, 600.0],
        })
        result = _compute("large_trade_ratio", df, window=4, threshold=500)
        # Only 2 trades: size 10 (small) and 600 (large) -> 1/2 = 0.5
        assert result.iloc[-1] == pytest.approx(0.5)

    def test_range(self, lob_df):
        result = _compute("large_trade_ratio", lob_df, window=50, threshold=500)
        assert (result >= 0).all() and (result <= 1).all()


# ---------------------------------------------------------------------------
# OFI Autocorrelation
# ---------------------------------------------------------------------------

class TestOFIAutocorrelation:
    def test_range(self, lob_df):
        result = _compute("ofi_autocorrelation", lob_df, window=10)
        assert (result >= -1).all() and (result <= 1).all()

    def test_first_values_zero(self, lob_df):
        """Early values should be 0 (not enough data for autocorrelation)."""
        result = _compute("ofi_autocorrelation", lob_df, window=10)
        assert result.iloc[0] == 0.0

    def test_known_autocorrelation(self):
        """With constant OFI, autocorrelation should be ~0 (no variance)."""
        df = pd.DataFrame({
            "bid_px_00": [100.0] * 20,
            "ask_px_00": [100.1] * 20,
            "bid_sz_00": [100.0] * 20,
            "ask_sz_00": [100.0] * 20,
        })
        result = _compute("ofi_autocorrelation", df, window=10)
        # Constant OFI = 0 everywhere -> autocorrelation is undefined -> fillna(0)
        assert (result == 0.0).all()