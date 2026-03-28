"""Limit Order Book (LOB) microstructure features.

These features are based on order book imbalance theory and high-frequency
trading microstructure research. References:
- Order Book Features documentation
- Microprice theory (Stoikov, et al.)
- Market microstructure analysis
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from trading_rl.features.base import Feature
from trading_rl.features.registry import register_feature


@register_feature("book_pressure")
class BookPressureFeature(Feature):
    """Book Pressure (Volume Imbalance) at a specific level.

    Formula: bkp_i = (BidVol_i - AskVol_i) / (BidVol_i + AskVol_i)

    Measures the imbalance of liquidity (volume) at a specific book level.
    Values range from -1 (all ask-side) to +1 (all bid-side).

    Params:
        level: Book level (0-9, default: 0 for best bid/ask)
        bid_size_col: Column name for bid size (default: "bid_sz_{level:02d}")
        ask_size_col: Column name for ask size (default: "ask_sz_{level:02d}")
    """

    def _get_level(self) -> int:
        return int(self.config.params.get("level", 0))

    def _get_bid_col(self) -> str:
        level = self._get_level()
        return self.config.params.get("bid_size_col", f"bid_sz_{level:02d}")

    def _get_ask_col(self) -> str:
        level = self._get_level()
        return self.config.params.get("ask_size_col", f"ask_sz_{level:02d}")

    def required_columns(self) -> list[str]:
        return [self._get_bid_col(), self._get_ask_col()]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        bid_vol = df[self._get_bid_col()]
        ask_vol = df[self._get_ask_col()]

        # Avoid division by zero
        total_vol = bid_vol + ask_vol
        pressure = pd.Series(0.0, index=df.index)

        mask = total_vol > 0
        pressure[mask] = (bid_vol[mask] - ask_vol[mask]) / total_vol[mask]

        return pressure


@register_feature("spread_bps")
class SpreadBpsFeature(Feature):
    """Spread in Basis Points.

    Formula: Spread_bps = (Ask_0 - Bid_0) / MidPrice * 10000

    Measures the relative cost of trading; regime indicator for liquidity.

    Params:
        bid_price_col: Column name for best bid (default: "bid_px_00")
        ask_price_col: Column name for best ask (default: "ask_px_00")
    """

    def _get_bid_col(self) -> str:
        return self.config.params.get("bid_price_col", "bid_px_00")

    def _get_ask_col(self) -> str:
        return self.config.params.get("ask_price_col", "ask_px_00")

    def required_columns(self) -> list[str]:
        return [self._get_bid_col(), self._get_ask_col()]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        bid = df[self._get_bid_col()]
        ask = df[self._get_ask_col()]

        mid_price = (bid + ask) / 2
        spread_bps = pd.Series(0.0, index=df.index)

        mask = mid_price > 0
        spread_bps[mask] = ((ask[mask] - bid[mask]) / mid_price[mask]) * 10000

        return spread_bps


@register_feature("microprice")
class MicropriceFeature(Feature):
    """Microprice (Volume-Weighted Fair Value).

    Formula: microprice = (AskVol * Bid + BidVol * Ask) / (BidVol + AskVol)

    Shifts fair price toward the side with less liquidity (more likely to move).
    Better predictor of next price movement than simple mid-price.

    Params:
        bid_price_col: Column name for best bid (default: "bid_px_00")
        ask_price_col: Column name for best ask (default: "ask_px_00")
        bid_size_col: Column name for bid size (default: "bid_sz_00")
        ask_size_col: Column name for ask size (default: "ask_sz_00")
    """

    def _get_cols(self) -> dict[str, str]:
        return {
            "bid_px": self.config.params.get("bid_price_col", "bid_px_00"),
            "ask_px": self.config.params.get("ask_price_col", "ask_px_00"),
            "bid_sz": self.config.params.get("bid_size_col", "bid_sz_00"),
            "ask_sz": self.config.params.get("ask_size_col", "ask_sz_00"),
        }

    def required_columns(self) -> list[str]:
        cols = self._get_cols()
        return list(cols.values())

    def compute(self, df: pd.DataFrame) -> pd.Series:
        cols = self._get_cols()
        bid_px = df[cols["bid_px"]]
        ask_px = df[cols["ask_px"]]
        bid_sz = df[cols["bid_sz"]]
        ask_sz = df[cols["ask_sz"]]

        total_size = bid_sz + ask_sz
        microprice = (bid_px + ask_px) / 2  # Fallback to mid-price

        mask = total_size > 0
        microprice[mask] = (
            (ask_sz[mask] * bid_px[mask] + bid_sz[mask] * ask_px[mask])
            / total_size[mask]
        )

        return microprice


@register_feature("microprice_divergence")
class MicropriceDivergenceFeature(Feature):
    """Microprice Divergence from Mid Price.

    Formula: MicroDiv = MicroPrice - MidPrice

    Signal for mean reversion or breakout. Positive values indicate
    more buying pressure, negative values indicate selling pressure.

    Params:
        bid_price_col: Column name for best bid (default: "bid_px_00")
        ask_price_col: Column name for best ask (default: "ask_px_00")
        bid_size_col: Column name for bid size (default: "bid_sz_00")
        ask_size_col: Column name for ask size (default: "ask_sz_00")
    """

    def _get_cols(self) -> dict[str, str]:
        return {
            "bid_px": self.config.params.get("bid_price_col", "bid_px_00"),
            "ask_px": self.config.params.get("ask_price_col", "ask_px_00"),
            "bid_sz": self.config.params.get("bid_size_col", "bid_sz_00"),
            "ask_sz": self.config.params.get("ask_size_col", "ask_sz_00"),
        }

    def required_columns(self) -> list[str]:
        cols = self._get_cols()
        return list(cols.values())

    def compute(self, df: pd.DataFrame) -> pd.Series:
        cols = self._get_cols()
        bid_px = df[cols["bid_px"]]
        ask_px = df[cols["ask_px"]]
        bid_sz = df[cols["bid_sz"]]
        ask_sz = df[cols["ask_sz"]]

        mid_price = (bid_px + ask_px) / 2

        total_size = bid_sz + ask_sz
        microprice = mid_price.copy()

        mask = total_size > 0
        microprice[mask] = (
            (ask_sz[mask] * bid_px[mask] + bid_sz[mask] * ask_px[mask])
            / total_size[mask]
        )

        return microprice - mid_price


@register_feature("depth_ratio")
class DepthRatioFeature(Feature):
    """Depth Ratio (Top vs. Deep Book).

    Formula: DepthRatio = Vol_L1 / sum(Vol_L2..L5)

    High ratio indicates a "thin" book behind the best price (fragile liquidity).
    Low ratio indicates deep liquidity support.

    Params:
        levels_deep: Number of deep levels to consider (default: 4, i.e., L2-L5)
        bid_size_prefix: Prefix for bid size columns (default: "bid_sz_")
        ask_size_prefix: Prefix for ask size columns (default: "ask_sz_")
    """

    def _get_params(self) -> dict:
        return {
            "levels_deep": int(self.config.params.get("levels_deep", 4)),
            "bid_sz_prefix": self.config.params.get("bid_size_prefix", "bid_sz_"),
            "ask_sz_prefix": self.config.params.get("ask_size_prefix", "ask_sz_"),
        }

    def required_columns(self) -> list[str]:
        params = self._get_params()
        cols = []
        # L1 (level 0)
        cols.extend([f"{params['bid_sz_prefix']}00", f"{params['ask_sz_prefix']}00"])
        # L2-L5 (levels 1-4)
        for i in range(1, params["levels_deep"] + 1):
            cols.extend([
                f"{params['bid_sz_prefix']}{i:02d}",
                f"{params['ask_sz_prefix']}{i:02d}"
            ])
        return cols

    def compute(self, df: pd.DataFrame) -> pd.Series:
        params = self._get_params()

        # L1 total volume
        l1_vol = (
            df[f"{params['bid_sz_prefix']}00"] +
            df[f"{params['ask_sz_prefix']}00"]
        )

        # L2-L5 total volume
        deep_vol = pd.Series(0.0, index=df.index)
        for i in range(1, params["levels_deep"] + 1):
            deep_vol += (
                df[f"{params['bid_sz_prefix']}{i:02d}"] +
                df[f"{params['ask_sz_prefix']}{i:02d}"]
            )

        depth_ratio = pd.Series(0.0, index=df.index)
        mask = deep_vol > 0
        depth_ratio[mask] = l1_vol[mask] / deep_vol[mask]

        return depth_ratio


@register_feature("vwmp_skew")
class VWMPSkewFeature(Feature):
    """Volume-Weighted Mid Price (VWMP) Skew.

    Formula: VWMP_Skew = (VWMP_L1..3 - MidPrice) / Spread

    Measures if deep liquidity supports the current mid-price.
    Positive skew indicates deep bid support, negative indicates deep ask support.

    Params:
        levels: Number of levels to include (default: 3)
        bid_price_prefix: Prefix for bid price columns (default: "bid_px_")
        ask_price_prefix: Prefix for ask price columns (default: "ask_px_")
        bid_size_prefix: Prefix for bid size columns (default: "bid_sz_")
        ask_size_prefix: Prefix for ask size columns (default: "ask_sz_")
    """

    def _get_params(self) -> dict:
        return {
            "levels": int(self.config.params.get("levels", 3)),
            "bid_px_prefix": self.config.params.get("bid_price_prefix", "bid_px_"),
            "ask_px_prefix": self.config.params.get("ask_price_prefix", "ask_px_"),
            "bid_sz_prefix": self.config.params.get("bid_size_prefix", "bid_sz_"),
            "ask_sz_prefix": self.config.params.get("ask_size_prefix", "ask_sz_"),
        }

    def required_columns(self) -> list[str]:
        params = self._get_params()
        cols = []
        for i in range(params["levels"]):
            cols.extend([
                f"{params['bid_px_prefix']}{i:02d}",
                f"{params['ask_px_prefix']}{i:02d}",
                f"{params['bid_sz_prefix']}{i:02d}",
                f"{params['ask_sz_prefix']}{i:02d}",
            ])
        return cols

    def compute(self, df: pd.DataFrame) -> pd.Series:
        params = self._get_params()

        # Calculate VWMP across specified levels
        total_vol = pd.Series(0.0, index=df.index)
        weighted_price = pd.Series(0.0, index=df.index)

        for i in range(params["levels"]):
            bid_px = df[f"{params['bid_px_prefix']}{i:02d}"]
            ask_px = df[f"{params['ask_px_prefix']}{i:02d}"]
            bid_sz = df[f"{params['bid_sz_prefix']}{i:02d}"]
            ask_sz = df[f"{params['ask_sz_prefix']}{i:02d}"]

            total_vol += bid_sz + ask_sz
            weighted_price += bid_px * bid_sz + ask_px * ask_sz

        vwmp = pd.Series(0.0, index=df.index)
        mask = total_vol > 0
        vwmp[mask] = weighted_price[mask] / total_vol[mask]

        # Calculate mid-price and spread from L1
        bid_0 = df[f"{params['bid_px_prefix']}00"]
        ask_0 = df[f"{params['ask_px_prefix']}00"]
        mid_price = (bid_0 + ask_0) / 2
        spread = ask_0 - bid_0

        # Calculate skew
        skew = pd.Series(0.0, index=df.index)
        spread_mask = spread > 0
        skew[spread_mask] = (vwmp[spread_mask] - mid_price[spread_mask]) / spread[spread_mask]

        return skew


@register_feature("bid_ask_slope")
class BidAskSlopeFeature(Feature):
    """Bid or Ask Slope (Elasticity).

    Formula (Bid): Slope_Bid = (BidPx_0 - BidPx_N) / sum(BidVol_0..N)
    Formula (Ask): Slope_Ask = (AskPx_N - AskPx_0) / sum(AskVol_0..N)

    Steeper slope implies lower impact cost for large orders.
    Measures the price elasticity of available liquidity.

    Params:
        side: "bid" or "ask" (required)
        levels: Number of levels to include (default: 5, i.e., L0-L4)
        price_prefix: Prefix for price columns (default: "bid_px_" or "ask_px_")
        size_prefix: Prefix for size columns (default: "bid_sz_" or "ask_sz_")
    """

    def _get_params(self) -> dict:
        side = self.config.params.get("side", "").lower()
        if side not in ["bid", "ask"]:
            raise ValueError(
                f"bid_ask_slope feature requires side='bid' or 'ask', got: {side}"
            )

        levels = int(self.config.params.get("levels", 5))
        price_prefix = self.config.params.get(
            "price_prefix", f"{side}_px_"
        )
        size_prefix = self.config.params.get(
            "size_prefix", f"{side}_sz_"
        )

        return {
            "side": side,
            "levels": levels,
            "price_prefix": price_prefix,
            "size_prefix": size_prefix,
        }

    def required_columns(self) -> list[str]:
        params = self._get_params()
        cols = []
        for i in range(params["levels"]):
            cols.extend([
                f"{params['price_prefix']}{i:02d}",
                f"{params['size_prefix']}{i:02d}",
            ])
        return cols

    def compute(self, df: pd.DataFrame) -> pd.Series:
        params = self._get_params()

        px_0 = df[f"{params['price_prefix']}00"]
        px_n = df[f"{params['price_prefix']}{params['levels']-1:02d}"]

        total_vol = pd.Series(0.0, index=df.index)
        for i in range(params["levels"]):
            total_vol += df[f"{params['size_prefix']}{i:02d}"]

        slope = pd.Series(0.0, index=df.index)
        mask = total_vol > 0

        if params["side"] == "bid":
            # Bid slope: higher is better (flatter price decay)
            slope[mask] = (px_0[mask] - px_n[mask]) / total_vol[mask]
        else:
            # Ask slope: higher is better (flatter price increase)
            slope[mask] = (px_n[mask] - px_0[mask]) / total_vol[mask]

        return slope


@register_feature("order_book_imbalance")
class OrderBookImbalanceFeature(Feature):
    """Multi-Level Order Book Imbalance.

    Formula: OBI = sum(BidVol_i - AskVol_i) / sum(BidVol_i + AskVol_i)

    Aggregates book pressure across multiple levels for a more robust signal.

    Params:
        levels: Number of levels to include (default: 3)
        bid_size_prefix: Prefix for bid size columns (default: "bid_sz_")
        ask_size_prefix: Prefix for ask size columns (default: "ask_sz_")
    """

    def _get_params(self) -> dict:
        return {
            "levels": int(self.config.params.get("levels", 3)),
            "bid_sz_prefix": self.config.params.get("bid_size_prefix", "bid_sz_"),
            "ask_sz_prefix": self.config.params.get("ask_size_prefix", "ask_sz_"),
        }

    def required_columns(self) -> list[str]:
        params = self._get_params()
        cols = []
        for i in range(params["levels"]):
            cols.extend([
                f"{params['bid_sz_prefix']}{i:02d}",
                f"{params['ask_sz_prefix']}{i:02d}",
            ])
        return cols

    def compute(self, df: pd.DataFrame) -> pd.Series:
        params = self._get_params()

        total_bid = pd.Series(0.0, index=df.index)
        total_ask = pd.Series(0.0, index=df.index)

        for i in range(params["levels"]):
            total_bid += df[f"{params['bid_sz_prefix']}{i:02d}"]
            total_ask += df[f"{params['ask_sz_prefix']}{i:02d}"]

        total_vol = total_bid + total_ask
        imbalance = pd.Series(0.0, index=df.index)

        mask = total_vol > 0
        imbalance[mask] = (total_bid[mask] - total_ask[mask]) / total_vol[mask]

        return imbalance


# =============================================================================
# Alpha-oriented HFT features
# =============================================================================


@register_feature("ofi")
class OrderFlowImbalanceFeature(Feature):
    """Order Flow Imbalance (OFI) — Cont, Kukanov & Stoikov (2014).

    Measures the net directional pressure from order book events at the best
    level between consecutive snapshots. Distinguishes aggressive order
    submission from cancellation and captures the flow that drives short-term
    price changes.

    Formula:
        dBid = bid_sz_00[t] - bid_sz_00[t-1]
        dAsk = ask_sz_00[t] - ask_sz_00[t-1]

        OFI = dBid * I(bid_px_00[t] >= bid_px_00[t-1])
            - dAsk * I(ask_px_00[t] <= ask_px_00[t-1])

    Positive: net buying pressure; Negative: net selling pressure.
    First row is set to 0.

    Params:
        bid_price_col: best bid price column (default: "bid_px_00")
        ask_price_col: best ask price column (default: "ask_px_00")
        bid_size_col:  best bid size column  (default: "bid_sz_00")
        ask_size_col:  best ask size column  (default: "ask_sz_00")
    """

    def _get_cols(self) -> dict[str, str]:
        return {
            "bid_px": self.config.params.get("bid_price_col", "bid_px_00"),
            "ask_px": self.config.params.get("ask_price_col", "ask_px_00"),
            "bid_sz": self.config.params.get("bid_size_col", "bid_sz_00"),
            "ask_sz": self.config.params.get("ask_size_col", "ask_sz_00"),
        }

    def required_columns(self) -> list[str]:
        return list(self._get_cols().values())

    def compute(self, df: pd.DataFrame) -> pd.Series:
        cols = self._get_cols()
        bid_px = df[cols["bid_px"]].astype(float)
        ask_px = df[cols["ask_px"]].astype(float)
        bid_sz = df[cols["bid_sz"]].astype(float)
        ask_sz = df[cols["ask_sz"]].astype(float)

        d_bid_sz = bid_sz.diff()
        d_ask_sz = ask_sz.diff()

        bid_px_up = (bid_px >= bid_px.shift(1)).astype(float)
        ask_px_dn = (ask_px <= ask_px.shift(1)).astype(float)

        ofi = d_bid_sz * bid_px_up - d_ask_sz * ask_px_dn
        ofi.iloc[0] = 0.0
        return ofi.fillna(0.0)


@register_feature("ofi_rolling")
class RollingOFIFeature(Feature):
    """Rolling signed order-flow imbalance over a fixed tick window.

    Sums OFI values over the last `window` rows to capture persistent
    directional pressure across multiple time horizons.

    Formula:
        RollingOFI_W[t] = sum(OFI[t-W+1 .. t])

    Params:
        window: look-back in ticks (default: 50)
        bid_price_col, ask_price_col, bid_size_col, ask_size_col:
            same defaults as OFI feature
    """

    def _get_cols(self) -> dict[str, str]:
        return {
            "bid_px": self.config.params.get("bid_price_col", "bid_px_00"),
            "ask_px": self.config.params.get("ask_price_col", "ask_px_00"),
            "bid_sz": self.config.params.get("bid_size_col", "bid_sz_00"),
            "ask_sz": self.config.params.get("ask_size_col", "ask_sz_00"),
        }

    def required_columns(self) -> list[str]:
        return list(self._get_cols().values())

    def compute(self, df: pd.DataFrame) -> pd.Series:
        cols = self._get_cols()
        window = int(self.config.params.get("window", 50))

        bid_px = df[cols["bid_px"]].astype(float)
        ask_px = df[cols["ask_px"]].astype(float)
        bid_sz = df[cols["bid_sz"]].astype(float)
        ask_sz = df[cols["ask_sz"]].astype(float)

        d_bid_sz = bid_sz.diff()
        d_ask_sz = ask_sz.diff()
        bid_px_up = (bid_px >= bid_px.shift(1)).astype(float)
        ask_px_dn = (ask_px <= ask_px.shift(1)).astype(float)

        ofi = (d_bid_sz * bid_px_up - d_ask_sz * ask_px_dn).fillna(0.0)
        return ofi.rolling(window=window, min_periods=1).sum()


@register_feature("queue_depletion")
class QueueDepletionFeature(Feature):
    """Queue Depletion Rate at the best bid or ask.

    Measures how fast the top-of-book queue is being consumed relative to its
    previous size. A rapidly depleting bid queue signals imminent downward
    price pressure; a depleting ask queue signals upward pressure.

    Formula:
        depletion = max(V[t-1] - V[t], 0) / V[t-1]

    Values in [0, 1]: 0 = queue grew or unchanged; 1 = queue fully cleared.
    First row is 0.

    Params:
        side: "bid" or "ask" (default: "bid")
        bid_size_col: bid size column (default: "bid_sz_00")
        ask_size_col: ask size column (default: "ask_sz_00")
    """

    def _get_col(self) -> str:
        side = self.config.params.get("side", "bid").lower()
        if side == "ask":
            return self.config.params.get("ask_size_col", "ask_sz_00")
        return self.config.params.get("bid_size_col", "bid_sz_00")

    def required_columns(self) -> list[str]:
        return [self._get_col()]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        sz = df[self._get_col()].astype(float)
        prev_sz = sz.shift(1)

        depletion = pd.Series(0.0, index=df.index)
        mask = prev_sz > 0
        depletion[mask] = np.clip(
            (prev_sz[mask] - sz[mask]) / prev_sz[mask], 0.0, 1.0
        )
        return depletion


@register_feature("mid_price_acceleration")
class MidPriceAccelerationFeature(Feature):
    """Mid-Price Acceleration (second finite difference).

    Captures whether the mid-price is accelerating or decelerating, which
    helps distinguish momentum from mean-reversion regimes at tick frequency.

    Formula:
        MidPrice[t]  = (bid_px_00[t] + ask_px_00[t]) / 2
        Accel[t]     = MidPrice[t] - 2 * MidPrice[t-1] + MidPrice[t-2]

    Positive: mid-price accelerating upward; Negative: decelerating / reversing.
    First two rows are 0.

    Params:
        bid_price_col: best bid price column (default: "bid_px_00")
        ask_price_col: best ask price column (default: "ask_px_00")
    """

    def required_columns(self) -> list[str]:
        return [
            self.config.params.get("bid_price_col", "bid_px_00"),
            self.config.params.get("ask_price_col", "ask_px_00"),
        ]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        bid = df[self.config.params.get("bid_price_col", "bid_px_00")].astype(float)
        ask = df[self.config.params.get("ask_price_col", "ask_px_00")].astype(float)
        mid = (bid + ask) / 2.0
        accel = mid - 2.0 * mid.shift(1) + mid.shift(2)
        return accel.fillna(0.0)


@register_feature("inter_event_time")
class InterEventTimeFeature(Feature):
    """Log inter-event time between consecutive order book updates.

    Sparse updates (large gap) signal low market activity and lower adverse-
    selection risk; dense updates signal informed order flow or a large order
    working through the book.

    Formula:
        gap_ns[t] = timestamp[t] - timestamp[t-1]   (nanoseconds)
        LogGap[t] = log(1 + gap_ns[t])

    Uses the DataFrame index (DatetimeIndex with nanosecond precision).
    First row is 0.

    Params: none
    """

    def required_columns(self) -> list[str]:
        return []

    def compute(self, df: pd.DataFrame) -> pd.Series:
        ts_ns = pd.Series(
            df.index.astype(np.int64), index=df.index, dtype=float
        )
        gap_ns = ts_ns.diff().fillna(0.0).clip(lower=0.0)
        return np.log1p(gap_ns)


@register_feature("spread_ratio")
class SpreadRatioFeature(Feature):
    """Spread-to-rolling-mean ratio — adverse selection / toxicity signal.

    A spread that is elevated relative to its recent average indicates a
    liquidity regime change: market makers are widening quotes due to
    perceived information asymmetry.

    Formula:
        SpreadBps[t]  = (ask_px_00 - bid_px_00) / MidPrice * 10000
        SpreadRatio[t] = SpreadBps[t] / RollingMean(SpreadBps, window)

    Values > 1: spread is wider than recent average (higher toxicity risk).
    Values < 1: tighter than average (lower cost environment).

    Params:
        window: rolling look-back in ticks (default: 100)
        bid_price_col: best bid price column (default: "bid_px_00")
        ask_price_col: best ask price column (default: "ask_px_00")
    """

    def required_columns(self) -> list[str]:
        return [
            self.config.params.get("bid_price_col", "bid_px_00"),
            self.config.params.get("ask_price_col", "ask_px_00"),
        ]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        window = int(self.config.params.get("window", 100))
        bid = df[self.config.params.get("bid_price_col", "bid_px_00")].astype(float)
        ask = df[self.config.params.get("ask_price_col", "ask_px_00")].astype(float)

        mid = (bid + ask) / 2.0
        spread_bps = pd.Series(0.0, index=df.index)
        mask = mid > 0
        spread_bps[mask] = ((ask[mask] - bid[mask]) / mid[mask]) * 10_000.0

        rolling_mean = spread_bps.rolling(window=window, min_periods=1).mean()
        ratio = pd.Series(1.0, index=df.index)
        nonzero = rolling_mean > 0
        ratio[nonzero] = spread_bps[nonzero] / rolling_mean[nonzero]
        return ratio


@register_feature("book_convexity")
class BookConvexityFeature(Feature):
    """Book Convexity — curvature of bid or ask price levels.

    Measures whether liquidity thins linearly or accelerates as you move away
    from the best price. Negative convexity (widening gaps) indicates the book
    has a discontinuity — large orders will face sudden price jumps beyond the
    visible depth.

    Formula (bid side):
        Convexity_bid = (P_bid_00 - P_bid_01) - (P_bid_01 - P_bid_02)

    Formula (ask side):
        Convexity_ask = (P_ask_02 - P_ask_01) - (P_ask_01 - P_ask_00)

    Positive: gaps narrowing deeper in book (unusual, tightly packed).
    Negative: gaps widening (standard; larger value = more discontinuous).

    Params:
        side: "bid" or "ask" (default: "bid")
        bid_price_prefix: prefix for bid price columns (default: "bid_px_")
        ask_price_prefix: prefix for ask price columns (default: "ask_px_")
    """

    def _get_params(self) -> dict:
        return {
            "side": self.config.params.get("side", "bid").lower(),
            "bid_px_prefix": self.config.params.get("bid_price_prefix", "bid_px_"),
            "ask_px_prefix": self.config.params.get("ask_price_prefix", "ask_px_"),
        }

    def required_columns(self) -> list[str]:
        p = self._get_params()
        prefix = p["bid_px_prefix"] if p["side"] == "bid" else p["ask_px_prefix"]
        return [f"{prefix}00", f"{prefix}01", f"{prefix}02"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        p = self._get_params()
        if p["side"] == "bid":
            px = p["bid_px_prefix"]
            p0 = df[f"{px}00"].astype(float)
            p1 = df[f"{px}01"].astype(float)
            p2 = df[f"{px}02"].astype(float)
            return (p0 - p1) - (p1 - p2)
        else:
            px = p["ask_px_prefix"]
            p0 = df[f"{px}00"].astype(float)
            p1 = df[f"{px}01"].astype(float)
            p2 = df[f"{px}02"].astype(float)
            return (p2 - p1) - (p1 - p0)


@register_feature("order_count_imbalance")
class OrderCountImbalanceFeature(Feature):
    """Order Count Imbalance at a specific book level.

    Uses the bid_ct / ask_ct columns (number of resting orders, not volume)
    to measure whether the book is dominated by many small orders (retail-like)
    or few large orders (institutional-like) on each side.

    Formula:
        OCI_i = (bid_ct_i - ask_ct_i) / (bid_ct_i + ask_ct_i)

    Complements book pressure (which uses volume): a level can show high volume
    imbalance from a single large order while OCI shows the opposite.
    Range: [-1, +1].

    Params:
        level: book level 0–9 (default: 0)
        bid_count_prefix: column prefix for bid order counts (default: "bid_ct_")
        ask_count_prefix: column prefix for ask order counts (default: "ask_ct_")
    """

    def _get_cols(self) -> tuple[str, str]:
        level = int(self.config.params.get("level", 0))
        bid_prefix = self.config.params.get("bid_count_prefix", "bid_ct_")
        ask_prefix = self.config.params.get("ask_count_prefix", "ask_ct_")
        return f"{bid_prefix}{level:02d}", f"{ask_prefix}{level:02d}"

    def required_columns(self) -> list[str]:
        return list(self._get_cols())

    def compute(self, df: pd.DataFrame) -> pd.Series:
        bid_col, ask_col = self._get_cols()
        bid_ct = df[bid_col].astype(float)
        ask_ct = df[ask_col].astype(float)

        total = bid_ct + ask_ct
        imbalance = pd.Series(0.0, index=df.index)
        mask = total > 0
        imbalance[mask] = (bid_ct[mask] - ask_ct[mask]) / total[mask]
        return imbalance


@register_feature("signed_trade_flow")
class SignedTradeFlowFeature(Feature):
    """Rolling signed trade flow (cumulative delta) from the trade tape.

    Aggregates buyer- vs seller-initiated trade volume over a rolling tick
    window. Requires the Databento MBP-10 `action` and `side` columns.

    Convention (Databento):
        action == 'T' and side == 'B'  → buyer-initiated trade  (+size)
        action == 'T' and side == 'A'  → seller-initiated trade  (-size)
        action != 'T'                   → book event, no trade    (0)

    Formula:
        signed_vol[t] = size[t] * sign  if action[t] == 'T'  else 0
        RollingDelta_W[t] = sum(signed_vol[t-W+1 .. t])

    Positive: net buying over window; Negative: net selling.

    Params:
        window:      rolling look-back in ticks (default: 100)
        action_col:  column identifying event type (default: "action")
        side_col:    column identifying aggressor side (default: "side")
        size_col:    column with trade/order size (default: "size")
    """

    def required_columns(self) -> list[str]:
        return [
            self.config.params.get("action_col", "action"),
            self.config.params.get("side_col", "side"),
            self.config.params.get("size_col", "size"),
        ]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        window = int(self.config.params.get("window", 100))
        action_col = self.config.params.get("action_col", "action")
        side_col = self.config.params.get("side_col", "side")
        size_col = self.config.params.get("size_col", "size")

        action = df[action_col].astype(str)
        side = df[side_col].astype(str)
        size = df[size_col].astype(float)

        is_trade = action == "T"
        sign = pd.Series(0.0, index=df.index)
        sign[is_trade & (side == "B")] = 1.0
        sign[is_trade & (side == "A")] = -1.0

        signed_vol = size * sign
        return signed_vol.rolling(window=window, min_periods=1).sum()


@register_feature("odd_lot_trade_ratio")
class OddLotTradeRatioFeature(Feature):
    """Rolling fraction of trades that are odd-lot sized.

    An odd lot is a trade with size strictly below the round-lot threshold
    (default 100 shares for US equities). Odd lot activity is predominantly
    retail-initiated uninformed flow. A high ratio signals a retail-dominated
    tape with lower adverse selection risk; a sudden drop may indicate
    institutional participation.

    Only rows where action == 'T' (trade events) contribute to the count.
    Non-trade rows are treated as zero observations and do not advance the
    numerator or denominator. The rolling window is over all ticks (book
    events + trades) so the value evolves continuously.

    Formula:
        is_odd[t]  = 1  if action[t] == 'T' and size[t] < round_lot  else 0
        is_trade[t]= 1  if action[t] == 'T'                          else 0

        OddLotRatio_W[t] = sum(is_odd[t-W+1..t]) / max(sum(is_trade[t-W+1..t]), 1)

    Range: [0, 1].

    Params:
        window:        rolling look-back in ticks (default: 200)
        round_lot:     round-lot threshold in shares (default: 100)
        action_col:    column identifying event type (default: "action")
        size_col:      column with trade size (default: "size")
    """

    def required_columns(self) -> list[str]:
        return [
            self.config.params.get("action_col", "action"),
            self.config.params.get("size_col", "size"),
        ]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        window = int(self.config.params.get("window", 200))
        round_lot = float(self.config.params.get("round_lot", 100))
        action_col = self.config.params.get("action_col", "action")
        size_col = self.config.params.get("size_col", "size")

        is_trade = (df[action_col].astype(str) == "T").astype(float)
        is_odd = (
            is_trade * (df[size_col].astype(float) < round_lot).astype(float)
        )

        rolling_odd = is_odd.rolling(window=window, min_periods=1).sum()
        rolling_trades = is_trade.rolling(window=window, min_periods=1).sum()

        ratio = pd.Series(0.0, index=df.index)
        mask = rolling_trades > 0
        ratio[mask] = rolling_odd[mask] / rolling_trades[mask]
        return ratio


@register_feature("odd_lot_imbalance")
class OddLotImbalanceFeature(Feature):
    """Rolling signed odd-lot flow imbalance.

    Measures the directional bias in odd-lot (retail) order flow.
    Buyer-initiated odd-lot trades indicate retail buying pressure;
    seller-initiated odd-lot trades indicate retail selling pressure.
    The net imbalance normalized by total odd-lot volume reveals which
    side retail is leaning toward.

    Only rows where action == 'T' and size < round_lot contribute.

    Formula:
        odd_buy[t]  = size[t]  if action[t]=='T', side[t]=='B', size[t] < round_lot
        odd_sell[t] = size[t]  if action[t]=='T', side[t]=='A', size[t] < round_lot
        (0 otherwise)

        RollingBuy_W  = sum(odd_buy[t-W+1..t])
        RollingSell_W = sum(odd_sell[t-W+1..t])

        OddLotImbalance_W = (RollingBuy_W - RollingSell_W)
                          / max(RollingBuy_W + RollingSell_W, 1)

    Range: [-1, +1].
    +1: all odd-lot flow is buyer-initiated (retail buying).
    -1: all odd-lot flow is seller-initiated (retail selling).

    Params:
        window:     rolling look-back in ticks (default: 200)
        round_lot:  round-lot threshold in shares (default: 100)
        action_col: column identifying event type (default: "action")
        side_col:   column identifying aggressor side (default: "side")
        size_col:   column with trade size (default: "size")
    """

    def required_columns(self) -> list[str]:
        return [
            self.config.params.get("action_col", "action"),
            self.config.params.get("side_col", "side"),
            self.config.params.get("size_col", "size"),
        ]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        window = int(self.config.params.get("window", 200))
        round_lot = float(self.config.params.get("round_lot", 100))
        action_col = self.config.params.get("action_col", "action")
        side_col = self.config.params.get("side_col", "side")
        size_col = self.config.params.get("size_col", "size")

        action = df[action_col].astype(str)
        side = df[side_col].astype(str)
        size = df[size_col].astype(float)

        is_odd_trade = (action == "T") & (size < round_lot)

        odd_buy = pd.Series(0.0, index=df.index)
        odd_sell = pd.Series(0.0, index=df.index)
        odd_buy[is_odd_trade & (side == "B")] = size[is_odd_trade & (side == "B")]
        odd_sell[is_odd_trade & (side == "A")] = size[is_odd_trade & (side == "A")]

        rolling_buy = odd_buy.rolling(window=window, min_periods=1).sum()
        rolling_sell = odd_sell.rolling(window=window, min_periods=1).sum()
        total = rolling_buy + rolling_sell

        imbalance = pd.Series(0.0, index=df.index)
        mask = total > 0
        imbalance[mask] = (rolling_buy[mask] - rolling_sell[mask]) / total[mask]
        return imbalance
