"""Limit Order Book (LOB) microstructure features.

These features are based on order book imbalance theory and high-frequency
trading microstructure research. References:
- Cont, Kukanov & Stoikov (2014) — Order Flow Imbalance
- Stoikov (2018) — Microprice
- Market microstructure analysis
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from trading_rl.features.base import Feature, FeatureConfig
from trading_rl.features.registry import register_feature


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
    fill: float = 0.0,
) -> pd.Series:
    """Divide two Series, returning `fill` wherever denominator is zero."""
    result = pd.Series(fill, index=numerator.index, dtype=float)
    mask = denominator != 0
    result[mask] = numerator[mask] / denominator[mask]
    return result


class LOBFeature(Feature):
    """Base class for LOB features with convenience helpers."""

    def _p(self, key: str, default):
        """Shorthand for self.config.params.get(key, default)."""
        return self.config.params.get(key, default)

    def _best_cols(self) -> dict[str, str]:
        """Resolve the canonical 4-column best-level names from params."""
        return {
            "bid_px": self._p("bid_price_col", "bid_px_00"),
            "ask_px": self._p("ask_price_col", "ask_px_00"),
            "bid_sz": self._p("bid_size_col", "bid_sz_00"),
            "ask_sz": self._p("ask_size_col", "ask_sz_00"),
        }

    def _trade_cols(self) -> tuple[str, str, str]:
        """Resolve trade-tape column names (action, side, size) from params."""
        return (
            self._p("action_col", "action"),
            self._p("side_col", "side"),
            self._p("size_col", "size"),
        )

    def _mid(self, df: pd.DataFrame, bid_col: str, ask_col: str) -> pd.Series:
        return (df[bid_col].astype(float) + df[ask_col].astype(float)) / 2.0


# ---------------------------------------------------------------------------
# Baseline LOB features
# ---------------------------------------------------------------------------

@register_feature("book_pressure")
class BookPressureFeature(LOBFeature):
    """Book Pressure (Volume Imbalance) at a specific level.

    Formula: bkp_i = (BidVol_i - AskVol_i) / (BidVol_i + AskVol_i)

    Measures the imbalance of liquidity (volume) at a specific book level.
    Values range from -1 (all ask-side) to +1 (all bid-side).

    Params:
        level: Book level (0-9, default: 0 for best bid/ask)
        bid_size_col: Column name for bid size (default: "bid_sz_{level:02d}")
        ask_size_col: Column name for ask size (default: "ask_sz_{level:02d}")
    """

    def _sz_cols(self) -> tuple[str, str]:
        lvl = int(self._p("level", 0))
        return (
            self._p("bid_size_col", f"bid_sz_{lvl:02d}"),
            self._p("ask_size_col", f"ask_sz_{lvl:02d}"),
        )

    def required_columns(self) -> list[str]:
        return list(self._sz_cols())

    def compute(self, df: pd.DataFrame) -> pd.Series:
        bid_col, ask_col = self._sz_cols()
        bid = df[bid_col].astype(float)
        ask = df[ask_col].astype(float)
        return safe_divide(bid - ask, bid + ask)


@register_feature("spread_bps")
class SpreadBpsFeature(LOBFeature):
    """Spread in Basis Points.

    Formula: Spread_bps = (Ask_0 - Bid_0) / MidPrice * 10000

    Measures the relative cost of trading; regime indicator for liquidity.

    Params:
        bid_price_col: Column name for best bid (default: "bid_px_00")
        ask_price_col: Column name for best ask (default: "ask_px_00")
    """

    def required_columns(self) -> list[str]:
        return [self._p("bid_price_col", "bid_px_00"),
                self._p("ask_price_col", "ask_px_00")]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        bid = df[self._p("bid_price_col", "bid_px_00")].astype(float)
        ask = df[self._p("ask_price_col", "ask_px_00")].astype(float)
        mid = (bid + ask) / 2.0
        return safe_divide((ask - bid) * 10_000.0, mid)


@register_feature("microprice")
class MicropriceFeature(LOBFeature):
    """Microprice (Volume-Weighted Fair Value).

    Formula: microprice = (AskVol * Bid + BidVol * Ask) / (BidVol + AskVol)

    Shifts fair price toward the side with less liquidity (more likely to move).
    Better predictor of next price movement than simple mid-price.

    Params:
        bid_price_col, ask_price_col, bid_size_col, ask_size_col:
            defaults: bid_px_00, ask_px_00, bid_sz_00, ask_sz_00
    """

    def required_columns(self) -> list[str]:
        return list(self._best_cols().values())

    def compute(self, df: pd.DataFrame) -> pd.Series:
        c = self._best_cols()
        bid_px = df[c["bid_px"]].astype(float)
        ask_px = df[c["ask_px"]].astype(float)
        bid_sz = df[c["bid_sz"]].astype(float)
        ask_sz = df[c["ask_sz"]].astype(float)
        total = bid_sz + ask_sz
        # Fallback to mid-price when total size is zero
        mid = (bid_px + ask_px) / 2.0
        weighted = ask_sz * bid_px + bid_sz * ask_px
        result = mid.copy()
        mask = total > 0
        result[mask] = weighted[mask] / total[mask]
        return result


@register_feature("microprice_divergence")
class MicropriceDivergenceFeature(LOBFeature):
    """Microprice Divergence from Mid Price.

    Formula: MicroDiv = MicroPrice - MidPrice

    Signal for mean reversion or breakout. Positive values indicate
    more buying pressure, negative values indicate selling pressure.

    Params:
        bid_price_col, ask_price_col, bid_size_col, ask_size_col:
            defaults: bid_px_00, ask_px_00, bid_sz_00, ask_sz_00
    """

    def required_columns(self) -> list[str]:
        return list(self._best_cols().values())

    def compute(self, df: pd.DataFrame) -> pd.Series:
        c = self._best_cols()
        bid_px = df[c["bid_px"]].astype(float)
        ask_px = df[c["ask_px"]].astype(float)
        bid_sz = df[c["bid_sz"]].astype(float)
        ask_sz = df[c["ask_sz"]].astype(float)
        mid = (bid_px + ask_px) / 2.0
        total = bid_sz + ask_sz
        microprice = mid.copy()
        mask = total > 0
        microprice[mask] = (ask_sz[mask] * bid_px[mask] + bid_sz[mask] * ask_px[mask]) / total[mask]
        return microprice - mid


@register_feature("depth_ratio")
class DepthRatioFeature(LOBFeature):
    """Depth Ratio (Top vs. Deep Book).

    Formula: DepthRatio = Vol_L0 / sum(Vol_L1..LN)

    High ratio indicates a "thin" book behind the best price (fragile liquidity).
    Low ratio indicates deep liquidity support.

    Params:
        levels_deep: Number of deep levels to consider (default: 4, i.e., L1-L4)
        bid_size_prefix: Prefix for bid size columns (default: "bid_sz_")
        ask_size_prefix: Prefix for ask size columns (default: "ask_sz_")
    """

    def required_columns(self) -> list[str]:
        bp = self._p("bid_size_prefix", "bid_sz_")
        ap = self._p("ask_size_prefix", "ask_sz_")
        n = int(self._p("levels_deep", 4))
        cols = [f"{bp}00", f"{ap}00"]
        for i in range(1, n + 1):
            cols += [f"{bp}{i:02d}", f"{ap}{i:02d}"]
        return cols

    def compute(self, df: pd.DataFrame) -> pd.Series:
        bp = self._p("bid_size_prefix", "bid_sz_")
        ap = self._p("ask_size_prefix", "ask_sz_")
        n = int(self._p("levels_deep", 4))
        l0_vol = df[f"{bp}00"].astype(float) + df[f"{ap}00"].astype(float)
        deep_vol = sum(
            df[f"{bp}{i:02d}"].astype(float) + df[f"{ap}{i:02d}"].astype(float)
            for i in range(1, n + 1)
        )
        return safe_divide(l0_vol, deep_vol)


@register_feature("vwmp_skew")
class VWMPSkewFeature(LOBFeature):
    """Volume-Weighted Mid Price (VWMP) Skew.

    Formula: VWMP_Skew = (VWMP_L0..N - MidPrice) / Spread

    Measures if deep liquidity supports the current mid-price.
    Positive skew indicates deep bid support, negative indicates deep ask support.

    Params:
        levels: Number of levels to include (default: 3)
        bid_price_prefix, ask_price_prefix: default "bid_px_", "ask_px_"
        bid_size_prefix, ask_size_prefix: default "bid_sz_", "ask_sz_"
    """

    def required_columns(self) -> list[str]:
        bpp = self._p("bid_price_prefix", "bid_px_")
        app = self._p("ask_price_prefix", "ask_px_")
        bsp = self._p("bid_size_prefix", "bid_sz_")
        asp = self._p("ask_size_prefix", "ask_sz_")
        n = int(self._p("levels", 3))
        return [
            col
            for i in range(n)
            for col in (f"{bpp}{i:02d}", f"{app}{i:02d}",
                        f"{bsp}{i:02d}", f"{asp}{i:02d}")
        ]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        bpp = self._p("bid_price_prefix", "bid_px_")
        app = self._p("ask_price_prefix", "ask_px_")
        bsp = self._p("bid_size_prefix", "bid_sz_")
        asp = self._p("ask_size_prefix", "ask_sz_")
        n = int(self._p("levels", 3))

        total_vol = pd.Series(0.0, index=df.index)
        weighted_price = pd.Series(0.0, index=df.index)
        for i in range(n):
            bid_px = df[f"{bpp}{i:02d}"].astype(float)
            ask_px = df[f"{app}{i:02d}"].astype(float)
            bid_sz = df[f"{bsp}{i:02d}"].astype(float)
            ask_sz = df[f"{asp}{i:02d}"].astype(float)
            total_vol += bid_sz + ask_sz
            weighted_price += bid_px * bid_sz + ask_px * ask_sz

        vwmp = safe_divide(weighted_price, total_vol)
        bid_0 = df[f"{bpp}00"].astype(float)
        ask_0 = df[f"{app}00"].astype(float)
        mid = (bid_0 + ask_0) / 2.0
        spread = ask_0 - bid_0
        return safe_divide(vwmp - mid, spread)


@register_feature("bid_ask_slope")
class BidAskSlopeFeature(LOBFeature):
    """Bid or Ask Slope (Elasticity).

    Formula (Bid): Slope_Bid = (BidPx_0 - BidPx_N) / sum(BidVol_0..N)
    Formula (Ask): Slope_Ask = (AskPx_N - AskPx_0) / sum(AskVol_0..N)

    Steeper slope implies lower impact cost for large orders.
    Measures the price elasticity of available liquidity.

    Params:
        side: "bid" or "ask" (required)
        levels: Number of levels to include (default: 5, i.e., L0-L4)
        price_prefix: default "bid_px_" or "ask_px_" based on side
        size_prefix: default "bid_sz_" or "ask_sz_" based on side
    """

    def _side_params(self) -> tuple[str, str, str, int]:
        side = self._p("side", "").lower()
        if side not in ("bid", "ask"):
            raise ValueError(f"bid_ask_slope requires side='bid' or 'ask', got: {side!r}")
        pp = self._p("price_prefix", f"{side}_px_")
        sp = self._p("size_prefix", f"{side}_sz_")
        n = int(self._p("levels", 5))
        return side, pp, sp, n

    def required_columns(self) -> list[str]:
        _, pp, sp, n = self._side_params()
        return [col for i in range(n) for col in (f"{pp}{i:02d}", f"{sp}{i:02d}")]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        side, pp, sp, n = self._side_params()
        px_0 = df[f"{pp}00"].astype(float)
        px_n = df[f"{pp}{n - 1:02d}"].astype(float)
        total_vol = sum(df[f"{sp}{i:02d}"].astype(float) for i in range(n))
        price_range = (px_0 - px_n) if side == "bid" else (px_n - px_0)
        return safe_divide(price_range, total_vol)


@register_feature("order_book_imbalance")
class OrderBookImbalanceFeature(LOBFeature):
    """Multi-Level Order Book Imbalance.

    Formula: OBI = sum(BidVol_i - AskVol_i) / sum(BidVol_i + AskVol_i)

    Aggregates book pressure across multiple levels for a more robust signal.

    Params:
        levels: Number of levels to include (default: 3)
        bid_size_prefix: Prefix for bid size columns (default: "bid_sz_")
        ask_size_prefix: Prefix for ask size columns (default: "ask_sz_")
    """

    def required_columns(self) -> list[str]:
        bp = self._p("bid_size_prefix", "bid_sz_")
        ap = self._p("ask_size_prefix", "ask_sz_")
        n = int(self._p("levels", 3))
        return [col for i in range(n) for col in (f"{bp}{i:02d}", f"{ap}{i:02d}")]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        bp = self._p("bid_size_prefix", "bid_sz_")
        ap = self._p("ask_size_prefix", "ask_sz_")
        n = int(self._p("levels", 3))
        total_bid = sum(df[f"{bp}{i:02d}"].astype(float) for i in range(n))
        total_ask = sum(df[f"{ap}{i:02d}"].astype(float) for i in range(n))
        return safe_divide(total_bid - total_ask, total_bid + total_ask)


# ---------------------------------------------------------------------------
# Alpha-oriented HFT features
# ---------------------------------------------------------------------------

@register_feature("ofi")
class OrderFlowImbalanceFeature(LOBFeature):
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
        bid_price_col, ask_price_col, bid_size_col, ask_size_col:
            defaults: bid_px_00, ask_px_00, bid_sz_00, ask_sz_00
    """

    def required_columns(self) -> list[str]:
        return list(self._best_cols().values())

    def compute(self, df: pd.DataFrame) -> pd.Series:
        c = self._best_cols()
        bid_px = df[c["bid_px"]].astype(float)
        ask_px = df[c["ask_px"]].astype(float)
        bid_sz = df[c["bid_sz"]].astype(float)
        ask_sz = df[c["ask_sz"]].astype(float)
        ofi = (
            bid_sz.diff() * (bid_px >= bid_px.shift(1)).astype(float)
            - ask_sz.diff() * (ask_px <= ask_px.shift(1)).astype(float)
        ).fillna(0.0)
        ofi.iloc[0] = 0.0
        return ofi


@register_feature("ofi_rolling")
class RollingOFIFeature(LOBFeature):
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

    def required_columns(self) -> list[str]:
        return list(self._best_cols().values())

    def compute(self, df: pd.DataFrame) -> pd.Series:
        c = self._best_cols()
        window = int(self._p("window", 50))
        bid_px = df[c["bid_px"]].astype(float)
        ask_px = df[c["ask_px"]].astype(float)
        bid_sz = df[c["bid_sz"]].astype(float)
        ask_sz = df[c["ask_sz"]].astype(float)
        ofi = (
            bid_sz.diff() * (bid_px >= bid_px.shift(1)).astype(float)
            - ask_sz.diff() * (ask_px <= ask_px.shift(1)).astype(float)
        ).fillna(0.0)
        return ofi.rolling(window=window, min_periods=1).sum()


@register_feature("queue_depletion")
class QueueDepletionFeature(LOBFeature):
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

    def _sz_col(self) -> str:
        side = self._p("side", "bid").lower()
        return self._p("ask_size_col", "ask_sz_00") if side == "ask" else self._p("bid_size_col", "bid_sz_00")

    def required_columns(self) -> list[str]:
        return [self._sz_col()]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        sz = df[self._sz_col()].astype(float)
        prev = sz.shift(1)
        return safe_divide(np.maximum(prev - sz, 0.0), prev).fillna(0.0)


@register_feature("mid_price_acceleration")
class MidPriceAccelerationFeature(LOBFeature):
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
        return [self._p("bid_price_col", "bid_px_00"),
                self._p("ask_price_col", "ask_px_00")]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        mid = self._mid(df, self._p("bid_price_col", "bid_px_00"),
                            self._p("ask_price_col", "ask_px_00"))
        return (mid - 2.0 * mid.shift(1) + mid.shift(2)).fillna(0.0)


@register_feature("inter_event_time")
class InterEventTimeFeature(LOBFeature):
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
        gap_ns = (
            pd.Series(df.index.astype(np.int64), index=df.index, dtype=float)
            .diff().fillna(0.0).clip(lower=0.0)
        )
        return np.log1p(gap_ns)


@register_feature("spread_ratio")
class SpreadRatioFeature(LOBFeature):
    """Spread-to-rolling-mean ratio — adverse selection / toxicity signal.

    A spread that is elevated relative to its recent average indicates a
    liquidity regime change: market makers are widening quotes due to
    perceived information asymmetry.

    Formula:
        SpreadBps[t]   = (ask - bid) / MidPrice * 10000
        SpreadRatio[t] = SpreadBps[t] / RollingMean(SpreadBps, window)

    Values > 1: spread wider than recent average (higher toxicity risk).
    Values < 1: tighter than average (lower cost environment).
    Defaults to 1.0 during warm-up when rolling mean is zero.

    Params:
        window: rolling look-back in ticks (default: 100)
        bid_price_col: best bid price column (default: "bid_px_00")
        ask_price_col: best ask price column (default: "ask_px_00")
    """

    def required_columns(self) -> list[str]:
        return [self._p("bid_price_col", "bid_px_00"),
                self._p("ask_price_col", "ask_px_00")]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        bid = df[self._p("bid_price_col", "bid_px_00")].astype(float)
        ask = df[self._p("ask_price_col", "ask_px_00")].astype(float)
        window = int(self._p("window", 100))
        mid = (bid + ask) / 2.0
        spread_bps = safe_divide((ask - bid) * 10_000.0, mid)
        rolling_mean = spread_bps.rolling(window=window, min_periods=1).mean()
        return safe_divide(spread_bps, rolling_mean, fill=1.0)


@register_feature("book_convexity")
class BookConvexityFeature(LOBFeature):
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

    def _px_prefix(self) -> str:
        side = self._p("side", "bid").lower()
        return self._p(
            "ask_price_prefix" if side == "ask" else "bid_price_prefix",
            f"{side}_px_",
        )

    def required_columns(self) -> list[str]:
        px = self._px_prefix()
        return [f"{px}00", f"{px}01", f"{px}02"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        px = self._px_prefix()
        side = self._p("side", "bid").lower()
        p0 = df[f"{px}00"].astype(float)
        p1 = df[f"{px}01"].astype(float)
        p2 = df[f"{px}02"].astype(float)
        if side == "bid":
            return (p0 - p1) - (p1 - p2)
        return (p2 - p1) - (p1 - p0)


@register_feature("order_count_imbalance")
class OrderCountImbalanceFeature(LOBFeature):
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

    def _ct_cols(self) -> tuple[str, str]:
        lvl = int(self._p("level", 0))
        return (
            f"{self._p('bid_count_prefix', 'bid_ct_')}{lvl:02d}",
            f"{self._p('ask_count_prefix', 'ask_ct_')}{lvl:02d}",
        )

    def required_columns(self) -> list[str]:
        return list(self._ct_cols())

    def compute(self, df: pd.DataFrame) -> pd.Series:
        bid_col, ask_col = self._ct_cols()
        bid = df[bid_col].astype(float)
        ask = df[ask_col].astype(float)
        return safe_divide(bid - ask, bid + ask)


@register_feature("signed_trade_flow")
class SignedTradeFlowFeature(LOBFeature):
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
        window:     rolling look-back in ticks (default: 100)
        action_col: column identifying event type (default: "action")
        side_col:   column identifying aggressor side (default: "side")
        size_col:   column with trade/order size (default: "size")
    """

    def required_columns(self) -> list[str]:
        return list(self._trade_cols())

    def compute(self, df: pd.DataFrame) -> pd.Series:
        action_col, side_col, size_col = self._trade_cols()
        window = int(self._p("window", 100))
        action = df[action_col].astype(str)
        side = df[side_col].astype(str)
        size = df[size_col].astype(float)
        is_trade = action == "T"
        sign = pd.Series(0.0, index=df.index)
        sign[is_trade & (side == "B")] = 1.0
        sign[is_trade & (side == "A")] = -1.0
        return (size * sign).rolling(window=window, min_periods=1).sum()


@register_feature("odd_lot_trade_ratio")
class OddLotTradeRatioFeature(LOBFeature):
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
        is_odd[t]   = 1  if action[t]=='T' and size[t] < round_lot  else 0
        is_trade[t] = 1  if action[t]=='T'                          else 0
        OddLotRatio_W[t] = sum(is_odd) / max(sum(is_trade), 1)

    Range: [0, 1].

    Params:
        window:     rolling look-back in ticks (default: 200)
        round_lot:  round-lot threshold in shares (default: 100)
        action_col: column identifying event type (default: "action")
        size_col:   column with trade size (default: "size")
    """

    def required_columns(self) -> list[str]:
        return [self._p("action_col", "action"), self._p("size_col", "size")]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        action_col = self._p("action_col", "action")
        size_col = self._p("size_col", "size")
        window = int(self._p("window", 200))
        round_lot = float(self._p("round_lot", 100))
        is_trade = (df[action_col].astype(str) == "T").astype(float)
        is_odd = is_trade * (df[size_col].astype(float) < round_lot).astype(float)
        rolling_odd = is_odd.rolling(window=window, min_periods=1).sum()
        rolling_trades = is_trade.rolling(window=window, min_periods=1).sum()
        return safe_divide(rolling_odd, rolling_trades)


def _impact_price_side(
    px_cols: list[str],
    sz_cols: list[str],
    df: pd.DataFrame,
    notional: pd.Series,
) -> pd.Series:
    """Walk one side of the order book for each row, consuming up to `notional`.

    Args:
        px_cols: Column names for price levels (best first).
        sz_cols: Column names for size levels (best first).
        df: Source DataFrame.
        notional: Per-row notional amount to consume.

    Returns:
        Series with volume-weighted average price for each row.
    """
    n_levels = len(px_cols)
    remaining = notional.copy()
    wsum = pd.Series(0.0, index=df.index, dtype=float)
    cumvol = pd.Series(0.0, index=df.index, dtype=float)

    for i in range(n_levels):
        px = df[px_cols[i]].astype(float)
        sz = df[sz_cols[i]].astype(float)
        notl = px * sz

        # Only take as much volume as needed
        mask = remaining < notl
        clipped_sz = sz.copy()
        clipped_sz[mask] = remaining[mask] / px[mask]

        wsum += px * clipped_sz
        cumvol += clipped_sz
        remaining -= px * clipped_sz

        if (remaining <= 1e-12).all():
            break

    return safe_divide(wsum, cumvol)


@register_feature("price_vamp")
class PriceVampFeature(LOBFeature):
    """Volume Adjusted Mid Price (VAMP).

    Computes a volume-weighted mid price that accounts for order book depth and
    imbalance, as opposed to the simple mid price which only considers the best
    bid and best ask. Originates from short-term crypto price prediction research
    (SSRN: 4351947).

    The algorithm walks into the book from both sides:
        1. impact_price(asks, bid_notional) -> volume-weighted ask price
        2. impact_price(bids, ask_notional) -> volume-weighted bid price
        3. VAMP = (P1 + P2) / 2

    A small notional produces VAMP close to the simple mid price (only BBA
    involved); larger notionals capture deeper structural imbalance.

    Params:
        levels: Number of book levels to walk (default: 5, i.e., L0-L4)
        notional: Dollar amount to walk into the book on each side (default: 1000)
        bid_notional: Separate notional for the bid side (falls back to notional if 0)
        ask_notional: Separate notional for the ask side (falls back to notional if 0)
        bid_price_prefix: Column prefix for bid prices (default: "bid_px_")
        ask_price_prefix: Column prefix for ask prices (default: "ask_px_")
        bid_size_prefix: Column prefix for bid sizes (default: "bid_sz_")
        ask_size_prefix: Column prefix for ask sizes (default: "ask_sz_")
    """

    def required_columns(self) -> list[str]:
        n = int(self._p("levels", 5))
        bpp = self._p("bid_price_prefix", "bid_px_")
        app = self._p("ask_price_prefix", "ask_px_")
        bsp = self._p("bid_size_prefix", "bid_sz_")
        asp = self._p("ask_size_prefix", "ask_sz_")
        return [
            col
            for i in range(n)
            for col in (
                f"{bpp}{i:02d}", f"{app}{i:02d}",
                f"{bsp}{i:02d}", f"{asp}{i:02d}",
            )
        ]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        n = int(self._p("levels", 5))
        notional = float(self._p("notional", 1000))
        bid_notional = float(self._p("bid_notional", 0))
        ask_notional = float(self._p("ask_notional", 0))

        # Fall back to `notional` if side-specific notionals are not set
        bid_notional = bid_notional if bid_notional > 0 else notional
        ask_notional = ask_notional if ask_notional > 0 else notional

        bpp = self._p("bid_price_prefix", "bid_px_")
        app = self._p("ask_price_prefix", "ask_px_")
        bsp = self._p("bid_size_prefix", "bid_sz_")
        asp = self._p("ask_size_prefix", "ask_sz_")

        bid_px_cols = [f"{bpp}{i:02d}" for i in range(n)]
        ask_px_cols = [f"{app}{i:02d}" for i in range(n)]
        bid_sz_cols = [f"{bsp}{i:02d}" for i in range(n)]
        ask_sz_cols = [f"{asp}{i:02d}" for i in range(n)]

        bid_not = pd.Series(bid_notional, index=df.index, dtype=float)
        ask_not = pd.Series(ask_notional, index=df.index, dtype=float)

        # P1: walk asks with bid-side notional (a buyer walks up the ask book)
        p1 = _impact_price_side(ask_px_cols, ask_sz_cols, df, bid_not)
        # P2: walk bids with ask-side notional (a seller walks down the bid book)
        p2 = _impact_price_side(bid_px_cols, bid_sz_cols, df, ask_not)

        return (p1 + p2) / 2.0


@register_feature("odd_lot_imbalance")
class OddLotImbalanceFeature(LOBFeature):
    """Rolling signed odd-lot flow imbalance.

    Measures the directional bias in odd-lot (retail) order flow.
    Buyer-initiated odd-lot trades indicate retail buying pressure;
    seller-initiated odd-lot trades indicate retail selling pressure.
    The net imbalance normalized by total odd-lot volume reveals which
    side retail is leaning toward.

    Only rows where action == 'T' and size < round_lot contribute.

    Formula:
        odd_buy[t]  = size[t]  if action=='T', side=='B', size < round_lot
        odd_sell[t] = size[t]  if action=='T', side=='A', size < round_lot
        OddLotImbalance_W = (sum(odd_buy) - sum(odd_sell))
                          / max(sum(odd_buy) + sum(odd_sell), 1)

    Range: [-1, +1].

    Params:
        window:     rolling look-back in ticks (default: 200)
        round_lot:  round-lot threshold in shares (default: 100)
        action_col: column identifying event type (default: "action")
        side_col:   column identifying aggressor side (default: "side")
        size_col:   column with trade size (default: "size")
    """

    def required_columns(self) -> list[str]:
        return list(self._trade_cols())

    def compute(self, df: pd.DataFrame) -> pd.Series:
        action_col, side_col, size_col = self._trade_cols()
        window = int(self._p("window", 200))
        round_lot = float(self._p("round_lot", 100))
        action = df[action_col].astype(str)
        side = df[side_col].astype(str)
        size = df[size_col].astype(float)
        is_odd = (action == "T") & (size < round_lot)
        odd_buy = pd.Series(0.0, index=df.index)
        odd_sell = pd.Series(0.0, index=df.index)
        odd_buy[is_odd & (side == "B")] = size[is_odd & (side == "B")]
        odd_sell[is_odd & (side == "A")] = size[is_odd & (side == "A")]
        rb = odd_buy.rolling(window=window, min_periods=1).sum()
        rs = odd_sell.rolling(window=window, min_periods=1).sum()
        return safe_divide(rb - rs, rb + rs)


# ---------------------------------------------------------------------------
# Flow and adverse selection features
# ---------------------------------------------------------------------------

@register_feature("cancel_to_trade_ratio")
class CancelToTradeRatioFeature(LOBFeature):
    """Rolling cancel-to-trade ratio — spoofing / adverse selection proxy.

    Measures the fraction of order book events that are cancellations relative
    to actual trades. A high ratio indicates that liquidity providers are
    repeatedly posting and cancelling orders without commitment (layering or
    spoofing), or that market makers are rapidly adapting to perceived
    information asymmetry.

    Only rows where action == 'C' (cancel) or action == 'T' (trade) contribute.
    Add and other events are excluded from both numerator and denominator.

    Formula:
        cancels[t]  = count(action == 'C')  over window
        trades[t]   = count(action == 'T')  over window
        C2T[t]     = cancels[t] / max(trades[t], 1)

    Range: [0, inf). Typical values for liquid US equities: 1–5.
    Values >> 5 indicate highly ephemeral liquidity.

    Params:
        window:     rolling look-back in ticks (default: 200)
        action_col: column identifying event type (default: "action")
    """

    def required_columns(self) -> list[str]:
        return [self._p("action_col", "action")]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        action_col = self._p("action_col", "action")
        window = int(self._p("window", 200))
        action = df[action_col].astype(str)
        cancels = (action == "C").astype(float)
        trades = (action == "T").astype(float)
        rolling_cancels = cancels.rolling(window=window, min_periods=1).sum()
        rolling_trades = trades.rolling(window=window, min_periods=1).sum()
        return safe_divide(rolling_cancels, rolling_trades)


@register_feature("ofi_multilevel")
class MultiLevelOFIFeature(LOBFeature):
    """Multi-level Order Flow Imbalance — Cont, Kukanov & Stoikov (2014).

    Extends the best-level OFI to multiple book levels, handling the three
    cases for each level as defined in the original paper:
      - Price improvement (bid up / ask down): all volume at new level is new flow
      - Price unchanged: only the queue size change contributes
      - Price deterioration (bid down / ask up): all volume at old level is
        withdrawn flow (negative contribution)

    Aggregates across N levels to produce a single OFI signal that captures
    order flow pressure deeper in the book.

    Formula (per level i, bid side):
        bid_event =  V^bid_i[t]                    if P^bid_i[t] > P^bid_i[t-1]
                     V^bid_i[t] - V^bid_i[t-1]     if P^bid_i[t] == P^bid_i[t-1]
                    -V^bid_i[t-1]                   if P^bid_i[t] < P^bid_i[t-1]

    Formula (per level i, ask side, analogous with sign flipped):
        ask_event =  V^ask_i[t]                    if P^ask_i[t] < P^ask_i[t-1]
                     V^ask_i[t] - V^ask_i[t-1]     if P^ask_i[t] == P^ask_i[t-1]
                    -V^ask_i[t-1]                   if P^ask_i[t] > P^ask_i[t-1]

    MultiLevelOFI = sum_i (bid_event_i - ask_event_i)

    Positive: net buying pressure across levels. Negative: net selling.
    First row is 0.

    Params:
        levels:     number of book levels to aggregate (default: 3)
        bid_price_prefix: default "bid_px_"
        ask_price_prefix: default "ask_px_"
        bid_size_prefix:  default "bid_sz_"
        ask_size_prefix:  default "ask_sz_"
    """

    def required_columns(self) -> list[str]:
        n = int(self._p("levels", 3))
        bpp = self._p("bid_price_prefix", "bid_px_")
        app = self._p("ask_price_prefix", "ask_px_")
        bsp = self._p("bid_size_prefix", "bid_sz_")
        asp = self._p("ask_size_prefix", "ask_sz_")
        return [
            col
            for i in range(n)
            for col in (f"{bpp}{i:02d}", f"{app}{i:02d}",
                        f"{bsp}{i:02d}", f"{asp}{i:02d}")
        ]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        n = int(self._p("levels", 3))
        bpp = self._p("bid_price_prefix", "bid_px_")
        app = self._p("ask_price_prefix", "ask_px_")
        bsp = self._p("bid_size_prefix", "bid_sz_")
        asp = self._p("ask_size_prefix", "ask_sz_")

        ofi_total = pd.Series(0.0, index=df.index)

        for i in range(n):
            bid_px = df[f"{bpp}{i:02d}"].astype(float)
            ask_px = df[f"{app}{i:02d}"].astype(float)
            bid_sz = df[f"{bsp}{i:02d}"].astype(float)
            ask_sz = df[f"{asp}{i:02d}"].astype(float)

            prev_bid_px = bid_px.shift(1)
            prev_ask_px = ask_px.shift(1)
            prev_bid_sz = bid_sz.shift(1)
            prev_ask_sz = ask_sz.shift(1)

            # Bid side: three cases from Cont et al. (2014)
            bid_up = bid_px > prev_bid_px
            bid_same = bid_px == prev_bid_px
            bid_down = bid_px < prev_bid_px

            bid_event = pd.Series(0.0, index=df.index)
            bid_event[bid_up] = bid_sz[bid_up]
            bid_event[bid_same] = (bid_sz - prev_bid_sz)[bid_same]
            bid_event[bid_down] = -prev_bid_sz[bid_down]

            # Ask side: three cases (mirrored)
            ask_down = ask_px < prev_ask_px
            ask_same = ask_px == prev_ask_px
            ask_up = ask_px > prev_ask_px

            ask_event = pd.Series(0.0, index=df.index)
            ask_event[ask_down] = ask_sz[ask_down]
            ask_event[ask_same] = (ask_sz - prev_ask_sz)[ask_same]
            ask_event[ask_up] = -prev_ask_sz[ask_up]

            ofi_total += bid_event - ask_event

        ofi_total.iloc[0] = 0.0
        return ofi_total


@register_feature("vpin")
class VPINFeature(LOBFeature):
    """Volume-synchronized Probability of Informed Trading (tick approximation).

    Tick-based approximation of VPIN (Easley, Lopez de Prado & O'Hara, 2012).
    The original VPIN uses volume-clock bucketing and bulk volume classification;
    this implementation uses a rolling tick window with actual trade side
    classification from the data, which is more accurate when side labels are
    available.

    Measures the imbalance between buyer- and seller-initiated trade volume
    normalized by total trade volume. High VPIN indicates toxic order flow
    (informed trading) and signals market makers to widen quotes.

    Formula:
        buy_vol[t]  = size[t]  if action[t]=='T' and side[t]=='B'  else 0
        sell_vol[t] = size[t]  if action[t]=='T' and side[t]=='A'  else 0
        VPIN_W[t]   = |sum(buy_vol) - sum(sell_vol)| / max(sum(buy_vol + sell_vol), 1)

    Range: [0, 1]. Values near 1 indicate one-directional informed flow.
    Values near 0 indicate balanced, uninformed flow.

    Params:
        window:     rolling look-back in ticks (default: 100)
        action_col: column identifying event type (default: "action")
        side_col:   column identifying aggressor side (default: "side")
        size_col:   column with trade size (default: "size")
    """

    def required_columns(self) -> list[str]:
        return list(self._trade_cols())

    def compute(self, df: pd.DataFrame) -> pd.Series:
        action_col, side_col, size_col = self._trade_cols()
        window = int(self._p("window", 100))
        action = df[action_col].astype(str)
        side = df[side_col].astype(str)
        size = df[size_col].astype(float)
        is_trade = action == "T"
        buy_vol = pd.Series(0.0, index=df.index)
        sell_vol = pd.Series(0.0, index=df.index)
        buy_vol[is_trade & (side == "B")] = size[is_trade & (side == "B")]
        sell_vol[is_trade & (side == "A")] = size[is_trade & (side == "A")]
        rb = buy_vol.rolling(window=window, min_periods=1).sum()
        rs = sell_vol.rolling(window=window, min_periods=1).sum()
        return safe_divide((rb - rs).abs(), rb + rs)
