"""Book-state and depth-derived limit order book features."""

from __future__ import annotations

import pandas as pd

from trading_rl.features.lob_common import LOBFeature, impact_price_side, safe_divide
from trading_rl.features.registry import register_feature


@register_feature("book_pressure")
class BookPressureFeature(LOBFeature):
    """Book Pressure (Volume Imbalance) at a specific level."""

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
    """Spread in basis points."""

    def required_columns(self) -> list[str]:
        return [
            self._p("bid_price_col", "bid_px_00"),
            self._p("ask_price_col", "ask_px_00"),
        ]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        bid = df[self._p("bid_price_col", "bid_px_00")].astype(float)
        ask = df[self._p("ask_price_col", "ask_px_00")].astype(float)
        mid = (bid + ask) / 2.0
        return safe_divide((ask - bid) * 10_000.0, mid)


@register_feature("microprice")
class MicropriceFeature(LOBFeature):
    """Microprice (volume-weighted fair value)."""

    def required_columns(self) -> list[str]:
        return list(self._best_cols().values())

    def compute(self, df: pd.DataFrame) -> pd.Series:
        c = self._best_cols()
        bid_px = df[c["bid_px"]].astype(float)
        ask_px = df[c["ask_px"]].astype(float)
        bid_sz = df[c["bid_sz"]].astype(float)
        ask_sz = df[c["ask_sz"]].astype(float)
        total = bid_sz + ask_sz
        mid = (bid_px + ask_px) / 2.0
        weighted = ask_sz * bid_px + bid_sz * ask_px
        result = mid.copy()
        mask = total > 0
        result[mask] = weighted[mask] / total[mask]
        return result


@register_feature("microprice_divergence")
class MicropriceDivergenceFeature(LOBFeature):
    """Microprice divergence from mid price."""

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
        microprice[mask] = (
            ask_sz[mask] * bid_px[mask] + bid_sz[mask] * ask_px[mask]
        ) / total[mask]
        return microprice - mid


@register_feature("depth_ratio")
class DepthRatioFeature(LOBFeature):
    """Depth ratio (top vs. deep book)."""

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
    """Volume-weighted mid-price skew."""

    def required_columns(self) -> list[str]:
        bpp = self._p("bid_price_prefix", "bid_px_")
        app = self._p("ask_price_prefix", "ask_px_")
        bsp = self._p("bid_size_prefix", "bid_sz_")
        asp = self._p("ask_size_prefix", "ask_sz_")
        n = int(self._p("levels", 3))
        return [
            col
            for i in range(n)
            for col in (
                f"{bpp}{i:02d}",
                f"{app}{i:02d}",
                f"{bsp}{i:02d}",
                f"{asp}{i:02d}",
            )
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
    """Bid or ask slope (elasticity)."""

    def _side_params(self) -> tuple[str, str, str, int]:
        side = self._p("side", "").lower()
        if side not in ("bid", "ask"):
            raise ValueError(
                f"bid_ask_slope requires side='bid' or 'ask', got: {side!r}"
            )
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
    """Multi-level order book imbalance."""

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


@register_feature("mid_price_acceleration")
class MidPriceAccelerationFeature(LOBFeature):
    """Mid-price acceleration (second finite difference)."""

    def required_columns(self) -> list[str]:
        return [
            self._p("bid_price_col", "bid_px_00"),
            self._p("ask_price_col", "ask_px_00"),
        ]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        mid = self._mid(
            df,
            self._p("bid_price_col", "bid_px_00"),
            self._p("ask_price_col", "ask_px_00"),
        )
        return (mid - 2.0 * mid.shift(1) + mid.shift(2)).fillna(0.0)


@register_feature("spread_ratio")
class SpreadRatioFeature(LOBFeature):
    """Spread-to-rolling-mean ratio with session boundary handling."""

    def required_columns(self) -> list[str]:
        return [
            self._p("bid_price_col", "bid_px_00"),
            self._p("ask_price_col", "ask_px_00"),
        ]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        from trading_rl.features.utils import apply_per_session_with_params

        bid = df[self._p("bid_price_col", "bid_px_00")].astype(float)
        ask = df[self._p("ask_price_col", "ask_px_00")].astype(float)
        window = int(self._p("window", 100))
        threshold_hours = float(self._p("session_break_threshold_hours", 1.0))
        mid = (bid + ask) / 2.0
        spread_bps = safe_divide((ask - bid) * 10_000.0, mid)

        # Compute rolling mean with session resets
        rolling_mean = apply_per_session_with_params(
            spread_bps,
            lambda s, w: s.rolling(window=w, min_periods=1).mean(),
            window,
            threshold_hours=threshold_hours,
        )

        return safe_divide(spread_bps, rolling_mean, fill=1.0)


@register_feature("book_convexity")
class BookConvexityFeature(LOBFeature):
    """Book convexity (curvature of bid or ask price levels)."""

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
    """Order count imbalance at a specific book level."""

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


@register_feature("price_vamp")
class PriceVampFeature(LOBFeature):
    """Volume adjusted mid price (VAMP)."""

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
                f"{bpp}{i:02d}",
                f"{app}{i:02d}",
                f"{bsp}{i:02d}",
                f"{asp}{i:02d}",
            )
        ]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        n = int(self._p("levels", 5))
        notional = float(self._p("notional", 1000))
        bid_notional = float(self._p("bid_notional", 0))
        ask_notional = float(self._p("ask_notional", 0))
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
        p1 = impact_price_side(ask_px_cols, ask_sz_cols, df, bid_not)
        p2 = impact_price_side(bid_px_cols, bid_sz_cols, df, ask_not)
        return (p1 + p2) / 2.0
