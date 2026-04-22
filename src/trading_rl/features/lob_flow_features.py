"""Order-flow and event-timing limit order book features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_rl.features.lob_common import LOBFeature, best_level_ofi, safe_divide
from trading_rl.features.registry import register_feature


@register_feature("ofi")
class OrderFlowImbalanceFeature(LOBFeature):
    """Best-level order flow imbalance."""

    def required_columns(self) -> list[str]:
        return list(self._best_cols().values())

    def compute(self, df: pd.DataFrame) -> pd.Series:
        c = self._best_cols()
        return best_level_ofi(
            df,
            bid_price_col=c["bid_px"],
            ask_price_col=c["ask_px"],
            bid_size_col=c["bid_sz"],
            ask_size_col=c["ask_sz"],
        )


@register_feature("ofi_rolling")
class RollingOFIFeature(LOBFeature):
    """Rolling signed order-flow imbalance over a fixed tick window."""

    def required_columns(self) -> list[str]:
        return list(self._best_cols().values())

    def compute(self, df: pd.DataFrame) -> pd.Series:
        c = self._best_cols()
        window = int(self._p("window", 50))
        ofi = best_level_ofi(
            df,
            bid_price_col=c["bid_px"],
            ask_price_col=c["ask_px"],
            bid_size_col=c["bid_sz"],
            ask_size_col=c["ask_sz"],
        )
        return ofi.rolling(window=window, min_periods=1).sum()


@register_feature("queue_depletion")
class QueueDepletionFeature(LOBFeature):
    """Queue depletion rate at the best bid or ask."""

    def _sz_col(self) -> str:
        side = self._p("side", "bid").lower()
        if side == "ask":
            return self._p("ask_size_col", "ask_sz_00")
        return self._p("bid_size_col", "bid_sz_00")

    def required_columns(self) -> list[str]:
        return [self._sz_col()]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        sz = df[self._sz_col()].astype(float)
        prev = sz.shift(1)
        return safe_divide(np.maximum(prev - sz, 0.0), prev).fillna(0.0)


@register_feature("inter_event_time")
class InterEventTimeFeature(LOBFeature):
    """Log inter-event time between consecutive order book updates."""

    def required_columns(self) -> list[str]:
        return []

    def compute(self, df: pd.DataFrame) -> pd.Series:
        gap_ns = (
            pd.Series(df.index.astype(np.int64), index=df.index, dtype=float)
            .diff()
            .fillna(0.0)
            .clip(lower=0.0)
        )
        return np.log1p(gap_ns)


@register_feature("cancel_to_trade_ratio")
class CancelToTradeRatioFeature(LOBFeature):
    """Rolling cancel-to-trade ratio."""

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
    """Multi-level order flow imbalance."""

    def required_columns(self) -> list[str]:
        n = int(self._p("levels", 3))
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

            bid_up = bid_px > prev_bid_px
            bid_same = bid_px == prev_bid_px
            bid_down = bid_px < prev_bid_px

            bid_event = pd.Series(0.0, index=df.index)
            bid_event[bid_up] = bid_sz[bid_up]
            bid_event[bid_same] = (bid_sz - prev_bid_sz)[bid_same]
            bid_event[bid_down] = -prev_bid_sz[bid_down]

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


@register_feature("trade_arrival_rate")
class TradeArrivalRateFeature(LOBFeature):
    """Rolling trade arrival rate."""

    def required_columns(self) -> list[str]:
        return [self._p("action_col", "action")]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        action_col = self._p("action_col", "action")
        window = int(self._p("window", 100))
        trades = (df[action_col].astype(str) == "T").astype(float)
        return trades.rolling(window=window, min_periods=1).sum()


@register_feature("ofi_autocorrelation")
class OFIAutocorrelationFeature(LOBFeature):
    """Rolling autocorrelation of order flow imbalance."""

    def required_columns(self) -> list[str]:
        return list(self._best_cols().values())

    def compute(self, df: pd.DataFrame) -> pd.Series:
        c = self._best_cols()
        window = int(self._p("window", 50))
        ofi = best_level_ofi(
            df,
            bid_price_col=c["bid_px"],
            ask_price_col=c["ask_px"],
            bid_size_col=c["bid_sz"],
            ask_size_col=c["ask_sz"],
        )
        return (
            ofi.rolling(window=window, min_periods=2)
            .apply(
                lambda x: x.autocorr(lag=1) if len(x) >= 2 else 0.0,
                raw=False,
            )
            .fillna(0.0)
        )
