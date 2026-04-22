"""Trade-tape, toxicity, and trade-size-derived LOB features."""

from __future__ import annotations

import pandas as pd

from trading_rl.features.lob_common import LOBFeature, safe_divide, split_trade_flow
from trading_rl.features.registry import register_feature


@register_feature("signed_trade_flow")
class SignedTradeFlowFeature(LOBFeature):
    """Rolling signed trade flow (cumulative delta)."""

    def required_columns(self) -> list[str]:
        return list(self._trade_cols())

    def compute(self, df: pd.DataFrame) -> pd.Series:
        action_col, side_col, size_col = self._trade_cols()
        window = int(self._p("window", 100))
        _is_trade, buy_vol, sell_vol = split_trade_flow(
            df,
            action_col=action_col,
            side_col=side_col,
            size_col=size_col,
        )
        return (buy_vol - sell_vol).rolling(window=window, min_periods=1).sum()


@register_feature("odd_lot_trade_ratio")
class OddLotTradeRatioFeature(LOBFeature):
    """Rolling fraction of trades that are odd-lot sized."""

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


@register_feature("odd_lot_imbalance")
class OddLotImbalanceFeature(LOBFeature):
    """Rolling signed odd-lot flow imbalance."""

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


@register_feature("vpin")
class VPINFeature(LOBFeature):
    """Volume-synchronized probability of informed trading."""

    def required_columns(self) -> list[str]:
        return list(self._trade_cols())

    def compute(self, df: pd.DataFrame) -> pd.Series:
        action_col, side_col, size_col = self._trade_cols()
        window = int(self._p("window", 100))
        _is_trade, buy_vol, sell_vol = split_trade_flow(
            df,
            action_col=action_col,
            side_col=side_col,
            size_col=size_col,
        )
        rb = buy_vol.rolling(window=window, min_periods=1).sum()
        rs = sell_vol.rolling(window=window, min_periods=1).sum()
        return safe_divide((rb - rs).abs(), rb + rs)


@register_feature("large_trade_ratio")
class LargeTradeRatioFeature(LOBFeature):
    """Rolling fraction of large trades."""

    def required_columns(self) -> list[str]:
        return [self._p("action_col", "action"), self._p("size_col", "size")]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        action_col = self._p("action_col", "action")
        size_col = self._p("size_col", "size")
        window = int(self._p("window", 200))
        threshold = float(self._p("threshold", 500))
        action = df[action_col].astype(str)
        size = df[size_col].astype(float)
        is_trade = (action == "T").astype(float)
        is_large = is_trade * (size >= threshold).astype(float)
        rolling_large = is_large.rolling(window=window, min_periods=1).sum()
        rolling_trades = is_trade.rolling(window=window, min_periods=1).sum()
        return safe_divide(rolling_large, rolling_trades)
