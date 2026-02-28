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
