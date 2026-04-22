"""Shared helpers for limit order book feature families."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_rl.features.base import Feature


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
        """Compute simple midpoint from bid and ask columns."""
        return (df[bid_col].astype(float) + df[ask_col].astype(float)) / 2.0


def impact_price_side(
    px_cols: list[str],
    sz_cols: list[str],
    df: pd.DataFrame,
    notional: pd.Series,
) -> pd.Series:
    """Walk one side of the order book for each row, consuming up to `notional`."""
    n_levels = len(px_cols)
    remaining = notional.copy()
    wsum = pd.Series(0.0, index=df.index, dtype=float)
    cumvol = pd.Series(0.0, index=df.index, dtype=float)

    for i in range(n_levels):
        px = df[px_cols[i]].astype(float)
        sz = df[sz_cols[i]].astype(float)
        notl = px * sz

        mask = remaining < notl
        clipped_sz = sz.copy()
        clipped_sz[mask] = remaining[mask] / px[mask]

        wsum += px * clipped_sz
        cumvol += clipped_sz
        remaining -= px * clipped_sz

        if (remaining <= 1e-12).all():
            break

    return safe_divide(wsum, cumvol)


def best_level_ofi(
    df: pd.DataFrame,
    *,
    bid_price_col: str,
    ask_price_col: str,
    bid_size_col: str,
    ask_size_col: str,
) -> pd.Series:
    """Compute best-level order flow imbalance from consecutive snapshots."""
    bid_px = df[bid_price_col].astype(float)
    ask_px = df[ask_price_col].astype(float)
    bid_sz = df[bid_size_col].astype(float)
    ask_sz = df[ask_size_col].astype(float)
    ofi = (
        bid_sz.diff() * (bid_px >= bid_px.shift(1)).astype(float)
        - ask_sz.diff() * (ask_px <= ask_px.shift(1)).astype(float)
    ).fillna(0.0)
    ofi.iloc[0] = 0.0
    return ofi


def split_trade_flow(
    df: pd.DataFrame,
    *,
    action_col: str,
    side_col: str,
    size_col: str,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Split trade tape into trade mask plus signed buy/sell volume series."""
    action = df[action_col].astype(str)
    side = df[side_col].astype(str)
    size = df[size_col].astype(float)
    is_trade = action == "T"

    buy_vol = pd.Series(0.0, index=df.index, dtype=float)
    sell_vol = pd.Series(0.0, index=df.index, dtype=float)
    buy_vol[is_trade & (side == "B")] = size[is_trade & (side == "B")]
    sell_vol[is_trade & (side == "A")] = size[is_trade & (side == "A")]
    return is_trade.astype(float), buy_vol, sell_vol
