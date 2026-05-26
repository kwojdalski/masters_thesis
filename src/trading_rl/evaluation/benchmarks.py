"""Benchmark evaluation helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from logger import get_logger
from trading_rl.config import DEFAULT_INITIAL_PORTFOLIO_VALUE
from trading_rl.constants import BenchmarkName
from trading_rl.evaluation.statistical_benchmarks import (
    compute_buy_and_hold_returns,

    compute_twap_returns,
    compute_vwap_returns,
    resolve_vwap_volume_series,
)

logger = get_logger(__name__)


def benchmarks_from_config(benchmarks_config: Any) -> frozenset[BenchmarkName]:
    """Build a frozenset of plot-relevant BenchmarkName values from a BenchmarksConfig.

    Returns only the names relevant to equity-curve plots (excludes RANDOM_ACTIONS,
    which is handled separately via statistical tests, not plotted as an equity line).
    """
    enabled_set = getattr(benchmarks_config, "enabled_set", None)
    if enabled_set is None:
        enabled_set = frozenset(getattr(benchmarks_config, "enabled", []))
    return frozenset(b for b in enabled_set if b != BenchmarkName.RANDOM_ACTIONS)


@dataclass
class BenchmarkSpec:
    """A benchmark defined by a name and a pure return-computation callable.

    ``compute_returns(max_steps)`` closes over whatever data it needs (price
    series, volume series) and returns a simple-returns array of length
    ``max_steps``.  No environment objects are involved.
    """

    name: str
    compute_returns: Callable[[int], np.ndarray]
    metadata: dict[str, Any] = field(default_factory=dict)


class BenchmarkReturnArray(np.ndarray):
    """Return array carrying benchmark position direction metadata."""

    benchmark_position_side: float | None

    def __new__(
        cls, values: np.ndarray, benchmark_position_side: float | None
    ) -> BenchmarkReturnArray:
        obj = np.asarray(values, dtype=float).view(cls)
        obj.benchmark_position_side = benchmark_position_side
        return obj

    def __array_finalize__(self, obj: Any) -> None:
        if obj is None:
            return
        self.benchmark_position_side = getattr(obj, "benchmark_position_side", None)


def _benchmark_returns(
    values: np.ndarray, benchmark_position_side: float | None
) -> BenchmarkReturnArray:
    return BenchmarkReturnArray(values, benchmark_position_side)


class BenchmarkEngine:
    """Factory for pure price-data benchmarks — no environment coupling."""

    @staticmethod
    def buy_and_hold(prices: pd.Series) -> BenchmarkSpec:
        return BenchmarkSpec(
            name=BenchmarkName.BUY_AND_HOLD,
            compute_returns=lambda steps: _benchmark_returns(
                compute_buy_and_hold_returns(prices, steps),
                benchmark_position_side=1.0,
            ),
        )

    @staticmethod
    def twap(prices: pd.Series) -> BenchmarkSpec:
        return BenchmarkSpec(
            name=BenchmarkName.TWAP,
            compute_returns=lambda steps: _benchmark_returns(
                compute_twap_returns(prices, steps),
                benchmark_position_side=1.0,
            ),
        )

    @staticmethod
    def vwap(
        prices: pd.Series,
        volumes: pd.Series,
        *,
        volume_source: str | None = None,
    ) -> BenchmarkSpec:
        meta: dict[str, Any] = {}
        if volume_source:
            meta["volume_source"] = volume_source
        return BenchmarkSpec(
            name=BenchmarkName.VWAP,
            compute_returns=lambda steps: _benchmark_returns(
                compute_vwap_returns(prices, volumes, steps),
                benchmark_position_side=1.0,
            ),
            metadata=meta,
        )

    @staticmethod
    def build(
        market_data: pd.DataFrame,
        config: Any,
        price_column: str = "close",
    ) -> tuple[list[BenchmarkSpec], dict[str, str]]:
        """Build all enabled benchmark specs from market data.

        Args:
            market_data: Full split DataFrame (prices + optional volume columns).
            config: ``BenchmarksConfig`` controlling which benchmarks are on.
            price_column: Preferred price column; falls back to ``"close"`` if absent.

        Returns:
            A ``(specs, metadata)`` tuple where ``metadata`` may contain
            ``"vwap_volume_source"`` if VWAP is enabled.
        """
        if price_column not in market_data.columns:
            if "close" in market_data.columns:
                price_column = "close"
            else:
                logger.warning(
                    "No price column '%s' or 'close' found; no benchmarks built.",
                    price_column,
                )
                return [], {}

        prices = market_data[price_column]
        specs: list[BenchmarkSpec] = []
        result_meta: dict[str, str] = {}
        _enabled_set = getattr(config, "enabled_set", None)
        _enabled_list = getattr(config, "enabled", None)
        if _enabled_set is not None:
            enabled = frozenset(_enabled_set)
        elif isinstance(_enabled_list, list):
            enabled = frozenset(BenchmarkName(v) for v in _enabled_list)
        else:
            # Backward compat: SimpleNamespace / old BenchmarksConfig with individual booleans.
            _flag_map = {
                "buy_and_hold":   BenchmarkName.BUY_AND_HOLD,
                "twap":           BenchmarkName.TWAP,
                "vwap":           BenchmarkName.VWAP,
            }
            enabled = frozenset(
                bname for attr, bname in _flag_map.items() if getattr(config, attr, False)
            )

        if BenchmarkName.BUY_AND_HOLD in enabled:
            specs.append(BenchmarkEngine.buy_and_hold(prices))

        if BenchmarkName.TWAP in enabled:
            specs.append(BenchmarkEngine.twap(prices))

        if BenchmarkName.VWAP in enabled:
            volumes, volume_source = resolve_vwap_volume_series(market_data)
            if volumes is None:
                logger.warning(
                    "VWAP benchmark skipped: no usable volume column found. "
                    "Expected action+size (MBP-10 T-events), or one of: "
                    "trade_volume, volume, last_size, qty, or bid_sz_00/ask_sz_00 for proxy."
                )
            else:
                if volume_source and "proxy" in str(volume_source):
                    logger.warning(
                        "VWAP is using %s. This is quote-size-weighted, not true traded volume.",
                        volume_source,
                    )
                specs.append(BenchmarkEngine.vwap(prices, volumes, volume_source=volume_source))
                if volume_source:
                    result_meta["vwap_volume_source"] = volume_source

        return specs, result_meta


def calculate_benchmark_dsr(
    df_prices,
    strategy: str | BenchmarkName = BenchmarkName.BUY_AND_HOLD,
    eta=0.01,
    epsilon=1e-8,
    max_steps=None,
    price_column: str = "close",
    initial_portfolio_value: float = DEFAULT_INITIAL_PORTFOLIO_VALUE,
):
    """Calculate differential Sharpe ratio for a benchmark trading strategy."""
    try:
        benchmark_name = BenchmarkName(strategy)
    except ValueError as exc:
        raise ValueError(f"Unknown strategy: {strategy}") from exc

    if price_column not in df_prices.columns:
        if "close" in df_prices.columns:
            logger.warning(
                "DSR benchmark price column '%s' not found; falling back to 'close'.",
                price_column,
            )
            price_column = "close"
        else:
            raise ValueError(
                f"DSR benchmark requires '{price_column}' or 'close' in df_prices columns."
            )

    prices = df_prices[price_column].values
    if max_steps is None:
        max_steps = len(prices) - 1
    else:
        max_steps = min(max_steps, len(prices) - 1)

    portfolio_values = [initial_portfolio_value]
    price_returns = np.diff(prices[: max_steps + 1]) / prices[:max_steps]

    if benchmark_name == BenchmarkName.BUY_AND_HOLD:
        positions = np.ones(len(price_returns))
    elif benchmark_name == BenchmarkName.MAX_PROFIT:
        positions = np.sign(price_returns)
        positions[positions == 0] = 1
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    for pos, ret in zip(positions, price_returns, strict=False):
        portfolio_values.append(portfolio_values[-1] * (1 + pos * ret))

    portfolio_values = np.array(portfolio_values)
    a_t = 0.0
    b_t = 0.0
    dsr_values = [0.0]

    for i in range(1, len(portfolio_values)):
        if portfolio_values[i - 1] <= 0 or portfolio_values[i] <= 0:
            logger.warning(
                "Invalid portfolio value at step %s: %s -> %s",
                i,
                portfolio_values[i - 1],
                portfolio_values[i],
            )
            dsr_values.append(0.0)
            # Decay EMAs so they don't remain frozen at pre-gap values,
            # which would cause a spike when the portfolio recovers.
            a_t = (1 - eta) * a_t
            b_t = (1 - eta) * b_t
            continue

        r_t = float(np.log(portfolio_values[i] / portfolio_values[i - 1]))
        delta_a = r_t - a_t
        delta_b = r_t**2 - b_t
        variance = b_t - a_t**2
        denominator = max(variance, 0.0) ** 1.5 + epsilon
        dsr = (b_t * delta_a - a_t * delta_b / 2) / denominator

        a_t = (1 - eta) * a_t + eta * r_t
        b_t = (1 - eta) * b_t + eta * (r_t**2)
        dsr_values.append(dsr)

    cumulative_dsr = np.cumsum(dsr_values)
    logger.debug(
        "Calculated %s DSR benchmark: %s steps, final DSR: %.4f, final portfolio: $%.2f",
        strategy,
        len(cumulative_dsr),
        cumulative_dsr[-1],
        portfolio_values[-1],
    )
    return cumulative_dsr, portfolio_values
