"""Benchmark evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from logger import get_logger
from trading_rl.config import DEFAULT_INITIAL_PORTFOLIO_VALUE
from trading_rl.evaluation.statistical_benchmarks import (
    compute_buy_and_hold_returns,
    compute_short_and_hold_returns,
    compute_twap_returns,
    compute_vwap_returns,
    resolve_vwap_volume_series,
)

logger = get_logger(__name__)


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


class BenchmarkEngine:
    """Factory for pure price-data benchmarks — no environment coupling."""

    @staticmethod
    def buy_and_hold(prices: pd.Series) -> BenchmarkSpec:
        return BenchmarkSpec(
            name="buy_and_hold",
            compute_returns=lambda steps: compute_buy_and_hold_returns(prices, steps),
        )

    @staticmethod
    def short_and_hold(prices: pd.Series) -> BenchmarkSpec:
        return BenchmarkSpec(
            name="short_and_hold",
            compute_returns=lambda steps: compute_short_and_hold_returns(prices, steps),
        )

    @staticmethod
    def twap(prices: pd.Series) -> BenchmarkSpec:
        return BenchmarkSpec(
            name="twap",
            compute_returns=lambda steps: compute_twap_returns(prices, steps),
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
            name="vwap",
            compute_returns=lambda steps: compute_vwap_returns(prices, volumes, steps),
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

        if getattr(config, "buy_and_hold", False):
            specs.append(BenchmarkEngine.buy_and_hold(prices))

        if getattr(config, "short_and_hold", False):
            specs.append(BenchmarkEngine.short_and_hold(prices))

        if getattr(config, "twap", False):
            specs.append(BenchmarkEngine.twap(prices))

        if getattr(config, "vwap", False):
            volumes, volume_source = resolve_vwap_volume_series(market_data)
            if volumes is None:
                logger.warning(
                    "VWAP benchmark skipped: no usable volume column found. "
                    "Expected one of: volume, trade_volume, last_size, size, qty, "
                    "or bid_sz_00/ask_sz_00 for proxy."
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
    strategy="buy_and_hold",
    eta=0.01,
    epsilon=1e-8,
    max_steps=None,
    price_column: str = "close",
    initial_portfolio_value: float = DEFAULT_INITIAL_PORTFOLIO_VALUE,
):
    """Calculate differential Sharpe ratio for a benchmark trading strategy."""
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

    if strategy == "buy_and_hold":
        positions = np.ones(len(price_returns))
    elif strategy == "max_profit":
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
