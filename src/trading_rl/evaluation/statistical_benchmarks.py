"""Benchmark return construction for statistical testing."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from trading_rl.evaluation.statistical_test_registry import _safe_div, _sharpe_ratio


def compute_buy_and_hold_returns(prices: pd.Series, max_steps: int) -> np.ndarray:
    """Compute buy-and-hold returns from price series."""
    if len(prices) < 2:
        raise ValueError("Price series must have at least 2 values")
    log_returns = np.log(prices / prices.shift(1)).fillna(0).to_numpy()[:max_steps]
    return np.exp(log_returns) - 1.0


def compute_short_and_hold_returns(prices: pd.Series, max_steps: int) -> np.ndarray:
    """Compute short-and-hold returns (inverse exposure to buy-and-hold)."""
    return -compute_buy_and_hold_returns(prices, max_steps)


def _normalize_execution_weights(weights: np.ndarray) -> np.ndarray:
    """Normalize execution weights to sum to 1 with robust fallbacks."""
    w = np.asarray(weights, dtype=float)
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w = np.clip(w, a_min=0.0, a_max=None)
    total = float(np.sum(w))
    if total <= 0.0:
        return np.full_like(w, 1.0 / max(len(w), 1), dtype=float)
    return w / total


def _execution_schedule_returns(
    prices: pd.Series,
    max_steps: int,
    weights: np.ndarray,
    direction: float = 1.0,
) -> np.ndarray:
    """Compute execution benchmark returns from progressive position buildup."""
    if len(prices) < 2:
        raise ValueError("Price series must have at least 2 values")

    simple_asset_returns = prices.pct_change().fillna(0.0).to_numpy()[:max_steps]
    n = len(simple_asset_returns)
    if n == 0:
        return np.array([], dtype=float)

    normalized_weights = _normalize_execution_weights(np.asarray(weights, dtype=float)[:n])
    cumulative_exposure = np.clip(np.cumsum(normalized_weights), 0.0, 1.0)
    lagged_exposure = np.concatenate(([0.0], cumulative_exposure[:-1]))
    return direction * lagged_exposure * simple_asset_returns


def compute_twap_returns(prices: pd.Series, max_steps: int) -> np.ndarray:
    """Compute TWAP benchmark returns via equal-time execution schedule."""
    if max_steps <= 0:
        return np.array([], dtype=float)
    return _execution_schedule_returns(prices, max_steps, np.ones(max_steps, dtype=float))


def compute_vwap_returns(
    prices: pd.Series,
    volumes: pd.Series,
    max_steps: int,
) -> np.ndarray:
    """Compute VWAP benchmark returns via volume-weighted execution schedule."""
    if len(volumes) < 1:
        raise ValueError("Volume series must have at least 1 value")
    if max_steps <= 0:
        return np.array([], dtype=float)
    weights = pd.Series(volumes).fillna(0.0).to_numpy()[:max_steps]
    return _execution_schedule_returns(prices, max_steps, weights)


def resolve_vwap_volume_series(
    market_data: pd.DataFrame | None,
) -> tuple[pd.Series | None, str | None]:
    """Resolve volume input for VWAP with explicit provenance."""
    if market_data is None or market_data.empty:
        return None, None

    direct_candidates = ["volume", "trade_volume", "last_size", "size", "qty"]
    for col in direct_candidates:
        if col in market_data.columns:
            return market_data[col], col

    if {"bid_sz_00", "ask_sz_00"}.issubset(market_data.columns):
        proxy_volume = (
            market_data["bid_sz_00"].fillna(0.0)
            + market_data["ask_sz_00"].fillna(0.0)
        )
        return proxy_volume, "bid_sz_00+ask_sz_00 (top-of-book size proxy)"

    return None, None


def _max_drawdown(simple_returns: np.ndarray) -> float:
    """Compute max drawdown from simple returns."""
    if simple_returns.size == 0:
        return np.nan
    equity = np.cumprod(1.0 + simple_returns)
    running_max = np.maximum.accumulate(equity)
    drawdown = equity / running_max - 1.0
    return float(np.min(drawdown))


def _performance_summary(
    simple_returns: np.ndarray,
    periods_per_year: int,
) -> dict[str, float]:
    """Compute compact benchmark performance summary metrics."""
    r = np.asarray(simple_returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return {
            "total_return": np.nan,
            "annualized_return_cagr": np.nan,
            "annualized_volatility": np.nan,
            "sharpe_ratio": np.nan,
            "sortino_ratio": np.nan,
            "max_drawdown": np.nan,
        }

    mu = float(np.mean(r))
    sigma = float(np.std(r, ddof=1)) if r.size > 1 else 0.0
    annualized_vol = sigma * np.sqrt(periods_per_year)
    downside = r[r < 0]
    downside_sigma = float(np.std(downside, ddof=1)) if downside.size > 1 else 0.0
    sharpe = _safe_div(mu * np.sqrt(periods_per_year), sigma)
    sortino = _safe_div(mu * np.sqrt(periods_per_year), downside_sigma)

    equity = np.cumprod(1.0 + r)
    total_return = float(equity[-1] - 1.0)
    years = max(r.size / periods_per_year, 1e-12)
    cagr = float(equity[-1] ** (1.0 / years) - 1.0)

    return {
        "total_return": total_return,
        "annualized_return_cagr": cagr,
        "annualized_volatility": float(annualized_vol),
        "sharpe_ratio": float(sharpe) if np.isfinite(sharpe) else np.nan,
        "sortino_ratio": float(sortino) if np.isfinite(sortino) else np.nan,
        "max_drawdown": _max_drawdown(r),
    }


def build_benchmark_comparison_table(
    strategy_returns: np.ndarray,
    benchmark_returns: dict[str, np.ndarray],
    periods_per_year: int = 252,
) -> list[dict[str, float | str]]:
    """Build a cross-benchmark comparison table with core performance metrics."""
    rows: list[dict[str, float | str]] = []
    rows.append(
        {"strategy": "agent", **_performance_summary(strategy_returns, periods_per_year)}
    )
    for name, returns in benchmark_returns.items():
        rows.append(
            {"strategy": name, **_performance_summary(returns, periods_per_year)}
        )
    return rows


def compute_random_baseline_returns(
    env: Any,
    max_steps: int,
    n_trials: int = 100,
    seed: int | None = None,
) -> list[np.ndarray]:
    """Generate random action baseline returns via Monte Carlo sampling."""
    import torch
    from tensordict.nn import InteractionType
    from torchrl.envs.utils import set_exploration_type

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    random_returns = []
    for trial in range(n_trials):
        env.set_seed(seed + trial if seed is not None else None)
        with torch.no_grad():
            with set_exploration_type(InteractionType.RANDOM):
                rollout = env.rollout(max_steps=max_steps)
        rewards = (
            rollout["next", "reward"].detach().cpu().reshape(-1).numpy()[:max_steps]
        )
        random_returns.append(np.exp(rewards) - 1.0)
    return random_returns


def summarize_random_baseline_trials(
    random_trials: list[np.ndarray],
) -> dict[str, float]:
    """Compute summary statistics across random baseline trials."""
    trial_sharpes = [_sharpe_ratio(trial) for trial in random_trials]
    return {
        "random_trials_sharpe_mean": float(np.mean(trial_sharpes)),
        "random_trials_sharpe_std": float(np.std(trial_sharpes)),
    }
