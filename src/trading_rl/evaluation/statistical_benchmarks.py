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
    if max_steps <= 0:
        return np.array([], dtype=float)
    price_window = prices.iloc[: max_steps + 1]
    return price_window.pct_change().iloc[1:].to_numpy(dtype=float)


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

    price_window = prices.iloc[: max_steps + 1]
    simple_asset_returns = price_window.pct_change().iloc[1:].to_numpy(dtype=float)
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
    equity = np.r_[1.0, np.cumprod(1.0 + simple_returns)]
    running_max = np.maximum.accumulate(equity)
    drawdown = equity / running_max - 1.0
    return float(np.min(drawdown))


def _performance_summary(
    simple_returns: np.ndarray,
    periods_per_year: int,
    risk_free_rate_annual: float = 0.0,
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

    rf_per_period = risk_free_rate_annual / periods_per_year
    excess_returns = r - rf_per_period
    mu = float(np.mean(r))
    mu_excess = float(np.mean(excess_returns))
    sigma = float(np.std(r, ddof=1)) if r.size > 1 else 0.0
    annualized_vol = sigma * np.sqrt(periods_per_year)
    downside = np.minimum(excess_returns, 0.0)
    downside_dev = float(np.sqrt(np.mean(np.square(downside))))
    sharpe = _safe_div(mu_excess * np.sqrt(periods_per_year), sigma)
    sortino = _safe_div(mu_excess * np.sqrt(periods_per_year), downside_dev)

    equity = np.cumprod(1.0 + r)
    total_return = float(equity[-1] - 1.0)
    years = max(r.size / periods_per_year, 1e-12)
    log_eq = np.log(max(float(equity[-1]), 1e-12))
    cagr = float(np.expm1(np.clip(log_eq / years, -50.0, 50.0)))

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
    risk_free_rate_annual: float = 0.0,
) -> list[dict[str, float | str]]:
    """Build a cross-benchmark comparison table with core performance metrics."""
    rows: list[dict[str, float | str]] = []
    rows.append(
        {"strategy": "agent", **_performance_summary(strategy_returns, periods_per_year, risk_free_rate_annual)}
    )
    for name, returns in benchmark_returns.items():
        rows.append(
            {"strategy": name, **_performance_summary(returns, periods_per_year, risk_free_rate_annual)}
        )
    return rows


def compute_random_baseline_returns(
    env: Any,
    max_steps: int,
    n_trials: int = 100,
    seed: int | None = None,
    reward_type: str = "log_return",
) -> list[np.ndarray]:
    """Generate random action baseline returns via Monte Carlo sampling."""
    import signal

    import torch
    from tensordict.nn import InteractionType
    from torchrl.envs.utils import set_exploration_type

    from trading_rl.constants import RewardType
    use_nlv = reward_type != RewardType.LOG_RETURN

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Set up signal handler for interrupting long-running random baseline computation
    original_sigint_handler = signal.getsignal(signal.SIGINT)

    def signal_handler(sig, frame):
        signal.signal(signal.SIGINT, original_sigint_handler)
        raise KeyboardInterrupt()

    signal.signal(signal.SIGINT, signal_handler)

    try:
        random_returns = []
        for trial in range(n_trials):
            env.set_seed(seed + trial if seed is not None else None)
            with torch.no_grad():
                with set_exploration_type(InteractionType.RANDOM):
                    rollout = env.rollout(max_steps=max_steps)

            if use_nlv:
                from trading_rl.evaluation.returns import extract_tradingenv_returns
                cumulative = extract_tradingenv_returns(env, max_steps)
                if cumulative is not None and len(cumulative) > 1:
                    step_log = np.diff(cumulative)
                    random_returns.append(np.exp(step_log) - 1.0)
                    continue

            rewards = (
                rollout["next", "reward"].detach().cpu().reshape(-1).numpy()[:max_steps]
            )
            random_returns.append(np.exp(rewards) - 1.0)
        return random_returns
    finally:
        signal.signal(signal.SIGINT, original_sigint_handler)


def summarize_random_baseline_trials(
    random_trials: list[np.ndarray],
) -> dict[str, float]:
    """Compute summary statistics across random baseline trials."""
    trial_sharpes = [_sharpe_ratio(trial) for trial in random_trials]
    return {
        "random_trials_sharpe_mean": float(np.mean(trial_sharpes)),
        "random_trials_sharpe_std": float(np.std(trial_sharpes)),
    }
