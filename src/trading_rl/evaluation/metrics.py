"""Quantitative finance evaluation metrics."""

from __future__ import annotations

import numpy as np


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0 or np.isnan(denominator):
        return np.nan
    return numerator / denominator


def _equity_curve(simple_returns: np.ndarray) -> np.ndarray:
    return np.cumprod(1.0 + simple_returns)


def _annualized_return_from_equity(equity_final: float, years: float) -> float:
    """Compute CAGR without overflowing on short horizons or extreme returns."""
    if not np.isfinite(equity_final) or equity_final <= 0:
        return np.nan

    exponent = np.log(equity_final) / max(years, 1e-12)
    max_exponent = np.log(np.finfo(float).max)
    return float(np.expm1(np.clip(exponent, a_min=None, a_max=max_exponent)))


def _drawdown_series(equity: np.ndarray) -> np.ndarray:
    running_max = np.maximum.accumulate(equity)
    return equity / running_max - 1.0


def _drawdown_stats(drawdowns: np.ndarray) -> tuple[float, float, int, int]:
    max_dd = float(np.min(drawdowns)) if drawdowns.size else np.nan
    avg_dd = float(np.mean(drawdowns[drawdowns < 0])) if np.any(drawdowns < 0) else 0.0

    max_duration = 0
    current = 0
    for d in drawdowns:
        if d < 0:
            current += 1
            max_duration = max(max_duration, current)
        else:
            current = 0

    trough_idx = int(np.argmin(drawdowns)) if drawdowns.size else -1
    recovery_time = np.nan
    if trough_idx >= 0:
        post = drawdowns[trough_idx:]
        recovered = np.where(post >= 0)[0]
        if recovered.size > 0:
            recovery_time = int(recovered[0])

    return max_dd, avg_dd, max_duration, recovery_time


def _tail_risk(simple_returns: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    if simple_returns.size == 0:
        return np.nan, np.nan
    var = float(np.quantile(simple_returns, alpha))
    tail = simple_returns[simple_returns <= var]
    cvar = float(np.mean(tail)) if tail.size > 0 else np.nan
    return var, cvar


def _turnover(actions: np.ndarray) -> float:
    if actions.size == 0:
        return np.nan
    if actions.ndim == 1:
        diffs = np.abs(np.diff(actions))
        return float(np.mean(diffs)) if diffs.size else 0.0
    diffs = np.abs(np.diff(actions, axis=0))
    per_step = np.sum(diffs, axis=1)
    return float(np.mean(per_step)) if per_step.size else 0.0


def _holding_period(actions: np.ndarray) -> float:
    if actions.size == 0:
        return np.nan
    if actions.ndim > 1:
        actions = np.argmax(actions, axis=1)
    actions = actions.reshape(-1)
    if actions.size <= 1:
        return float(actions.size)
    change_points = np.where(np.diff(actions) != 0)[0]
    boundaries = np.concatenate(([-1], change_points, [actions.size - 1]))
    lengths = np.diff(boundaries)
    return float(np.mean(lengths)) if lengths.size else float(actions.size)


def build_metric_report(
    strategy_simple_returns: np.ndarray,
    benchmark_simple_returns: np.ndarray | None,
    actions: np.ndarray | None,
    periods_per_year: int,
    risk_free_rate_annual: float = 0.0,
) -> dict[str, float]:
    """Compute 25 standard quantitative finance metrics."""
    r = np.asarray(strategy_simple_returns, dtype=float)
    r = r[np.isfinite(r)]

    if r.size == 0:
        return dict.fromkeys(_metric_keys(), np.nan)

    rf_per_period = risk_free_rate_annual / periods_per_year
    mu = float(np.mean(r))
    sigma = float(np.std(r, ddof=1)) if r.size > 1 else 0.0
    annual_vol = sigma * np.sqrt(periods_per_year)

    downside = np.minimum(r - rf_per_period, 0.0)
    downside_dev = np.sqrt(np.mean(np.square(downside))) * np.sqrt(periods_per_year)
    sharpe = _safe_div((mu - rf_per_period) * np.sqrt(periods_per_year), sigma)
    sortino = _safe_div((mu - rf_per_period) * periods_per_year, downside_dev)

    equity = _equity_curve(r)
    total_return = float(equity[-1] - 1.0)
    years = max(r.size / periods_per_year, 1e-12)
    cagr = _annualized_return_from_equity(float(equity[-1]), years)
    drawdowns = _drawdown_series(equity)
    max_dd, avg_dd, max_dd_duration, recovery_time = _drawdown_stats(drawdowns)
    calmar = _safe_div(cagr, abs(max_dd))

    var_95, cvar_95 = _tail_risk(r, alpha=0.05)
    skew = float(np.mean(((r - mu) / sigma) ** 3)) if sigma > 0 else np.nan
    kurt = float(np.mean(((r - mu) / sigma) ** 4)) if sigma > 0 else np.nan

    wins = r[r > 0]
    losses = r[r < 0]
    hit_rate = float(np.mean(r > 0))
    gross_profit = float(np.sum(wins)) if wins.size else 0.0
    gross_loss = float(np.sum(np.abs(losses))) if losses.size else 0.0
    profit_factor = _safe_div(gross_profit, gross_loss)
    payoff_ratio = _safe_div(float(np.mean(wins)) if wins.size else 0.0, abs(float(np.mean(losses))) if losses.size else 0.0)
    expectancy = float(mu)

    actions_arr = (
        np.asarray(actions, dtype=float)
        if actions is not None and len(actions) > 0
        else np.array([])
    )
    turnover = _turnover(actions_arr)
    avg_holding = _holding_period(actions_arr)

    beta = np.nan
    alpha = np.nan
    info_ratio = np.nan
    tracking_error = np.nan
    if benchmark_simple_returns is not None:
        b = np.asarray(benchmark_simple_returns, dtype=float)
        n = min(r.size, b.size)
        if n > 1:
            rs = r[:n]
            bs = b[:n]
            cov = np.cov(rs, bs, ddof=1)
            var_b = cov[1, 1]
            beta = _safe_div(cov[0, 1], var_b)
            alpha = (np.mean(rs) - rf_per_period - beta * (np.mean(bs) - rf_per_period)) * periods_per_year
            active = rs - bs
            active_std = float(np.std(active, ddof=1))
            tracking_error = active_std * np.sqrt(periods_per_year)
            info_ratio = _safe_div(float(np.mean(active)) * np.sqrt(periods_per_year), active_std)

    return {
        "total_return": total_return,
        "annualized_return_cagr": cagr,
        "annualized_volatility": annual_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "max_drawdown": max_dd,
        "average_drawdown": avg_dd,
        "max_drawdown_duration": float(max_dd_duration),
        "recovery_time_from_max_drawdown": float(recovery_time) if np.isfinite(recovery_time) else np.nan,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "downside_deviation": downside_dev,
        "return_skewness": skew,
        "return_kurtosis": kurt,
        "win_rate": hit_rate,
        "hit_rate": hit_rate,
        "profit_factor": profit_factor,
        "payoff_ratio": payoff_ratio,
        "expectancy_per_period": expectancy,
        "turnover": turnover,
        "average_holding_period": avg_holding,
        "beta": beta,
        "alpha": float(alpha) if np.isfinite(alpha) else np.nan,
        "information_ratio": info_ratio,
        "tracking_error": tracking_error,
    }


def _metric_keys() -> list[str]:
    return [
        "total_return",
        "annualized_return_cagr",
        "annualized_volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
        "max_drawdown",
        "average_drawdown",
        "max_drawdown_duration",
        "recovery_time_from_max_drawdown",
        "var_95",
        "cvar_95",
        "downside_deviation",
        "return_skewness",
        "return_kurtosis",
        "win_rate",
        "hit_rate",
        "profit_factor",
        "payoff_ratio",
        "expectancy_per_period",
        "turnover",
        "average_holding_period",
        "beta",
        "alpha",
        "information_ratio",
        "tracking_error",
    ]
