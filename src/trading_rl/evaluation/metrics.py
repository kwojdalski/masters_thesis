"""Quantitative finance evaluation metrics."""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import numpy as np

from trading_rl.constants import ReportingFrequency
from trading_rl.evaluation.returns import ReturnSeries

# Ordered coarsest-first: pick the largest bar size still yielding ≥ _MIN_SR_OBSERVATIONS.
_REPORTING_LADDER: list[ReportingFrequency] = [
    ReportingFrequency.DAILY,
    ReportingFrequency.HOURLY,
    ReportingFrequency.MIN_15,
    ReportingFrequency.MIN_1,
    ReportingFrequency.SEC_5,
]
_MIN_SR_OBSERVATIONS = 50


def _compound_return_chunks(
    simple_returns: np.ndarray,
    steps_per_bar: int,
) -> np.ndarray:
    """Compound returns into fixed-width chunks, keeping the final partial chunk."""
    if simple_returns.size == 0:
        return np.array([], dtype=float)
    steps_per_bar = max(1, steps_per_bar)
    return np.asarray(
        [
            np.prod(1.0 + simple_returns[start : start + steps_per_bar]) - 1.0
            for start in range(0, len(simple_returns), steps_per_bar)
        ],
        dtype=float,
    )


def aggregate_to_reporting_frequency(
    simple_returns: np.ndarray,
    periods_per_year: int,
) -> tuple[np.ndarray, int]:
    """Compound sub-daily returns to the coarsest bar size giving ≥ 50 observations.

    At tick/second frequency the return series is ~90% zeros (no trade between
    consecutive LOB events). This collapses σ faster than μ, inflating the
    annualised Sharpe ratio by √N. Compounding to a coarser bar eliminates the
    zero-inflation and restores the GBM scaling assumption.

    Returns the aggregated return array and the corresponding periods_per_year.
    If the raw series already has a daily-or-lower frequency, it is returned
    unchanged. If no bar size in the ladder yields ≥ 50 bars, the finest ladder
    entry (5-second) is used as a best-effort fallback.
    """
    if periods_per_year <= ReportingFrequency.DAILY.periods_per_year:
        return simple_returns, periods_per_year

    for freq in _REPORTING_LADDER:
        steps_per_bar = max(1, round(periods_per_year / freq.periods_per_year))
        n_bars = math.ceil(len(simple_returns) / steps_per_bar)
        if n_bars >= _MIN_SR_OBSERVATIONS:
            return _compound_return_chunks(
                simple_returns, steps_per_bar
            ), freq.periods_per_year

    # Fallback: finest ladder entry (5-second) even if < 50 bars.
    finest = _REPORTING_LADDER[-1]
    steps_per_bar = max(1, round(periods_per_year / finest.periods_per_year))
    n_bars = math.ceil(len(simple_returns) / steps_per_bar)
    if n_bars < 2:
        # The eval window covers less than one bar at the finest supported
        # resolution (5-second). We cannot aggregate, so we keep the raw tick
        # series and cap ppy at finest.periods_per_year. This understates
        # annualised vol but avoids both the NaN problem and the astronomical
        # inflation from using the raw tick ppy.
        return simple_returns, finest.periods_per_year
    return _compound_return_chunks(
        simple_returns, steps_per_bar
    ), finest.periods_per_year


_NAN = float("nan")


@dataclass
class MetricReport:
    """Typed container for the 43 standard quantitative finance metrics.

    All fields default to NaN so that callers can safely access any metric
    even when computation was skipped (e.g. empty returns).
    """

    # Metadata / scale
    n_periods: float = _NAN  # number of raw return observations
    n_bars: float = _NAN  # number of bars after frequency aggregation
    periods_per_year_used: float = _NAN  # effective ppy used for annualised ratios

    # Return / growth
    total_return: float = _NAN
    annualized_return_cagr: float = _NAN
    annualized_volatility: float = _NAN
    mean_return: float = _NAN  # per-bar mean return (μ)
    std_return: float = _NAN  # per-bar standard deviation (σ)

    # Risk-adjusted ratios
    sharpe_ratio: float = _NAN
    sharpe_expost: float = _NAN  # sharpe_raw × √n_bars; t-stat interpretation
    sortino_ratio: float = _NAN
    calmar_ratio: float = _NAN
    omega_ratio: float = _NAN

    # Drawdown
    max_drawdown: float = _NAN
    average_drawdown: float = _NAN
    max_drawdown_duration: float = _NAN
    recovery_time_from_max_drawdown: float = _NAN
    ulcer_index: float = _NAN  # sqrt(mean(drawdown²)); RMS drawdown pain

    # Tail risk
    var_95: float = _NAN
    cvar_95: float = _NAN
    var_99: float = _NAN
    cvar_99: float = _NAN
    downside_deviation: float = _NAN
    tail_ratio: float = _NAN  # 95th-pctile return / |5th-pctile return|

    # Distribution shape
    return_skewness: float = _NAN
    return_kurtosis: float = _NAN

    # Trade statistics
    win_rate: float = _NAN
    lose_rate: float = _NAN
    profit_factor: float = _NAN
    payoff_ratio: float = _NAN
    expectancy_per_period: float = _NAN
    gross_profit: float = _NAN  # sum of all positive returns
    gross_loss: float = _NAN  # absolute sum of all negative returns

    # Execution / position
    turnover: float = _NAN
    average_holding_period: float = _NAN
    n_trades: float = _NAN  # number of position changes
    pct_long: float = _NAN
    pct_short: float = _NAN
    pct_neutral: float = _NAN  # fraction of periods with zero position

    # Benchmark-relative (NaN when no benchmark provided)
    beta: float = _NAN
    alpha: float = _NAN
    information_ratio: float = _NAN
    tracking_error: float = _NAN

    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, float]:
        """Return a plain dict representation (keys identical to field names)."""
        return dataclasses.asdict(self)

    def to_filtered_dict(self, enabled: frozenset[str] | None) -> dict[str, float]:
        """Return a dict containing only the specified metric names.

        Args:
            enabled: Set of field names to include.  None means return all fields.
        """
        full = dataclasses.asdict(self)
        if enabled is None:
            return full
        return {k: v for k, v in full.items() if k in enabled}

    @classmethod
    def all_nan(cls) -> MetricReport:
        """Return a MetricReport with every field set to NaN."""
        return cls()

    def __contains__(self, key: object) -> bool:
        """Support `key in report` — checks field names, mirrors dict behaviour."""
        return (
            isinstance(key, str)
            and hasattr(self, key)
            and key in {f.name for f in dataclasses.fields(self)}
        )

    def __getitem__(self, key: str) -> float:
        """Support `report[key]` — mirrors dict behaviour."""
        if key not in self:
            raise KeyError(key)
        return getattr(self, key)

    def __bool__(self) -> bool:
        """True when at least one field is finite (i.e. not all NaN)."""
        return any(math.isfinite(v) for v in dataclasses.asdict(self).values())


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0 or np.isnan(denominator):
        return np.nan
    return numerator / denominator


# ---------------------------------------------------------------------------
# Three Sharpe / Sortino variants
# ---------------------------------------------------------------------------


def sharpe_raw(mu_excess: float, sigma: float) -> float:
    """Raw per-period Sharpe: μ_excess / σ.  No frequency scaling."""
    return _safe_div(mu_excess, sigma)


def sharpe_expost(mu_excess: float, sigma: float, n_bars: int) -> float:
    """Ex-post Sharpe scaled by √n: (μ_excess / σ) × √n_bars.

    Equals the t-statistic for H0: μ = 0 (up to a constant).  Valid for any
    window length; coincides with the annualised Sharpe when n_bars = ppy.
    """
    return _safe_div(mu_excess * np.sqrt(n_bars), sigma)


def sharpe_annualized(mu_excess: float, sigma: float, periods_per_year: int) -> float:
    """Annualised Sharpe: (μ_excess / σ) × √periods_per_year.

    Standard industry formula.  Unreliable when the observation window is much
    shorter than one year (standard error ≈ √(ppy / n_bars)).
    """
    return _safe_div(mu_excess * np.sqrt(periods_per_year), sigma)


def sortino_raw(mu_excess: float, downside_dev_per_bar: float) -> float:
    """Raw per-period Sortino: μ_excess / downside_deviation.  No scaling."""
    return _safe_div(mu_excess, downside_dev_per_bar)


def sortino_expost(mu_excess: float, downside_dev_per_bar: float, n_bars: int) -> float:
    """Ex-post Sortino scaled by √n: (μ_excess / downside_dev) × √n_bars."""
    return _safe_div(mu_excess * np.sqrt(n_bars), downside_dev_per_bar)


def sortino_annualized(
    mu_excess: float, downside_dev_per_bar: float, periods_per_year: int
) -> float:
    """Annualised Sortino: (μ_excess / downside_dev) × √periods_per_year."""
    return _safe_div(mu_excess * np.sqrt(periods_per_year), downside_dev_per_bar)


def _equity_curve(simple_returns: np.ndarray) -> np.ndarray:
    # Prepend 1.0 so the running-max starts at the true initial value (1.0).
    # Without this, drawdown[0] is always 0 and any decline in the first period
    # is invisible to the drawdown series.
    return np.r_[1.0, np.cumprod(1.0 + simple_returns)]


def _annualized_return_from_equity(equity_final: float, years: float) -> float:
    """Compute CAGR without overflowing on short horizons or extreme returns."""
    if not np.isfinite(equity_final) or equity_final <= 0:
        return np.nan

    exponent = np.log(equity_final) / max(years, 1e-12)
    max_exponent = np.log(np.finfo(float).max)
    return float(np.expm1(np.clip(exponent, a_min=None, a_max=max_exponent)))


def _should_compute_annualized_growth(periods_per_year: int) -> bool:
    """CAGR/Calmar are meaningful for daily-or-lower frequencies, not HFT."""
    return 0 < periods_per_year <= 252


def _drawdown_series(equity: np.ndarray) -> np.ndarray:
    running_max = np.maximum.accumulate(equity)
    return equity / running_max - 1.0


def _drawdown_stats(drawdowns: np.ndarray) -> tuple[float, float, int, int]:
    max_dd = float(np.min(drawdowns)) if drawdowns.size else np.nan
    avg_dd = float(np.mean(drawdowns[drawdowns < 0])) if np.any(drawdowns < 0) else 0.0

    if drawdowns.size and np.any(drawdowns < 0):
        is_dd = drawdowns < 0
        # Assign a unique id to each contiguous run by counting the number of
        # times the series is *not* in drawdown up to each point.  Elements
        # within the same drawdown run share the same group_id.
        group_ids = np.cumsum(~is_dd)
        run_lengths = np.bincount(group_ids[is_dd])
        max_duration = int(run_lengths.max())
    else:
        max_duration = 0

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
    strategy_simple_returns: np.ndarray | ReturnSeries,
    benchmark_simple_returns: np.ndarray | ReturnSeries | None,
    actions: np.ndarray | None,
    periods_per_year: int,
    risk_free_rate_annual: float = 0.0,
) -> MetricReport:
    """Compute 43 standard quantitative finance metrics."""
    benchmark_position_side = (
        getattr(strategy_simple_returns, "benchmark_position_side", None)
        if actions is None
        else None
    )
    if isinstance(strategy_simple_returns, ReturnSeries):
        strategy_simple_returns = strategy_simple_returns.to_simple().values
    _r_orig = np.asarray(
        strategy_simple_returns, dtype=float
    )  # kept with NaNs for position-aligned benchmark pairing
    _orig_ppy = periods_per_year
    r_all, periods_per_year = aggregate_to_reporting_frequency(
        _r_orig[np.isfinite(_r_orig)], periods_per_year
    )
    r = r_all
    # ppy > 0 after aggregation; annualised metrics (vol, Sharpe, Sortino) use
    # the effective ppy returned by aggregate_to_reporting_frequency.  When the
    # window is sub-minute, aggregate_to_reporting_frequency caps ppy at
    # finest.periods_per_year (98 280) so the result is finite but understated.
    _can_annualize = periods_per_year > 0

    if r.size == 0:
        return MetricReport.all_nan()

    rf_per_period = risk_free_rate_annual / max(periods_per_year, 1)
    mu = float(np.mean(r))
    sigma = float(np.std(r, ddof=1)) if r.size > 1 else 0.0

    if _can_annualize:
        annual_vol = sigma * np.sqrt(periods_per_year)
        downside = np.minimum(r - rf_per_period, 0.0)
        downside_dev = np.sqrt(np.mean(np.square(downside))) * np.sqrt(periods_per_year)
        # When annualised vol is negligibly small the equity curve is essentially flat
        # (e.g. an untrained agent with near-constant actions). mu/sigma is then
        # dominated by numerical noise, not signal, producing absurd ratios like 4000+.
        _MIN_ANNUAL_VOL = 1e-3
        if annual_vol >= _MIN_ANNUAL_VOL:
            _dd_raw = float(np.sqrt(np.mean(np.square(downside))))
            sharpe = sharpe_raw(mu - rf_per_period, sigma)
            sortino = sortino_raw(mu - rf_per_period, _dd_raw)
        else:
            sharpe = np.nan
            sortino = np.nan
    else:
        annual_vol = np.nan
        downside_dev = np.nan
        sharpe = np.nan
        sortino = np.nan

    equity = _equity_curve(r)
    total_return = float(equity[-1] - 1.0)
    if _should_compute_annualized_growth(periods_per_year):
        years = max(r.size / periods_per_year, 1e-12)
        cagr = _annualized_return_from_equity(float(equity[-1]), years)
    else:
        cagr = np.nan
    drawdowns = _drawdown_series(equity)
    max_dd, avg_dd, max_dd_duration, recovery_time = _drawdown_stats(drawdowns)
    calmar = _safe_div(cagr, abs(max_dd))
    ulcer_idx = float(np.sqrt(np.mean(drawdowns**2))) if drawdowns.size else np.nan

    var_95, cvar_95 = _tail_risk(r, alpha=0.05)
    var_99, cvar_99 = _tail_risk(r, alpha=0.01)
    p95 = float(np.quantile(r, 0.95)) if r.size else np.nan
    p05 = float(np.quantile(r, 0.05)) if r.size else np.nan
    tail_ratio = _safe_div(p95, abs(p05))
    skew = float(np.mean(((r - mu) / sigma) ** 3)) if sigma > 0 else np.nan
    kurt = float(np.mean(((r - mu) / sigma) ** 4) - 3) if sigma > 0 else np.nan

    wins = r[r > 0]
    losses = r[r < 0]
    hit_rate = float(np.mean(r > 0))
    lose_rate = float(np.mean(r < 0))
    gross_profit = float(np.sum(wins)) if wins.size else 0.0
    gross_loss = float(np.sum(np.abs(losses))) if losses.size else 0.0
    profit_factor = _safe_div(gross_profit, gross_loss)
    payoff_ratio = _safe_div(
        float(np.mean(wins)) if wins.size else 0.0,
        abs(float(np.mean(losses))) if losses.size else 0.0,
    )
    expectancy = float(mu)

    # Omega ratio at the risk-free threshold: E[max(r - rf, 0)] / E[max(rf - r, 0)]
    # Generalises profit_factor (equal to it when rf_per_period == 0).
    _gains = np.mean(np.maximum(r - rf_per_period, 0.0))
    _losses_omega = np.mean(np.maximum(rf_per_period - r, 0.0))
    omega_ratio = _safe_div(float(_gains), float(_losses_omega))

    if actions is not None and len(actions) > 0:
        actions_arr = np.asarray(actions, dtype=float)
        if actions_arr.size == _r_orig.size:
            actions_arr = actions_arr[np.isfinite(_r_orig)]
    elif benchmark_position_side is not None and r.size > 0:
        actions_arr = np.full(r.size, float(benchmark_position_side), dtype=float)
    else:
        actions_arr = np.array([])
    turnover = _turnover(actions_arr)
    avg_holding = _holding_period(actions_arr)
    pct_long = float(np.mean(actions_arr > 0)) if actions_arr.size > 0 else np.nan
    pct_short = float(np.mean(actions_arr < 0)) if actions_arr.size > 0 else np.nan
    pct_neutral = float(np.mean(actions_arr == 0.0)) if actions_arr.size > 0 else np.nan
    if actions_arr.size > 1:
        n_trades = float(np.sum(np.diff(actions_arr) != 0))
    else:
        n_trades = 0.0

    # Ex-post Sharpe: sharpe_raw × √n_bars — has t-statistic interpretation for H0: μ=0
    # Use the same _MIN_ANNUAL_VOL guard as sharpe_ratio for consistency
    sharpe_expost_val = (
        sharpe_raw(mu - rf_per_period, sigma) * np.sqrt(r.size)
        if _can_annualize and annual_vol >= _MIN_ANNUAL_VOL and sigma > 0 and r.size > 1
        else np.nan
    )

    beta = np.nan
    alpha = np.nan
    info_ratio = np.nan
    tracking_error = np.nan
    if benchmark_simple_returns is not None:
        if isinstance(benchmark_simple_returns, ReturnSeries):
            benchmark_simple_returns = benchmark_simple_returns.to_simple().values
        b_raw = np.asarray(benchmark_simple_returns, dtype=float)
        if _orig_ppy > 252:
            # Align strategy and benchmark before aggregation. Dropping NaNs
            # independently shifts timestamps and fabricates relative metrics.
            n = min(_r_orig.size, b_raw.size)
            if n > 1:
                paired_mask = np.isfinite(_r_orig[:n]) & np.isfinite(b_raw[:n])
                # rs/bs come from the same paired_mask array length, so both
                # calls land on the same ladder tier. Their returned ppy can
                # differ from the outer `periods_per_year` (computed from the
                # strategy-only isfinite mask, usually a longer array) — use
                # the paired ppy for these metrics so annualisation matches
                # the actual bar spacing of rs/bs.
                rs, ppy_paired = aggregate_to_reporting_frequency(
                    _r_orig[:n][paired_mask], _orig_ppy
                )
                bs, _ = aggregate_to_reporting_frequency(
                    b_raw[:n][paired_mask], _orig_ppy
                )
                n = min(rs.size, bs.size)
                rs, bs = rs[:n], bs[:n]
            else:
                rs, bs = np.array([]), np.array([])
                ppy_paired = periods_per_year
        else:
            # Daily-or-lower: use original arrays so the paired NaN mask aligns
            # strategy and benchmark by position before dropping missing values.
            n = min(_r_orig.size, b_raw.size)
            if n > 1:
                paired_mask = np.isfinite(_r_orig[:n]) & np.isfinite(b_raw[:n])
                rs, bs = _r_orig[:n][paired_mask], b_raw[:n][paired_mask]
            else:
                rs, bs = np.array([]), np.array([])
            ppy_paired = periods_per_year
        if n > 1 and rs.size > 1:
            cov = np.cov(rs, bs, ddof=1)
            var_b = cov[1, 1]
            beta = _safe_div(cov[0, 1], var_b)
            if ppy_paired > 0:
                alpha = (
                    np.mean(rs) - rf_per_period - beta * (np.mean(bs) - rf_per_period)
                ) * ppy_paired
                active = rs - bs
                active_std = float(np.std(active, ddof=1))
                tracking_error = active_std * np.sqrt(ppy_paired)
                info_ratio = _safe_div(
                    float(np.mean(active)) * np.sqrt(ppy_paired), active_std
                )

    return MetricReport(
        n_periods=float(_r_orig.size),
        n_bars=float(r.size),
        periods_per_year_used=float(periods_per_year),
        total_return=total_return,
        annualized_return_cagr=cagr,
        annualized_volatility=annual_vol,
        mean_return=mu,
        std_return=sigma,
        sharpe_ratio=sharpe,
        sharpe_expost=sharpe_expost_val,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        omega_ratio=omega_ratio,
        max_drawdown=max_dd,
        average_drawdown=avg_dd,
        max_drawdown_duration=float(max_dd_duration),
        recovery_time_from_max_drawdown=float(recovery_time)
        if np.isfinite(recovery_time)
        else np.nan,
        ulcer_index=ulcer_idx,
        var_95=var_95,
        cvar_95=cvar_95,
        var_99=var_99,
        cvar_99=cvar_99,
        downside_deviation=downside_dev,
        tail_ratio=tail_ratio,
        return_skewness=skew,
        return_kurtosis=kurt,
        win_rate=hit_rate,
        lose_rate=lose_rate,
        profit_factor=profit_factor,
        payoff_ratio=payoff_ratio,
        expectancy_per_period=expectancy,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        turnover=turnover,
        average_holding_period=avg_holding,
        n_trades=n_trades,
        pct_long=pct_long,
        pct_short=pct_short,
        pct_neutral=pct_neutral,
        beta=beta,
        alpha=float(alpha) if np.isfinite(alpha) else np.nan,
        information_ratio=info_ratio,
        tracking_error=tracking_error,
    )
