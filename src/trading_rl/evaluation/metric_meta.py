"""Single source of truth for metric display metadata.

Every place that shows metric names, formats, or groups them into sections
should import from here rather than defining its own dicts or tuples.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricMeta:
    key: str
    label: str
    fmt: str
    section: str
    description: str = ""
    is_pct: bool = False


# Full ordered registry — all MetricReport fields.
# fmt strings follow Python format-spec syntax (used as f"{val:{fmt}}").
# is_pct=True means the raw float is a fraction and should be displayed as %.
METRIC_META: list[MetricMeta] = [
    # ── Observation counts ────────────────────────────────────────────────────
    MetricMeta(
        "n_periods", "Periods", ".0f", "Metadata",
        "Total number of raw return observations in the evaluation window.",
    ),
    MetricMeta(
        "n_bars", "Bars", ".0f", "Metadata",
        "Observations after frequency aggregation (e.g. 1-min ticks compounded to 5-min bars) "
        "to remove zero-inflation and restore GBM scaling.",
    ),
    MetricMeta(
        "periods_per_year_used", "Periods / Year", ".0f", "Metadata",
        "Effective annualisation factor used for all ratio calculations (auto-selected from "
        "a frequency ladder: annual / quarterly / monthly / weekly / daily / hourly / 5-min).",
    ),

    # ── Return ────────────────────────────────────────────────────────────────
    MetricMeta(
        "total_return", "Total Return", ".2%", "Return",
        "Cumulative portfolio return over the full evaluation horizon: (V_T - V_0) / V_0.",
        is_pct=True,
    ),
    MetricMeta(
        "annualized_return_cagr", "Ann. Return (CAGR)", ".2%", "Return",
        "Compound Annual Growth Rate: (1 + total_return)^(periods_per_year / n_periods) - 1. "
        "Misleading on sub-day HFT data — use total_return instead.",
        is_pct=True,
    ),
    MetricMeta(
        "annualized_volatility", "Ann. Volatility", ".2%", "Return",
        "Standard deviation of bar returns scaled to annual frequency: σ_bar × √periods_per_year.",
        is_pct=True,
    ),
    MetricMeta(
        "mean_return", "Mean Return", ".4%", "Return",
        "Average per-bar return (μ). Positive values indicate the strategy earns on average.",
        is_pct=True,
    ),
    MetricMeta(
        "std_return", "Std Return", ".4%", "Return",
        "Per-bar standard deviation of returns (σ). Measures raw dispersion before annualisation.",
        is_pct=True,
    ),

    # ── Risk-adjusted ─────────────────────────────────────────────────────────
    MetricMeta(
        "sharpe_ratio", "Sharpe Ratio", ".3f", "Risk-adjusted",
        "Mean excess bar return divided by its standard deviation: μ / σ. "
        "Not annualised — comparable across evaluation windows of different length.",
    ),
    MetricMeta(
        "sharpe_expost", "Sharpe (ex-post)", ".3f", "Risk-adjusted",
        "Sharpe ratio scaled by √n_bars: (μ / σ) × √n_bars. "
        "Equals the t-statistic for H₀: μ = 0; values above ~2 indicate statistical significance.",
    ),
    MetricMeta(
        "sortino_ratio", "Sortino Ratio", ".3f", "Risk-adjusted",
        "Like Sharpe but penalises only downside deviation (returns below zero): μ / σ_down. "
        "Higher is better; rewards strategies that avoid losses without punishing upside volatility.",
    ),
    MetricMeta(
        "calmar_ratio", "Calmar Ratio", ".3f", "Risk-adjusted",
        "Annualised return divided by maximum drawdown: CAGR / |max_drawdown|. "
        "Emphasises recovery from the worst loss; misleading on short HFT windows.",
    ),
    MetricMeta(
        "omega_ratio", "Omega Ratio", ".3f", "Risk-adjusted",
        "Ratio of the probability-weighted area above the threshold (gains) to the area below "
        "(losses): Σ max(r,0) / Σ max(-r,0). Values > 1 indicate net positive expectation.",
    ),

    # ── Drawdown ──────────────────────────────────────────────────────────────
    MetricMeta(
        "max_drawdown", "Max Drawdown", ".2%", "Drawdown",
        "Largest peak-to-trough decline in portfolio value over the evaluation period. "
        "Measures the worst case an investor would have experienced.",
        is_pct=True,
    ),
    MetricMeta(
        "average_drawdown", "Avg Drawdown", ".2%", "Drawdown",
        "Mean of all individual drawdown magnitudes (not just the maximum). "
        "Reflects the typical pain level rather than the worst case.",
        is_pct=True,
    ),
    MetricMeta(
        "max_drawdown_duration", "Max DD Duration", ".1f", "Drawdown",
        "Number of bars (after aggregation) from the peak before the maximum drawdown "
        "to its subsequent trough.",
    ),
    MetricMeta(
        "recovery_time_from_max_drawdown", "Recovery Time", ".1f", "Drawdown",
        "Bars required to recover from the maximum drawdown back to the prior equity peak. "
        "NaN if the portfolio has not yet recovered by the end of the window.",
    ),
    MetricMeta(
        "ulcer_index", "Ulcer Index", ".4f", "Drawdown",
        "RMS of all drawdown values: √(mean(drawdown²)). Captures both depth and duration of "
        "underwater periods; lower values indicate smoother equity curves.",
    ),

    # ── Risk ──────────────────────────────────────────────────────────────────
    MetricMeta(
        "var_95", "VaR 95%", ".2%", "Risk",
        "Value at Risk at 95% confidence: the 5th-percentile bar return. "
        "On a bad day (1-in-20), losses are expected to exceed this threshold.",
        is_pct=True,
    ),
    MetricMeta(
        "cvar_95", "CVaR 95%", ".2%", "Risk",
        "Conditional VaR (Expected Shortfall) at 95%: mean return in the worst 5% of bars. "
        "More informative than VaR as it measures the average tail loss, not just the cutoff.",
        is_pct=True,
    ),
    MetricMeta(
        "var_99", "VaR 99%", ".2%", "Risk",
        "Value at Risk at 99% confidence: the 1st-percentile bar return. "
        "Captures more extreme tail risk than VaR 95%.",
        is_pct=True,
    ),
    MetricMeta(
        "cvar_99", "CVaR 99%", ".2%", "Risk",
        "Conditional VaR (Expected Shortfall) at 99%: mean return in the worst 1% of bars. "
        "The most conservative tail-risk measure reported.",
        is_pct=True,
    ),
    MetricMeta(
        "downside_deviation", "Downside Dev", ".2%", "Risk",
        "Standard deviation computed only on negative returns: √(mean(min(r,0)²)). "
        "Measures the volatility of losses; used in the Sortino ratio denominator.",
        is_pct=True,
    ),
    MetricMeta(
        "tail_ratio", "Tail Ratio", ".3f", "Risk",
        "95th-percentile return divided by the absolute 5th-percentile return. "
        "Values > 1 indicate the right tail is fatter than the left — good upside vs downside.",
    ),
    MetricMeta(
        "return_skewness", "Skewness", ".3f", "Risk",
        "Third standardised moment of the bar return distribution. "
        "Positive = right tail (large gains possible); negative = left tail (large losses).",
    ),
    MetricMeta(
        "return_kurtosis", "Kurtosis", ".3f", "Risk",
        "Excess kurtosis (fourth moment minus 3). "
        "Values > 0 indicate fat tails relative to a normal distribution — more extreme events.",
    ),

    # ── Trading ───────────────────────────────────────────────────────────────
    MetricMeta(
        "win_rate", "Win Rate", ".2%", "Trading",
        "Fraction of bars where the portfolio return was strictly positive.",
        is_pct=True,
    ),
    MetricMeta(
        "lose_rate", "Lose Rate", ".2%", "Trading",
        "Fraction of bars where the portfolio return was strictly negative.",
        is_pct=True,
    ),
    MetricMeta(
        "profit_factor", "Profit Factor", ".3f", "Trading",
        "Gross profit divided by gross loss: Σ max(r,0) / Σ max(-r,0). "
        "Values > 1 mean total gains exceed total losses; < 1 means the strategy loses money.",
    ),
    MetricMeta(
        "payoff_ratio", "Payoff Ratio", ".3f", "Trading",
        "Average winning bar return divided by the average losing bar return (in absolute terms). "
        "Measures reward-to-risk per individual bar regardless of win/lose frequency.",
    ),
    MetricMeta(
        "expectancy_per_period", "Expectancy / Bar", ".6f", "Trading",
        "Expected return per bar: win_rate × avg_win - lose_rate × avg_loss. "
        "Positive values indicate the strategy has positive edge per step on average.",
    ),
    MetricMeta(
        "gross_profit", "Gross Profit", ".4%", "Trading",
        "Sum of all positive bar returns. Represents total upside captured over the window.",
        is_pct=True,
    ),
    MetricMeta(
        "gross_loss", "Gross Loss", ".4%", "Trading",
        "Absolute sum of all negative bar returns. Represents total downside absorbed.",
        is_pct=True,
    ),
    MetricMeta(
        "turnover", "Turnover", ".3f", "Trading",
        "Mean absolute position change per bar. High turnover incurs more trading costs and "
        "indicates an active strategy; low turnover indicates buy-and-hold-like behaviour.",
    ),
    MetricMeta(
        "average_holding_period", "Avg Hold Period", ".1f", "Trading",
        "Average number of consecutive bars the agent holds the same position before switching. "
        "Longer holding periods imply lower turnover and fewer transaction costs.",
    ),
    MetricMeta(
        "n_trades", "# Trades", ".0f", "Trading",
        "Total number of position changes (entries and exits) over the evaluation window.",
    ),
    MetricMeta(
        "pct_long", "% Long", ".2%", "Trading",
        "Fraction of bars where the agent held a long position.",
        is_pct=True,
    ),
    MetricMeta(
        "pct_short", "% Short", ".2%", "Trading",
        "Fraction of bars where the agent held a short position.",
        is_pct=True,
    ),
    MetricMeta(
        "pct_neutral", "% Neutral", ".2%", "Trading",
        "Fraction of bars where the agent held a zero (flat) position.",
        is_pct=True,
    ),

    # ── Benchmark ─────────────────────────────────────────────────────────────
    MetricMeta(
        "beta", "Beta", ".3f", "Benchmark",
        "Covariance of strategy returns with the buy-and-hold benchmark divided by the "
        "benchmark variance. Beta = 1 means the strategy moves in lockstep with the asset.",
    ),
    MetricMeta(
        "alpha", "Alpha", ".4f", "Benchmark",
        "Intercept from an OLS regression of strategy returns on benchmark returns. "
        "Positive alpha indicates returns in excess of what beta alone predicts.",
    ),
    MetricMeta(
        "information_ratio", "Info Ratio", ".3f", "Benchmark",
        "Mean active return (strategy minus benchmark) divided by tracking error. "
        "Measures consistency of outperformance relative to the benchmark.",
    ),
    MetricMeta(
        "tracking_error", "Tracking Error", ".4f", "Benchmark",
        "Standard deviation of the difference between strategy and benchmark bar returns. "
        "Low values indicate the strategy closely follows the benchmark.",
        is_pct=True,
    ),
]

# Lookup by key for O(1) access.
METRIC_META_BY_KEY: dict[str, MetricMeta] = {m.key: m for m in METRIC_META}

# Sections in display order, each with its ordered list of MetricMeta.
METRIC_SECTIONS: dict[str, list[MetricMeta]] = {}
for _m in METRIC_META:
    METRIC_SECTIONS.setdefault(_m.section, []).append(_m)

# Legend auto-generated from MetricMeta.description — every metric that has a
# non-empty description appears here.  Hand-written entries for non-metric rows
# (Env Steps, Episodes, etc.) are merged in afterwards so they remain available
# to callers that use METRIC_LEGEND directly.
METRIC_LEGEND: dict[str, str] = {
    m.label: m.description for m in METRIC_META if m.description
}
# Training-context rows not in MetricReport
METRIC_LEGEND.update({
    "Env Steps":       "Total environment steps collected during training.",
    "Episode Length":  "Steps per episode (streaming window or full dataset).",
    "Episodes":        "Number of full episodes completed (resets) during training.",
    "Optimizer Steps": "Number of gradient update steps taken by the optimiser.",
    "Eval Horizon":    "Number of steps used for each evaluation rollout.",
    "Final Reward":    "Raw reward signal at the last training step.",
})


def fmt_metric(key: str, val: float) -> str:
    """Format a metric value using the registered format spec.

    Non-finite values are always returned as 'N/A' regardless of format.
    is_pct values are multiplied by 100 before formatting so .2% renders
    correctly (Python's % format already multiplies by 100, so we only need
    the raw float and the .2% spec).
    """
    import math
    if not math.isfinite(val):
        return "N/A"
    meta = METRIC_META_BY_KEY.get(key)
    if meta is None:
        return f"{val:.4f}"
    return f"{val:{meta.fmt}}"
