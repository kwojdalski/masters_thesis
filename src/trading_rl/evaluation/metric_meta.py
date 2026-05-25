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
    is_pct: bool = False


# Full ordered registry — all 29 MetricReport fields.
# fmt strings follow Python format-spec syntax (used as f"{val:{fmt}}").
# is_pct=True means the raw float is a fraction and should be displayed as %.
METRIC_META: list[MetricMeta] = [
    # Return
    MetricMeta("total_return",              "Total Return",          ".2%", "Return",       is_pct=True),
    MetricMeta("annualized_return_cagr",    "Ann. Return (CAGR)",    ".2%", "Return",       is_pct=True),
    MetricMeta("annualized_volatility",     "Ann. Volatility",       ".2%", "Return",       is_pct=True),
    # Risk-adjusted
    MetricMeta("sharpe_ratio",              "Sharpe Ratio",          ".3f", "Risk-adjusted"),
    MetricMeta("sortino_ratio",             "Sortino Ratio",         ".3f", "Risk-adjusted"),
    MetricMeta("calmar_ratio",              "Calmar Ratio",          ".3f", "Risk-adjusted"),
    MetricMeta("omega_ratio",               "Omega Ratio",           ".3f", "Risk-adjusted"),
    # Drawdown
    MetricMeta("max_drawdown",              "Max Drawdown",          ".2%", "Drawdown",     is_pct=True),
    MetricMeta("average_drawdown",          "Avg Drawdown",          ".2%", "Drawdown",     is_pct=True),
    MetricMeta("max_drawdown_duration",     "Max DD Duration",       ".1f", "Drawdown"),
    MetricMeta("recovery_time_from_max_drawdown", "Recovery Time",   ".1f", "Drawdown"),
    # Risk
    MetricMeta("var_95",                    "VaR 95%",               ".2%", "Risk",         is_pct=True),
    MetricMeta("cvar_95",                   "CVaR 95%",              ".2%", "Risk",         is_pct=True),
    MetricMeta("downside_deviation",        "Downside Dev",          ".2%", "Risk",         is_pct=True),
    MetricMeta("return_skewness",           "Skewness",              ".3f", "Risk"),
    MetricMeta("return_kurtosis",           "Kurtosis",              ".3f", "Risk"),
    # Trading
    MetricMeta("win_rate",                  "Win Rate",              ".2%", "Trading",      is_pct=True),
    MetricMeta("lose_rate",                 "Lose Rate",             ".2%", "Trading",      is_pct=True),
    MetricMeta("profit_factor",             "Profit Factor",         ".3f", "Trading"),
    MetricMeta("payoff_ratio",              "Payoff Ratio",          ".3f", "Trading"),
    MetricMeta("expectancy_per_period",     "Expectancy / Period",   ".6f", "Trading"),
    MetricMeta("turnover",                  "Turnover",              ".3f", "Trading"),
    MetricMeta("average_holding_period",    "Avg Hold Period",       ".1f", "Trading"),
    MetricMeta("pct_long",                  "% Long",                ".2%", "Trading",      is_pct=True),
    MetricMeta("pct_short",                 "% Short",               ".2%", "Trading",      is_pct=True),
    # Benchmark
    MetricMeta("beta",                      "Beta",                  ".3f", "Benchmark"),
    MetricMeta("alpha",                     "Alpha",                 ".4f", "Benchmark"),
    MetricMeta("information_ratio",         "Info Ratio",            ".3f", "Benchmark"),
    MetricMeta("tracking_error",            "Tracking Error",        ".4f", "Benchmark",    is_pct=True),
]

# Lookup by key for O(1) access.
METRIC_META_BY_KEY: dict[str, MetricMeta] = {m.key: m for m in METRIC_META}

# Sections in display order, each with its ordered list of MetricMeta.
METRIC_SECTIONS: dict[str, list[MetricMeta]] = {}
for _m in METRIC_META:
    METRIC_SECTIONS.setdefault(_m.section, []).append(_m)

# Plain-text descriptions — single source of truth for console legend and PNG footer.
METRIC_LEGEND: dict[str, str] = {
    "Env Steps":       "Total environment steps collected during training.",
    "Episode Length":  "Steps per episode (streaming window or full dataset). Episodes = Env Steps / Episode Length.",
    "Episodes":        "Number of full episodes completed (resets) during training.",
    "Optimizer Steps": "Number of gradient update steps taken.",
    "Eval Horizon":    "Number of steps used for each evaluation rollout.",
    "Final Reward":    "Raw reward signal at the last training step.",
    "Total Return":    "Cumulative portfolio return over the evaluation horizon.",
    "Sharpe Ratio":    "Mean excess return divided by its standard deviation, annualised.",
    "Sortino Ratio":   "Like Sharpe but penalises only downside deviation.",
    "Max Drawdown":    "Largest peak-to-trough decline in portfolio value.",
    "Win Rate":        "Fraction of steps where the portfolio return was positive.",
    "Lose Rate":       "Fraction of steps where the portfolio return was negative.",
    "Profit Factor":   "Gross profit divided by gross loss (> 1 means net profitable).",
}


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
