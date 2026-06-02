"""Evaluation plotting helpers for rollout and benchmark comparisons."""

from __future__ import annotations

import time
import numpy as np
import pandas as pd
from plotnine import (
    aes,
    element_text,
    geom_line,
    ggplot,
    guide_legend,
    guides,
    labs,
    scale_color_gradient,
    scale_color_manual,
    scale_linetype_manual,
    scale_x_datetime,
    theme,
)

from trading_rl.evaluation.asset_meta import write_asset_meta
from trading_rl.evaluation.thesis_theme import FIGURE_WIDTH, LINETYPE, PALETTE, thesis_theme
from torch import allclose

from logger import get_logger
from trading_rl.config import DEFAULT_INITIAL_PORTFOLIO_VALUE
from trading_rl.evaluation.returns import (
    ReturnKind,
    ReturnSeries,
    extract_tradingenv_return_series,
)

logger = get_logger(__name__)

_RUN_DESCRIPTIONS: dict[str, str] = {
    "Deterministic": "greedy policy with no exploration noise",
    "Random": "uniformly sampled actions used as a baseline",
    "Buy-and-Hold": "buy at step 0 and hold for the full evaluation horizon",
    "Short-and-Hold": "short at step 0 and hold for the full evaluation horizon",
    "Max Profit (Unleveraged)": "perfect-foresight upper bound — always trades in the correct direction",
    "TWAP": "equal-time execution — builds a long position uniformly over the evaluation horizon",
    "VWAP": "volume-weighted execution — builds a long position proportionally to market volume",
}

# Canonical display order for legend entries; Deterministic always first.
_RUN_ORDER: list[str] = [
    "Deterministic",
    "Random",
    "Buy-and-Hold",
    "Short-and-Hold",
    "TWAP",
    "VWAP",
    "Max Profit (Unleveraged)",
]


def _as_ordered_run_categorical(series: "pd.Series") -> "pd.Categorical":
    """Convert a Run string column to an ordered Categorical with Deterministic first.

    Any run name not in _RUN_ORDER is appended after the known entries in the
    order they were first encountered, so unknown names don't get dropped.
    """
    present = list(series.unique())
    known = [r for r in _RUN_ORDER if r in present]
    unknown = [r for r in present if r not in known]
    return pd.Categorical(series, categories=known + unknown, ordered=True)


def _build_run_caption(
    prefix: str,
    runs: list[str],
    training_steps: int | None = None,
    training_episodes: int | None = None,
    date_range: str = "",
) -> str:
    """Build a caption line per run present in the plot."""
    lines = [prefix]
    if date_range:
        lines.append(f"Date range: {date_range}.")
    if training_steps is not None and training_episodes is not None:
        lines.append(f"Policy trained for {training_steps:,} steps ({training_episodes:,} episodes).")
    elif training_steps is not None:
        lines.append(f"Policy trained for {training_steps:,} steps.")
    for run in runs:
        desc = _RUN_DESCRIPTIONS.get(run)
        if desc:
            lines.append(f"{run}: {desc}.")
    return "\n".join(lines)


def _portfolio_values_from_actual_returns(
    actual_returns,
    initial_portfolio_value: float,
    n_obs: int,
) -> np.ndarray:
    if isinstance(actual_returns, ReturnSeries):
        equity = actual_returns.to_equity(initial_portfolio_value).values
        if actual_returns.kind == ReturnKind.EQUITY and not actual_returns.includes_initial:
            return equity[:n_obs]
        return equity[1 : n_obs + 1]

    # Legacy callers pass cumulative log returns.
    cumulative_log_returns = np.asarray(actual_returns[:n_obs], dtype=float)
    return initial_portfolio_value * np.exp(cumulative_log_returns)


def _date_range_str(df: pd.DataFrame | None, n_obs: int) -> str:
    """Return 'May 1, 2025 13:45 – May 22, 2025 09:30' from df.index, or '' if not datetime."""
    if df is None or not pd.api.types.is_datetime64_any_dtype(df.index):
        return ""
    idx = df.index[: n_obs + 1]
    if len(idx) < 2:
        return ""
    fmt = "%Y-%m-%d %H:%M:%S"
    start_s = idx[0].strftime(fmt)
    end_s = idx[-1].strftime(fmt)
    return start_s if start_s == end_s else f"{start_s} – {end_s}"


_REWARD_TYPE_LABELS: dict[str, str] = {
    "log_return": "Log Return",
    "differential_sharpe": "Differential Sharpe Ratio",
}


# ---------------------------------------------------------------------------
# Rollout plot data builders and standalone plotters
# ---------------------------------------------------------------------------


def build_rollout_plot_data(
    rollouts,
    n_obs: int,
    is_portfolio: bool = False,
    training_steps: int | None = None,
    training_episodes: int | None = None,
    df: pd.DataFrame | None = None,
    reward_type: str | None = None,
    max_plot_points: int | None = None,
    show_allocation_ma: bool = True,
    allocation_ma_window: int = 500,
    show_benchmarks: bool = False,
    benchmark_price_column: str = "close",
) -> dict:
    """Build DataFrames for rollout comparison plots.

    Separates data preparation from rendering so QMD can call plot_rewards /
    plot_actions with custom figure dimensions without re-running the rollout.

    Returns a dict with keys:
        rewards: DataFrame(Steps, Cumulative_Reward, Run)
        actions: DataFrame(Steps, Actions, Run)
        actions_ma: DataFrame(Steps, MA, Run) or None
        stride, date_str, reward_type, is_portfolio,
        training_steps, training_episodes, n_obs, allocation_ma_window
    """
    all_actions = []
    for rollout in rollouts:
        action = rollout["action"].squeeze()
        if action.ndim > 1 and action.shape[-1] > 1:
            processed_action = action.argmax(dim=-1)
        else:
            processed_action = action
        all_actions.append(processed_action[:n_obs])

    all_rewards = [rollout["next"]["reward"][:n_obs] for rollout in rollouts]

    for i in range(len(rollouts)):
        for j in range(i + 1, len(rollouts)):
            min_a = min(all_actions[i].shape[0], all_actions[j].shape[0])
            min_r = min(all_rewards[i].shape[0], all_rewards[j].shape[0])
            actions_equal = bool(allclose(all_actions[i][:min_a].float(), all_actions[j][:min_a].float()))
            rewards_equal = bool(allclose(all_rewards[i][:min_r].float(), all_rewards[j][:min_r].float()))
            logger.info(
                "Run {} vs Run {} | actions_identical={} rewards_identical={}",
                i + 1,
                j + 1,
                actions_equal,
                rewards_equal,
            )

    stride = max(1, n_obs // max_plot_points) if max_plot_points and max_plot_points < n_obs else 1

    rewards_data = []
    for i, rewards in enumerate(all_rewards):
        rewards_np = rewards.detach().cpu().numpy()
        cumsum = np.cumsum(rewards_np)
        idx = np.arange(len(cumsum))[::stride]
        rewards_data.extend(
            [
                {
                    "Steps": int(step),
                    "Cumulative_Reward": float(val),
                    "Run": "Deterministic" if i == 0 else "Random",
                }
                for step, val in zip(idx, cumsum[::stride])
            ]
        )
    df_rewards = pd.DataFrame(rewards_data)
    df_rewards["Run"] = _as_ordered_run_categorical(df_rewards["Run"])

    # Add benchmark reward curves if requested and reward_type is log_return
    if show_benchmarks and df is not None and reward_type == "log_return":
        price_col = benchmark_price_column
        if price_col not in df.columns and "close" in df.columns:
            price_col = "close"
        if price_col in df.columns:
            price_series = df[price_col].iloc[: n_obs + 1]
            log_returns = np.log(price_series / price_series.shift(1)).iloc[1:].to_numpy(dtype=float)

            n_bad = int(np.sum(~np.isfinite(log_returns)))
            if n_bad == 0:
                idx = np.arange(len(log_returns))[::stride]
                for bench_name, sign in [("Buy-and-Hold", 1.0), ("Short-and-Hold", -1.0)]:
                    cumsum = (sign * log_returns).cumsum()
                    rewards_data.extend(
                        {
                            "Steps": int(s),
                            "Cumulative_Reward": float(cumsum[s]),
                            "Run": bench_name,
                        }
                        for i, s in enumerate(idx) if s < len(cumsum)
                    )
                twap_positions = np.arange(0, len(log_returns), dtype=float) / len(log_returns)
                twap_cumsum = (twap_positions * log_returns).cumsum()
                rewards_data.extend(
                    {
                        "Steps": int(s),
                        "Cumulative_Reward": float(twap_cumsum[s]),
                        "Run": "TWAP",
                    }
                    for i, s in enumerate(idx) if s < len(twap_cumsum)
                )
                df_rewards = pd.DataFrame(rewards_data)
                df_rewards["Run"] = _as_ordered_run_categorical(df_rewards["Run"])
            else:
                logger.warning(
                    "Benchmark reward curves skipped: {} non-finite returns in price series",
                    n_bad,
                )

    # Add DSR benchmark curves if requested and reward_type is differential_sharpe
    if show_benchmarks and df is not None and reward_type == "differential_sharpe":
        from trading_rl.evaluation.benchmarks import calculate_benchmark_dsr, calculate_twap_dsr
        from trading_rl.constants import BenchmarkName

        price_col = benchmark_price_column
        if price_col not in df.columns and "close" in df.columns:
            price_col = "close"
        if price_col in df.columns:
            eta = 0.01
            for bench_name, strategy in [
                ("Buy-and-Hold", BenchmarkName.BUY_AND_HOLD),
                ("Short-and-Hold", BenchmarkName.SHORT_AND_HOLD),
            ]:
                dsr_cumsum, _ = calculate_benchmark_dsr(
                    df, strategy=strategy, eta=eta, max_steps=n_obs, price_column=price_col,
                )
                idx = np.arange(len(dsr_cumsum))[::stride]
                rewards_data.extend(
                    {
                        "Steps": int(s),
                        "Cumulative_Reward": float(dsr_cumsum[s]),
                        "Run": bench_name,
                    }
                    for i, s in enumerate(idx) if s < len(dsr_cumsum)
                )
            twap_dsr_cumsum, _ = calculate_twap_dsr(
                df, eta=eta, max_steps=n_obs, price_column=price_col,
            )
            idx = np.arange(len(twap_dsr_cumsum))[::stride]
            rewards_data.extend(
                {
                    "Steps": int(s),
                    "Cumulative_Reward": float(twap_dsr_cumsum[s]),
                    "Run": "TWAP",
                }
                for i, s in enumerate(idx) if s < len(twap_dsr_cumsum)
            )
            df_rewards = pd.DataFrame(rewards_data)
            df_rewards["Run"] = _as_ordered_run_categorical(df_rewards["Run"])

    actions_data = []
    for i, actions in enumerate(all_actions):
        actions_np = actions.detach().cpu().numpy()
        idx = np.arange(len(actions_np))[::stride]
        actions_data.extend(
            [
                {
                    "Steps": int(step),
                    "Actions": float(val),
                    "Run": "Deterministic" if i == 0 else "Random",
                }
                for step, val in zip(idx, actions_np[::stride])
            ]
        )
    df_actions = pd.DataFrame(actions_data)
    df_actions["Run"] = _as_ordered_run_categorical(df_actions["Run"])

    date_str = _date_range_str(df, n_obs)

    df_ma = None
    if is_portfolio and show_allocation_ma:
        ma_rows = []
        for run_name, grp in df_actions.groupby("Run", sort=False):
            ma_vals = grp["Actions"].rolling(window=allocation_ma_window, min_periods=1).mean()
            ma_rows.append(pd.DataFrame({
                "Steps": grp["Steps"].values,
                "MA": ma_vals.values,
                "Run": run_name,
            }))
        df_ma = pd.concat(ma_rows, ignore_index=True)

    return {
        "rewards": df_rewards,
        "actions": df_actions,
        "actions_ma": df_ma,
        "stride": stride,
        "date_str": date_str,
        "reward_type": reward_type,
        "is_portfolio": is_portfolio,
        "training_steps": training_steps,
        "training_episodes": training_episodes,
        "n_obs": n_obs,
        "allocation_ma_window": allocation_ma_window if (is_portfolio and show_allocation_ma) else None,
    }


def plot_rewards(
    df_rewards: pd.DataFrame,
    training_steps: int | None = None,
    training_episodes: int | None = None,
    reward_type: str | None = None,
    stride: int = 1,
    n_obs: int | None = None,
    date_str: str = "",
) -> "ggplot":
    """Build cumulative rewards ggplot from a pre-built DataFrame.

    Accepts the DataFrame returned by build_rollout_plot_data (key "rewards"),
    or any DataFrame with columns Steps, Cumulative_Reward, Run.
    """
    reward_label = _REWARD_TYPE_LABELS.get(reward_type or "", reward_type or "")
    reward_prefix = "Cumulative sum of per-step rewards received by the agent."
    if reward_label:
        reward_prefix += f" Reward function: {reward_label}."
    if stride > 1 and n_obs is not None:
        n_plotted = len(range(0, n_obs, stride))
        reward_prefix += f"\nRollout: {n_obs:,} steps total; showing {n_plotted:,} points (every {stride}th step)."

    reward_runs = list(df_rewards["Run"].unique())
    return (
        ggplot(df_rewards, aes(x="Steps", y="Cumulative_Reward", color="Run"))
        + geom_line(size=0.32)
        + labs(
            title="Cumulative Rewards",
            x="Steps",
            y="Cumulative Reward",
            caption=_build_run_caption(
                reward_prefix,
                reward_runs,
                training_steps=training_steps,
                training_episodes=training_episodes,
                date_range=date_str,
            ),
        )
        + scale_color_manual(values=PALETTE, name="Strategy")
        + thesis_theme()
        + guides(color=guide_legend(title="Strategy"))
    )


def plot_actions(
    df_actions: pd.DataFrame,
    df_ma: "pd.DataFrame | None" = None,
    is_portfolio: bool = False,
    training_steps: int | None = None,
    training_episodes: int | None = None,
    stride: int = 1,
    n_obs: int | None = None,
    date_str: str = "",
    allocation_ma_window: int = 500,
) -> "ggplot":
    """Build actions/portfolio-allocation ggplot from a pre-built DataFrame.

    Accepts the DataFrame returned by build_rollout_plot_data (key "actions"),
    or any DataFrame with columns Steps, Actions, Run.
    """
    if is_portfolio:
        y_label = "Portfolio Weight"
        title = "Portfolio Allocation Comparison"
        action_prefix = "Portfolio weight output by the agent at each step.\nRange [-1, 1]: -1 = fully short, 0 = flat, +1 = fully long."
        if df_ma is not None:
            action_prefix += f"\nDashed line represents the mean position over {allocation_ma_window} steps."
    else:
        y_label = "Actions"
        title = "Actions Comparison"
        action_prefix = "Discrete action selected by the agent at each step."

    action_runs = list(df_actions["Run"].unique())
    action_plot = (
        ggplot(df_actions, aes(x="Steps", y="Actions", color="Run"))
        + geom_line(size=0.32)
        + labs(
            title=title,
            x="Steps",
            y=y_label,
            caption=_build_run_caption(
                action_prefix,
                action_runs,
                training_steps=training_steps,
                training_episodes=training_episodes,
                date_range=date_str,
            ),
        )
        + scale_color_manual(values=PALETTE, name="Strategy")
        + thesis_theme()
        + guides(color=guide_legend(title="Strategy"))
    )

    if is_portfolio and df_ma is not None:
        action_plot = action_plot + geom_line(
            data=df_ma,
            mapping=aes(x="Steps", y="MA", group="Run"),
            color="black",
            linetype="dashed",
            size=0.7,
            inherit_aes=False,
        )

    return action_plot


def compare_rollouts(
    rollouts,
    n_obs,
    is_portfolio: bool = False,
    training_steps: int | None = None,
    training_episodes: int | None = None,
    df: pd.DataFrame | None = None,
    reward_type: str | None = None,
    max_plot_points: int | None = None,
    show_allocation_ma: bool = True,
    allocation_ma_window: int = 500,
    show_benchmarks: bool = False,
    benchmark_price_column: str = "close",
):
    """Compare multiple rollouts and visualize their actions and rewards."""
    data = build_rollout_plot_data(
        rollouts, n_obs,
        is_portfolio=is_portfolio,
        training_steps=training_steps,
        training_episodes=training_episodes,
        df=df,
        reward_type=reward_type,
        max_plot_points=max_plot_points,
        show_allocation_ma=show_allocation_ma,
        allocation_ma_window=allocation_ma_window,
        show_benchmarks=show_benchmarks,
        benchmark_price_column=benchmark_price_column,
    )
    reward_plot = plot_rewards(
        data["rewards"],
        training_steps=data["training_steps"],
        training_episodes=data["training_episodes"],
        reward_type=data["reward_type"],
        stride=data["stride"],
        n_obs=data["n_obs"],
        date_str=data["date_str"],
    )
    action_plot = plot_actions(
        data["actions"],
        df_ma=data.get("actions_ma"),
        is_portfolio=data["is_portfolio"],
        training_steps=data["training_steps"],
        training_episodes=data["training_episodes"],
        stride=data["stride"],
        n_obs=data["n_obs"],
        date_str=data["date_str"],
        allocation_ma_window=data.get("allocation_ma_window") or allocation_ma_window,
    )
    return reward_plot, action_plot


# ---------------------------------------------------------------------------
# Equity curve data builder and standalone plotter
# ---------------------------------------------------------------------------


def build_equity_plot_data(
    rollouts,
    n_obs: int,
    df_prices=None,
    env=None,
    actual_returns_list=None,
    initial_portfolio_value: float = DEFAULT_INITIAL_PORTFOLIO_VALUE,
    benchmark_price_column: str = "close",
    initial_capital: float | None = None,
    benchmarks: "frozenset | None" = None,
    training_steps: int | None = None,
    training_episodes: int | None = None,
    n_total_symbols: int | None = None,
    policy_mode: str = "deterministic",
    max_plot_points: int | None = None,
    reward_type: str | None = None,
) -> dict:
    """Build DataFrame for the equity curve plot.

    Separates data preparation from rendering so QMD can call plot_equity_curve
    with custom figure dimensions.

    Returns a dict with keys:
        returns: DataFrame(Steps, Portfolio_Value, Run)
        initial_portfolio_value, policy_mode, training_steps, training_episodes,
        date_str, n_obs, stride, symbols, n_total_symbols
    """
    from trading_rl.constants import BenchmarkName
    if benchmarks is None:
        benchmarks = frozenset({BenchmarkName.BUY_AND_HOLD})
    show_buy_and_hold = BenchmarkName.BUY_AND_HOLD in benchmarks
    show_max_profit   = BenchmarkName.MAX_PROFIT    in benchmarks
    show_twap         = BenchmarkName.TWAP          in benchmarks
    show_vwap         = BenchmarkName.VWAP          in benchmarks

    if initial_capital is not None:
        initial_portfolio_value = initial_capital
    if initial_portfolio_value <= 0:
        raise ValueError(f"initial_portfolio_value must be > 0, got {initial_portfolio_value}")

    t0 = time.monotonic()
    returns_data = []
    stride = max(1, n_obs // max_plot_points) if max_plot_points and max_plot_points < n_obs else 1
    logger.trace("build_equity_plot_data start n_obs={} stride={}", n_obs, stride)

    def _extend_with_stride(run_name: str, values: np.ndarray) -> None:
        idx = np.arange(len(values))[::stride]
        returns_data.extend(
            {"Steps": int(s), "Portfolio_Value": float(v), "Run": run_name}
            for s, v in zip(idx, values[::stride])
        )

    if rollouts is None and actual_returns_list:
        for i, actual_returns in enumerate(actual_returns_list):
            run_name = "Deterministic" if i == 0 else f"Run_{i}"
            if actual_returns is not None:
                logger.trace("{}: Using actual portfolio returns from provided list", run_name)
                portfolio_values = _portfolio_values_from_actual_returns(
                    actual_returns, initial_portfolio_value, n_obs,
                )
                _extend_with_stride(run_name, portfolio_values)
    else:
        for i, rollout in enumerate(rollouts):
            run_name = "Deterministic" if i == 0 else "Random"
            if actual_returns_list and i < len(actual_returns_list):
                actual_returns = actual_returns_list[i]
            else:
                actual_returns = extract_tradingenv_return_series(env, n_obs) if env else None

            if actual_returns is not None:
                logger.trace("{}: Using actual portfolio returns from TradingEnv broker", run_name)
                portfolio_values = _portfolio_values_from_actual_returns(
                    actual_returns, initial_portfolio_value, n_obs,
                )
                _extend_with_stride(run_name, portfolio_values)
            elif reward_type in (None, "log_return"):
                rewards = rollout["next"]["reward"][:n_obs].detach().cpu().numpy()
                cumulative_log_returns = np.cumsum(rewards)
                logger.trace("{}: Using rollout rewards as log-return fallback", run_name)
                portfolio_values = initial_portfolio_value * np.exp(cumulative_log_returns)
                _extend_with_stride(run_name, portfolio_values)
            else:
                logger.warning(
                    "{}: Cannot derive portfolio values — reward_type='{}' rewards are not "
                    "log returns and no broker NLV is available. Skipping series.",
                    run_name, reward_type,
                )

    logger.trace("returns_data built n_points={} elapsed_s={:.2f}", len(returns_data), time.monotonic() - t0)

    price_series = None
    if df_prices is not None:
        benchmark_col = benchmark_price_column
        if benchmark_col not in df_prices.columns:
            if benchmark_col != "close":
                raise ValueError(
                    f"Benchmark price column '{benchmark_col}' not found in df_prices. "
                    f"Available columns: {list(df_prices.columns)}"
                )
            elif "close" in df_prices.columns:
                benchmark_col = "close"
            else:
                logger.warning("No benchmark price column available; skipping benchmarks.")
                benchmark_col = ""
        price_series = df_prices[benchmark_col] if benchmark_col else None
        if price_series is None:
            df_prices = None

    if df_prices is not None:
        price_window = price_series.iloc[: n_obs + 1]
        benchmark_returns = price_window.pct_change().iloc[1:].to_numpy(dtype=float)
        n_bad = int(np.sum(~np.isfinite(benchmark_returns)))
        if n_bad > 0:
            logger.warning(
                "benchmark price series has {} non-finite return(s) "
                "(likely cross-symbol boundary in concatenated val_df); skipping benchmark lines",
                n_bad,
            )
            df_prices = None
        else:
            buy_and_hold_values = max_profit_values = twap_values = vwap_values = None
            if show_buy_and_hold:
                buy_and_hold_values = initial_portfolio_value * np.exp(
                    np.asarray(np.log1p(benchmark_returns).cumsum(), dtype=float)
                )
            if show_max_profit:
                max_profit_values = initial_portfolio_value * np.exp(
                    np.asarray(np.log1p(np.abs(benchmark_returns)).cumsum(), dtype=float)
                )
            if show_twap or show_vwap:
                from trading_rl.evaluation.statistical_benchmarks import (
                    compute_twap_returns,
                    compute_vwap_returns,
                    resolve_vwap_volume_series,
                )
                if show_twap:
                    twap_simple = compute_twap_returns(price_series, n_obs)
                    twap_values = initial_portfolio_value * np.cumprod(1.0 + twap_simple)
                if show_vwap:
                    volumes, _vol_source = resolve_vwap_volume_series(df_prices)
                    if volumes is None:
                        logger.warning("VWAP benchmark skipped: no usable volume column in df_prices")
                        show_vwap = False
                    else:
                        vwap_simple = compute_vwap_returns(price_series, volumes, n_obs)
                        vwap_values = initial_portfolio_value * np.cumprod(1.0 + vwap_simple)

            if show_buy_and_hold and buy_and_hold_values is not None:
                _extend_with_stride("Buy-and-Hold", buy_and_hold_values)
            if show_max_profit and max_profit_values is not None:
                _extend_with_stride("Max Profit (Unleveraged)", max_profit_values)
            if show_twap and twap_values is not None:
                _extend_with_stride("TWAP", twap_values)
            if show_vwap and vwap_values is not None:
                _extend_with_stride("VWAP", vwap_values)

    logger.trace("benchmark data appended total_points={} elapsed_s={:.2f}", len(returns_data), time.monotonic() - t0)

    df_returns = pd.DataFrame(returns_data)
    df_returns["Run"] = _as_ordered_run_categorical(df_returns["Run"])
    logger.trace("DataFrame built elapsed_s={:.2f}", time.monotonic() - t0)

    symbols: list[str] = []
    if df_prices is not None and "symbol" in df_prices.columns:
        symbols = sorted(df_prices["symbol"].dropna().unique().tolist())

    date_range_str = _date_range_str(df_prices, n_obs)

    return {
        "returns": df_returns,
        "initial_portfolio_value": initial_portfolio_value,
        "policy_mode": policy_mode,
        "training_steps": training_steps,
        "training_episodes": training_episodes,
        "date_str": date_range_str,
        "n_obs": n_obs,
        "stride": stride,
        "symbols": symbols,
        "n_total_symbols": n_total_symbols,
    }


def plot_equity_curve(
    df_returns: pd.DataFrame,
    initial_portfolio_value: float = DEFAULT_INITIAL_PORTFOLIO_VALUE,
    policy_mode: str = "deterministic",
    training_steps: int | None = None,
    training_episodes: int | None = None,
    date_str: str = "",
    n_obs: int | None = None,
    stride: int = 1,
    symbols: "list[str] | None" = None,
    n_total_symbols: int | None = None,
) -> "ggplot":
    """Build portfolio equity curve ggplot from a pre-built DataFrame.

    Accepts the DataFrame returned by build_equity_plot_data (key "returns"),
    or any DataFrame with columns Steps, Portfolio_Value, Run.
    """
    symbols = symbols or []
    is_sample = n_total_symbols is not None and n_total_symbols > len(symbols) and len(symbols) > 0
    asset_str = f" — {', '.join(symbols)}" if (symbols and not is_sample) else ""
    full_title = f"Portfolio Value{asset_str}"

    pooled_note = (
        f"Evaluation shown on {len(symbols)} representative symbol(s) ({', '.join(symbols)}); "
        f"model trained on {n_total_symbols} symbols."
        if is_sample
        else ""
    )

    returns_runs = list(df_returns["Run"].unique())
    _policy_label = {
        "deterministic": "deterministic (no exploration noise)",
        "stochastic": "stochastic (exploration noise active)",
    }.get(policy_mode, policy_mode)
    caption_prefix = (
        f"Portfolio value in \\$ reconstructed from broker NLV at each step."
        f" Initial capital: \\${initial_portfolio_value:,.0f}."
        f" Policy: {_policy_label}."
    )
    if pooled_note:
        caption_prefix = f"{caption_prefix}\n{pooled_note}"
    if stride > 1 and n_obs is not None:
        n_plotted = len(range(0, n_obs, stride))
        caption_prefix += f"\nRollout: {n_obs:,} steps total; showing {n_plotted:,} points (every {stride}th step)."

    return (
        ggplot(df_returns, aes(x="Steps", y="Portfolio_Value", color="Run", linetype="Run"))
        + geom_line(size=0.32)
        + labs(
            title=full_title,
            x="Steps",
            y="Portfolio Value",
            caption=_build_run_caption(
                caption_prefix,
                returns_runs,
                training_steps=training_steps,
                training_episodes=training_episodes,
                date_range=date_str,
            ),
        )
        + scale_color_manual(values=PALETTE, name="Strategy")
        + scale_linetype_manual(values=LINETYPE, name="Strategy")
        + thesis_theme()
    )


def create_equity_curve_plot(
    rollouts,
    n_obs,
    df_prices=None,
    env=None,
    actual_returns_list=None,
    initial_portfolio_value: float = DEFAULT_INITIAL_PORTFOLIO_VALUE,
    benchmark_price_column: str = "close",
    initial_capital: float | None = None,
    benchmarks: "frozenset | None" = None,
    training_steps: int | None = None,
    training_episodes: int | None = None,
    n_total_symbols: int | None = None,
    policy_mode: str = "deterministic",
    max_plot_points: int | None = None,
    reward_type: str | None = None,
):
    """Create a plot showing actual portfolio returns, not training rewards."""
    data = build_equity_plot_data(
        rollouts, n_obs,
        df_prices=df_prices,
        env=env,
        actual_returns_list=actual_returns_list,
        initial_portfolio_value=initial_portfolio_value,
        benchmark_price_column=benchmark_price_column,
        initial_capital=initial_capital,
        benchmarks=benchmarks,
        training_steps=training_steps,
        training_episodes=training_episodes,
        n_total_symbols=n_total_symbols,
        policy_mode=policy_mode,
        max_plot_points=max_plot_points,
        reward_type=reward_type,
    )
    return plot_equity_curve(
        data["returns"],
        initial_portfolio_value=data["initial_portfolio_value"],
        policy_mode=data["policy_mode"],
        training_steps=data["training_steps"],
        training_episodes=data["training_episodes"],
        date_str=data["date_str"],
        n_obs=data["n_obs"],
        stride=data["stride"],
        symbols=data["symbols"],
        n_total_symbols=data["n_total_symbols"],
    )


def create_equity_progression_plot(
    history: "list[tuple[int, ReturnSeries]]",
    initial_portfolio_value: float = DEFAULT_INITIAL_PORTFOLIO_VALUE,
    max_plot_points: int | None = None,
):
    """Equity curve at each training checkpoint with a single blue hue gradient.

    Args:
        history: List of (training_step, ReturnSeries) pairs, ordered by step.
        initial_portfolio_value: Starting portfolio value for equity reconstruction.
        max_plot_points: Downsample each series to at most this many points.

    Returns:
        A plotnine ggplot, or None if fewer than two checkpoints are available.
    """
    if len(history) < 2:
        return None

    rows = []
    for training_step, returns in history:
        equity = returns.to_equity(initial_portfolio_value).values
        n = len(equity)
        stride = max(1, n // max_plot_points) if max_plot_points and max_plot_points < n else 1
        idx = np.arange(n)[::stride]
        for s, v in zip(idx, equity[::stride]):
            rows.append({
                "Steps": int(s),
                "Portfolio_Value": float(v),
                "Training_Step": int(training_step),
            })

    df = pd.DataFrame(rows)
    n_checkpoints = df["Training_Step"].nunique()

    return (
        ggplot(df, aes(x="Steps", y="Portfolio_Value", color="Training_Step", group="Training_Step"))
        + geom_line(size=0.32)
        + scale_color_gradient(
            low="#f5cccc", high="#CC0000", name="Training Step",
            labels=lambda vals: [f"{int(v / 1_000)}k" for v in vals],
        )
        + labs(
            title="Equity Curve Progression",
            x="Steps",
            y="Portfolio Value",
            caption=(
                f"Each line shows the deterministic policy at a training checkpoint"
                f" ({n_checkpoints} checkpoints).\n"
                "Lighter blue = earlier in training; darker blue = later."
            ),
        )
        + thesis_theme()
    )


def create_train_val_progression_plot(
    train_history: "list[tuple[int, ReturnSeries]] | None",
    val_history: "list[tuple[int, ReturnSeries]] | None",
    initial_portfolio_value: float = DEFAULT_INITIAL_PORTFOLIO_VALUE,
    metric: str = "total_return",
):
    """Train vs val progression at checkpoints — shows overfitting/learning.

    Args:
        train_history: List of (training_step, ReturnSeries) for train split.
        val_history: List of (training_step, ReturnSeries) for val split.
        initial_portfolio_value: Starting portfolio value.
        metric: "total_return" or "final_portfolio_value".

    Returns:
        A plotnine ggplot, or None if fewer than two checkpoints total.
    """
    train_history = train_history or []
    val_history = val_history or []

    if len(train_history) + len(val_history) < 2:
        return None

    rows = []

    for training_step, returns in train_history:
        if metric == "total_return":
            y = float(returns.to_cumulative_log(include_initial=False).values[-1])
        elif metric == "final_portfolio_value":
            equity = returns.to_equity(initial_portfolio_value)
            y = float(equity.values[-1])
        else:
            raise ValueError(f"Unknown metric: {metric}")
        rows.append({
            "Training_Step": int(training_step),
            "Value": y,
            "Split": "Train",
        })

    for training_step, returns in val_history:
        if metric == "total_return":
            y = float(returns.to_cumulative_log(include_initial=False).values[-1])
        elif metric == "final_portfolio_value":
            equity = returns.to_equity(initial_portfolio_value)
            y = float(equity.values[-1])
        else:
            raise ValueError(f"Unknown metric: {metric}")
        rows.append({
            "Training_Step": int(training_step),
            "Value": y,
            "Split": "Val",
        })

    df = pd.DataFrame(rows)

    y_label = "Total Return" if metric == "total_return" else "Final Portfolio Value"
    title = "Learning Progression: Train vs Val"

    caption_text = (
        f"Each point shows the deterministic policy evaluated on {metric.replace('_', ' ')} "
        "at that training checkpoint."
    )
    if metric == "total_return":
        caption_text += " (log scale)"

    return (
        ggplot(df, aes(x="Training_Step", y="Value", color="Split", linetype="Split"))
        + geom_line(size=0.7)
        + labs(
            title=title,
            x="Training Step",
            y=y_label,
            caption=caption_text,
        )
        + thesis_theme()
        + guides(color=guide_legend(title="Split"), linetype=guide_legend(title="Split"))
    )


def create_price_plot(
    df: pd.DataFrame,
    price_column: str = "close",
    max_points: int = 5_000,
) -> "ggplot | None":
    """Line plot of the underlying close price for an evaluation split.

    Downsampled to at most *max_points* rows so the PNG stays small.
    Returns None when *price_column* is not present in *df*.
    """
    if price_column not in df.columns:
        return None

    prices = df[price_column]
    n = len(prices)
    if n > max_points:
        step = max(1, n // max_points)
        prices = prices.iloc[::step]

    use_datetime = isinstance(df.index, pd.DatetimeIndex)
    if use_datetime:
        plot_df = pd.DataFrame({"x": prices.index, "price": prices.values})
        x_label = "Time"
    else:
        plot_df = pd.DataFrame({"x": range(len(prices)), "price": prices.values})
        x_label = "Step"

    plot = (
        ggplot(plot_df, aes(x="x", y="price"))
        + geom_line(color=PALETTE.get("Deterministic", "#CC0000"), size=0.4)
        + labs(x=x_label, y="Price")
        + thesis_theme()
        + theme(axis_text_x=element_text(angle=90, hjust=1))
    )
    if use_datetime:
        plot = plot + scale_x_datetime(date_labels="%H:%M:%S")
    return plot


def _render_table_on_ax(
    ax: "matplotlib.axes.Axes",
    rows: "list[tuple[str, str, str, str]]",
    base_size: int,
) -> None:
    """Draw a styled metrics table onto *ax*."""
    ax.axis("off")
    col_labels = ["", "Metric", "Value", "Description"]
    table_data = [(r[0], r[1], r[2], r[3]) for r in rows]
    tbl = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="left",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(base_size)

    for col in range(4):
        cell = tbl[0, col]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")

    for row_idx, (sec, *_) in enumerate(rows, start=1):
        bg = "#f0f4f8" if row_idx % 2 == 0 else "white"
        for col in range(4):
            cell = tbl[row_idx, col]
            cell.set_facecolor(bg)
            if col == 0 and sec:
                cell.set_text_props(fontweight="bold", color="#2c3e50")

    for col, w in zip(range(4), [0.10, 0.18, 0.12, 0.60]):
        for row_idx in range(len(rows) + 1):
            tbl[row_idx, col].set_width(w)


def create_metrics_table_figure(
    metric_report: "MetricReport",
    step: int | None = None,
    split: str | None = None,
    df: "pd.DataFrame | None" = None,
    max_steps: int | None = None,
) -> "matplotlib.figure.Figure":
    """Render a MetricReport as a matplotlib table figure.

    All metrics are rendered in a single column regardless of row count.
    Uses the thesis font size so the PNG looks consistent with other evaluation plots.
    """
    import matplotlib.pyplot as plt
    from trading_rl.evaluation.metric_meta import METRIC_SECTIONS, fmt_metric

    report_dict = metric_report.to_dict()

    rows: list[tuple[str, str, str, str]] = []
    for section, metas in METRIC_SECTIONS.items():
        first = True
        for meta in metas:
            if meta.key not in report_dict:
                continue
            val = report_dict[meta.key]
            sec_label = section if first else ""
            first = False
            rows.append((sec_label, meta.label, fmt_metric(meta.key, val), meta.description))

    BASE_SIZE = 11

    fig_height = max(3.5, len(rows) * 0.30 + 1.2) * 1.3 * 1.25 * 1.15

    title_parts = []
    if split:
        title_parts.append(f"Split: {split}")
    if step is not None:
        title_parts.append(f"Step: {step:,}")
    date_str = _date_range_str(df, max_steps or (len(df) - 1 if df is not None else 0))
    if date_str:
        title_parts.append(date_str)
    title = "  |  ".join(title_parts) if title_parts else None

    fig, ax = plt.subplots(figsize=(14.3 * 1.15, fig_height))
    if title:
        ax.set_title(title, fontsize=BASE_SIZE + 1, pad=8)
    _render_table_on_ax(ax, rows, BASE_SIZE)
    fig.tight_layout()

    return fig


_MERGED_PANEL_HEIGHT = 9.0  # inches per panel — sized for 22pt base font


def create_merged_comparison_plot(reward_plot, action_plot, equity_curve_plot=None, save_path=None):
    """Merge reward, action, and (optionally) equity curve plots into a vertical layout."""
    if equity_curve_plot is not None:
        merged_plot = reward_plot / action_plot / equity_curve_plot
    else:
        merged_plot = reward_plot / action_plot
    n_panels = 3 if equity_curve_plot is not None else 2
    merged_plot = merged_plot + theme(figure_size=(FIGURE_WIDTH * 2, round(_MERGED_PANEL_HEIGHT * n_panels, 1)))
    if save_path:
        logger.info("save merged comparison plot path={}", save_path)
        from trading_rl.evaluation.thesis_theme import PLOT_DPI
        merged_plot.save(save_path, dpi=PLOT_DPI, verbose=False)
        write_asset_meta(save_path, generator="evaluation/plots.py")
    return merged_plot
