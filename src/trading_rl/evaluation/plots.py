"""Evaluation plotting helpers for rollout and benchmark comparisons."""

from __future__ import annotations

import time
import numpy as np
import pandas as pd
from plotnine import (
    aes,
    geom_line,
    ggplot,
    guide_legend,
    guides,
    labs,
    scale_color_manual,
    scale_linetype_manual,
    theme,
)

from trading_rl.evaluation.thesis_theme import FIGURE_HEIGHT, FIGURE_WIDTH, LINETYPE, PALETTE, thesis_theme
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
):
    """Compare multiple rollouts and visualize their actions and rewards."""
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
            # One rollout may terminate early (e.g. bankruptcy); truncate to the
            # shorter length so allclose receives same-shape tensors.
            min_a = min(all_actions[i].shape[0], all_actions[j].shape[0])
            min_r = min(all_rewards[i].shape[0], all_rewards[j].shape[0])
            actions_equal = bool(allclose(all_actions[i][:min_a].float(), all_actions[j][:min_a].float()))
            rewards_equal = bool(allclose(all_rewards[i][:min_r].float(), all_rewards[j][:min_r].float()))
            logger.info(
                "Run %s vs Run %s | actions_identical=%s rewards_identical=%s",
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
        # Downsample after cumsum so in-between steps still contribute.
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

    date_str = _date_range_str(df, n_obs)

    reward_label = _REWARD_TYPE_LABELS.get(reward_type or "", reward_type or "")
    reward_prefix = "Cumulative sum of per-step rewards received by the agent."
    if reward_label:
        reward_prefix += f" Reward function: {reward_label}."

    reward_runs = list(df_rewards["Run"].unique())
    reward_plot = (
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
        + thesis_theme()
        + guides(color=guide_legend(title="Strategy"))
    )

    action_runs = list(df_actions["Run"].unique())
    if is_portfolio:
        y_label = "Portfolio Weight"
        title = "Portfolio Allocation Comparison"
        action_prefix = "Portfolio weight output by the agent at each step.\nRange [-1, 1]: -1 = fully short, 0 = flat, +1 = fully long."
        if show_allocation_ma:
            action_prefix += f"\nDashed line represents the mean position over {allocation_ma_window} steps."
    else:
        y_label = "Actions"
        title = "Actions Comparison"
        action_prefix = "Discrete action selected by the agent at each step."

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
        + thesis_theme()
        + guides(color=guide_legend(title="Strategy"))
    )

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
        action_plot = action_plot + geom_line(
            data=df_ma,
            mapping=aes(x="Steps", y="MA", group="Run"),
            color="black",
            linetype="dashed",
            size=0.7,
            inherit_aes=False,
        )

    return reward_plot, action_plot


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
    from trading_rl.constants import BenchmarkName
    if benchmarks is None:
        benchmarks = frozenset({BenchmarkName.BUY_AND_HOLD})
    show_buy_and_hold   = BenchmarkName.BUY_AND_HOLD   in benchmarks
    show_short_and_hold = BenchmarkName.SHORT_AND_HOLD  in benchmarks
    show_max_profit     = BenchmarkName.MAX_PROFIT      in benchmarks
    show_twap           = BenchmarkName.TWAP            in benchmarks
    show_vwap           = BenchmarkName.VWAP            in benchmarks

    if initial_capital is not None:
        initial_portfolio_value = initial_capital
    if initial_portfolio_value <= 0:
        raise ValueError(
            f"initial_portfolio_value must be > 0, got {initial_portfolio_value}"
        )

    t0 = time.monotonic()
    returns_data = []
    stride = max(1, n_obs // max_plot_points) if max_plot_points and max_plot_points < n_obs else 1
    logger.debug("create_equity_curve_plot start n_obs=%d stride=%d", n_obs, stride)

    def _extend_with_stride(run_name: str, values: np.ndarray) -> None:
        idx = np.arange(len(values))[::stride]
        returns_data.extend(
            {"Steps": int(s), "Portfolio_Value": float(v), "Run": run_name}
            for s, v in zip(idx, values[::stride])
        )

    # Handle case where rollouts is None but actual_returns_list is provided
    if rollouts is None and actual_returns_list:
        for i, actual_returns in enumerate(actual_returns_list):
            run_name = "Deterministic" if i == 0 else f"Run_{i}"

            if actual_returns is not None:
                logger.debug(
                    "%s: Using actual portfolio returns from provided list",
                    run_name,
                )

                portfolio_values = _portfolio_values_from_actual_returns(
                    actual_returns,
                    initial_portfolio_value,
                    n_obs,
                )
                _extend_with_stride(run_name, portfolio_values)
    else:
        # Original logic for when rollouts are provided
        for i, rollout in enumerate(rollouts):
            run_name = "Deterministic" if i == 0 else "Random"
            if actual_returns_list and i < len(actual_returns_list):
                actual_returns = actual_returns_list[i]
            else:
                actual_returns = extract_tradingenv_return_series(env, n_obs) if env else None

            if actual_returns is not None:
                logger.debug(
                    "%s: Using actual portfolio returns from TradingEnv broker",
                    run_name,
                )
                portfolio_values = _portfolio_values_from_actual_returns(
                    actual_returns,
                    initial_portfolio_value,
                    n_obs,
                )
                _extend_with_stride(run_name, portfolio_values)
            elif reward_type in (None, "log_return"):
                rewards = rollout["next"]["reward"][:n_obs].detach().cpu().numpy()
                cumulative_log_returns = np.cumsum(rewards)
                logger.debug("%s: Using rollout rewards as log-return fallback", run_name)
                portfolio_values = initial_portfolio_value * np.exp(cumulative_log_returns)
                _extend_with_stride(run_name, portfolio_values)
            else:
                logger.warning(
                    "%s: Cannot derive portfolio values — reward_type='%s' rewards are not "
                    "log returns and no broker NLV is available. Skipping series.",
                    run_name,
                    reward_type,
                )

    logger.debug("returns_data built n_points=%d elapsed=%.2fs", len(returns_data), time.monotonic() - t0)

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
                logger.warning(
                    "No benchmark price column available; skipping benchmarks.",
                )
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
                "benchmark price series has %d non-finite return(s) "
                "(likely cross-symbol boundary in concatenated val_df); skipping benchmark lines",
                n_bad,
            )
            df_prices = None
        else:
            if show_buy_and_hold:
                buy_and_hold = np.log1p(benchmark_returns).cumsum()
                buy_and_hold_values = initial_portfolio_value * np.exp(
                    np.asarray(buy_and_hold, dtype=float)
                )

            if show_short_and_hold:
                short_and_hold = np.log1p(-benchmark_returns).cumsum()
                short_and_hold_values = initial_portfolio_value * np.exp(
                    np.asarray(short_and_hold, dtype=float)
                )

            if show_max_profit:
                max_profit = np.log1p(np.abs(benchmark_returns)).cumsum()
                max_profit_values = initial_portfolio_value * np.exp(
                    np.asarray(max_profit, dtype=float)
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
                    volumes, vol_source = resolve_vwap_volume_series(df_prices)
                    if volumes is None:
                        logger.warning("VWAP benchmark skipped: no usable volume column in df_prices")
                        show_vwap = False
                    else:
                        vwap_simple = compute_vwap_returns(price_series, volumes, n_obs)
                        vwap_values = initial_portfolio_value * np.cumprod(1.0 + vwap_simple)

    if df_prices is not None:
        if show_buy_and_hold:
            _extend_with_stride("Buy-and-Hold", buy_and_hold_values)
        if show_short_and_hold:
            _extend_with_stride("Short-and-Hold", short_and_hold_values)
        if show_max_profit:
            _extend_with_stride("Max Profit (Unleveraged)", max_profit_values)
        if show_twap:
            _extend_with_stride("TWAP", twap_values)
        if show_vwap:
            _extend_with_stride("VWAP", vwap_values)

    logger.debug("benchmark data appended total_points=%d elapsed=%.2fs", len(returns_data), time.monotonic() - t0)

    logger.debug("building DataFrame for plot n_rows=%d", len(returns_data))
    df_returns = pd.DataFrame(returns_data)
    logger.debug("DataFrame built elapsed=%.2fs", time.monotonic() - t0)

    # Build title components: asset composition and datetime range.
    symbols: list[str] = []
    if df_prices is not None and "symbol" in df_prices.columns:
        symbols = sorted(df_prices["symbol"].dropna().unique().tolist())

    is_sample = n_total_symbols is not None and n_total_symbols > len(symbols) and len(symbols) > 0
    if symbols and not is_sample:
        asset_str = f" — {', '.join(symbols)}"
    else:
        asset_str = ""

    date_range_str = _date_range_str(df_prices, n_obs)

    full_title = f"Portfolio Value{asset_str}"

    pooled_note = (
        f"Evaluation shown on {len(symbols)} representative symbol(s) ({', '.join(symbols)}); "
        f"model trained on {n_total_symbols} symbols."
        if is_sample
        else ""
    )

    returns_runs = list(df_returns["Run"].unique())
    logger.debug("constructing ggplot object")
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
    plot = (
        ggplot(df_returns, aes(x="Steps", y="Portfolio_Value", color="Run", linetype="Run"))
        + geom_line(size=0.32)
        + labs(
            title=full_title,
            x="Steps",
            y="Portfolio Value (\\$)",
            caption=_build_run_caption(
                caption_prefix,
                returns_runs,
                training_steps=training_steps,
                training_episodes=training_episodes,
                date_range=date_range_str,
            ),
        )
        + scale_color_manual(values=PALETTE, name="Strategy")
        + scale_linetype_manual(values=LINETYPE, name="Strategy")
        + thesis_theme()
    )
    logger.debug("ggplot object constructed elapsed=%.2fs", time.monotonic() - t0)
    return plot


def create_merged_comparison_plot(reward_plot, action_plot, equity_curve_plot=None, save_path=None):
    """Merge reward, action, and (optionally) equity curve plots into a vertical layout."""
    if equity_curve_plot is not None:
        merged_plot = reward_plot / action_plot / equity_curve_plot
    else:
        merged_plot = reward_plot / action_plot
    n_panels = 3 if equity_curve_plot is not None else 2
    merged_plot = merged_plot + theme(figure_size=(FIGURE_WIDTH * 2, round(FIGURE_HEIGHT * n_panels, 1)))
    if save_path:
        logger.info("save merged comparison plot path=%s", save_path)
        from trading_rl.evaluation.thesis_theme import PLOT_DPI
        merged_plot.save(save_path, dpi=PLOT_DPI, verbose=False)
    return merged_plot
