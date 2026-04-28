"""Evaluation plotting helpers for rollout and benchmark comparisons."""

from __future__ import annotations

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
    scale_color_manual,
    theme,
)
from torch import allclose

from logger import get_logger
from trading_rl.config import DEFAULT_INITIAL_PORTFOLIO_VALUE
from trading_rl.evaluation.returns import extract_tradingenv_returns

logger = get_logger(__name__)


def compare_rollouts(rollouts, n_obs, is_portfolio: bool = False):
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
            actions_equal = bool(allclose(all_actions[i].float(), all_actions[j].float()))
            rewards_equal = bool(allclose(all_rewards[i].float(), all_rewards[j].float()))
            logger.info(
                "Run %s vs Run %s | actions_identical=%s rewards_identical=%s",
                i + 1,
                j + 1,
                actions_equal,
                rewards_equal,
            )

    rewards_data = []
    for i, rewards in enumerate(all_rewards):
        rewards_np = rewards.detach().cpu().numpy()
        rewards_data.extend(
            [
                {
                    "Steps": step,
                    "Cumulative_Reward": val,
                    "Run": "Deterministic" if i == 0 else "Random",
                }
                for step, val in enumerate(np.cumsum(rewards_np))
            ]
        )
    df_rewards = pd.DataFrame(rewards_data)

    actions_data = []
    for i, actions in enumerate(all_actions):
        actions_np = actions.detach().cpu().numpy()
        actions_data.extend(
            [
                {
                    "Steps": step,
                    "Actions": val,
                    "Run": "Deterministic" if i == 0 else "Random",
                }
                for step, val in enumerate(actions_np)
            ]
        )
    df_actions = pd.DataFrame(actions_data)

    reward_plot = (
        ggplot(df_rewards, aes(x="Steps", y="Cumulative_Reward", color="Run"))
        + geom_line()
        + labs(title="Cumulative Rewards Comparison", x="Steps", y="Cumulative Reward")
        + theme(
            figure_size=(13, 7.8),
            legend_position="bottom",
            legend_title=element_text(weight="bold", size=11),
            legend_text=element_text(size=10),
            subplots_adjust={"left": 0.10, "right": 0.95},
        )
        + guides(color=guide_legend(title="Strategy"))
    )

    if is_portfolio:
        y_label = "Portfolio Weight"
        title = "Portfolio Allocation Comparison"
    else:
        y_label = "Actions"
        title = "Actions Comparison"

    action_plot = (
        ggplot(df_actions, aes(x="Steps", y="Actions", color="Run"))
        + geom_line()
        + labs(title=title, x="Steps", y=y_label)
        + theme(
            figure_size=(13, 7.8),
            legend_position="bottom",
            legend_title=element_text(weight="bold", size=11),
            legend_text=element_text(size=10),
            subplots_adjust={"left": 0.10, "right": 0.95},
        )
        + guides(color=guide_legend(title="Strategy"))
    )

    return reward_plot, action_plot


def create_actual_returns_plot(
    rollouts,
    n_obs,
    df_prices=None,
    env=None,
    actual_returns_list=None,
    initial_portfolio_value: float = DEFAULT_INITIAL_PORTFOLIO_VALUE,
    benchmark_price_column: str = "close",
    initial_capital: float | None = None,
):
    """Create a plot showing actual portfolio returns, not training rewards."""
    if initial_capital is not None:
        initial_portfolio_value = initial_capital
    if initial_portfolio_value <= 0:
        raise ValueError(
            f"initial_portfolio_value must be > 0, got {initial_portfolio_value}"
        )

    returns_data = []
    for i, rollout in enumerate(rollouts):
        run_name = "Deterministic" if i == 0 else "Random"
        if actual_returns_list and i < len(actual_returns_list):
            actual_returns = actual_returns_list[i]
        else:
            actual_returns = extract_tradingenv_returns(env, n_obs) if env else None

        if actual_returns is not None:
            cumulative_log_returns = np.asarray(actual_returns[:n_obs], dtype=float)
            logger.debug(
                "%s: Using actual portfolio returns from TradingEnv broker",
                run_name,
            )
        else:
            rewards = rollout["next"]["reward"][:n_obs].detach().cpu().numpy()
            cumulative_log_returns = np.cumsum(rewards)
            logger.debug("%s: Using rollout rewards as fallback", run_name)

        portfolio_values = initial_portfolio_value * np.exp(cumulative_log_returns)
        returns_data.extend(
            [
                {"Steps": step, "Portfolio_Value": val, "Run": run_name}
                for step, val in enumerate(portfolio_values)
            ]
        )

    if df_prices is not None:
        benchmark_col = benchmark_price_column
        if benchmark_col not in df_prices.columns:
            if "close" in df_prices.columns:
                logger.warning(
                    "Benchmark price column '%s' not found; falling back to 'close'.",
                    benchmark_col,
                )
                benchmark_col = "close"
            else:
                logger.warning(
                    "Benchmark price column '%s' missing and no 'close' fallback available; skipping benchmarks.",
                    benchmark_col,
                )
                benchmark_col = ""

        price_series = df_prices[benchmark_col] if benchmark_col else None
        if price_series is None:
            df_prices = None

    if df_prices is not None:
        buy_and_hold = (
            np.log(price_series / price_series.shift(1)).fillna(0).cumsum()[:n_obs]
        )
        max_profit = (
            np.log(abs(price_series / price_series.shift(1) - 1) + 1)
            .fillna(0)
            .cumsum()[:n_obs]
        )

        buy_and_hold_values = initial_portfolio_value * np.exp(
            np.asarray(buy_and_hold, dtype=float)
        )
        max_profit_values = initial_portfolio_value * np.exp(
            np.asarray(max_profit, dtype=float)
        )

        for step, (bh_val, mp_val) in enumerate(
            zip(buy_and_hold_values, max_profit_values, strict=False)
        ):
            returns_data.extend(
                [
                    {"Steps": step, "Portfolio_Value": bh_val, "Run": "Buy-and-Hold"},
                    {
                        "Steps": step,
                        "Portfolio_Value": mp_val,
                        "Run": "Max Profit (Unleveraged)",
                    },
                ]
            )

    df_returns = pd.DataFrame(returns_data)
    return (
        ggplot(df_returns, aes(x="Steps", y="Portfolio_Value", color="Run"))
        + geom_line()
        + labs(
            title=f"Actual Portfolio Value (Start ${initial_portfolio_value:,.0f})",
            x="Steps",
            y="Portfolio Value ($)",
        )
        + scale_color_manual(
            values={
                "Deterministic": "#F8766D",
                "Random": "#00BFC4",
                "Buy-and-Hold": "violet",
                "Max Profit (Unleveraged)": "green",
            }
        )
        + theme(figure_size=(13, 7.8))
    )


def create_merged_comparison_plot(reward_plot, action_plot, save_path=None):
    """Merge reward and action comparison plots into a single vertical layout."""
    merged_plot = reward_plot / action_plot
    if save_path:
        logger.info("save merged comparison plot path=%s", save_path)
        merged_plot.save(save_path, dpi=150, verbose=False)
    return merged_plot
