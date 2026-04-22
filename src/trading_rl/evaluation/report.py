"""Evaluation report generation from deterministic rollouts."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from tensordict.nn import InteractionType
from torchrl.envs.utils import set_exploration_type

from logger import get_logger
from trading_rl.evaluation.metrics import build_metric_report
from trading_rl.evaluation.returns import extract_tradingenv_returns

logger = get_logger(__name__)


def periods_per_year_from_timeframe(timeframe: str) -> int:
    """Convert timeframe strings to annualization factor."""
    mapping = {
        "1m": 365 * 24 * 60,
        "5m": 365 * 24 * 12,
        "15m": 365 * 24 * 4,
        "30m": 365 * 24 * 2,
        "1h": 365 * 24,
        "4h": 365 * 6,
        "1d": 365,
        "1w": 52,
    }
    return mapping.get(str(timeframe).lower(), 252)


def _extract_action_array(rollout, is_portfolio: bool) -> np.ndarray:
    action_tensor = rollout.get("action", None)
    if not isinstance(action_tensor, torch.Tensor):
        return np.array([])
    action_tensor = action_tensor.detach().cpu().squeeze()
    if not is_portfolio and action_tensor.ndim > 1 and action_tensor.shape[-1] > 1:
        action_tensor = action_tensor.argmax(dim=-1)
    return action_tensor.numpy()


def build_evaluation_report_for_trainer(
    trainer: Any,
    df_prices: pd.DataFrame,
    max_steps: int,
    config: Any,
    eval_env: Any | None = None,
) -> dict[str, float]:
    """Build the 25-metric evaluation report for deterministic policy rollout."""
    env_to_use = eval_env or trainer.env
    with torch.no_grad():
        try:
            with set_exploration_type(InteractionType.MODE):
                rollout = env_to_use.rollout(max_steps=max_steps, policy=trainer.actor)
        except RuntimeError:
            with set_exploration_type(InteractionType.DETERMINISTIC):
                rollout = env_to_use.rollout(max_steps=max_steps, policy=trainer.actor)

    reward_type = getattr(getattr(config, "env", None), "reward_type", "log_return")
    backend = getattr(getattr(config, "env", None), "backend", None)
    benchmark_price_column = getattr(
        getattr(config, "env", None), "price_column", None
    )
    if not benchmark_price_column:
        benchmark_price_column = "close"

    # Reward can be a shaped signal (e.g., differential Sharpe), so only interpret
    # rollout reward as log-return when reward_type explicitly says so.
    if str(reward_type).lower() == "log_return":
        strategy_log_returns = (
            rollout["next", "reward"].detach().cpu().reshape(-1).numpy()[:max_steps]
        )
        strategy_simple_returns = np.exp(strategy_log_returns) - 1.0
    else:
        strategy_simple_returns = np.array([], dtype=float)

        # TradingEnv backend can expose true NLV path via broker.track_record.
        if str(backend).lower() == "tradingenv":
            cumulative_log_returns = extract_tradingenv_returns(env_to_use, max_steps)
            if cumulative_log_returns is not None and len(cumulative_log_returns) > 0:
                cumulative_log_returns = np.asarray(cumulative_log_returns, dtype=float)
                step_log_returns = np.diff(cumulative_log_returns, prepend=0.0)
                strategy_simple_returns = np.exp(step_log_returns) - 1.0
                logger.info(
                    "Evaluation metrics using actual TradingEnv broker returns "
                    "(reward_type=%s, %d steps).",
                    reward_type,
                    len(strategy_simple_returns),
                )
            else:
                logger.warning(
                    "Could not extract TradingEnv broker returns for evaluation "
                    "(reward_type=%s); falling back to reward-as-log-return proxy.",
                    reward_type,
                )

        if strategy_simple_returns.size == 0:
            proxy_log_returns = (
                rollout["next", "reward"].detach().cpu().reshape(-1).numpy()[:max_steps]
            )
            strategy_simple_returns = np.exp(proxy_log_returns) - 1.0
            logger.warning(
                "Evaluation metrics fallback is using reward stream as log-return proxy; "
                "return/CAGR metrics may be invalid when reward_type=%s.",
                reward_type,
            )

    if benchmark_price_column in df_prices.columns:
        benchmark_series = df_prices[benchmark_price_column]
    elif "close" in df_prices.columns:
        benchmark_series = df_prices["close"]
        logger.warning(
            "Evaluation report price column '%s' missing; falling back to 'close'.",
            benchmark_price_column,
        )
    else:
        raise ValueError(
            "Evaluation report requires env.price_column or 'close' in dataframe."
        )

    benchmark_log_returns = (
        np.log(benchmark_series / benchmark_series.shift(1)).fillna(0).to_numpy()
    )[:max_steps]
    benchmark_simple_returns = np.exp(benchmark_log_returns) - 1.0

    is_portfolio = backend == "tradingenv"
    actions = _extract_action_array(rollout, is_portfolio=is_portfolio)
    timeframe = getattr(getattr(config, "data", None), "timeframe", "1d")
    periods_per_year = periods_per_year_from_timeframe(timeframe)

    return build_metric_report(
        strategy_simple_returns=strategy_simple_returns,
        benchmark_simple_returns=benchmark_simple_returns,
        actions=actions,
        periods_per_year=periods_per_year,
        risk_free_rate_annual=0.0,
    )
