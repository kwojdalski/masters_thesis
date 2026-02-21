"""Evaluation report generation from deterministic rollouts."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from tensordict.nn import InteractionType
from torchrl.envs.utils import set_exploration_type

from trading_rl.evaluation.metrics import build_metric_report


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
) -> dict[str, float]:
    """Build the 25-metric evaluation report for deterministic policy rollout."""
    with torch.no_grad():
        try:
            with set_exploration_type(InteractionType.MODE):
                rollout = trainer.env.rollout(max_steps=max_steps, policy=trainer.actor)
        except RuntimeError:
            with set_exploration_type(InteractionType.DETERMINISTIC):
                rollout = trainer.env.rollout(max_steps=max_steps, policy=trainer.actor)

    strategy_log_returns = (
        rollout["next", "reward"].detach().cpu().reshape(-1).numpy()[:max_steps]
    )
    strategy_simple_returns = np.exp(strategy_log_returns) - 1.0

    benchmark_log_returns = (
        np.log(df_prices["close"] / df_prices["close"].shift(1)).fillna(0).to_numpy()
    )[:max_steps]
    benchmark_simple_returns = np.exp(benchmark_log_returns) - 1.0

    backend = getattr(config.env, "backend", None) if config is not None else None
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
