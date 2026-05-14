"""Strategy evaluator service - decoupled from training logic.

This module provides a pure evaluation service that can:
- Evaluate policies on data splits
- Extract returns (NLV-based or reward-based)
- Compute financial metrics
- Generate plots

All without coupling to training, MLflow, or specific algorithm details.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import torch

from trading_rl.constants import RewardType
from trading_rl.evaluation.metrics import build_metric_report
from trading_rl.evaluation.plots import compare_rollouts
from trading_rl.evaluation.returns import extract_tradingenv_returns


@dataclass(frozen=True)
class EnvConfig:
    """Environment configuration for evaluation."""

    name: str = ""  # Environment name
    positions: list[int] | None = None  # Will use TradePosition default if None
    mode: str = "mft"  # Feature regime mode
    trading_fees: float = 0.0
    borrow_interest_rate: float = 0.0
    initial_portfolio_value: float = 10000.0
    price_column: str = "close"  # Price column for environment


@dataclass(frozen=True)
class EvaluationConfig:
    """Simplified configuration for evaluation.

    Decoupled from full ExperimentConfig - contains only what's needed
    for evaluating a policy on price data.
    """

    reward_type: str = "log_return"
    backend: str = "tradingenv"
    max_steps: int | None = None  # Resolve from DF if None
    price_column: str = "close"
    enable_plots: bool = True
    enable_metrics: bool = True
    periods_per_year: int = 252
    env: EnvConfig = field(default_factory=EnvConfig)  # Environment configuration


@dataclass(frozen=True)
class SplitEvaluationResult:
    """Results from evaluating a policy on one data split.

    Pure data structures - no MLflow artifacts or plotting side effects.
    """

    final_reward: float
    last_positions: list[Any]
    simple_returns: np.ndarray
    cumulative_returns: np.ndarray | None = None  # For debugging
    metrics: dict[str, float] | None = None
    plots: dict[str, Any] | None = None  # Raw plot objects


class StrategyEvaluator:
    """Pure evaluation service for trading strategies.

    Decoupled from training logic. Can be used:
    - During training for periodic evaluation
    - After training for final evaluation
    - Standalone for inference-only scenarios
    - In unit tests (mocking only env and policy)
    """

    def __init__(
        self,
        env_factory: Callable[[pd.DataFrame, EvaluationConfig], Any],
        policy: Any,
        config: EvaluationConfig,
    ):
        """Initialize evaluator.

        Args:
            env_factory: Function that creates environments from dataframes.
                Signature: (df: pd.DataFrame, config: EvaluationConfig) -> env
            policy: The trained policy to evaluate (actor, Q-network, etc.)
            config: Evaluation configuration
        """
        self.env_factory = env_factory
        self.policy = policy
        self.config = config

    def _build_env(self, df: pd.DataFrame) -> Any:
        """Build evaluation environment from factory.

        Args:
            df: Price data for the environment

        Returns:
            Environment instance
        """
        return self.env_factory(df, self.config)

    def _run_rollout(self, env: Any, max_steps: int) -> Any:
        """Run deterministic rollout with given policy.

        Args:
            env: Environment to run rollout on
            max_steps: Maximum number of steps

        Returns:
            Rollout TensorDict
        """
        from tensordict.nn import InteractionType
        from torchrl.envs.utils import set_exploration_type

        with torch.no_grad():
            try:
                with set_exploration_type(InteractionType.MODE):
                    return env.rollout(max_steps=max_steps, policy=self.policy)
            except RuntimeError:
                # Fallback for distributions without analytical mode
                with set_exploration_type(InteractionType.DETERMINISTIC):
                    return env.rollout(max_steps=max_steps, policy=self.policy)

    def _extract_returns(
        self, env: Any, rollout: Any, max_steps: int
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Extract strategy returns based on reward type and backend.

        Args:
            env: Environment (for NLV extraction)
            rollout: Rollout TensorDict (for reward extraction)
            max_steps: Number of steps

        Returns:
            (simple_returns, cumulative_returns) tuple
        """
        # Extract NLV-based returns for TradingEnv backend
        if self.config.backend.lower() == "tradingenv":
            cumulative = extract_tradingenv_returns(env, max_steps)
            if cumulative is not None and len(cumulative) > 1:
                step_log = np.diff(cumulative)
                simple_returns = np.exp(step_log) - 1.0
                return simple_returns, cumulative

        # Extract reward-stream returns for log_return
        if self.config.reward_type == RewardType.LOG_RETURN:
            rewards = rollout["next", "reward"].detach().cpu().numpy()[:max_steps]
            simple_returns = np.exp(rewards) - 1.0
            return simple_returns, None

        # Fallback: compute from reward stream (not ideal, preserves behavior)
        rewards = rollout["next", "reward"].detach().cpu().numpy()[:max_steps]
        simple_returns = np.exp(rewards) - 1.0
        return simple_returns, None

    def _compute_metrics(self, simple_returns: np.ndarray, df: pd.DataFrame) -> dict[str, float]:
        """Compute financial metrics from strategy returns.

        Args:
            simple_returns: Strategy simple returns
            df: DataFrame with price column for benchmark

        Returns:
            Dictionary of metrics (same as build_metric_report output)
        """
        # Get price column for benchmark comparison
        price_column = self.config.price_column
        if price_column not in df.columns and "close" in df.columns:
            price_column = "close"

        if price_column not in df.columns:
            return {}  # No benchmark possible

        price_series = df[price_column]

        # Compute benchmark returns
        if len(price_series) > 1:
            benchmark_simple_returns = (
                price_series.pct_change().iloc[1:].to_numpy(dtype=float)
            )
        else:
            benchmark_simple_returns = np.array([])

        # Build full metric report
        return build_metric_report(
            strategy_simple_returns=simple_returns,
            benchmark_simple_returns=benchmark_simple_returns,
            actions=None,
            periods_per_year=self.config.periods_per_year,
            risk_free_rate_annual=0.0,
        )

    def _extract_last_positions(self, actions: Any, max_steps: int) -> list[Any]:
        """Extract final positions from rollout actions.

        Handles both discrete (-1, 0, 1) and continuous (portfolio weights 0-1).

        Args:
            actions: Action tensor or similar
            max_steps: Maximum number of steps

        Returns:
            List of positions (one per step)
        """
        if actions is None:
            return []

        action_tensor = actions.squeeze()

        # Handle continuous portfolio actions
        if action_tensor.ndim > 1 and action_tensor.shape[-1] > 1:
            # Multi-asset: return mean allocation per asset
            return action_tensor.mean(dim=0).tolist()
        else:
            # Single-asset or discrete
            flat_actions = (
                action_tensor.flatten().numpy()
                if hasattr(action_tensor, "flatten")
                else np.array([action_tensor])
            )
            return flat_actions[:max_steps].tolist()

    def evaluate_split(
        self,
        split: str,
        df: pd.DataFrame,
        env: Any | None = None,
    ) -> SplitEvaluationResult:
        """Evaluate strategy on one data split.

        Args:
            split: Split name ("train", "val", or "test")
            df: DataFrame with OHLCV data for this split
            env: Optional pre-built environment. Supplying this avoids rebuilding
                a different backend or observation shape from a reduced config.

        Returns:
            SplitEvaluationResult with returns, metrics, and plots
        """
        if len(df) < 2:
            # Skip tiny splits (need at least 2 for pct_change)
            return SplitEvaluationResult(
                final_reward=float("nan"),
                last_positions=[],
                simple_returns=np.array([]),
                metrics={},
                plots=None,
            )

        max_steps = self.config.max_steps or len(df) - 1

        # Use the caller-provided environment when available. Training code
        # already builds split-specific envs from the full ExperimentConfig.
        env = env if env is not None else self._build_env(df)

        # Run deterministic rollout
        rollout = self._run_rollout(env, max_steps)

        # Extract returns
        simple_returns, cumulative_returns = self._extract_returns(env, rollout, max_steps)

        # Compute metrics
        metrics = self._compute_metrics(simple_returns, df) if self.config.enable_metrics else None

        # Generate plots
        plots = None
        if self.config.enable_plots:
            is_portfolio = self.config.backend.lower() == "tradingenv"
            reward_plot, action_plot = compare_rollouts([rollout], max_steps, is_portfolio=is_portfolio)
            plots = {
                "reward_plot": reward_plot,
                "action_plot": action_plot,
            }

        # Extract last positions
        actions = rollout.get("action", None)
        last_positions = self._extract_last_positions(actions, max_steps) if actions is not None else []

        return SplitEvaluationResult(
            final_reward=float(rollout["next", "reward"].sum().item()),
            last_positions=last_positions,
            simple_returns=simple_returns,
            cumulative_returns=cumulative_returns,
            metrics=metrics,
            plots=plots,
        )

    def evaluate_all_splits(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> dict[str, SplitEvaluationResult]:
        """Evaluate strategy on all data splits.

        Args:
            train_df: Training data
            val_df: Validation data
            test_df: Test data

        Returns:
            Dict mapping split names to SplitEvaluationResult
        """
        results = {}

        for split, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            results[split] = self.evaluate_split(split, df)

        return results
