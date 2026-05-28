"""Strategy evaluator service - decoupled from training logic.

This module provides a pure evaluation service that can:
- Evaluate policies on data splits
- Extract returns (NLV-based or reward-based)
- Compute financial metrics
- Generate plots

All without coupling to training, MLflow, or specific algorithm details.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import torch

from logger import get_logger
from trading_rl.constants import BenchmarkName, EnvBackend, EnvMode, RewardType
from trading_rl.evaluation.metrics import MetricReport, build_metric_report
from trading_rl.evaluation.plots import (
    build_equity_plot_data,
    build_rollout_plot_data,
    plot_actions,
    plot_equity_curve,
    plot_rewards,
)
from trading_rl.evaluation.returns import (
    ReturnKind,
    ReturnSeries,
    RewardSeries,
    extract_tradingenv_return_series,
)

logger = get_logger(__name__)


@dataclass(frozen=True)
class EvaluatorEnvConfig:
    """Environment configuration for evaluation."""

    name: str = ""  # Environment name
    positions: list[int] | None = None  # Will use TradePosition default if None
    mode: EnvMode = EnvMode.MFT
    trading_fees: float = 0.0
    borrow_interest_rate: float = 0.0
    initial_portfolio_value: float = 10000.0
    price_column: str = "close"  # Price column for environment


@dataclass(frozen=True)
class StrategyEvaluatorConfig:
    """Configuration for StrategyEvaluator.

    Decoupled from full ExperimentConfig - contains only what's needed
    for evaluating a policy on price data.
    """

    reward_type: str = RewardType.LOG_RETURN
    backend: str = EnvBackend.TRADINGENV
    max_steps: int | None = None  # Resolve from DF if None
    price_column: str = "close"
    enable_plots: bool = True
    enable_metrics: bool = True
    periods_per_year: int = 252
    env: EvaluatorEnvConfig = field(default_factory=EvaluatorEnvConfig)  # Environment configuration
    max_plot_points: int | None = None  # Cap the number of plotted points per series; None = plot all
    show_allocation_ma: bool = True  # Overlay moving-average line on Portfolio Allocation plot
    allocation_ma_window: int = 500  # Rolling window size for the allocation MA
    eval_plots: tuple[str, ...] = ("rewards", "positions", "portfolio_value")  # Which plots to generate
    training_steps: int | None = None  # Steps the policy was trained for (shown in captions)
    training_episodes: int | None = None  # Episodes the policy was trained for (shown in captions)
    benchmarks: frozenset[BenchmarkName] = frozenset({BenchmarkName.BUY_AND_HOLD})
    show_reward_benchmarks: bool = False  # Show benchmark reward curves on the reward plot


@dataclass(frozen=True)
class SplitEvaluationResult:
    """Results from evaluating a policy on one data split.

    Pure data structures - no MLflow artifacts or plotting side effects.
    """

    final_reward: float
    last_positions: list[Any]
    simple_returns: np.ndarray
    rollout: Any | None = None
    cumulative_returns: np.ndarray | None = None  # For debugging
    return_series: ReturnSeries | None = None
    metrics: MetricReport | None = None
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
        env_factory: Callable[[pd.DataFrame, StrategyEvaluatorConfig], Any],
        policy: Any,
        config: StrategyEvaluatorConfig,
    ):
        """Initialize evaluator.

        Args:
            env_factory: Function that creates environments from dataframes.
                Signature: (df: pd.DataFrame, config: StrategyEvaluatorConfig) -> env
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
        import time as _time

        from tensordict.nn import InteractionType
        from torchrl.envs.utils import set_exploration_type

        logger.debug("rollout start max_steps=%d", max_steps)
        _t = _time.monotonic()
        with torch.no_grad():
            try:
                with set_exploration_type(InteractionType.MODE):
                    rollout = env.rollout(max_steps=max_steps, policy=self.policy)
            except (NotImplementedError, RuntimeError) as exc:
                if not (
                    isinstance(exc, NotImplementedError)
                    or "does not have a mode" in str(exc)
                    or "analytical mode" in str(exc).lower()
                ):
                    raise
                # Fallback for distributions without analytical mode
                with set_exploration_type(InteractionType.DETERMINISTIC):
                    rollout = env.rollout(max_steps=max_steps, policy=self.policy)
        actual_steps = rollout.shape[0] if rollout.ndim > 0 else 1
        logger.debug(
            "rollout done requested=%d actual=%d elapsed=%.2fs",
            max_steps, actual_steps, _time.monotonic() - _t,
        )
        return rollout

    def _extract_return_series(
        self, env: Any, rollout: Any, max_steps: int
    ) -> ReturnSeries | None:
        """Extract strategy returns based on reward type and backend.

        Args:
            env: Environment (for NLV extraction)
            rollout: Rollout TensorDict (for reward extraction)
            max_steps: Number of steps

        Returns:
            ReturnSeries when a true return path is available.
        """
        # Extract NLV-based returns for TradingEnv backend
        if self.config.backend.lower() == EnvBackend.TRADINGENV:
            series = extract_tradingenv_return_series(env, max_steps)
            if series is not None:
                return series

        # Extract reward-stream returns for log_return
        if self.config.reward_type == RewardType.LOG_RETURN:
            rewards = rollout["next", "reward"].detach().cpu().numpy()[:max_steps]
            return RewardSeries(rewards, self.config.reward_type).to_return_series()

        return None

    def _compute_metrics(
        self,
        simple_returns: np.ndarray,
        df: pd.DataFrame,
        positions: list[Any] | None = None,
    ) -> MetricReport:
        """Compute financial metrics from strategy returns.

        Args:
            simple_returns: Strategy simple returns
            df: DataFrame with price column for benchmark
            positions: Per-step position values (portfolio weights or discrete
                actions) used to compute pct_long / pct_short.

        Returns:
            MetricReport with all computed metrics.
        """
        # Get price column for benchmark comparison
        price_column = self.config.price_column
        if price_column not in df.columns and "close" in df.columns:
            price_column = "close"

        if price_column not in df.columns:
            return MetricReport.all_nan()

        price_series = df[price_column]

        # Compute benchmark returns
        if len(price_series) > 1:
            benchmark_simple_returns = (
                price_series.pct_change().iloc[1:].to_numpy(dtype=float)
            )
        else:
            benchmark_simple_returns = np.array([])

        actions_array = np.asarray(positions, dtype=float) if positions else None

        # Build full metric report
        return build_metric_report(
            strategy_simple_returns=simple_returns,
            benchmark_simple_returns=benchmark_simple_returns,
            actions=actions_array,
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

        action_tensor = actions.detach().cpu() if hasattr(actions, "detach") else actions
        is_portfolio = str(self.config.backend).lower() == EnvBackend.TRADINGENV

        # Handle continuous portfolio actions
        if is_portfolio:
            flat_actions = np.asarray(action_tensor, dtype=float).reshape(-1)
            return flat_actions[:max_steps].tolist()

        if action_tensor.ndim > 1 and action_tensor.shape[-1] > 1:
            action_tensor = action_tensor.argmax(dim=-1)

        flat_actions = np.asarray(action_tensor, dtype=float).reshape(-1)[:max_steps]
        positions = self.config.env.positions
        if positions and flat_actions.size:
            indices = flat_actions.astype(int)
            if np.allclose(flat_actions, indices) and np.all(
                (0 <= indices) & (indices < len(positions))
            ):
                return [positions[int(i)] for i in indices]
        return flat_actions.tolist()

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
                metrics=MetricReport.all_nan(),
                plots=None,
            )

        max_steps = self.config.max_steps or len(df) - 1
        logger.debug("evaluate_split split=%s max_steps=%d df_rows=%d", split, max_steps, len(df))

        # Use the caller-provided environment when available. Training code
        # already builds split-specific envs from the full ExperimentConfig.
        env = env if env is not None else self._build_env(df)

        # Run deterministic rollout
        _t = time.monotonic()
        rollout = self._run_rollout(env, max_steps)
        logger.debug("evaluate_split: rollout elapsed=%.2fs steps=%d", time.monotonic() - _t, max_steps)

        # Extract returns
        _t = time.monotonic()
        return_series = self._extract_return_series(env, rollout, max_steps)
        logger.debug("evaluate_split: extract_returns elapsed=%.2fs", time.monotonic() - _t)
        if return_series is None:
            simple_returns = np.array([], dtype=float)
            cumulative_returns = None
        else:
            simple_returns = return_series.to_simple().values
            cumulative_returns = return_series.to_cumulative_log(include_initial=True).values

        # Extract last positions before metrics so we can pass them in
        actions = rollout.get("action", None)
        last_positions = self._extract_last_positions(actions, max_steps) if actions is not None else []

        # Compute metrics (pass positions for pct_long / pct_short)
        _t = time.monotonic()
        metrics = (
            self._compute_metrics(simple_returns, df, last_positions)
            if self.config.enable_metrics
            else None
        )
        logger.debug("evaluate_split: compute_metrics elapsed=%.2fs", time.monotonic() - _t)

        # Generate plots
        plots = None
        if self.config.enable_plots:
            enabled = set(self.config.eval_plots)
            plots = {}

            if "rewards" in enabled or "positions" in enabled:
                _t = time.monotonic()
                is_portfolio = self.config.backend.lower() == EnvBackend.TRADINGENV
                rollout_data = build_rollout_plot_data(
                    [rollout], max_steps, is_portfolio=is_portfolio, df=df,
                    reward_type=self.config.reward_type,
                    max_plot_points=self.config.max_plot_points,
                    show_allocation_ma=self.config.show_allocation_ma,
                    allocation_ma_window=self.config.allocation_ma_window,
                    training_steps=self.config.training_steps,
                    training_episodes=self.config.training_episodes,
                    show_benchmarks=self.config.show_reward_benchmarks,
                    benchmark_price_column=self.config.price_column,
                )
                logger.debug("evaluate_split: build_rollout_plot_data elapsed=%.2fs", time.monotonic() - _t)
                plots["_rollout_plot_data"] = rollout_data
                if "rewards" in enabled:
                    plots["reward_plot"] = plot_rewards(
                        rollout_data["rewards"],
                        training_steps=rollout_data["training_steps"],
                        training_episodes=rollout_data["training_episodes"],
                        reward_type=rollout_data["reward_type"],
                        stride=rollout_data["stride"],
                        n_obs=rollout_data["n_obs"],
                        date_str=rollout_data["date_str"],
                    )
                if "positions" in enabled:
                    plots["action_plot"] = plot_actions(
                        rollout_data["actions"],
                        df_ma=rollout_data.get("actions_ma"),
                        is_portfolio=rollout_data["is_portfolio"],
                        training_steps=rollout_data["training_steps"],
                        training_episodes=rollout_data["training_episodes"],
                        stride=rollout_data["stride"],
                        n_obs=rollout_data["n_obs"],
                        date_str=rollout_data["date_str"],
                        allocation_ma_window=rollout_data.get("allocation_ma_window") or self.config.allocation_ma_window,
                    )

            if "portfolio_value" in enabled:
                plot_series = return_series or (
                    ReturnSeries(simple_returns, ReturnKind.SIMPLE) if simple_returns.size else None
                )
                if plot_series is not None:
                    try:
                        _t = time.monotonic()
                        equity_data = build_equity_plot_data(
                            None,
                            max_steps,
                            df_prices=df,
                            actual_returns_list=[plot_series],
                            initial_portfolio_value=self.config.env.initial_portfolio_value,
                            benchmark_price_column=self.config.price_column,
                            reward_type=self.config.reward_type,
                            max_plot_points=self.config.max_plot_points,
                            training_steps=self.config.training_steps,
                            training_episodes=self.config.training_episodes,
                            benchmarks=self.config.benchmarks,
                        )
                        plots["_equity_plot_data"] = equity_data
                        plots["portfolio_value_plot"] = plot_equity_curve(
                            equity_data["returns"],
                            initial_portfolio_value=equity_data["initial_portfolio_value"],
                            policy_mode=equity_data["policy_mode"],
                            training_steps=equity_data["training_steps"],
                            training_episodes=equity_data["training_episodes"],
                            date_str=equity_data["date_str"],
                            n_obs=equity_data["n_obs"],
                            stride=equity_data["stride"],
                            symbols=equity_data["symbols"],
                            n_total_symbols=equity_data["n_total_symbols"],
                        )
                        logger.debug("evaluate_split: portfolio_value_plot elapsed=%.2fs", time.monotonic() - _t)
                    except Exception:
                        logger.warning("evaluate_split: portfolio value plot failed", exc_info=True)

        return SplitEvaluationResult(
            final_reward=float(rollout["next", "reward"].sum().item()),
            last_positions=last_positions,
            simple_returns=simple_returns,
            rollout=rollout,
            cumulative_returns=cumulative_returns,
            return_series=return_series,
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
