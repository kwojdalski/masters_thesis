"""Strategy evaluator service - decoupled from training logic.

This module provides a pure evaluation service that can:
- Evaluate policies on data splits
- Extract returns (NLV-based or reward-based)
- Compute financial metrics
- Generate plots

All without coupling to training, MLflow, or specific algorithm details.
"""

from __future__ import annotations

import gc
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

    @classmethod
    def from_experiment_config(cls, config: "Any") -> "StrategyEvaluatorConfig":
        """Project an ExperimentConfig to the narrow evaluation config.

        This factory makes the actual evaluation dependencies explicit and
        allows standalone evaluation without a full ExperimentConfig.
        """
        from trading_rl.evaluation.report import periods_per_year_from_timeframe

        env = getattr(config, "env", None)
        data = getattr(config, "data", None)
        benchmarks_cfg = getattr(config, "benchmarks", None)

        price_column = getattr(env, "price_column", None) or "close"
        timeframe = getattr(data, "timeframe", "1d") or "1d"
        periods_per_year = periods_per_year_from_timeframe(timeframe) or 252

        benchmark_set: frozenset[BenchmarkName]
        if benchmarks_cfg is not None and hasattr(benchmarks_cfg, "enabled_set"):
            benchmark_set = benchmarks_cfg.enabled_set
        else:
            benchmark_set = frozenset({BenchmarkName.BUY_AND_HOLD})

        eval_env = EvaluatorEnvConfig(
            name=getattr(env, "name", ""),
            positions=getattr(env, "positions", None),
            trading_fees=getattr(env, "trading_fees", 0.0),
            borrow_interest_rate=getattr(env, "borrow_interest_rate", 0.0),
            initial_portfolio_value=getattr(env, "initial_portfolio_value", 10_000.0),
            price_column=price_column,
        )

        return cls(
            reward_type=getattr(env, "reward_type", RewardType.LOG_RETURN),
            backend=getattr(env, "backend", EnvBackend.TRADINGENV),
            price_column=price_column,
            periods_per_year=periods_per_year,
            env=eval_env,
            benchmarks=benchmark_set,
        )


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

        # Profiling: cumulative totals across full rollout
        _cb_time_total = [0.0]
        _cb_samples: list[float] = []
        _cb_reward_time = [0.0]
        _cb_action_time = [0.0]
        _cb_done_time = [0.0]
        _inter_cb_total = [0.0]   # time outside callback (env step + policy)
        _pol_time_total = [0.0]   # policy forward pass only
        _last_cb_end: list[float | None] = [None]

        # Wrap the policy to measure forward-pass time separately from env step.
        # env_time per step = inter_cb_time - pol_time  (both measured over the same window)
        _original_policy = self.policy

        class _TimedPolicy:
            def __call__(self_inner, td: Any) -> Any:
                t = _time.monotonic()
                result = _original_policy(td)
                elapsed = _time.monotonic() - t
                _pol_time_total[0] += elapsed
                _fine_pol[0] += elapsed
                return result

        timed_policy = _TimedPolicy()

        _fine_interval = max(1, max_steps // 100)   # 1% intervals for compact trace
        _coarse_interval = max(1, max_steps // 10)  # 10% intervals for full stats
        _step_counter = [0]
        _t_last_fine = [_time.monotonic()]
        _t_last_coarse = [_time.monotonic()]
        # Coarse accumulators (reset each 10%)
        _r_sum = [0.0]; _r_min = [float("inf")]; _r_max = [float("-inf")]; _r_n = [0]
        _a_sum = [0.0]; _a_long = [0]; _a_short = [0]; _a_n = [0]
        _ep_done = [0]
        _cum_reward = [0.0]
        # Fine accumulators (reset each 1%)
        _fine_cb = [0.0]; _fine_inter = [0.0]; _fine_pol = [0.0]
        _fine_reward = [0.0]; _fine_action = [0.0]; _fine_done = [0.0]

        def _progress_cb(env: Any, td: Any) -> None:  # noqa: ARG001
            cb_start = _time.monotonic()
            # Inter-callback time = env step + policy forward pass
            if _last_cb_end[0] is not None:
                inter_elapsed = cb_start - _last_cb_end[0]
                _inter_cb_total[0] += inter_elapsed
                _fine_inter[0] += inter_elapsed
            _step_counter[0] += 1

            # 1. Reward extraction
            t_reward = _time.monotonic()
            try:
                r = float(td["next", "reward"].mean().item())
                _cum_reward[0] += r
                _r_sum[0] += r
                _r_n[0] += 1
                if r < _r_min[0]:
                    _r_min[0] = r
                if r > _r_max[0]:
                    _r_max[0] = r
            except Exception as exc:  # noqa: BLE001
                logger.debug("_progress_cb reward extraction failed step={} err={}", _step_counter[0], exc)
            reward_elapsed = _time.monotonic() - t_reward
            _cb_reward_time[0] += reward_elapsed
            _fine_reward[0] += reward_elapsed

            # 2. Action extraction
            t_action = _time.monotonic()
            try:
                a = float(td["action"].float().mean().item())
                _a_sum[0] += a
                _a_n[0] += 1
                if a > 0:
                    _a_long[0] += 1
                elif a < 0:
                    _a_short[0] += 1
            except Exception as exc:  # noqa: BLE001
                logger.debug("_progress_cb action extraction failed step={} err={}", _step_counter[0], exc)
            action_elapsed = _time.monotonic() - t_action
            _cb_action_time[0] += action_elapsed
            _fine_action[0] += action_elapsed

            # 3. Done check
            t_done = _time.monotonic()
            try:
                if bool(td["next", "done"].any().item()):
                    _ep_done[0] += 1
            except Exception as exc:  # noqa: BLE001
                logger.debug("_progress_cb done check failed step={} err={}", _step_counter[0], exc)
            done_elapsed = _time.monotonic() - t_done
            _cb_done_time[0] += done_elapsed
            _fine_done[0] += done_elapsed

            cb_elapsed = _time.monotonic() - cb_start
            _cb_time_total[0] += cb_elapsed
            _fine_cb[0] += cb_elapsed
            if len(_cb_samples) < 1000:
                _cb_samples.append(cb_elapsed)
            _last_cb_end[0] = _time.monotonic()

            # Fine-grain: every 1% — compact per-step timing breakdown
            step = _step_counter[0]
            if step % _fine_interval == 0 and step % _coarse_interval != 0:
                now = _last_cb_end[0]
                pct = 100.0 * step / max_steps
                interval_s = now - _t_last_fine[0]
                steps_s = _fine_interval / max(interval_s, 1e-9)
                n = _fine_interval
                inter_us = _fine_inter[0] / n * 1e6
                pol_us   = _fine_pol[0]   / n * 1e6
                env_us   = max(0.0, inter_us - pol_us)
                cb_us    = _fine_cb[0]    / n * 1e6
                r_us     = _fine_reward[0] / n * 1e6
                a_us     = _fine_action[0] / n * 1e6
                d_us     = _fine_done[0]   / n * 1e6
                total_us = inter_us + cb_us
                env_pct = 100.0 * env_us  / total_us if total_us > 0 else 0.0
                pol_pct = 100.0 * pol_us  / total_us if total_us > 0 else 0.0
                logger.trace(
                    "rollout: pct={}% spd={}/s | step_us={:.0f} env_us={:.0f} env_pct={:.1f}% pol_us={:.0f} pol_pct={:.1f}% cb_us={:.0f} r_us={:.0f} a_us={:.0f} d_us={:.0f}",
                    int(pct), int(steps_s),
                    total_us, env_us, env_pct, pol_us, pol_pct, cb_us, r_us, a_us, d_us,
                )
                _t_last_fine[0] = now
                _fine_cb[0] = 0.0; _fine_inter[0] = 0.0; _fine_pol[0] = 0.0
                _fine_reward[0] = 0.0; _fine_action[0] = 0.0; _fine_done[0] = 0.0

            # Coarse: every 10% — full reward/action stats
            if step % _coarse_interval == 0:
                now = _last_cb_end[0]
                pct = 100.0 * step / max_steps
                elapsed = now - _t
                eta = (elapsed / step) * (max_steps - step)
                interval_s = now - _t_last_coarse[0]
                steps_s = _coarse_interval / max(interval_s, 1e-9)

                r_min  = _r_min[0] if _r_n[0] else float("nan")
                r_max  = _r_max[0] if _r_n[0] else float("nan")
                a_mean = _a_sum[0] / _a_n[0] if _a_n[0] else float("nan")
                long_pct  = 100.0 * _a_long[0] / _a_n[0] if _a_n[0] else float("nan")
                short_pct = 100.0 * _a_short[0] / _a_n[0] if _a_n[0] else float("nan")

                # Memory tracking
                gc.collect()
                torch_mb = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0

                # Detailed timing breakdown (first 10% interval only)
                if step == _coarse_interval:
                    pol_us   = _pol_time_total[0] / step * 1e6
                    inter_us = _inter_cb_total[0] / step * 1e6
                    env_us   = max(0.0, inter_us - pol_us)
                    cb_us    = _cb_time_total[0]  / step * 1e6
                    total_us = elapsed / step * 1e6
                    env_pct  = 100.0 * env_us / total_us if total_us > 0 else 0.0
                    pol_pct  = 100.0 * pol_us / total_us if total_us > 0 else 0.0
                    cb_pct_v = 100.0 * cb_us  / total_us if total_us > 0 else 0.0
                    logger.debug(
                        "rollout profiling: steps={} spd={}/s | "
                        "step_us={:.0f} env_us={:.0f} env_pct={:.1f}% pol_us={:.0f} pol_pct={:.1f}% cb_us={:.0f} cb_pct={:.1f}% | "
                        "cb: r_us={:.0f} a_us={:.0f} d_us={:.0f} | mem_mb={:.0f}",
                        step, int(steps_s),
                        total_us, env_us, env_pct, pol_us, pol_pct, cb_us, cb_pct_v,
                        _cb_reward_time[0] / step * 1e6,
                        _cb_action_time[0] / step * 1e6,
                        _cb_done_time[0]   / step * 1e6,
                        torch_mb,
                    )

                logger.trace(
                    "rollout: step={}/{} pct={}% elapsed_s={:.1f} eta_s={:.1f} spd={}/s | "
                    "r={:.4f} r_min={:.5f} r_max={:.5f} | a={:.2f} long_pct={:.0f}% short_pct={:.0f}%",
                    step, max_steps, int(pct),
                    elapsed, eta, int(steps_s),
                    _cum_reward[0], r_min, r_max,
                    a_mean, long_pct, short_pct,
                )
                _t_last_coarse[0] = now
                _t_last_fine[0] = now
                # Reset coarse accumulators
                _r_sum[0] = 0.0; _r_min[0] = float("inf"); _r_max[0] = float("-inf"); _r_n[0] = 0
                _a_sum[0] = 0.0; _a_long[0] = 0; _a_short[0] = 0; _a_n[0] = 0
                _ep_done[0] = 0
                # Reset fine accumulators (coarse boundary doubles as fine)
                _fine_cb[0] = 0.0; _fine_inter[0] = 0.0; _fine_pol[0] = 0.0
                _fine_reward[0] = 0.0; _fine_action[0] = 0.0; _fine_done[0] = 0.0

        # gym_trading_env's History uses dtype='O' (Python object array), accumulating
        # ~82 Python object references per step. Python's gen-2 GC scans all live objects
        # every ~300 steps; at step k that scan is O(k), producing O(k) overhead per step.
        # Disabling auto-GC eliminates this. Manual gc.collect() calls in the callback
        # at 10% intervals still run (gc.collect() ignores the disabled flag) and bound
        # memory growth to one episode's worth of objects between collections.
        gc.collect()
        gc.disable()
        logger.debug("rollout start max_steps={}", max_steps)
        _t = _time.monotonic()
        try:
            with torch.no_grad():
                try:
                    with set_exploration_type(InteractionType.MODE):
                        rollout = env.rollout(max_steps=max_steps, policy=timed_policy, callback=_progress_cb)
                except (NotImplementedError, RuntimeError) as exc:
                    if not (
                        isinstance(exc, NotImplementedError)
                        or "does not have a mode" in str(exc)
                        or "analytical mode" in str(exc).lower()
                    ):
                        raise
                    # Fallback for distributions without analytical mode
                    with set_exploration_type(InteractionType.DETERMINISTIC):
                        rollout = env.rollout(max_steps=max_steps, policy=timed_policy, callback=_progress_cb)
        finally:
            gc.enable()
            gc.collect()
        actual_steps = rollout.shape[0] if rollout.ndim > 0 else 1
        elapsed_total = _time.monotonic() - _t
        n = actual_steps if actual_steps > 0 else 1
        pol_avg_us   = _pol_time_total[0]  / n * 1e6
        inter_avg_us = _inter_cb_total[0]  / n * 1e6
        env_avg_us   = max(0.0, inter_avg_us - pol_avg_us)
        cb_avg_us    = _cb_time_total[0]   / n * 1e6
        env_pct  = 100.0 * env_avg_us  / elapsed_total * n / 1e6 if elapsed_total > 0 else 0
        pol_pct  = 100.0 * pol_avg_us  / elapsed_total * n / 1e6 if elapsed_total > 0 else 0
        cb_pct   = 100.0 * cb_avg_us   / elapsed_total * n / 1e6 if elapsed_total > 0 else 0
        logger.info(
            "rollout done: steps={}/{} elapsed_s={:.1f} spd={}/s | "
            "env_us={:.0f} env_pct={:.1f}% pol_us={:.0f} pol_pct={:.1f}% cb_us={:.0f} cb_pct={:.1f}%",
            actual_steps, max_steps, elapsed_total, int(actual_steps / elapsed_total),
            env_avg_us, env_pct, pol_avg_us, pol_pct, cb_avg_us, cb_pct,
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
        logger.debug("evaluate_split split={} max_steps={} df_rows={}", split, max_steps, len(df))

        # Use the caller-provided environment when available. Training code
        # already builds split-specific envs from the full ExperimentConfig.
        env = env if env is not None else self._build_env(df)

        # Run deterministic rollout
        _t = time.monotonic()
        rollout = self._run_rollout(env, max_steps)
        logger.trace("evaluate_split: rollout elapsed={:.2f}s steps={}", time.monotonic() - _t, max_steps)

        # Extract returns
        _t = time.monotonic()
        return_series = self._extract_return_series(env, rollout, max_steps)
        logger.trace("evaluate_split: extract_returns elapsed={:.2f}s", time.monotonic() - _t)
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
        logger.trace("evaluate_split: compute_metrics elapsed={:.2f}s", time.monotonic() - _t)

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
                logger.trace("evaluate_split: build_rollout_plot_data elapsed={:.2f}s", time.monotonic() - _t)
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
                        logger.trace("evaluate_split: portfolio_value_plot elapsed={:.2f}s", time.monotonic() - _t)
                    except Exception:
                        logger.opt(exception=True).warning("evaluate_split: portfolio value plot failed")

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
