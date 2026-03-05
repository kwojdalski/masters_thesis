"""Statistical significance testing for trading strategy equity curves.

This module provides a modular, extensible framework for statistical tests
comparing trading strategy performance against baselines.

Design:
- StatisticalTest: Abstract base class defining test interface
- BootstrapTest: Template for bootstrap-based ratio tests (DRY)
- PermutationTest: Template for permutation-based tests (DRY)
- Concrete tests: TTest, MannWhitneyTest, etc.
- Registry: Type-safe dispatch without string matching

See STATISTICAL_TESTS_README.md for extension guide.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# Utility functions
# ============================================================================


def _safe_div(numerator: float, denominator: float) -> float:
    """Safe division handling zero/nan denominators."""
    if denominator == 0 or np.isnan(denominator):
        return np.nan
    return numerator / denominator


def _sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """Compute Sharpe ratio from returns."""
    if len(returns) == 0:
        return np.nan
    excess_returns = returns - risk_free_rate
    return _safe_div(
        float(np.mean(excess_returns)), float(np.std(excess_returns, ddof=1))
    )


def _sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """Compute Sortino ratio from returns (uses downside deviation only)."""
    if len(returns) == 0:
        return np.nan
    excess_returns = returns - risk_free_rate
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0:
        downside_dev = float(np.std(returns, ddof=1))
    else:
        downside_dev = float(np.std(downside_returns, ddof=1))

    return _safe_div(float(np.mean(excess_returns)), downside_dev)


# ============================================================================
# Abstract base classes
# ============================================================================


class StatisticalTest(ABC):
    """Abstract base class for all statistical significance tests.

    Defines the interface that all tests must implement.
    Subclasses override `name` and `run()` to provide specific test logic.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this test (used in config)."""
        pass

    @abstractmethod
    def run(
        self, strategy_returns: np.ndarray, baseline_returns: np.ndarray, **params
    ) -> dict[str, Any]:
        """Run the statistical test.

        Args:
            strategy_returns: Strategy simple returns
            baseline_returns: Baseline simple returns
            **params: Test-specific parameters from config

        Returns:
            Dictionary with test results including:
                - test_name: str
                - p_value: float
                - significant: bool
                - test-specific metrics
        """
        pass


class BootstrapTest(StatisticalTest, ABC):
    """Template for bootstrap-based ratio tests.

    Eliminates code duplication between Sharpe, Sortino, and future ratio tests.
    Subclasses only need to implement `compute_metric()`.
    """

    @abstractmethod
    def compute_metric(self, returns: np.ndarray) -> float:
        """Compute the metric to bootstrap (e.g., Sharpe ratio, Sortino ratio)."""
        pass

    @property
    def metric_name(self) -> str:
        """Name of the metric (e.g., 'sharpe', 'sortino')."""
        # Default: extract from test name by removing '_bootstrap'
        return self.name.replace("_bootstrap", "")

    def run(
        self,
        strategy_returns: np.ndarray,
        baseline_returns: np.ndarray,
        n_bootstrap: int = 10000,
        confidence_level: float = 0.95,
        seed: int | None = None,
        **params,
    ) -> dict[str, Any]:
        """Run bootstrap test with confidence intervals."""
        if seed is not None:
            np.random.seed(seed)

        # Observed metrics
        strategy_metric = self.compute_metric(strategy_returns)
        baseline_metric = self.compute_metric(baseline_returns)
        observed_diff = strategy_metric - baseline_metric

        # Optional pre-run validation (can be overridden)
        self._validate_metrics(strategy_metric, baseline_metric)

        # Bootstrap distribution
        strategy_metrics = []
        baseline_metrics = []
        diff_metrics = []

        n_strategy = len(strategy_returns)
        n_baseline = len(baseline_returns)

        for _ in range(n_bootstrap):
            # Resample with replacement
            strategy_sample = np.random.choice(
                strategy_returns, size=n_strategy, replace=True
            )
            baseline_sample = np.random.choice(
                baseline_returns, size=n_baseline, replace=True
            )

            s_metric = self.compute_metric(strategy_sample)
            b_metric = self.compute_metric(baseline_sample)

            strategy_metrics.append(s_metric)
            baseline_metrics.append(b_metric)
            diff_metrics.append(s_metric - b_metric)

        strategy_metrics = np.array(strategy_metrics)
        baseline_metrics = np.array(baseline_metrics)
        diff_metrics = np.array(diff_metrics)

        # Remove NaN values
        strategy_metrics = strategy_metrics[~np.isnan(strategy_metrics)]
        baseline_metrics = baseline_metrics[~np.isnan(baseline_metrics)]
        diff_metrics = diff_metrics[~np.isnan(diff_metrics)]

        # Confidence intervals
        alpha = 1 - confidence_level
        strategy_ci_lower = float(np.percentile(strategy_metrics, 100 * alpha / 2))
        strategy_ci_upper = float(
            np.percentile(strategy_metrics, 100 * (1 - alpha / 2))
        )
        baseline_ci_lower = float(np.percentile(baseline_metrics, 100 * alpha / 2))
        baseline_ci_upper = float(
            np.percentile(baseline_metrics, 100 * (1 - alpha / 2))
        )
        diff_ci_lower = float(np.percentile(diff_metrics, 100 * alpha / 2))
        diff_ci_upper = float(np.percentile(diff_metrics, 100 * (1 - alpha / 2)))

        # P-value: proportion of bootstrap diffs with opposite sign
        if observed_diff > 0:
            p_value = np.mean(diff_metrics <= 0)
        else:
            p_value = np.mean(diff_metrics >= 0)

        # Significant if CI doesn't include 0
        significant = not (diff_ci_lower <= 0 <= diff_ci_upper)

        metric = self.metric_name
        return {
            "test_name": self.name,
            f"strategy_{metric}": float(strategy_metric),
            f"baseline_{metric}": float(baseline_metric),
            f"{metric}_difference": float(observed_diff),
            f"strategy_{metric}_ci_lower": strategy_ci_lower,
            f"strategy_{metric}_ci_upper": strategy_ci_upper,
            f"baseline_{metric}_ci_lower": baseline_ci_lower,
            f"baseline_{metric}_ci_upper": baseline_ci_upper,
            "difference_ci_lower": diff_ci_lower,
            "difference_ci_upper": diff_ci_upper,
            "p_value": float(p_value),
            "confidence_level": confidence_level,
            "n_bootstrap": n_bootstrap,
            "significant": significant,
        }

    def _validate_metrics(self, strategy_metric: float, baseline_metric: float) -> None:
        """Optional validation hook for subclasses."""
        pass


class PermutationTest(StatisticalTest, ABC):
    """Template for permutation-based tests.

    Eliminates code duplication for permutation tests.
    Subclasses implement `compute_test_statistic()`.
    """

    @abstractmethod
    def compute_test_statistic(
        self, strategy_returns: np.ndarray, baseline_returns: np.ndarray
    ) -> float:
        """Compute test statistic (e.g., difference in means)."""
        pass

    def run(
        self,
        strategy_returns: np.ndarray,
        baseline_returns: np.ndarray,
        n_permutations: int = 10000,
        seed: int | None = None,
        **params,
    ) -> dict[str, Any]:
        """Run permutation test."""
        if seed is not None:
            np.random.seed(seed)

        # Observed test statistic
        observed_stat = self.compute_test_statistic(strategy_returns, baseline_returns)

        # Pool all returns
        pooled = np.concatenate([strategy_returns, baseline_returns])
        n_strategy = len(strategy_returns)

        # Permutation distribution
        perm_stats = []
        for _ in range(n_permutations):
            np.random.shuffle(pooled)
            perm_strategy = pooled[:n_strategy]
            perm_baseline = pooled[n_strategy:]
            perm_stat = self.compute_test_statistic(perm_strategy, perm_baseline)
            perm_stats.append(perm_stat)

        perm_stats = np.array(perm_stats)

        # Two-tailed p-value: proportion of permutations as extreme as observed
        p_value = np.mean(np.abs(perm_stats) >= np.abs(observed_stat))

        return {
            "test_name": self.name,
            "observed_statistic": float(observed_stat),
            "p_value": float(p_value),
            "n_permutations": n_permutations,
            "significant": p_value < 0.05,
        }


# ============================================================================
# Concrete test implementations
# ============================================================================


class TTest(StatisticalTest):
    """Two-sample t-test for comparing mean returns."""

    @property
    def name(self) -> str:
        return "t_test"

    def run(
        self, strategy_returns: np.ndarray, baseline_returns: np.ndarray, **params
    ) -> dict[str, Any]:
        """Run two-sample t-test."""
        t_stat, p_value = stats.ttest_ind(strategy_returns, baseline_returns)

        return {
            "test_name": self.name,
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "strategy_mean": float(np.mean(strategy_returns)),
            "baseline_mean": float(np.mean(baseline_returns)),
            "significant": p_value < 0.05,
        }


class MannWhitneyTest(StatisticalTest):
    """Mann-Whitney U test for comparing distributions (non-parametric)."""

    @property
    def name(self) -> str:
        return "mann_whitney"

    def run(
        self, strategy_returns: np.ndarray, baseline_returns: np.ndarray, **params
    ) -> dict[str, Any]:
        """Run Mann-Whitney U test."""
        u_stat, p_value = stats.mannwhitneyu(
            strategy_returns, baseline_returns, alternative="two-sided"
        )

        return {
            "test_name": self.name,
            "u_statistic": float(u_stat),
            "p_value": float(p_value),
            "strategy_median": float(np.median(strategy_returns)),
            "baseline_median": float(np.median(baseline_returns)),
            "significant": p_value < 0.05,
        }


class PermutationMeanTest(PermutationTest):
    """Permutation test for difference in means (distribution-free)."""

    @property
    def name(self) -> str:
        return "permutation_test"

    def compute_test_statistic(
        self, strategy_returns: np.ndarray, baseline_returns: np.ndarray
    ) -> float:
        """Compute difference in means."""
        return np.mean(strategy_returns) - np.mean(baseline_returns)


class SharpeBootstrapTest(BootstrapTest):
    """Bootstrap test for Sharpe ratio significance."""

    @property
    def name(self) -> str:
        return "sharpe_bootstrap"

    def compute_metric(self, returns: np.ndarray) -> float:
        """Compute Sharpe ratio."""
        return _sharpe_ratio(returns)

    def _validate_metrics(self, strategy_sharpe: float, baseline_sharpe: float) -> None:
        """Warn if Sharpe ratios are negative."""
        if strategy_sharpe < 0 and baseline_sharpe < 0:
            logger.warning(
                "Both Sharpe ratios are negative (strategy=%.3f, baseline=%.3f). "
                "Interpretation is problematic: higher volatility makes Sharpe less negative. "
                "Enable 'sortino_bootstrap' test for more robust comparison with negative returns.",
                strategy_sharpe,
                baseline_sharpe,
            )
        elif strategy_sharpe < 0:
            logger.warning(
                "Strategy Sharpe ratio is negative (%.3f). "
                "This indicates mean returns below risk-free rate. "
                "Enable 'sortino_bootstrap' test for more interpretable risk-adjusted metric.",
                strategy_sharpe,
            )


class SortinoBootstrapTest(BootstrapTest):
    """Bootstrap test for Sortino ratio significance.

    More robust than Sharpe for:
    - Negative return periods
    - Asymmetric return distributions
    - Trading strategies (upside volatility is desirable)
    """

    @property
    def name(self) -> str:
        return "sortino_bootstrap"

    def compute_metric(self, returns: np.ndarray) -> float:
        """Compute Sortino ratio."""
        return _sortino_ratio(returns)


# ============================================================================
# Test registry and orchestration
# ============================================================================

# Registry maps test names to test classes
TEST_REGISTRY: dict[str, type[StatisticalTest]] = {
    "t_test": TTest,
    "mann_whitney": MannWhitneyTest,
    "permutation_test": PermutationMeanTest,
    "sharpe_bootstrap": SharpeBootstrapTest,
    "sortino_bootstrap": SortinoBootstrapTest,
}


def get_test(test_name: str) -> StatisticalTest | None:
    """Factory function to create test instances.

    Args:
        test_name: Test identifier from config

    Returns:
        Test instance or None if not found
    """
    test_class = TEST_REGISTRY.get(test_name)
    if test_class:
        return test_class()
    return None


def register_test(test_class: type[StatisticalTest]) -> None:
    """Register a new test class (for extensibility).

    Example:
        >>> class MyCustomTest(StatisticalTest):
        ...     @property
        ...     def name(self): return "my_test"
        ...     def run(self, ...): ...
        >>> register_test(MyCustomTest)
    """
    instance = test_class()
    TEST_REGISTRY[instance.name] = test_class
    logger.info(f"Registered statistical test: {instance.name}")


def list_available_tests() -> list[str]:
    """List all registered test names."""
    return list(TEST_REGISTRY.keys())


# ============================================================================
# Baseline computation
# ============================================================================


def compute_buy_and_hold_returns(prices: pd.Series, max_steps: int) -> np.ndarray:
    """Compute buy-and-hold returns from price series.

    Args:
        prices: Price series (must have at least 2 values)
        max_steps: Number of steps to compute returns for

    Returns:
        Array of simple returns for buy-and-hold strategy
    """
    if len(prices) < 2:
        raise ValueError("Price series must have at least 2 values")

    # Compute log returns and convert to simple returns
    log_returns = np.log(prices / prices.shift(1)).fillna(0).to_numpy()[:max_steps]
    simple_returns = np.exp(log_returns) - 1.0

    return simple_returns


def compute_short_and_hold_returns(prices: pd.Series, max_steps: int) -> np.ndarray:
    """Compute short-and-hold returns (inverse exposure to buy-and-hold)."""
    return -compute_buy_and_hold_returns(prices, max_steps)


def _normalize_execution_weights(weights: np.ndarray) -> np.ndarray:
    """Normalize execution weights to sum to 1 with robust fallbacks."""
    w = np.asarray(weights, dtype=float)
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w = np.clip(w, a_min=0.0, a_max=None)
    total = float(np.sum(w))
    if total <= 0.0:
        return np.full_like(w, 1.0 / max(len(w), 1), dtype=float)
    return w / total


def _execution_schedule_returns(
    prices: pd.Series,
    max_steps: int,
    weights: np.ndarray,
    direction: float = 1.0,
) -> np.ndarray:
    """Compute execution benchmark returns from progressive position buildup.

    The benchmark starts flat and builds toward full target exposure over the
    evaluation horizon using the provided schedule weights.
    """
    if len(prices) < 2:
        raise ValueError("Price series must have at least 2 values")

    simple_asset_returns = prices.pct_change().fillna(0.0).to_numpy()[:max_steps]
    n = len(simple_asset_returns)
    if n == 0:
        return np.array([], dtype=float)

    normalized_weights = _normalize_execution_weights(np.asarray(weights, dtype=float)[:n])
    cumulative_exposure = np.clip(np.cumsum(normalized_weights), 0.0, 1.0)
    lagged_exposure = np.concatenate(([0.0], cumulative_exposure[:-1]))
    return direction * lagged_exposure * simple_asset_returns


def compute_twap_returns(prices: pd.Series, max_steps: int) -> np.ndarray:
    """Compute TWAP benchmark returns via equal-time execution schedule."""
    if max_steps <= 0:
        return np.array([], dtype=float)
    weights = np.ones(max_steps, dtype=float)
    return _execution_schedule_returns(prices, max_steps, weights)


def compute_vwap_returns(
    prices: pd.Series,
    volumes: pd.Series,
    max_steps: int,
) -> np.ndarray:
    """Compute VWAP benchmark returns via volume-weighted execution schedule."""
    if len(volumes) < 1:
        raise ValueError("Volume series must have at least 1 value")
    if max_steps <= 0:
        return np.array([], dtype=float)
    weights = pd.Series(volumes).fillna(0.0).to_numpy()[:max_steps]
    return _execution_schedule_returns(prices, max_steps, weights)


def _resolve_vwap_volume_series(market_data: pd.DataFrame | None) -> tuple[pd.Series | None, str | None]:
    """Resolve volume input for VWAP with explicit provenance."""
    if market_data is None or market_data.empty:
        return None, None

    direct_candidates = ["volume", "trade_volume", "last_size", "size", "qty"]
    for col in direct_candidates:
        if col in market_data.columns:
            return market_data[col], col

    if {"bid_sz_00", "ask_sz_00"}.issubset(market_data.columns):
        proxy_volume = market_data["bid_sz_00"].fillna(0.0) + market_data["ask_sz_00"].fillna(0.0)
        return proxy_volume, "bid_sz_00+ask_sz_00 (top-of-book size proxy)"

    return None, None


def _max_drawdown(simple_returns: np.ndarray) -> float:
    """Compute max drawdown from simple returns."""
    if simple_returns.size == 0:
        return np.nan
    equity = np.cumprod(1.0 + simple_returns)
    running_max = np.maximum.accumulate(equity)
    drawdown = equity / running_max - 1.0
    return float(np.min(drawdown))


def _performance_summary(simple_returns: np.ndarray, periods_per_year: int) -> dict[str, float]:
    """Compute compact benchmark performance summary metrics."""
    r = np.asarray(simple_returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return {
            "total_return": np.nan,
            "annualized_return_cagr": np.nan,
            "annualized_volatility": np.nan,
            "sharpe_ratio": np.nan,
            "sortino_ratio": np.nan,
            "max_drawdown": np.nan,
        }

    mu = float(np.mean(r))
    sigma = float(np.std(r, ddof=1)) if r.size > 1 else 0.0
    annualized_vol = sigma * np.sqrt(periods_per_year)
    downside = r[r < 0]
    downside_sigma = float(np.std(downside, ddof=1)) if downside.size > 1 else 0.0
    sharpe = _safe_div(mu * np.sqrt(periods_per_year), sigma)
    sortino = _safe_div(mu * np.sqrt(periods_per_year), downside_sigma)

    equity = np.cumprod(1.0 + r)
    total_return = float(equity[-1] - 1.0)
    years = max(r.size / periods_per_year, 1e-12)
    cagr = float(equity[-1] ** (1.0 / years) - 1.0)

    return {
        "total_return": total_return,
        "annualized_return_cagr": cagr,
        "annualized_volatility": float(annualized_vol),
        "sharpe_ratio": float(sharpe) if np.isfinite(sharpe) else np.nan,
        "sortino_ratio": float(sortino) if np.isfinite(sortino) else np.nan,
        "max_drawdown": _max_drawdown(r),
    }


def build_benchmark_comparison_table(
    strategy_returns: np.ndarray,
    benchmark_returns: dict[str, np.ndarray],
    periods_per_year: int = 252,
) -> list[dict[str, float | str]]:
    """Build a cross-benchmark comparison table with core performance metrics."""
    rows: list[dict[str, float | str]] = []
    strategy_metrics = _performance_summary(strategy_returns, periods_per_year)
    rows.append({"strategy": "agent", **strategy_metrics})

    for name, returns in benchmark_returns.items():
        benchmark_metrics = _performance_summary(returns, periods_per_year)
        rows.append({"strategy": name, **benchmark_metrics})

    return rows


def compute_random_baseline_returns(
    env: Any,
    max_steps: int,
    n_trials: int = 100,
    seed: int | None = None,
) -> list[np.ndarray]:
    """Generate random action baseline returns via Monte Carlo sampling.

    Args:
        env: Trading environment (TorchRL environment)
        max_steps: Number of steps per trial
        n_trials: Number of random trials to run
        seed: Random seed for reproducibility

    Returns:
        List of return arrays, one per trial
    """
    import torch
    from tensordict.nn import InteractionType
    from torchrl.envs.utils import set_exploration_type

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    random_returns = []

    for trial in range(n_trials):
        env.set_seed(seed + trial if seed is not None else None)

        # Random policy: sample actions uniformly
        with torch.no_grad():
            with set_exploration_type(InteractionType.RANDOM):
                rollout = env.rollout(max_steps=max_steps)

        # Extract returns from rollout
        # For shaped rewards, we approximate returns from the reward signal
        rewards = (
            rollout["next", "reward"].detach().cpu().reshape(-1).numpy()[:max_steps]
        )

        # Treat rewards as log returns for approximation
        simple_returns = np.exp(rewards) - 1.0
        random_returns.append(simple_returns)

    return random_returns


# ============================================================================
# Test orchestration
# ============================================================================


def run_statistical_tests(
    strategy_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    baseline_name: str,
    config: Any,
) -> dict[str, Any]:
    """Run configured statistical tests comparing strategy to baseline.

    Uses registry pattern for type-safe, extensible dispatch.

    Args:
        strategy_returns: Strategy simple returns
        benchmark_returns: Baseline/benchmark simple returns
        baseline_name: Name of baseline for logging
        config: StatisticalTestingConfig with test parameters

    Returns:
        Dict with test results for each configured test
    """
    results = {
        "baseline": baseline_name,
        "n_strategy_samples": len(strategy_returns),
        "n_baseline_samples": len(benchmark_returns),
    }

    # Run configured tests using registry
    for test_name in config.tests:
        test = get_test(test_name)

        if test is None:
            logger.warning(
                f"Unknown test: {test_name}. Available: {list(TEST_REGISTRY.keys())}"
            )
            continue

        try:
            # Extract parameters for this test
            test_params = {
                "n_bootstrap": config.n_bootstrap_samples,
                "n_permutations": config.n_permutations,
                "confidence_level": config.confidence_level,
                "seed": config.random_seed,
            }

            test_result = test.run(
                strategy_returns=strategy_returns,
                baseline_returns=benchmark_returns,
                **test_params,
            )
            results[test_name] = test_result

        except Exception as e:
            logger.error(f"Failed to run {test_name} for {baseline_name}: {e}")
            results[test_name] = {"error": str(e)}

    return results


def run_all_statistical_tests(
    strategy_returns: np.ndarray,
    prices: pd.Series | None,
    env: Any,
    max_steps: int,
    config: Any,
    *,
    market_data: pd.DataFrame | None = None,
    periods_per_year: int = 252,
) -> dict[str, Any]:
    """Run all configured statistical significance tests.

    Args:
        strategy_returns: Strategy simple returns
        prices: Price series for buy-and-hold comparison (optional)
        env: Trading environment for random baseline
        max_steps: Number of steps for evaluation
        config: StatisticalTestingConfig

    Returns:
        Dict with all test results organized by baseline
    """
    if not config.enabled:
        return {"enabled": False}

    logger.info("Running statistical significance tests...")

    all_results = {
        "enabled": True,
        "tests_configured": config.tests,
        "baselines": [],
    }
    benchmark_returns_map: dict[str, np.ndarray] = {}

    # Test against buy-and-hold benchmark
    if config.compare_to_buy_and_hold and prices is not None:
        try:
            logger.info("Computing buy-and-hold baseline...")
            bh_returns = compute_buy_and_hold_returns(prices, max_steps)
            benchmark_returns_map["buy_and_hold"] = bh_returns

            logger.info(
                f"Running tests against buy-and-hold (n={len(bh_returns)} samples)..."
            )
            bh_results = run_statistical_tests(
                strategy_returns, bh_returns, "buy_and_hold", config
            )
            all_results["baselines"].append(bh_results)
            logger.info("Buy-and-hold tests complete")
        except Exception as e:
            logger.error(f"Failed to run buy-and-hold comparison: {e}")
            all_results["baselines"].append(
                {"baseline": "buy_and_hold", "error": str(e)}
            )

    # Test against short-and-hold benchmark
    if getattr(config, "compare_to_short_and_hold", False) and prices is not None:
        try:
            logger.info("Computing short-and-hold baseline...")
            sh_returns = compute_short_and_hold_returns(prices, max_steps)
            benchmark_returns_map["short_and_hold"] = sh_returns
            sh_results = run_statistical_tests(
                strategy_returns, sh_returns, "short_and_hold", config
            )
            all_results["baselines"].append(sh_results)
            logger.info("Short-and-hold tests complete")
        except Exception as e:
            logger.error(f"Failed to run short-and-hold comparison: {e}")
            all_results["baselines"].append(
                {"baseline": "short_and_hold", "error": str(e)}
            )

    # Test against TWAP execution benchmark
    if getattr(config, "compare_to_twap", False) and prices is not None:
        try:
            logger.info("Computing TWAP baseline...")
            twap_returns = compute_twap_returns(prices, max_steps)
            benchmark_returns_map["twap"] = twap_returns
            twap_results = run_statistical_tests(
                strategy_returns, twap_returns, "twap", config
            )
            all_results["baselines"].append(twap_results)
            logger.info("TWAP tests complete")
        except Exception as e:
            logger.error(f"Failed to run TWAP comparison: {e}")
            all_results["baselines"].append({"baseline": "twap", "error": str(e)})

    # Test against VWAP execution benchmark
    if getattr(config, "compare_to_vwap", False) and prices is not None:
        try:
            volume_series, volume_source = _resolve_vwap_volume_series(market_data)
            if volume_series is None:
                msg = (
                    "VWAP baseline skipped: no usable volume column found. "
                    "Expected one of: volume, trade_volume, last_size, size, qty, "
                    "or bid_sz_00/ask_sz_00 for proxy."
                )
                logger.warning(msg)
                all_results["baselines"].append({"baseline": "vwap", "error": msg})
            else:
                if "proxy" in str(volume_source):
                    logger.warning(
                        "VWAP is using %s. This is quote-size-weighted, not true traded volume.",
                        volume_source,
                    )
                vwap_returns = compute_vwap_returns(
                    prices, volume_series, max_steps=max_steps
                )
                benchmark_returns_map["vwap"] = vwap_returns
                vwap_results = run_statistical_tests(
                    strategy_returns, vwap_returns, "vwap", config
                )
                vwap_results["volume_source"] = volume_source
                all_results["baselines"].append(vwap_results)
                all_results["vwap_volume_source"] = volume_source
                logger.info("VWAP tests complete")
        except Exception as e:
            logger.error(f"Failed to run VWAP comparison: {e}")
            all_results["baselines"].append({"baseline": "vwap", "error": str(e)})

    # Test against random action baseline
    if config.compare_to_random:
        try:
            logger.info(
                f"Computing random baseline ({config.n_random_trials} trials)..."
            )
            random_trials = compute_random_baseline_returns(
                env, max_steps, n_trials=config.n_random_trials, seed=config.random_seed
            )

            # Aggregate random trials (use mean across trials)
            random_returns_mean = np.mean(random_trials, axis=0)

            logger.info(
                f"Running tests against random baseline (n={len(random_returns_mean)} samples)..."
            )
            random_results = run_statistical_tests(
                strategy_returns, random_returns_mean, "random_actions", config
            )

            # Also compute summary statistics across trials
            random_results["random_trials_sharpe_mean"] = float(
                np.mean([_sharpe_ratio(trial) for trial in random_trials])
            )
            random_results["random_trials_sharpe_std"] = float(
                np.std([_sharpe_ratio(trial) for trial in random_trials])
            )
            benchmark_returns_map["random_actions"] = random_returns_mean

            all_results["baselines"].append(random_results)
            logger.info("Random baseline tests complete")
        except Exception as e:
            logger.error(f"Failed to run random baseline comparison: {e}")
            all_results["baselines"].append(
                {"baseline": "random_actions", "error": str(e)}
            )

    all_results["benchmark_comparison_table"] = build_benchmark_comparison_table(
        strategy_returns=strategy_returns,
        benchmark_returns=benchmark_returns_map,
        periods_per_year=periods_per_year,
    )

    logger.info("Statistical significance testing complete")
    return all_results


# ============================================================================
# Backward-compatible function wrappers
# ============================================================================


def t_test_mean_returns(
    strategy_returns: np.ndarray,
    baseline_returns: np.ndarray,
) -> dict[str, float]:
    """Two-sample t-test for comparing mean returns."""
    test = TTest()
    return test.run(strategy_returns, baseline_returns)


def mann_whitney_test(
    strategy_returns: np.ndarray,
    baseline_returns: np.ndarray,
) -> dict[str, float]:
    """Mann-Whitney U test for comparing return distributions (non-parametric)."""
    test = MannWhitneyTest()
    return test.run(strategy_returns, baseline_returns)


def permutation_test(
    strategy_returns: np.ndarray,
    baseline_returns: np.ndarray,
    n_permutations: int = 10000,
    seed: int | None = None,
) -> dict[str, float]:
    """Permutation test for comparing mean returns (distribution-free)."""
    test = PermutationMeanTest()
    return test.run(
        strategy_returns,
        baseline_returns,
        n_permutations=n_permutations,
        seed=seed,
    )


def sharpe_ratio_bootstrap_test(
    strategy_returns: np.ndarray,
    baseline_returns: np.ndarray,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    seed: int | None = None,
) -> dict[str, float]:
    """Bootstrap test for Sharpe ratio significance."""
    test = SharpeBootstrapTest()
    return test.run(
        strategy_returns,
        baseline_returns,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        seed=seed,
    )


def sortino_ratio_bootstrap_test(
    strategy_returns: np.ndarray,
    baseline_returns: np.ndarray,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    seed: int | None = None,
) -> dict[str, float]:
    """Bootstrap test for Sortino ratio significance."""
    test = SortinoBootstrapTest()
    return test.run(
        strategy_returns,
        baseline_returns,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        seed=seed,
    )


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    "TEST_REGISTRY",
    # Core classes (for advanced users/extensions)
    "BootstrapTest",
    "MannWhitneyTest",
    "PermutationMeanTest",
    "PermutationTest",
    "SharpeBootstrapTest",
    "SortinoBootstrapTest",
    "StatisticalTest",
    "TTest",
    "build_benchmark_comparison_table",
    # Baseline computation
    "compute_buy_and_hold_returns",
    "compute_random_baseline_returns",
    "compute_short_and_hold_returns",
    "compute_twap_returns",
    "compute_vwap_returns",
    # Factory and registry
    "get_test",
    "list_available_tests",
    # Individual test functions (backward compatible)
    "mann_whitney_test",
    "permutation_test",
    "register_test",
    "run_all_statistical_tests",
    "run_statistical_tests",
    "sharpe_ratio_bootstrap_test",
    "sortino_ratio_bootstrap_test",
    "t_test_mean_returns",
]
