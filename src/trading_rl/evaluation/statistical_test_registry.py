"""Statistical test primitives, registry, and per-baseline execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from scipy import stats

from logger import get_logger

logger = get_logger(__name__)


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
        float(np.mean(excess_returns)),
        float(np.std(excess_returns, ddof=1)),
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


class StatisticalTest(ABC):
    """Abstract base class for all statistical significance tests."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this test (used in config)."""

    @abstractmethod
    def run(
        self,
        strategy_returns: np.ndarray,
        baseline_returns: np.ndarray,
        **params,
    ) -> dict[str, Any]:
        """Run the statistical test."""


class BootstrapTest(StatisticalTest, ABC):
    """Template for bootstrap-based ratio tests."""

    @abstractmethod
    def compute_metric(self, returns: np.ndarray) -> float:
        """Compute the metric to bootstrap (e.g., Sharpe ratio, Sortino ratio)."""

    @property
    def metric_name(self) -> str:
        """Name of the metric (e.g., 'sharpe', 'sortino')."""
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

        strategy_metric = self.compute_metric(strategy_returns)
        baseline_metric = self.compute_metric(baseline_returns)
        observed_diff = strategy_metric - baseline_metric
        self._validate_metrics(strategy_metric, baseline_metric)

        strategy_metrics = []
        baseline_metrics = []
        diff_metrics = []

        n_strategy = len(strategy_returns)
        n_baseline = len(baseline_returns)

        for _ in range(n_bootstrap):
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

        strategy_metrics = strategy_metrics[~np.isnan(strategy_metrics)]
        baseline_metrics = baseline_metrics[~np.isnan(baseline_metrics)]
        diff_metrics = diff_metrics[~np.isnan(diff_metrics)]

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

        if observed_diff > 0:
            p_value = np.mean(diff_metrics <= 0)
        else:
            p_value = np.mean(diff_metrics >= 0)

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


class PermutationTest(StatisticalTest, ABC):
    """Template for permutation-based tests."""

    @abstractmethod
    def compute_test_statistic(
        self,
        strategy_returns: np.ndarray,
        baseline_returns: np.ndarray,
    ) -> float:
        """Compute test statistic (e.g., difference in means)."""

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

        observed_stat = self.compute_test_statistic(strategy_returns, baseline_returns)
        pooled = np.concatenate([strategy_returns, baseline_returns])
        n_strategy = len(strategy_returns)

        perm_stats = []
        for _ in range(n_permutations):
            np.random.shuffle(pooled)
            perm_strategy = pooled[:n_strategy]
            perm_baseline = pooled[n_strategy:]
            perm_stats.append(
                self.compute_test_statistic(perm_strategy, perm_baseline)
            )

        perm_stats = np.array(perm_stats)
        p_value = np.mean(np.abs(perm_stats) >= np.abs(observed_stat))
        return {
            "test_name": self.name,
            "observed_statistic": float(observed_stat),
            "p_value": float(p_value),
            "n_permutations": n_permutations,
            "significant": p_value < 0.05,
        }


class TTest(StatisticalTest):
    """Two-sample t-test for comparing mean returns."""

    @property
    def name(self) -> str:
        return "t_test"

    def run(
        self,
        strategy_returns: np.ndarray,
        baseline_returns: np.ndarray,
        **params,
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
        self,
        strategy_returns: np.ndarray,
        baseline_returns: np.ndarray,
        **params,
    ) -> dict[str, Any]:
        """Run Mann-Whitney U test."""
        u_stat, p_value = stats.mannwhitneyu(
            strategy_returns,
            baseline_returns,
            alternative="two-sided",
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
        self,
        strategy_returns: np.ndarray,
        baseline_returns: np.ndarray,
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
    """Bootstrap test for Sortino ratio significance."""

    @property
    def name(self) -> str:
        return "sortino_bootstrap"

    def compute_metric(self, returns: np.ndarray) -> float:
        """Compute Sortino ratio."""
        return _sortino_ratio(returns)


TEST_REGISTRY: dict[str, type[StatisticalTest]] = {
    "t_test": TTest,
    "mann_whitney": MannWhitneyTest,
    "permutation_test": PermutationMeanTest,
    "sharpe_bootstrap": SharpeBootstrapTest,
    "sortino_bootstrap": SortinoBootstrapTest,
}


def get_test(test_name: str) -> StatisticalTest | None:
    """Factory function to create test instances."""
    test_class = TEST_REGISTRY.get(test_name)
    if test_class:
        return test_class()
    return None


def register_test(test_class: type[StatisticalTest]) -> None:
    """Register a new test class (for extensibility)."""
    instance = test_class()
    TEST_REGISTRY[instance.name] = test_class
    logger.info("registered statistical test name=%s", instance.name)


def list_available_tests() -> list[str]:
    """List all registered test names."""
    return list(TEST_REGISTRY.keys())


def run_statistical_tests(
    strategy_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    baseline_name: str,
    config: Any,
) -> dict[str, Any]:
    """Run configured statistical tests comparing strategy to one baseline."""
    results = {
        "baseline": baseline_name,
        "n_strategy_samples": len(strategy_returns),
        "n_baseline_samples": len(benchmark_returns),
    }

    for test_name in config.tests:
        test = get_test(test_name)
        if test is None:
            logger.warning(
                "Unknown test: %s. Available: %s",
                test_name,
                list(TEST_REGISTRY.keys()),
            )
            continue

        try:
            test_params = {
                "n_bootstrap": config.n_bootstrap_samples,
                "n_permutations": config.n_permutations,
                "confidence_level": config.confidence_level,
                "seed": config.random_seed,
            }
            results[test_name] = test.run(
                strategy_returns=strategy_returns,
                baseline_returns=benchmark_returns,
                **test_params,
            )
        except Exception as exc:
            logger.error(
                "Failed to run %s for %s: %s",
                test_name,
                baseline_name,
                exc,
            )
            results[test_name] = {"error": str(exc)}

    return results


def t_test_mean_returns(
    strategy_returns: np.ndarray,
    baseline_returns: np.ndarray,
) -> dict[str, float]:
    """Two-sample t-test for comparing mean returns."""
    return TTest().run(strategy_returns, baseline_returns)


def mann_whitney_test(
    strategy_returns: np.ndarray,
    baseline_returns: np.ndarray,
) -> dict[str, float]:
    """Mann-Whitney U test for comparing return distributions."""
    return MannWhitneyTest().run(strategy_returns, baseline_returns)


def permutation_test(
    strategy_returns: np.ndarray,
    baseline_returns: np.ndarray,
    n_permutations: int = 10000,
    seed: int | None = None,
) -> dict[str, float]:
    """Permutation test for comparing mean returns."""
    return PermutationMeanTest().run(
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
    return SharpeBootstrapTest().run(
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
    return SortinoBootstrapTest().run(
        strategy_returns,
        baseline_returns,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        seed=seed,
    )
