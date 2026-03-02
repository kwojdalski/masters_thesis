"""Statistical significance testing for trading strategy equity curves."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
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
    return _safe_div(float(np.mean(excess_returns)), float(np.std(excess_returns, ddof=1)))


def _sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """Compute Sortino ratio from returns (uses downside deviation only).

    Better alternative to Sharpe ratio when returns are negative, as it only
    penalizes downside volatility.
    """
    if len(returns) == 0:
        return np.nan
    excess_returns = returns - risk_free_rate
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0:
        # No downside risk - use std of all returns as denominator
        downside_dev = float(np.std(returns, ddof=1))
    else:
        downside_dev = float(np.std(downside_returns, ddof=1))

    return _safe_div(float(np.mean(excess_returns)), downside_dev)


def compute_buy_and_hold_returns(
    prices: pd.Series, max_steps: int
) -> np.ndarray:
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
    from torchrl.envs.utils import set_exploration_type
    from tensordict.nn import InteractionType

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
        rewards = rollout["next", "reward"].detach().cpu().reshape(-1).numpy()[:max_steps]

        # Treat rewards as log returns for approximation
        simple_returns = np.exp(rewards) - 1.0
        random_returns.append(simple_returns)

    return random_returns


def t_test_mean_returns(
    strategy_returns: np.ndarray,
    baseline_returns: np.ndarray,
) -> dict[str, float]:
    """Two-sample t-test for comparing mean returns.

    Tests null hypothesis: mean(strategy) = mean(baseline)
    Alternative: mean(strategy) != mean(baseline)

    Args:
        strategy_returns: Strategy simple returns
        baseline_returns: Baseline simple returns

    Returns:
        Dict with t_statistic, p_value, strategy_mean, baseline_mean
    """
    t_stat, p_value = stats.ttest_ind(strategy_returns, baseline_returns)

    return {
        "test_name": "t_test",
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "strategy_mean": float(np.mean(strategy_returns)),
        "baseline_mean": float(np.mean(baseline_returns)),
        "significant": p_value < 0.05,
    }


def mann_whitney_test(
    strategy_returns: np.ndarray,
    baseline_returns: np.ndarray,
) -> dict[str, float]:
    """Mann-Whitney U test for comparing return distributions (non-parametric).

    Tests null hypothesis: distributions are the same
    Alternative: strategy distribution is stochastically greater/less than baseline

    Args:
        strategy_returns: Strategy simple returns
        baseline_returns: Baseline simple returns

    Returns:
        Dict with u_statistic, p_value, strategy_median, baseline_median
    """
    u_stat, p_value = stats.mannwhitneyu(
        strategy_returns, baseline_returns, alternative="two-sided"
    )

    return {
        "test_name": "mann_whitney",
        "u_statistic": float(u_stat),
        "p_value": float(p_value),
        "strategy_median": float(np.median(strategy_returns)),
        "baseline_median": float(np.median(baseline_returns)),
        "significant": p_value < 0.05,
    }


def permutation_test(
    strategy_returns: np.ndarray,
    baseline_returns: np.ndarray,
    n_permutations: int = 10000,
    seed: int | None = None,
) -> dict[str, float]:
    """Permutation test for comparing mean returns (distribution-free).

    Tests null hypothesis: strategy and baseline have the same distribution
    by randomly permuting labels and computing test statistic.

    Args:
        strategy_returns: Strategy simple returns
        baseline_returns: Baseline simple returns
        n_permutations: Number of permutations
        seed: Random seed for reproducibility

    Returns:
        Dict with observed_diff, p_value, significant
    """
    if seed is not None:
        np.random.seed(seed)

    # Observed difference in means
    observed_diff = np.mean(strategy_returns) - np.mean(baseline_returns)

    # Pool all returns
    pooled = np.concatenate([strategy_returns, baseline_returns])
    n_strategy = len(strategy_returns)

    # Permutation distribution
    perm_diffs = []
    for _ in range(n_permutations):
        np.random.shuffle(pooled)
        perm_strategy = pooled[:n_strategy]
        perm_baseline = pooled[n_strategy:]
        perm_diff = np.mean(perm_strategy) - np.mean(perm_baseline)
        perm_diffs.append(perm_diff)

    perm_diffs = np.array(perm_diffs)

    # Two-tailed p-value: proportion of permutations as extreme as observed
    p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))

    return {
        "test_name": "permutation_test",
        "observed_difference": float(observed_diff),
        "p_value": float(p_value),
        "n_permutations": n_permutations,
        "significant": p_value < 0.05,
    }


def sharpe_ratio_bootstrap_test(
    strategy_returns: np.ndarray,
    baseline_returns: np.ndarray,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    seed: int | None = None,
) -> dict[str, float]:
    """Bootstrap test for Sharpe ratio significance.

    Computes bootstrap confidence intervals for Sharpe ratios and tests
    if strategy Sharpe is significantly different from baseline.

    NOTE: Sharpe ratio interpretation when negative:
        - Negative Sharpe ratios are valid but problematic for comparison
        - Higher volatility makes Sharpe "less negative" (closer to 0) even if worse
        - Consider enabling 'sortino_bootstrap' test for more robust risk-adjusted comparison
        - This test still provides valid p-values, but interpretation requires care

    NOTE: Differential Sharpe Ratio (DSR) vs Sharpe Ratio:
        - DSR is a shaped REWARD SIGNAL used during RL training
        - This test uses ACTUAL RETURNS extracted from environment rollouts
        - DSR is NOT used for evaluation or statistical testing

    Args:
        strategy_returns: Strategy simple returns (actual returns, not DSR rewards)
        baseline_returns: Baseline simple returns
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95)
        seed: Random seed for reproducibility

    Returns:
        Dict with Sharpe ratios, confidence intervals, and p_value
    """
    if seed is not None:
        np.random.seed(seed)

    # Observed Sharpe ratios
    strategy_sharpe = _sharpe_ratio(strategy_returns)
    baseline_sharpe = _sharpe_ratio(baseline_returns)
    observed_diff = strategy_sharpe - baseline_sharpe

    # Warn if both Sharpe ratios are negative
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

    # Bootstrap distribution
    strategy_sharpes = []
    baseline_sharpes = []
    diff_sharpes = []

    n_strategy = len(strategy_returns)
    n_baseline = len(baseline_returns)

    for _ in range(n_bootstrap):
        # Resample with replacement
        strategy_sample = np.random.choice(strategy_returns, size=n_strategy, replace=True)
        baseline_sample = np.random.choice(baseline_returns, size=n_baseline, replace=True)

        s_sharpe = _sharpe_ratio(strategy_sample)
        b_sharpe = _sharpe_ratio(baseline_sample)

        strategy_sharpes.append(s_sharpe)
        baseline_sharpes.append(b_sharpe)
        diff_sharpes.append(s_sharpe - b_sharpe)

    strategy_sharpes = np.array(strategy_sharpes)
    baseline_sharpes = np.array(baseline_sharpes)
    diff_sharpes = np.array(diff_sharpes)

    # Remove NaN values
    strategy_sharpes = strategy_sharpes[~np.isnan(strategy_sharpes)]
    baseline_sharpes = baseline_sharpes[~np.isnan(baseline_sharpes)]
    diff_sharpes = diff_sharpes[~np.isnan(diff_sharpes)]

    # Confidence intervals
    alpha = 1 - confidence_level
    strategy_ci_lower = float(np.percentile(strategy_sharpes, 100 * alpha / 2))
    strategy_ci_upper = float(np.percentile(strategy_sharpes, 100 * (1 - alpha / 2)))
    baseline_ci_lower = float(np.percentile(baseline_sharpes, 100 * alpha / 2))
    baseline_ci_upper = float(np.percentile(baseline_sharpes, 100 * (1 - alpha / 2)))
    diff_ci_lower = float(np.percentile(diff_sharpes, 100 * alpha / 2))
    diff_ci_upper = float(np.percentile(diff_sharpes, 100 * (1 - alpha / 2)))

    # P-value: proportion of bootstrap diffs with opposite sign
    if observed_diff > 0:
        p_value = np.mean(diff_sharpes <= 0)
    else:
        p_value = np.mean(diff_sharpes >= 0)

    # Significant if CI doesn't include 0
    significant = not (diff_ci_lower <= 0 <= diff_ci_upper)

    return {
        "test_name": "sharpe_bootstrap",
        "strategy_sharpe": float(strategy_sharpe),
        "baseline_sharpe": float(baseline_sharpe),
        "sharpe_difference": float(observed_diff),
        "strategy_sharpe_ci_lower": strategy_ci_lower,
        "strategy_sharpe_ci_upper": strategy_ci_upper,
        "baseline_sharpe_ci_lower": baseline_ci_lower,
        "baseline_sharpe_ci_upper": baseline_ci_upper,
        "difference_ci_lower": diff_ci_lower,
        "difference_ci_upper": diff_ci_upper,
        "p_value": float(p_value),
        "confidence_level": confidence_level,
        "n_bootstrap": n_bootstrap,
        "significant": significant,
    }


def sortino_ratio_bootstrap_test(
    strategy_returns: np.ndarray,
    baseline_returns: np.ndarray,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    seed: int | None = None,
) -> dict[str, float]:
    """Bootstrap test for Sortino ratio significance.

    Computes bootstrap confidence intervals for Sortino ratios and tests
    if strategy Sortino is significantly different from baseline.

    Sortino ratio uses downside deviation (volatility of losses) instead of
    total volatility, making it more interpretable for:
    - Negative return periods (avoids Sharpe ratio's counterintuitive ranking)
    - Asymmetric return distributions (common in trading)
    - Strategies where upside volatility is desirable

    Args:
        strategy_returns: Strategy simple returns
        baseline_returns: Baseline simple returns
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95)
        seed: Random seed for reproducibility

    Returns:
        Dict with Sortino ratios, confidence intervals, and p_value
    """
    if seed is not None:
        np.random.seed(seed)

    # Observed Sortino ratios
    strategy_sortino = _sortino_ratio(strategy_returns)
    baseline_sortino = _sortino_ratio(baseline_returns)
    observed_diff = strategy_sortino - baseline_sortino

    # Bootstrap distribution
    strategy_sortinos = []
    baseline_sortinos = []
    diff_sortinos = []

    n_strategy = len(strategy_returns)
    n_baseline = len(baseline_returns)

    for _ in range(n_bootstrap):
        # Resample with replacement
        strategy_sample = np.random.choice(strategy_returns, size=n_strategy, replace=True)
        baseline_sample = np.random.choice(baseline_returns, size=n_baseline, replace=True)

        s_sortino = _sortino_ratio(strategy_sample)
        b_sortino = _sortino_ratio(baseline_sample)

        strategy_sortinos.append(s_sortino)
        baseline_sortinos.append(b_sortino)
        diff_sortinos.append(s_sortino - b_sortino)

    strategy_sortinos = np.array(strategy_sortinos)
    baseline_sortinos = np.array(baseline_sortinos)
    diff_sortinos = np.array(diff_sortinos)

    # Remove NaN values
    strategy_sortinos = strategy_sortinos[~np.isnan(strategy_sortinos)]
    baseline_sortinos = baseline_sortinos[~np.isnan(baseline_sortinos)]
    diff_sortinos = diff_sortinos[~np.isnan(diff_sortinos)]

    # Confidence intervals
    alpha = 1 - confidence_level
    strategy_ci_lower = float(np.percentile(strategy_sortinos, 100 * alpha / 2))
    strategy_ci_upper = float(np.percentile(strategy_sortinos, 100 * (1 - alpha / 2)))
    baseline_ci_lower = float(np.percentile(baseline_sortinos, 100 * alpha / 2))
    baseline_ci_upper = float(np.percentile(baseline_sortinos, 100 * (1 - alpha / 2)))
    diff_ci_lower = float(np.percentile(diff_sortinos, 100 * alpha / 2))
    diff_ci_upper = float(np.percentile(diff_sortinos, 100 * (1 - alpha / 2)))

    # P-value: proportion of bootstrap diffs with opposite sign
    if observed_diff > 0:
        p_value = np.mean(diff_sortinos <= 0)
    else:
        p_value = np.mean(diff_sortinos >= 0)

    # Significant if CI doesn't include 0
    significant = not (diff_ci_lower <= 0 <= diff_ci_upper)

    return {
        "test_name": "sortino_bootstrap",
        "strategy_sortino": float(strategy_sortino),
        "baseline_sortino": float(baseline_sortino),
        "sortino_difference": float(observed_diff),
        "strategy_sortino_ci_lower": strategy_ci_lower,
        "strategy_sortino_ci_upper": strategy_ci_upper,
        "baseline_sortino_ci_lower": baseline_ci_lower,
        "baseline_sortino_ci_upper": baseline_ci_upper,
        "difference_ci_lower": diff_ci_lower,
        "difference_ci_upper": diff_ci_upper,
        "p_value": float(p_value),
        "confidence_level": confidence_level,
        "n_bootstrap": n_bootstrap,
        "significant": significant,
    }


def run_statistical_tests(
    strategy_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    baseline_name: str,
    config: Any,
) -> dict[str, Any]:
    """Run configured statistical tests comparing strategy to baseline.

    Args:
        strategy_returns: Strategy simple returns
        benchmark_returns: Baseline/benchmark simple returns
        baseline_name: Name of baseline for logging (e.g., "buy_and_hold", "random")
        config: StatisticalTestingConfig with test parameters

    Returns:
        Dict with test results for each configured test
    """
    results = {
        "baseline": baseline_name,
        "n_strategy_samples": len(strategy_returns),
        "n_baseline_samples": len(benchmark_returns),
    }

    # Run configured tests
    for test_name in config.tests:
        try:
            if test_name == "t_test":
                results["t_test"] = t_test_mean_returns(strategy_returns, benchmark_returns)
            elif test_name == "mann_whitney":
                results["mann_whitney"] = mann_whitney_test(strategy_returns, benchmark_returns)
            elif test_name == "permutation_test":
                results["permutation_test"] = permutation_test(
                    strategy_returns,
                    benchmark_returns,
                    n_permutations=config.n_permutations,
                    seed=config.random_seed,
                )
            elif test_name == "sharpe_bootstrap":
                results["sharpe_bootstrap"] = sharpe_ratio_bootstrap_test(
                    strategy_returns,
                    benchmark_returns,
                    n_bootstrap=config.n_bootstrap_samples,
                    confidence_level=config.confidence_level,
                    seed=config.random_seed,
                )
            elif test_name == "sortino_bootstrap":
                results["sortino_bootstrap"] = sortino_ratio_bootstrap_test(
                    strategy_returns,
                    benchmark_returns,
                    n_bootstrap=config.n_bootstrap_samples,
                    confidence_level=config.confidence_level,
                    seed=config.random_seed,
                )
            else:
                logger.warning(f"Unknown test: {test_name}")
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

    # Test against buy-and-hold benchmark
    if config.compare_to_buy_and_hold and prices is not None:
        try:
            logger.info("Computing buy-and-hold baseline...")
            bh_returns = compute_buy_and_hold_returns(prices, max_steps)

            logger.info(f"Running tests against buy-and-hold (n={len(bh_returns)} samples)...")
            bh_results = run_statistical_tests(
                strategy_returns, bh_returns, "buy_and_hold", config
            )
            all_results["baselines"].append(bh_results)
            logger.info("Buy-and-hold tests complete")
        except Exception as e:
            logger.error(f"Failed to run buy-and-hold comparison: {e}")
            all_results["baselines"].append({
                "baseline": "buy_and_hold",
                "error": str(e)
            })

    # Test against random action baseline
    if config.compare_to_random:
        try:
            logger.info(f"Computing random baseline ({config.n_random_trials} trials)...")
            random_trials = compute_random_baseline_returns(
                env, max_steps, n_trials=config.n_random_trials, seed=config.random_seed
            )

            # Aggregate random trials (use mean across trials)
            random_returns_mean = np.mean([trial for trial in random_trials], axis=0)

            logger.info(f"Running tests against random baseline (n={len(random_returns_mean)} samples)...")
            random_results = run_statistical_tests(
                strategy_returns, random_returns_mean, "random_actions", config
            )

            # Also compute summary statistics across trials
            random_results["random_trials_sharpe_mean"] = float(np.mean([
                _sharpe_ratio(trial) for trial in random_trials
            ]))
            random_results["random_trials_sharpe_std"] = float(np.std([
                _sharpe_ratio(trial) for trial in random_trials
            ]))

            all_results["baselines"].append(random_results)
            logger.info("Random baseline tests complete")
        except Exception as e:
            logger.error(f"Failed to run random baseline comparison: {e}")
            all_results["baselines"].append({
                "baseline": "random_actions",
                "error": str(e)
            })

    logger.info("Statistical significance testing complete")
    return all_results
