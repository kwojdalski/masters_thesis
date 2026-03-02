"""Statistical significance testing for trading strategy equity curves.

This module provides the public API for statistical significance testing.
The implementation uses a modular Strategy pattern (see statistical_tests_refactored.py).

Public API (backward compatible):
- Individual test functions (for manual use)
- run_statistical_tests (orchestrates multiple tests for one baseline)
- run_all_statistical_tests (orchestrates all baselines and tests)
- Baseline computation functions
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from logger import get_logger

# Import refactored implementation
from trading_rl.evaluation.statistical_tests_refactored import (
    TEST_REGISTRY,
    StatisticalTest,
    BootstrapTest,
    PermutationTest,
    TTest,
    MannWhitneyTest,
    PermutationMeanTest,
    SharpeBootstrapTest,
    SortinoBootstrapTest,
    get_test,
    register_test,
    list_available_tests,
    run_statistical_tests,
)

logger = get_logger(__name__)


# ============================================================================
# Baseline computation (not tests themselves)
# ============================================================================

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


# ============================================================================
# Backward-compatible individual test functions
# ============================================================================

def t_test_mean_returns(
    strategy_returns: np.ndarray,
    baseline_returns: np.ndarray,
) -> dict[str, float]:
    """Two-sample t-test for comparing mean returns.

    Backward-compatible wrapper around TTest class.
    """
    test = TTest()
    return test.run(strategy_returns, baseline_returns)


def mann_whitney_test(
    strategy_returns: np.ndarray,
    baseline_returns: np.ndarray,
) -> dict[str, float]:
    """Mann-Whitney U test for comparing return distributions (non-parametric).

    Backward-compatible wrapper around MannWhitneyTest class.
    """
    test = MannWhitneyTest()
    return test.run(strategy_returns, baseline_returns)


def permutation_test(
    strategy_returns: np.ndarray,
    baseline_returns: np.ndarray,
    n_permutations: int = 10000,
    seed: int | None = None,
) -> dict[str, float]:
    """Permutation test for comparing mean returns (distribution-free).

    Backward-compatible wrapper around PermutationMeanTest class.
    """
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
    """Bootstrap test for Sharpe ratio significance.

    Backward-compatible wrapper around SharpeBootstrapTest class.
    """
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
    """Bootstrap test for Sortino ratio significance.

    Backward-compatible wrapper around SortinoBootstrapTest class.
    """
    test = SortinoBootstrapTest()
    return test.run(
        strategy_returns,
        baseline_returns,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        seed=seed,
    )


# ============================================================================
# High-level orchestration
# ============================================================================

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
            from trading_rl.evaluation.statistical_tests_refactored import _sharpe_ratio
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


# ============================================================================
# Re-export for backward compatibility
# ============================================================================

__all__ = [
    # Core classes (for advanced users/extensions)
    "StatisticalTest",
    "BootstrapTest",
    "PermutationTest",
    # Concrete test classes
    "TTest",
    "MannWhitneyTest",
    "PermutationMeanTest",
    "SharpeBootstrapTest",
    "SortinoBootstrapTest",
    # Factory and registry
    "get_test",
    "register_test",
    "list_available_tests",
    "TEST_REGISTRY",
    # Individual test functions (backward compatible)
    "t_test_mean_returns",
    "mann_whitney_test",
    "permutation_test",
    "sharpe_ratio_bootstrap_test",
    "sortino_ratio_bootstrap_test",
    # Baseline computation
    "compute_buy_and_hold_returns",
    "compute_random_baseline_returns",
    # Orchestration
    "run_statistical_tests",
    "run_all_statistical_tests",
]
