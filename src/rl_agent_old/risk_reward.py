"""
Risk and Reward Module for statistical arbitrage.

This module provides functions for estimating risk measures and
evaluating trading performance.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats


@dataclass
class RiskMetrics:
    """Class to store risk estimation results."""

    overall_scale: float
    sd: float
    var: float
    es: float


def estimate_agent_risk(
    observations: np.ndarray,
    position: int,
    tail_fraction: float = 0.05,
) -> RiskMetrics:
    """
    Estimate risk metrics for an agent's position.

    Args:
        observations: Array of log price differences
        position: Position direction (1 for long, -1 for short)
        tail_fraction: Quantile for risk estimation

    Returns:
        RiskMetrics containing risk measures
    """
    # Calculate basic statistics
    data_size = len(observations)
    scale = np.std(observations)

    # Fit APARCH model
    model = arch_model(
        observations,
        vol="APARCH",
        p=1,
        q=1,
        dist="studentt",
        rescale=True,
    )
    results = model.fit(disp="off")

    # Get standardized residuals
    res = results.resid / results.conditional_volatility

    # Fit GPD to tails
    if position == 1:
        # For long positions, fit to lower tail
        tail_data = res[res < np.quantile(res, tail_fraction)]
        xi, mu, beta = stats.genpareto.fit(tail_data)
        var = stats.genpareto.ppf(tail_fraction, xi, mu, beta)
        es = stats.genpareto.mean(xi, mu, beta)
        es = var - es
    else:
        # For short positions, fit to upper tail
        tail_data = res[res > np.quantile(res, 1 - tail_fraction)]
        xi, mu, beta = stats.genpareto.fit(-tail_data)
        var = -stats.genpareto.ppf(1 - tail_fraction, xi, mu, beta)
        es = stats.genpareto.mean(xi, mu, beta)
        es = var + es

    # Rescale using conditional volatility
    var = var * results.conditional_volatility[-1]
    es = es * results.conditional_volatility[-1]
    sd = results.conditional_volatility[-1]

    return RiskMetrics(
        overall_scale=scale,
        sd=sd,
        var=var,
        es=es,
    )


@dataclass
class PerformanceMetrics:
    """Class to store performance evaluation results."""

    ntrans: np.ndarray
    final_pnl_net: float
    total_cost: float
    sharpe_ratio: float
    net_pnl: np.ndarray
    gross_pnl: float
    gross_sharpe_ratio: float


def performance_pair_trading(
    position: np.ndarray,
    pair: pd.DataFrame,
    coef: np.ndarray,
    spread: pd.Series,
    cost: float = 0.001,
    continuous: bool = True,
) -> PerformanceMetrics:
    """
    Evaluate performance of pair trading strategy.

    Args:
        position: Array of positions
        pair: DataFrame of pair prices
        coef: Cointegration coefficients
        spread: Cointegration spread
        cost: Transaction cost
        continuous: Whether to use continuous returns

    Returns:
        PerformanceMetrics containing performance measures
    """
    n = len(spread)

    # Calculate position changes
    diff_pos = np.diff(position, prepend=0)

    # Initialize PnL arrays
    gross_pnl = np.zeros(n)
    gross_pnl[0] = 1

    if continuous:
        # Calculate continuous returns
        gross_pnl = position[:n] * (
            np.diff(np.log(pair.iloc[:, 0]), prepend=np.log(pair.iloc[0, 0]))
            - coef[1]
            * np.diff(np.log(pair.iloc[:, 1]), prepend=np.log(pair.iloc[0, 1]))
        )
        gross_pnl_cum = np.cumsum(gross_pnl)
    else:
        # Calculate discrete returns
        gross_pnl = position[:n] * (
            np.diff(pair.iloc[:, 0], prepend=pair.iloc[0, 0]) / pair.iloc[:, 0]
            - coef[1]
            * np.diff(pair.iloc[:, 1], prepend=pair.iloc[0, 1])
            / pair.iloc[:, 1]
        )
        gross_pnl_cum = np.cumprod(1 + gross_pnl) - 1

    # Calculate transaction costs
    ntrans_temp = np.abs(diff_pos) * cost
    ntrans_temp_cum = np.cumsum(ntrans_temp)

    # Calculate net PnL
    net_pnl = gross_pnl - ntrans_temp[:n]
    if continuous:
        net_pnl_cum = np.cumsum(net_pnl)
    else:
        net_pnl_cum = np.cumprod(1 + net_pnl)

    # Calculate Sharpe ratios
    net_sharpe = np.mean(net_pnl_cum) / np.std(net_pnl_cum) * np.sqrt(252 * 6.5 * 60)
    gross_sharpe = (
        np.mean(gross_pnl_cum) / np.std(gross_pnl_cum) * np.sqrt(252 * 6.5 * 60)
    )

    print(
        f"\nThe Net Profit is: {net_pnl_cum[-1]:.4f}\n"
        f"... and the Sharpe Ratio for this dataset is: {net_sharpe:.6f}\n"
    )

    return PerformanceMetrics(
        ntrans=ntrans_temp,
        final_pnl_net=net_pnl_cum[-1],
        total_cost=ntrans_temp_cum[-1],
        sharpe_ratio=net_sharpe,
        net_pnl=net_pnl_cum,
        gross_pnl=gross_pnl_cum[-1],
        gross_sharpe_ratio=gross_sharpe,
    )


@dataclass
class SingleAssetPerformance:
    """Class to store single asset performance metrics."""

    ntrans: np.ndarray
    agg_ntrans: pd.Series
    final_pnl_net: float
    number_of_transactions: float
    sharpe_ratio: float
    pnl_net: np.ndarray
    gross_pnl: float
    gross_sharpe_ratio: float


def performance(
    x: np.ndarray,
    underlying: pd.Series,
    cost: float = 4,
    index_point: float = 50,
) -> SingleAssetPerformance:
    """
    Evaluate performance of single asset trading strategy.

    Args:
        x: Array of positions
        underlying: Series of underlying asset prices
        cost: Transaction cost
        index_point: Index point value

    Returns:
        SingleAssetPerformance containing performance measures
    """
    # Calculate gross PnL
    temp_gross = np.where(
        np.isnan(x * np.diff(underlying, prepend=underlying[0])),
        0,
        x * np.diff(underlying, prepend=underlying[0]) * index_point,
    )
    temp_cumsum = np.cumsum(temp_gross)

    # Calculate transaction costs
    ntrans_temp = np.abs(np.diff(x, prepend=x[0]))
    ntrans_temp[0] = 0
    ntrans_temp = pd.Series(ntrans_temp, index=underlying.index)
    agg_ntrans = ntrans_temp.resample("D").sum()
    ntrans_temp2 = np.cumsum(ntrans_temp)

    # Calculate net PnL
    pnl_net = temp_gross - ntrans_temp * cost
    pnl_net2 = np.cumsum(pnl_net)
    gross2 = np.cumsum(temp_gross)

    # Calculate Sharpe ratios
    ind_sharpe = np.mean(pnl_net2) / np.std(pnl_net2) * np.sqrt(252 * 6.5 * 60)
    gross_sharpe = np.mean(gross2) / np.std(gross2) * np.sqrt(252 * 6.5 * 60)

    print(
        f"\nThe average daily number of transactions is: {agg_ntrans.mean():.2f}\n"
        f"The Net Profit is: {pnl_net2[-1]:.4f}\n"
        f"... and the Sharpe Ratio for this dataset is: {ind_sharpe:.6f}\n"
    )

    return SingleAssetPerformance(
        ntrans=ntrans_temp,
        agg_ntrans=agg_ntrans,
        final_pnl_net=pnl_net2[-1],
        number_of_transactions=ntrans_temp2[-1],
        sharpe_ratio=ind_sharpe,
        pnl_net=pnl_net2,
        gross_pnl=gross2[-1],
        gross_sharpe_ratio=gross_sharpe,
    )
