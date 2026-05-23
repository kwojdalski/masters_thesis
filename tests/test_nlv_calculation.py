"""Verify broker NLV tracking and extract_tradingenv_return_series.

Setup: always-long (weight = +1.0), 6 known prices, no fees.

Key TradingEnv timing: at step t the agent acts at price[t] and the broker
records NLV *at price[t]* (not at price[t+1]).  Consequences:

  r[0] = 0         -- buying at price[0], NLV stays at initial_cash
  r[t] = price[t] / price[t-1] - 1   for t >= 1
  total_return     = price[n_steps-1] / price[0] - 1
                     (last price PRICES[-1] is never entered into NLV)

The benchmark window in report.py / evaluate_command.py uses
  prices.iloc[:max_steps + 1]
which includes price[n_steps] — one extra price relative to the strategy's
NLV path.  Impact is 1/n_steps (~5 ppm for 193 K steps), but the alignment
is documented here so the discrepancy is not mistaken for a bug.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from trading_rl.envs.tradingenvxy_wrapper import TradingEnvXYFactory
from trading_rl.evaluation.returns import extract_tradingenv_return_series

INITIAL_CASH = 10_000.0
PRICES = [100.0, 101.0, 102.0, 101.0, 100.0, 99.0]


def _make_df(prices: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01 09:30:00", periods=len(prices), freq="s")
    return pd.DataFrame(
        {
            "close": prices,
            "feature_signal": [float(i) / 100.0 for i in range(len(prices))],
        },
        index=idx,
    )


def _make_env(df: pd.DataFrame):
    from trading_rl.config import ExperimentConfig

    config = ExperimentConfig.from_dict(
        {
            "env": {
                "backend": "tradingenv",
                "price_column": "close",
                "trading_fees": 0.0,
                "reward_type": "log_return",
                "initial_portfolio_value": INITIAL_CASH,
            }
        }
    )
    factory = TradingEnvXYFactory(config)
    return factory.make(df, config=config, feature_columns=["feature_signal"], price_column="close")


def _run_always_long(env, n_steps: int):
    """Roll out n_steps of always-long (weight +1.0)."""
    from tensordict.nn import InteractionType
    from torchrl.envs.utils import set_exploration_type

    # Action spec is shape (1,) float64.
    def _policy(td):
        td["action"] = torch.ones(1, dtype=torch.float64)
        return td

    with torch.no_grad():
        with set_exploration_type(InteractionType.DETERMINISTIC):
            rollout = env.rollout(max_steps=n_steps, policy=_policy)
    return rollout


class TestNLVCalculation:
    """NLV path should match TradingEnv's at-trade-price valuation semantics."""

    def setup_method(self):
        self.df = _make_df(PRICES)
        self.env = _make_env(self.df)
        # Run one fewer step than prices so the last price is "tomorrow"
        self.n_steps = len(PRICES) - 1  # 5 steps from 6 prices
        self.rollout = _run_always_long(self.env, self.n_steps)

    def test_rollout_steps(self):
        assert self.rollout.shape[0] == self.n_steps

    def test_extract_returns_not_none(self):
        series = extract_tradingenv_return_series(self.env, self.n_steps)
        assert series is not None, "extract_tradingenv_return_series returned None"

    def test_nlv_length(self):
        series = extract_tradingenv_return_series(self.env, self.n_steps)
        # context_pre of step 0  + context_post of each step → n_steps + 1 values
        assert len(series.values) == self.n_steps + 1, (
            f"Expected {self.n_steps + 1} NLV values, got {len(series.values)}"
        )

    def test_initial_nlv(self):
        series = extract_tradingenv_return_series(self.env, self.n_steps)
        assert abs(series.values[0] - INITIAL_CASH) < 1.0, (
            f"Initial NLV {series.values[0]:.2f} != expected {INITIAL_CASH:.2f}"
        )

    def test_nlv_path_matches_tradingenv_semantics(self):
        """NLV[t] = initial_cash * PRICES[t-1] / PRICES[0] for t >= 1.

        The broker records NLV at the trade price (PRICES[t-1]), not the
        next price (PRICES[t]).  PRICES[-1]=99 is never incorporated.
        """
        series = extract_tradingenv_return_series(self.env, self.n_steps)
        nlv = series.values  # length n_steps + 1

        # NLV[0] is initial cash (pre-first-trade)
        # NLV[t] for t>=1 reflects PRICES[t-1]
        expected_nlv = [INITIAL_CASH] + [
            INITIAL_CASH * PRICES[t] / PRICES[0] for t in range(self.n_steps)
        ]

        print("\nNLV path (always-long, buy at trade price):")
        print(f"{'t':>3}  {'price_used':>10}  {'nlv_actual':>12}  {'nlv_expected':>12}")
        for t, (actual, expected) in enumerate(zip(nlv, expected_nlv)):
            price_label = f"PRICES[{t-1}]={PRICES[t-1]:.0f}" if t > 0 else "initial"
            print(f"{t:>3}  {price_label:>10}  {actual:>12.4f}  {expected:>12.4f}")

        for t, (actual, expected) in enumerate(zip(nlv, expected_nlv)):
            assert abs(actual - expected) < 0.01, (
                f"NLV mismatch at t={t}: actual={actual:.6f} expected={expected:.6f}"
            )

    def test_first_return_is_zero(self):
        """r[0] is always 0: the agent buys at the current price, NLV unchanged."""
        series = extract_tradingenv_return_series(self.env, self.n_steps)
        r = series.to_simple().values
        assert abs(r[0]) < 1e-10, f"Expected r[0]=0 but got {r[0]}"

    def test_subsequent_returns_match_price_relatives(self):
        """r[t] = PRICES[t] / PRICES[t-1] - 1 for t >= 1."""
        series = extract_tradingenv_return_series(self.env, self.n_steps)
        r = series.to_simple().values  # length n_steps

        print("\nReturn comparison:")
        print(f"{'t':>3}  {'actual':>12}  {'expected':>12}  {'diff':>12}")
        for t in range(1, self.n_steps):
            expected = PRICES[t] / PRICES[t - 1] - 1.0
            print(f"{t:>3}  {r[t]:>12.8f}  {expected:>12.8f}  {r[t]-expected:>12.8f}")
            assert abs(r[t] - expected) < 1e-6, (
                f"Return mismatch at t={t}: actual={r[t]:.8f} expected={expected:.8f}"
            )

    def test_total_return_covers_observed_prices_only(self):
        """Total return = PRICES[n_steps-1] / PRICES[0] - 1.

        The last price PRICES[n_steps]=99 is not reached, so the strategy
        ends at PRICES[n_steps-1]=100 → total return = 0%, not -1%.
        This also confirms the strategy vs benchmark off-by-one: the
        benchmark window uses prices[:max_steps+1] which *does* include
        PRICES[n_steps], adding one extra price move to the comparison.
        """
        series = extract_tradingenv_return_series(self.env, self.n_steps)
        r = series.to_simple().values
        equity = np.cumprod(1.0 + r)
        total_return = equity[-1] - 1.0

        # Strategy ends at PRICES[n_steps-1]=PRICES[4]=100, not PRICES[5]=99
        expected = PRICES[self.n_steps - 1] / PRICES[0] - 1.0  # = 0.0
        assert abs(total_return - expected) < 1e-6, (
            f"Total return {total_return:.6f} != expected {expected:.6f}\n"
            f"Note: PRICES[-1]={PRICES[-1]} is never seen; last observed is "
            f"PRICES[{self.n_steps - 1}]={PRICES[self.n_steps - 1]}"
        )
