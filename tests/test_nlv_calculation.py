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

General NLV formula (no fees, any weight sequence):

  NLV[0]   = initial_cash               (pre-first-trade)
  NLV[1]   = initial_cash               (entry at price[0] — no P&L yet)
  NLV[t+1] = NLV[t] * (1 + w[t-1] * (price[t] / price[t-1] - 1))  for t >= 1

where w[t] is the portfolio weight set at step t and held until step t+1.
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
    return factory.make(
        df, config=config, feature_columns=["feature_signal"], price_column="close"
    )


def _expected_nlv(
    initial_cash: float, prices: list[float], weights: list[float]
) -> list[float]:
    """Reference implementation of the broker NLV formula (no fees).

    NLV[0] = initial_cash (pre-trade)
    NLV[1] = initial_cash (entry step — no P&L at entry price)
    NLV[t+1] = NLV[t] * (1 + weights[t-1] * (prices[t] / prices[t-1] - 1)) for t >= 1
    """
    nlv = [initial_cash, initial_cash]
    for t in range(1, len(weights)):
        ret = weights[t - 1] * (prices[t] / prices[t - 1] - 1.0)
        nlv.append(nlv[-1] * (1.0 + ret))
    return nlv


def _run_policy_from_weights(env, weights: list[float], n_steps: int):
    """Roll out n_steps using a deterministic sequence of portfolio weights."""
    from tensordict.nn import InteractionType
    from torchrl.envs.utils import set_exploration_type

    step_counter = [0]

    def _policy(td):
        i = step_counter[0]
        w = weights[i] if i < len(weights) else 0.0
        td["action"] = torch.tensor([w], dtype=torch.float64)
        step_counter[0] += 1
        return td

    with torch.no_grad():
        with set_exploration_type(InteractionType.DETERMINISTIC):
            rollout = env.rollout(max_steps=n_steps, policy=_policy)
    return rollout


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
        assert (
            len(series.values) == self.n_steps + 1
        ), f"Expected {self.n_steps + 1} NLV values, got {len(series.values)}"

    def test_initial_nlv(self):
        series = extract_tradingenv_return_series(self.env, self.n_steps)
        assert (
            abs(series.values[0] - INITIAL_CASH) < 1.0
        ), f"Initial NLV {series.values[0]:.2f} != expected {INITIAL_CASH:.2f}"

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
        for t, (actual, expected) in enumerate(zip(nlv, expected_nlv, strict=False)):
            price_label = f"PRICES[{t - 1}]={PRICES[t - 1]:.0f}" if t > 0 else "initial"
            print(f"{t:>3}  {price_label:>10}  {actual:>12.4f}  {expected:>12.4f}")

        for t, (actual, expected) in enumerate(zip(nlv, expected_nlv, strict=False)):
            assert (
                abs(actual - expected) < 0.01
            ), f"NLV mismatch at t={t}: actual={actual:.6f} expected={expected:.6f}"

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
            print(f"{t:>3}  {r[t]:>12.8f}  {expected:>12.8f}  {r[t] - expected:>12.8f}")
            assert (
                abs(r[t] - expected) < 1e-6
            ), f"Return mismatch at t={t}: actual={r[t]:.8f} expected={expected:.8f}"

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


class TestNLVAlwaysShort:
    """NLV path for always-short (weight = -1.0) must mirror always-long by symmetry.

    With w=-1 at every step:
      r[0] = 0
      r[t] = -1 * (price[t] / price[t-1] - 1)  for t >= 1
    """

    def setup_method(self):
        self.df = _make_df(PRICES)
        self.env = _make_env(self.df)
        self.n_steps = len(PRICES) - 1
        self.rollout = _run_policy_from_weights(
            self.env, [-1.0] * self.n_steps, self.n_steps
        )

    def test_first_return_is_zero(self):
        series = extract_tradingenv_return_series(self.env, self.n_steps)
        r = series.to_simple().values
        assert abs(r[0]) < 1e-10, f"Expected r[0]=0 but got {r[0]}"

    def test_nlv_path_matches_formula(self):
        series = extract_tradingenv_return_series(self.env, self.n_steps)
        nlv = series.values
        expected = _expected_nlv(INITIAL_CASH, PRICES, [-1.0] * self.n_steps)

        print("\nNLV path (always-short):")
        print(f"{'t':>3}  {'actual':>12}  {'expected':>12}")
        for t, (act, exp) in enumerate(zip(nlv, expected, strict=False)):
            print(f"{t:>3}  {act:>12.4f}  {exp:>12.4f}")
            assert (
                abs(act - exp) < 0.01
            ), f"NLV mismatch at t={t}: actual={act:.6f} expected={exp:.6f}"

    def test_short_loses_when_price_rises(self):
        """Price[1]=101 > Price[0]=100 → short position loses at step 1."""
        series = extract_tradingenv_return_series(self.env, self.n_steps)
        nlv = series.values
        assert (
            nlv[2] < INITIAL_CASH
        ), f"Short should lose when price rises 100→101 but NLV[2]={nlv[2]:.2f}"

    def test_short_gains_when_price_falls(self):
        """Price[4]=100 < Price[3]=101 → short position gains at step 4."""
        series = extract_tradingenv_return_series(self.env, self.n_steps)
        nlv = series.values
        # At t=4, price fell from PRICES[3]=101 to PRICES[4]=100.
        # NLV[4] reflects holding -1 over that move; compare with NLV[3].
        assert (
            nlv[4] > nlv[3]
        ), f"Short should gain when price falls 101→100: NLV[3]={nlv[3]:.2f} NLV[4]={nlv[4]:.2f}"

    def test_long_and_short_returns_are_negatives(self):
        """r_short[t] = -r_long[t] for t >= 1 (same prices, opposite weights)."""
        long_env = _make_env(_make_df(PRICES))
        _run_always_long(long_env, self.n_steps)
        r_long = (
            extract_tradingenv_return_series(long_env, self.n_steps).to_simple().values
        )
        r_short = (
            extract_tradingenv_return_series(self.env, self.n_steps).to_simple().values
        )

        for t in range(1, self.n_steps):
            assert (
                abs(r_short[t] + r_long[t]) < 1e-8
            ), f"r_short[{t}]={r_short[t]:.8f} + r_long[{t}]={r_long[t]:.8f} != 0"


# Variable-weight scenario:
#   prices  = [100, 110, 99, 88, 110]
#   weights = [+1,  -1,   0,  0]   (long → short → flat → flat)
#
# Expected NLV (manual):
#   NLV[0] = 10000
#   NLV[1] = 10000  (entry at 100, no P&L)
#   NLV[2] = 11000  (long 100→110, +10%)
#   NLV[3] = 12100  (short 110→99, +10%  — price fell while short)
#   NLV[4] = 12100  (flat 99→88,   0%   — zero weight)
_PRICES_VAR = [100.0, 110.0, 99.0, 88.0, 110.0]
_WEIGHTS_VAR = [+1.0, -1.0, 0.0, 0.0]
_EXPECTED_NLV_VAR = _expected_nlv(INITIAL_CASH, _PRICES_VAR, _WEIGHTS_VAR)


class TestNLVVariableWeights:
    """NLV with long→short→flat weight sequence on prices that stress each phase."""

    def setup_method(self):
        self.df = _make_df(_PRICES_VAR)
        self.env = _make_env(self.df)
        self.n_steps = len(_PRICES_VAR) - 1  # 4
        self.rollout = _run_policy_from_weights(self.env, _WEIGHTS_VAR, self.n_steps)

    def test_rollout_steps(self):
        assert self.rollout.shape[0] == self.n_steps

    def test_nlv_path_matches_formula(self):
        series = extract_tradingenv_return_series(self.env, self.n_steps)
        nlv = series.values
        assert len(nlv) == self.n_steps + 1

        print("\nNLV path (long→short→flat):")
        print(
            f"{'t':>3}  {'weight':>7}  {'price':>7}  {'actual':>12}  {'expected':>12}"
        )
        labels = ["entry"] + [f"w={w:+.0f}" for w in _WEIGHTS_VAR]
        prices = [_PRICES_VAR[0], *_PRICES_VAR]
        for t, (act, exp) in enumerate(zip(nlv, _EXPECTED_NLV_VAR, strict=False)):
            print(
                f"{t:>3}  {labels[t]:>7}  {prices[t]:>7.1f}  {act:>12.4f}  {exp:>12.4f}"
            )
            assert (
                abs(act - exp) < 0.01
            ), f"NLV mismatch at t={t}: actual={act:.6f} expected={exp:.6f}"

    def test_long_phase_profits_on_price_rise(self):
        """w=+1 while price 100→110: NLV[2] = 11000 > 10000."""
        nlv = extract_tradingenv_return_series(self.env, self.n_steps).values
        assert (
            abs(nlv[2] - 11_000.0) < 0.01
        ), f"Long phase NLV[2]={nlv[2]:.2f}, expected 11000"

    def test_short_phase_profits_on_price_drop(self):
        """w=-1 while price 110→99: NLV[3] = 12100 > 11000."""
        nlv = extract_tradingenv_return_series(self.env, self.n_steps).values
        assert (
            abs(nlv[3] - 12_100.0) < 0.01
        ), f"Short phase NLV[3]={nlv[3]:.2f}, expected 12100"

    def test_flat_phase_holds_nlv_constant(self):
        """w=0 while price 99→88 and 88→110: NLV[4] = NLV[3] = 12100."""
        nlv = extract_tradingenv_return_series(self.env, self.n_steps).values
        assert (
            abs(nlv[4] - nlv[3]) < 0.01
        ), f"Flat phase changed NLV: NLV[3]={nlv[3]:.2f} NLV[4]={nlv[4]:.2f}"

    def test_returns_match_weight_times_price_return(self):
        """r[t] = w[t-1] * (price[t] / price[t-1] - 1) for t >= 1."""
        series = extract_tradingenv_return_series(self.env, self.n_steps)
        r = series.to_simple().values

        assert abs(r[0]) < 1e-10, f"r[0] should be 0 but got {r[0]}"
        for t in range(1, self.n_steps):
            expected_r = _WEIGHTS_VAR[t - 1] * (
                _PRICES_VAR[t] / _PRICES_VAR[t - 1] - 1.0
            )
            assert abs(r[t] - expected_r) < 1e-8, (
                f"r[{t}]={r[t]:.8f}, expected w={_WEIGHTS_VAR[t - 1]:+.0f} * "
                f"({_PRICES_VAR[t]}/{_PRICES_VAR[t - 1]}-1) = {expected_r:.8f}"
            )
