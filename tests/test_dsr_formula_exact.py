"""Exact numeric value tests for DifferentialSharpeRatio.

These tests verify the formula uses OLD EMA values (A_{t-1}, B_{t-1}) to compute
the DSR numerator and THEN updates the EMAs — the order mandated by Moody & Saffell (2001).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from trading_rl.rewards import DifferentialSharpeRatio


def _run_dsr(nlvs: list[float], eta: float = 0.01, epsilon: float = 1e-8):
    """Feed nlv sequence through DifferentialSharpeRatio; return (rewards, dsr)."""
    dsr = DifferentialSharpeRatio(eta=eta, epsilon=epsilon)
    rewards = []
    for nlv in nlvs:
        env = MagicMock()
        env.broker.net_liquidation_value.return_value = float(nlv)
        rewards.append(dsr.calculate(env))
    return rewards, dsr


class TestDSRFirstStep:
    def test_first_call_zero(self):
        rewards, _ = _run_dsr([1000.0])
        assert rewards[0] == 0.0

    def test_second_step_numerator_is_zero(self):
        """Step 2: A_{t-1}=B_{t-1}=0, so numerator=B*ΔA - A*ΔB/2=0 regardless of return."""
        rewards, _ = _run_dsr([100.0, 110.0])
        assert rewards[1] == pytest.approx(0.0, abs=1e-12)

    def test_second_step_zero_for_any_return(self):
        for factor in [0.8, 1.0, 1.5, 2.0]:
            rewards, _ = _run_dsr([100.0, 100.0 * factor])
            assert rewards[1] == pytest.approx(0.0, abs=1e-12), f"factor={factor}"


class TestEMAStateAfterFirstReturn:
    def test_ema_state_exact(self):
        """After step 2: A_t = eta*R_1, B_t = eta*R_1^2."""
        eta = 0.3
        R1 = np.log(110.0 / 100.0)
        _, dsr = _run_dsr([100.0, 110.0], eta=eta)
        assert dsr.A_t == pytest.approx(eta * R1, rel=1e-10)
        assert dsr.B_t == pytest.approx(eta * R1 ** 2, rel=1e-10)


class TestDSRExactValue:
    def test_constant_return_step3(self):
        """Three steps with constant 10 % return and eta=0.5: verify exact DSR."""
        eta = 0.5
        epsilon = 1e-8
        R = np.log(1.1)

        # After step 2: A_old = eta*R, B_old = eta*R^2
        A_old = eta * R
        B_old = eta * R ** 2
        delta_A = R - A_old
        delta_B = R ** 2 - B_old
        variance = B_old - A_old ** 2
        denominator = max(variance, 0.0) ** 1.5 + epsilon
        expected = (B_old * delta_A - A_old * delta_B / 2) / denominator

        rewards, _ = _run_dsr([100.0, 110.0, 121.0], eta=eta, epsilon=epsilon)
        assert rewards[2] == pytest.approx(expected, rel=1e-6)

    def test_mixed_return_step3(self):
        """Negative third step return; verify exact DSR against reference formula."""
        eta = 0.4
        epsilon = 1e-8
        R1 = np.log(1100.0 / 1000.0)
        R2 = np.log(1045.0 / 1100.0)

        A_old = eta * R1
        B_old = eta * R1 ** 2
        delta_A = R2 - A_old
        delta_B = R2 ** 2 - B_old
        variance = B_old - A_old ** 2
        denominator = max(variance, 0.0) ** 1.5 + epsilon
        expected = (B_old * delta_A - A_old * delta_B / 2) / denominator

        rewards, _ = _run_dsr([1000.0, 1100.0, 1045.0], eta=eta, epsilon=epsilon)
        assert rewards[2] == pytest.approx(expected, rel=1e-6)


class TestOldEMAsUsed:
    def test_old_emas_not_new(self):
        """Implementation must use A_{t-1}/B_{t-1} for the numerator, not the updated values."""
        eta = 0.5
        epsilon = 1e-8
        R1 = np.log(110.0 / 100.0)
        R2 = np.log(105.0 / 110.0)  # negative return

        A_old = eta * R1
        B_old = eta * R1 ** 2

        # Correct formula: OLD EMAs in numerator
        delta_A = R2 - A_old
        delta_B = R2 ** 2 - B_old
        variance = B_old - A_old ** 2
        denom = max(variance, 0.0) ** 1.5 + epsilon
        correct = (B_old * delta_A - A_old * delta_B / 2) / denom

        # Wrong formula: updated EMAs in numerator
        A_new = (1 - eta) * A_old + eta * R2
        B_new = (1 - eta) * B_old + eta * R2 ** 2
        delta_A_w = R2 - A_new
        delta_B_w = R2 ** 2 - B_new
        variance_w = B_new - A_new ** 2
        denom_w = max(variance_w, 0.0) ** 1.5 + epsilon
        wrong = (B_new * delta_A_w - A_new * delta_B_w / 2) / denom_w

        rewards, _ = _run_dsr([100.0, 110.0, 105.0], eta=eta, epsilon=epsilon)

        assert rewards[2] == pytest.approx(correct, rel=1e-6)
        # The two formulas produce clearly different results
        assert abs(rewards[2] - wrong) > 0.5
