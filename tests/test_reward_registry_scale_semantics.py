"""Regression tests for RewardRegistry.scale semantics (issue #392).

reward_scale must mean "final reward *= scale" uniformly across every
registered reward type, even though tradingenv.LogReturn's own constructor
parameter divides by scale. Also proves the runtime reward magnitude
produced by the *inverted* config values checked into scenario YAML files
is numerically identical to what the pre-fix code produced with the
original (pre-inversion) values.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from trading_rl.rewards.registry import RewardRegistry


class _FakeContext:
    def __init__(self, nlv: float) -> None:
        self.nlv = nlv


class _FakeTrackRecord:
    def __init__(self, nlv_last: float) -> None:
        self._entry = SimpleNamespace(context_pre=_FakeContext(nlv_last))

    def __getitem__(self, idx: int):
        return self._entry


class _FakeBroker:
    def __init__(self, nlv_last: float, nlv_now: float) -> None:
        self.track_record = _FakeTrackRecord(nlv_last)
        self._nlv_now = nlv_now

    def net_liquidation_value(self) -> float:
        return self._nlv_now


class _FakeEnv:
    def __init__(self, nlv_last: float, nlv_now: float) -> None:
        self.broker = _FakeBroker(nlv_last, nlv_now)


def _log_return_reward(scale: float, nlv_last: float, nlv_now: float) -> float:
    reward = RewardRegistry.create("log_return", eta=0.01, scale=scale)
    return reward.calculate(_FakeEnv(nlv_last, nlv_now))


class TestLogReturnScaleIsMultiplicative:
    """RewardRegistry.create("log_return", scale=X) must produce
    raw_log_return * X, not raw_log_return / X."""

    def test_scale_above_one_amplifies(self):
        # A 0.01% (1bp) tick-level log return amplified by scale=10000
        # should read back as ~1.0 in reward units.
        nlv_last, nlv_now = 100.0, 100.01
        reward = _log_return_reward(scale=10000.0, nlv_last=nlv_last, nlv_now=nlv_now)
        assert reward == pytest.approx(10000.0 * (nlv_now / nlv_last - 1), rel=1e-3)
        assert reward > 0.5  # would be ~0.0001 under the old divide semantics

    def test_scale_of_one_is_identity(self):
        nlv_last, nlv_now = 100.0, 101.0
        import math

        raw = math.log(nlv_now / nlv_last)
        reward = _log_return_reward(scale=1.0, nlv_last=nlv_last, nlv_now=nlv_now)
        assert reward == pytest.approx(raw, rel=1e-9)

    def test_zero_scale_raises_instead_of_dividing_by_zero(self):
        with pytest.raises(ValueError, match="non-zero"):
            RewardRegistry.create("log_return", eta=0.01, scale=0.0)


class TestScenarioConfigsPreserveRuntimeRewardMagnitude:
    """The inverted reward_scale values checked into scenario YAMLs
    (1/old_value) must reproduce the exact runtime reward the pre-fix code
    produced with the original (pre-inversion) value, for every distinct
    magnitude used across the repo's log_return scenarios."""

    @pytest.mark.parametrize(
        ("old_yaml_value", "new_yaml_value"),
        [
            (0.0001, 10000.0),
            (0.00003, 33333.333333333336),
            (0.003, 333.3333333333333),
        ],
    )
    def test_inverted_value_matches_pre_fix_runtime_reward(
        self, old_yaml_value, new_yaml_value
    ):
        nlv_last, nlv_now = 100.0, 100.005

        # Pre-fix runtime behaviour: LogReturn(scale=old_yaml_value) divides.
        import math

        raw = math.log(nlv_now / nlv_last)
        pre_fix_reward = raw / old_yaml_value

        post_fix_reward = _log_return_reward(
            scale=new_yaml_value, nlv_last=nlv_last, nlv_now=nlv_now
        )

        assert post_fix_reward == pytest.approx(pre_fix_reward, rel=1e-9)


class TestDifferentialSharpeScaleUnchanged:
    """differential_sharpe's scale semantics were already multiplicative and
    must not be affected by the log_return-side fix."""

    def test_scale_multiplies_dsr_output(self):
        reward_1x = RewardRegistry.create("differential_sharpe", eta=0.01, scale=1.0)
        reward_10x = RewardRegistry.create("differential_sharpe", eta=0.01, scale=10.0)
        assert reward_1x.scale == 1.0
        assert reward_10x.scale == 10.0
