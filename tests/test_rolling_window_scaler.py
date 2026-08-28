"""Regression test: RollingWindowScaler.transform must clip +-inf like its
sibling scalers (RunningMeanStd, TimeWeightedRunningMeanStd) do (issue #475)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_rl.features.base import RollingWindowScaler


class TestRollingWindowScalerInfHandling:
    def test_inf_input_value_does_not_leak_into_output(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, np.inf, 5.0, 6.0])
        scaler = RollingWindowScaler(window=3, min_periods=1).fit(s)

        result = scaler.transform(s)

        assert np.all(np.isfinite(result.values))

    def test_neg_inf_input_value_does_not_leak_into_output(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, -np.inf, 5.0, 6.0])
        scaler = RollingWindowScaler(window=3, min_periods=1).fit(s)

        result = scaler.transform(s)

        assert np.all(np.isfinite(result.values))

    def test_finite_input_is_unaffected(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 3.5, 5.0, 6.0])
        scaler = RollingWindowScaler(window=3, min_periods=1).fit(s)

        result = scaler.transform(s)

        assert np.all(np.isfinite(result.values))
        # values past the warmup window should be genuinely nonzero, not
        # incidentally clipped by the inf-guard
        assert not np.any(result.values[2:] == 0.0)
