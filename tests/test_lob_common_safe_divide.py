"""Regression test: safe_divide must return `fill` for NaN denominators too,
not just zero denominators (issue #478)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_rl.features.lob_common import safe_divide


class TestSafeDivideHandlesNanDenominator:
    def test_nan_denominator_returns_fill_not_nan(self):
        result = safe_divide(
            pd.Series([1.0, 2.0, 3.0]),
            pd.Series([0.0, np.nan, 5.0]),
            fill=0.0,
        )

        assert result.tolist() == [0.0, 0.0, 0.6]

    def test_zero_denominator_still_returns_fill(self):
        result = safe_divide(pd.Series([1.0]), pd.Series([0.0]), fill=-1.0)

        assert result.tolist() == [-1.0]

    def test_finite_nonzero_denominator_divides_normally(self):
        result = safe_divide(pd.Series([6.0]), pd.Series([3.0]))

        assert result.tolist() == [2.0]
