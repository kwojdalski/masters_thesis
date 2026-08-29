"""Tests for feature look-ahead / data leakage detection.

Two independent checks for the same property — a feature at time t must not
encode information that is only available after time t:

  Causality test (equivalent to "lag-1 equivalence" investigation step 3)
  -----------------------------------------------------------------------
  Compute a feature on a prefix of the data, then on the full dataset.
  For a causal feature the values at every row in the prefix are identical
  regardless of how many future rows exist.  A feature that uses shift(-1)
  violates this: the last row of the prefix changes once the next row is
  appended.

  Correlation bound test (investigation step 2)
  ----------------------------------------------
  On synthetic random-walk LOB data, the correlation between feature[t] and
  the next-step mid-price return (t → t+1) must be below a threshold.  A
  feature that directly encodes the next step's price change will have
  correlation close to 1.0.

Both tests include a sanity-check case using ``mid_price_future_velocity``,
the one intentionally-leaking feature in the codebase, to prove the test is
sensitive enough to catch real look-ahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_rl.features.base import FeatureConfig
from trading_rl.features.registry import FeatureRegistry

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

_N_ROWS = 300
_N_PREFIX = 150  # first half used as the "prefix"
_WARMUP = 70  # skip; rolling features need history to stabilise


@pytest.fixture(scope="module")
def lob_df() -> pd.DataFrame:
    """Synthetic LOB snapshot DataFrame with all columns required by the
    scenario features.  Mid-price follows a random walk so adjacent values
    are never identical (guarantees non-trivial diffs for the sanity check).
    """
    rng = np.random.default_rng(42)
    mid = 100.0 + rng.standard_normal(_N_ROWS).cumsum() * 0.10
    spread = 0.05
    data: dict[str, np.ndarray] = {}
    for i in range(5):
        data[f"bid_px_{i:02d}"] = mid - spread / 2 - i * 0.10
        data[f"ask_px_{i:02d}"] = mid + spread / 2 + i * 0.10
        data[f"bid_sz_{i:02d}"] = rng.uniform(50, 200, _N_ROWS)
        data[f"ask_sz_{i:02d}"] = rng.uniform(50, 200, _N_ROWS)
    data["bid_ct_00"] = rng.integers(1, 20, _N_ROWS)
    data["ask_ct_00"] = rng.integers(1, 20, _N_ROWS)
    data["action"] = rng.choice(["A", "C", "T"], _N_ROWS, p=[0.5, 0.3, 0.2])
    data["side"] = rng.choice(["B", "A"], _N_ROWS)
    data["size"] = rng.uniform(1, 500, _N_ROWS)
    idx = pd.date_range("2024-01-02 09:30:00", periods=_N_ROWS, freq="100ms")
    return pd.DataFrame(data, index=idx)


def _compute(feature_type: str, df: pd.DataFrame, **params) -> pd.Series:
    cfg = FeatureConfig(
        name=f"test_{feature_type}",
        feature_type=feature_type,
        domain="hft",
        params=params if params else None,
    )
    return FeatureRegistry.create(cfg).compute(df)


def _mid_next_return(df: pd.DataFrame) -> pd.Series:
    """Simple return from t to t+1 using mid-price — the target for correlation checks."""
    mid = (df["bid_px_00"] + df["ask_px_00"]) / 2.0
    return mid.shift(-1) / mid - 1


# ---------------------------------------------------------------------------
# Production features used in the selected-DSR scenario (observation.yaml)
# ---------------------------------------------------------------------------

_SCENARIO_FEATURES: list[tuple[str, dict, str]] = [
    ("book_pressure", {"level": 0}, "book_pressure_l0"),
    ("order_book_imbalance", {"levels": 3}, "order_book_imbalance_3l"),
    ("order_count_imbalance", {"level": 0}, "order_count_imbalance_l0"),
    ("microprice", {}, "microprice"),
    ("microprice_divergence", {}, "microprice_divergence"),
    ("bid_ask_slope", {"side": "bid", "levels": 5}, "bid_slope"),
    ("bid_ask_slope", {"side": "ask", "levels": 5}, "ask_slope"),
    ("ofi", {}, "ofi"),
    ("ofi_rolling", {"window": 50}, "ofi_rolling_50"),
    (
        "signed_trade_flow",
        {"window": 50, "action_col": "action", "side_col": "side", "size_col": "size"},
        "signed_trade_flow_50",
    ),
    ("mid_price_velocity", {}, "mid_price_velocity"),
]

_SCENARIO_IDS = [label for _, _, label in _SCENARIO_FEATURES]
_SCENARIO_PARAMS = [(ft, p) for ft, p, _ in _SCENARIO_FEATURES]


# ---------------------------------------------------------------------------
# Causality tests (investigation step 3)
# ---------------------------------------------------------------------------


class TestFeatureCausality:
    """Adding future rows to a DataFrame must not change the feature value at
    any existing row.  A feature using ``shift(-1)`` violates this because the
    last row in the prefix acquires the value of the next (now-visible) row.
    """

    @pytest.mark.parametrize("feature_type,params", _SCENARIO_PARAMS, ids=_SCENARIO_IDS)
    def test_no_future_bleed(
        self,
        lob_df: pd.DataFrame,
        feature_type: str,
        params: dict,
    ) -> None:
        df_prefix = lob_df.iloc[:_N_PREFIX]

        result_prefix = _compute(feature_type, df_prefix, **params)
        result_full = _compute(feature_type, lob_df, **params)

        # Including the last row of the prefix (_N_PREFIX - 1): a causal feature
        # at row t only reads rows 0..t, so the value is unchanged by appending
        # rows beyond t.
        prefix_vals = result_prefix.iloc[_WARMUP:].to_numpy(dtype=float)
        full_vals = result_full.iloc[_WARMUP:_N_PREFIX].to_numpy(dtype=float)

        np.testing.assert_allclose(
            prefix_vals,
            full_vals,
            rtol=1e-6,
            atol=1e-10,
            err_msg=(
                f"{feature_type}: value at row t changed when future rows were appended "
                f"— this indicates look-ahead / data leakage"
            ),
        )

    def test_intentional_lookahead_fails_causality(self, lob_df: pd.DataFrame) -> None:
        """``mid_price_future_velocity`` uses shift(-1) and must fail the
        causality test — proving the test is sensitive enough to catch real
        look-ahead.  The last row of the prefix is 0 (fillna), but in the
        full computation it holds the actual next-step price diff.
        """
        df_prefix = lob_df.iloc[:_N_PREFIX]

        result_prefix = _compute("mid_price_future_velocity", df_prefix)
        result_full = _compute("mid_price_future_velocity", lob_df)

        prefix_vals = result_prefix.iloc[_WARMUP:].to_numpy(dtype=float)
        full_vals = result_full.iloc[_WARMUP:_N_PREFIX].to_numpy(dtype=float)

        # The arrays must NOT be equal — if they were, the test above would fail
        # to catch real look-ahead.
        assert not np.allclose(
            prefix_vals, full_vals, rtol=1e-6, atol=1e-10, equal_nan=False
        ), (
            "mid_price_future_velocity should fail the causality check "
            "(shift(-1) makes the last prefix row depend on the first future row). "
            "If this assertion fails the fixture data has a zero diff — increase _N_ROWS."
        )


# ---------------------------------------------------------------------------
# Correlation-bound tests (investigation step 2)
# ---------------------------------------------------------------------------

_CORR_THRESHOLD = 0.5  # |corr| above this on random data signals look-ahead


class TestFeatureReturnCorrelation:
    """On synthetic random-walk LOB data, a legitimate feature at time t must
    not be strongly correlated with the next-step mid-price return.  High
    correlation (above the threshold) is only achievable by encoding future
    price information.
    """

    @pytest.mark.parametrize("feature_type,params", _SCENARIO_PARAMS, ids=_SCENARIO_IDS)
    def test_correlation_with_next_return_is_bounded(
        self,
        lob_df: pd.DataFrame,
        feature_type: str,
        params: dict,
    ) -> None:
        feature_vals = _compute(feature_type, lob_df, **params)
        next_ret = _mid_next_return(lob_df)

        combined = (
            pd.DataFrame({"f": feature_vals, "r": next_ret})
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .iloc[_WARMUP:]
        )
        assert len(combined) >= 50, (
            f"Too few clean rows to compute correlation for {feature_type} "
            f"(got {len(combined)}, need >= 50)"
        )

        corr = abs(float(combined["f"].corr(combined["r"])))
        assert corr < _CORR_THRESHOLD, (
            f"{feature_type}: |corr(feature[t], return[t→t+1])| = {corr:.3f} "
            f">= threshold {_CORR_THRESHOLD} on random-walk data — "
            f"possible look-ahead bias"
        )

    def test_intentional_lookahead_exceeds_correlation_threshold(
        self, lob_df: pd.DataFrame
    ) -> None:
        """``mid_price_future_velocity`` encodes exactly the next-step price
        change and must have near-unity correlation with the next return,
        proving the threshold is tight enough to catch real look-ahead.
        """
        feature_vals = _compute("mid_price_future_velocity", lob_df)
        next_ret = _mid_next_return(lob_df)

        combined = (
            pd.DataFrame({"f": feature_vals, "r": next_ret})
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .iloc[_WARMUP:]
        )
        corr = abs(float(combined["f"].corr(combined["r"])))
        assert corr >= _CORR_THRESHOLD, (
            f"mid_price_future_velocity |corr| = {corr:.3f} < threshold {_CORR_THRESHOLD}. "
            f"The correlation test may not be sensitive enough to catch real look-ahead."
        )
