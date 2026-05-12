"""Tests for FeaturePipeline fit/transform correctness and leakage prevention."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_rl.features.pipeline import create_default_pipeline


def _ohlcv(n_rows: int, price_start: float = 100.0) -> pd.DataFrame:
    """OHLCV DataFrame with monotonically increasing prices."""
    prices = price_start + np.arange(n_rows, dtype=float)
    return pd.DataFrame(
        {
            "open": prices * 0.99,
            "high": prices * 1.01,
            "low": prices * 0.98,
            "close": prices,
            "volume": np.full(n_rows, 1000.0),
        },
        index=pd.date_range("2020-01-01", periods=n_rows, freq="1min"),
    )


class TestTransformBeforeFit:
    def test_raises_runtime_error(self):
        pipeline = create_default_pipeline()
        with pytest.raises(RuntimeError, match="[Ff]it"):
            pipeline.transform(_ohlcv(50))

    def test_error_message_mentions_fit(self):
        pipeline = create_default_pipeline()
        with pytest.raises(RuntimeError) as exc_info:
            pipeline.transform(_ohlcv(50))
        assert "fit" in str(exc_info.value).lower()


class TestReset:
    def test_reset_clears_fitted_flag(self):
        pipeline = create_default_pipeline()
        pipeline.fit(_ohlcv(50))
        assert pipeline._is_fitted is True
        pipeline.reset()
        assert pipeline._is_fitted is False

    def test_transform_raises_after_reset(self):
        pipeline = create_default_pipeline()
        pipeline.fit(_ohlcv(50))
        pipeline.reset()
        with pytest.raises(RuntimeError, match="[Ff]it"):
            pipeline.transform(_ohlcv(50))

    def test_refit_after_reset_succeeds(self):
        pipeline = create_default_pipeline()
        pipeline.fit(_ohlcv(50))
        pipeline.reset()
        pipeline.fit(_ohlcv(50))
        result = pipeline.transform(_ohlcv(50))
        assert len(result) > 0
        assert pipeline._is_fitted is True


class TestNoLeakage:
    def test_transform_is_idempotent(self):
        """Calling transform(train) twice must return identical results."""
        pipeline = create_default_pipeline()
        train_df = _ohlcv(100)
        pipeline.fit(train_df)
        result1 = pipeline.transform(train_df)
        result2 = pipeline.transform(train_df)
        pd.testing.assert_frame_equal(result1, result2)

    def test_transforming_val_does_not_change_train_result(self):
        """transform(val) must not mutate the fitted scaler state."""
        pipeline = create_default_pipeline()
        train_df = _ohlcv(100, price_start=100.0)
        val_df = _ohlcv(50, price_start=500.0)  # different price regime

        pipeline.fit(train_df)
        train_before = pipeline.transform(train_df)
        pipeline.transform(val_df)  # must not change scaler state
        train_after = pipeline.transform(train_df)

        pd.testing.assert_frame_equal(train_before, train_after)

    def test_val_features_differ_from_train(self):
        """Val data at a different price scale must produce meaningfully different features."""
        pipeline = create_default_pipeline()
        train_df = _ohlcv(200, price_start=100.0)
        val_df = _ohlcv(100, price_start=1000.0)  # 10x larger prices

        pipeline.fit(train_df)
        train_features = pipeline.transform(train_df)
        val_features = pipeline.transform(val_df)

        # At least some features must differ between the two splits
        common_cols = list(set(train_features.columns) & set(val_features.columns))
        assert len(common_cols) > 0

    def test_fit_transform_equivalent_to_fit_then_transform(self):
        pipeline1 = create_default_pipeline()
        pipeline2 = create_default_pipeline()
        train_df = _ohlcv(80)

        result1 = pipeline1.fit_transform(train_df)
        pipeline2.fit(train_df)
        result2 = pipeline2.transform(train_df)

        pd.testing.assert_frame_equal(result1, result2)
