"""Tests for prepare_data() split boundary logic.

Verifies that prepare_data splits BEFORE feature engineering so train/val/test
have the correct row counts and no temporal overlap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_rl.data_utils import PrepareDataConfig, prepare_data


def _write_ohlcv_parquet(tmp_path, n_rows: int, price_start: float = 100.0) -> str:
    """Write a minimal OHLCV parquet file and return its path."""
    prices = price_start + np.arange(n_rows, dtype=float)
    df = pd.DataFrame(
        {
            "open": prices * 0.99,
            "high": prices * 1.01,
            "low": prices * 0.98,
            "close": prices,
            "volume": np.full(n_rows, 1000.0),
        },
        index=pd.date_range("2020-01-01", periods=n_rows, freq="1min"),
    )
    path = str(tmp_path / "data.parquet")
    df.to_parquet(path)
    return path


def _cfg(**kwargs) -> PrepareDataConfig:
    """Build PrepareDataConfig with feature_cache_dir=None to skip caching."""
    return PrepareDataConfig(feature_cache_dir=None, **kwargs)


class TestExplicitSizes:
    def test_train_size_respected(self, tmp_path):
        path = _write_ohlcv_parquet(tmp_path, n_rows=200)
        train_df, _, _ = prepare_data(path, _cfg(train_size=100, validation_size=50, test_size=50))
        assert len(train_df) == 100

    def test_val_size_respected(self, tmp_path):
        path = _write_ohlcv_parquet(tmp_path, n_rows=200)
        _, val_df, _ = prepare_data(path, _cfg(train_size=100, validation_size=50, test_size=50))
        assert len(val_df) == 50

    def test_test_size_respected(self, tmp_path):
        path = _write_ohlcv_parquet(tmp_path, n_rows=200)
        _, _, test_df = prepare_data(path, _cfg(train_size=100, validation_size=50, test_size=50))
        assert len(test_df) == 50


class TestTemporalOrdering:
    def test_no_index_overlap_train_val(self, tmp_path):
        path = _write_ohlcv_parquet(tmp_path, n_rows=180)
        train_df, val_df, _ = prepare_data(
            path, _cfg(train_size=90, validation_size=45, test_size=45)
        )
        assert set(train_df.index).isdisjoint(set(val_df.index))

    def test_no_index_overlap_val_test(self, tmp_path):
        path = _write_ohlcv_parquet(tmp_path, n_rows=180)
        _, val_df, test_df = prepare_data(
            path, _cfg(train_size=90, validation_size=45, test_size=45)
        )
        assert set(val_df.index).isdisjoint(set(test_df.index))

    def test_val_starts_after_train(self, tmp_path):
        """Last train timestamp must be strictly before first val timestamp."""
        path = _write_ohlcv_parquet(tmp_path, n_rows=180)
        train_df, val_df, _ = prepare_data(
            path, _cfg(train_size=90, validation_size=45, test_size=45)
        )
        assert train_df.index.max() < val_df.index.min()

    def test_test_starts_after_val(self, tmp_path):
        path = _write_ohlcv_parquet(tmp_path, n_rows=180)
        _, val_df, test_df = prepare_data(
            path, _cfg(train_size=90, validation_size=45, test_size=45)
        )
        assert val_df.index.max() < test_df.index.min()


class TestDefaultSizes:
    def test_validation_size_none_splits_remaining_evenly(self, tmp_path):
        """When validation_size=None and test_size=None, remaining is split 50/50."""
        n_rows = 200
        train_size = 100
        path = _write_ohlcv_parquet(tmp_path, n_rows=n_rows)
        _, val_df, test_df = prepare_data(path, _cfg(train_size=train_size))
        remaining = n_rows - train_size
        expected_each = remaining // 2
        assert len(val_df) == expected_each
        assert len(test_df) == expected_each


class TestErrorCases:
    def test_train_size_too_large_raises(self, tmp_path):
        path = _write_ohlcv_parquet(tmp_path, n_rows=100)
        with pytest.raises(ValueError, match="train_size"):
            prepare_data(path, _cfg(train_size=100, validation_size=1, test_size=1))

    def test_validation_size_too_large_raises(self, tmp_path):
        path = _write_ohlcv_parquet(tmp_path, n_rows=100)
        with pytest.raises(ValueError):
            prepare_data(path, _cfg(train_size=50, validation_size=51))
