"""LazyDataFrame accessors that must not materialise the frame (#514, #519)."""

from __future__ import annotations

import pandas as pd
import pytest

from trading_rl.data_loading import LazyDataFrame


@pytest.fixture
def parquet(tmp_path):
    df = pd.DataFrame(
        {"close": range(500), "volume": range(500, 1000)},
        index=pd.date_range("2026-01-01", periods=500, freq="s"),
    )
    path = tmp_path / "split.parquet"
    df.to_parquet(path)
    return path, df


def test_n_rows_does_not_cache_the_frame(parquet):
    path, df = parquet
    lazy = LazyDataFrame(path)

    assert lazy.n_rows == len(df)
    assert lazy._df is None, "n_rows must not pin the frame in memory"


def test_len_still_materialises(parquet):
    """len() keeps its loading behaviour; only n_rows is the cheap path."""
    path, df = parquet
    lazy = LazyDataFrame(path)

    assert len(lazy) == len(df)
    assert lazy._df is not None


def test_head_rows_matches_iloc_including_index(parquet):
    path, df = parquet
    lazy = LazyDataFrame(path)

    head = lazy.head_rows(64)

    assert head.equals(df.iloc[:64])
    assert head.index.equals(df.iloc[:64].index)
    assert isinstance(head.index, pd.DatetimeIndex)


def test_head_rows_does_not_cache_the_frame(parquet):
    path, _ = parquet
    lazy = LazyDataFrame(path)

    lazy.head_rows(64)

    assert lazy._df is None, "head_rows must not pin the frame in memory"


def test_head_rows_caps_at_available_rows(parquet):
    path, df = parquet
    lazy = LazyDataFrame(path)

    assert len(lazy.head_rows(10_000)) == len(df)


def test_head_rows_uses_cache_when_already_loaded(parquet):
    path, df = parquet
    lazy = LazyDataFrame(path)
    len(lazy)  # force the load

    assert lazy.head_rows(32).equals(df.iloc[:32])
