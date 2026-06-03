from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from trading_rl.constants import SplitName
from trading_rl.data.loading import build_feature_pipeline_with_state
from trading_rl.data_loading import (
    LazyDataFrame,
    load_memmap_paths,
    load_prepared_splits,
    save_prepared_splits,
    save_symbol_memmap,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "close": [100.0, 101.0, 102.0],
            "feature_signal": [0.0, 0.1, 0.2],
            "symbol": ["AAPL", "AAPL", "AAPL"],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="min"),
    )


def _feature_config(tmp_path) -> str:
    path = tmp_path / "features.yaml"
    path.write_text(
        "features:\n"
        "  - name: px\n"
        "    feature_type: column_value\n"
        "    output_name: feature_px\n"
        "    reset_on_session_break: false\n"
        "    params:\n"
        "      column: close\n",
        encoding="utf-8",
    )
    return str(path)


def test_lazy_dataframe_repr_shows_cache_state(tmp_path) -> None:
    path = tmp_path / "data.parquet"
    _frame().to_parquet(path)
    lazy = LazyDataFrame(path)

    assert "not loaded" in repr(lazy)
    assert len(lazy) == 3
    assert "cached" in repr(lazy)


def test_lazy_dataframe_column_access_matches_parquet(tmp_path) -> None:
    path = tmp_path / "data.parquet"
    df = _frame()
    df.to_parquet(path)

    lazy = LazyDataFrame(path)

    pd.testing.assert_series_equal(lazy["close"], df["close"], check_freq=False)
    assert lazy.shape == df.shape
    assert list(lazy.columns) == list(df.columns)


def test_lazy_dataframe_private_attribute_raises_attribute_error(tmp_path) -> None:
    path = tmp_path / "data.parquet"
    _frame().to_parquet(path)
    lazy = LazyDataFrame(path)

    with pytest.raises(AttributeError):
        _ = lazy._missing_private_attr


def test_save_prepared_splits_writes_all_split_files(tmp_path) -> None:
    paths = save_prepared_splits(
        _frame(), _frame().iloc[:2], _frame().iloc[:1], tmp_path
    )

    assert set(paths) == {SplitName.TRAIN, SplitName.VAL, SplitName.TEST}
    for path in paths.values():
        assert path.exists()


def test_load_prepared_splits_returns_lazy_dataframes(tmp_path) -> None:
    save_prepared_splits(_frame(), _frame().iloc[:2], _frame().iloc[:1], tmp_path)

    splits = load_prepared_splits(tmp_path)

    assert set(splits) == {SplitName.TRAIN, SplitName.VAL, SplitName.TEST}
    assert isinstance(splits[SplitName.TRAIN], LazyDataFrame)
    assert len(splits[SplitName.TEST]) == 1


def test_load_prepared_splits_raises_when_split_missing(tmp_path) -> None:
    (_frame()).to_parquet(tmp_path / "train_prepared.parquet")

    with pytest.raises(FileNotFoundError, match="val_prepared"):
        load_prepared_splits(tmp_path)


def test_save_symbol_memmap_drops_non_numeric_columns_and_preserves_symbol(
    tmp_path,
) -> None:
    paths = save_symbol_memmap(_frame(), tmp_path, prefix="0", symbol="AAPL")

    assert paths.symbol == "AAPL"
    assert paths.columns == ["close", "feature_signal"]
    data = np.load(paths.data_path)
    assert data.dtype == np.float32
    assert data.shape == (3, 2)


def test_save_symbol_memmap_stores_datetime_index_as_nanoseconds(tmp_path) -> None:
    df = _frame()
    paths = save_symbol_memmap(df, tmp_path, prefix="0")

    index_ns = np.load(paths.index_path)

    np.testing.assert_array_equal(index_ns, df.index.astype(np.int64).to_numpy())


def test_save_symbol_memmap_falls_back_to_range_index_for_non_datetime_index(
    tmp_path,
) -> None:
    df = _frame().reset_index(drop=True)
    paths = save_symbol_memmap(df, tmp_path, prefix="0")

    index_values = np.load(paths.index_path)

    np.testing.assert_array_equal(index_values, np.arange(len(df), dtype=np.int64))


def test_load_memmap_paths_discovers_complete_memmaps_sorted_by_prefix(
    tmp_path,
) -> None:
    save_symbol_memmap(_frame(), tmp_path, prefix="1", symbol="MSFT")
    save_symbol_memmap(_frame(), tmp_path, prefix="0", symbol="AAPL")

    paths = load_memmap_paths(tmp_path)

    assert [path.symbol for path in paths] == ["AAPL", "MSFT"]
    assert [path.n_rows for path in paths] == [3, 3]


def test_load_memmap_paths_skips_incomplete_memmap(tmp_path) -> None:
    np.save(tmp_path / "0_train_data.npy", np.ones((2, 2), dtype=np.float32))

    assert load_memmap_paths(tmp_path) == []


def test_load_memmap_paths_reads_legacy_memmap_without_symbol(tmp_path) -> None:
    np.save(tmp_path / "0_train_data.npy", np.ones((2, 2), dtype=np.float32))
    np.save(tmp_path / "0_train_index.npy", np.arange(2, dtype=np.int64))
    (tmp_path / "0_columns.json").write_text(json.dumps(["a", "b"]))

    paths = load_memmap_paths(tmp_path)

    assert len(paths) == 1
    assert paths[0].symbol == ""


def test_build_feature_pipeline_with_state_restores_provided_scaler_state(
    tmp_path,
) -> None:
    state = {"feature_px": {"mean": 100.0, "var": 25.0, "count": 10}}

    result = build_feature_pipeline_with_state(
        _feature_config(tmp_path),
        feature_pipeline_state=state,
    )
    features = result.pipeline.transform(_frame())

    assert result.restored is True
    assert result.state_size == 1
    assert result.source == "provided"
    assert np.isclose(features["feature_px"].iloc[0], 0.0)
    assert np.isclose(features["feature_px"].iloc[1], 0.2, atol=1e-4)


def test_build_feature_pipeline_with_state_loads_checkpoint_state(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "feature_pipeline_state": {
                "feature_px": {"mean": 100.0, "var": 25.0, "count": 10}
            }
        },
        checkpoint_path,
    )

    result = build_feature_pipeline_with_state(
        _feature_config(tmp_path),
        checkpoint_path=checkpoint_path,
    )

    assert result.restored is True
    assert result.state_size == 1
    assert result.source == "checkpoint"
    assert np.isclose(
        result.pipeline.transform(_frame())["feature_px"].iloc[2],
        0.4,
        atol=1e-4,
    )
