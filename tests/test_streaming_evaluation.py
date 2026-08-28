import logging
from pathlib import Path

import pandas as pd
import pytest

import trading_rl.pipeline.evaluation as evaluation_module
from trading_rl.config import ExperimentConfig
from trading_rl.data.cache import (
    _load_prepared_pipeline_state,
    _write_prepared_cache_metadata,
    _write_prepared_pipeline_state,
)
from trading_rl.data_loading import save_prepared_splits, save_symbol_memmap
from trading_rl.data_utils import _prepared_cache_compatible


def _priced_frame(n_rows: int) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n_rows, freq="s")
    close = pd.Series([100.0 + i for i in range(n_rows)], index=idx, dtype=float)
    return pd.DataFrame(
        {
            "close": close,
            "feature_signal": close.pct_change().fillna(0.0),
        },
        index=idx,
    )


def _cache_config(tmp_path: Path, data_path: Path) -> ExperimentConfig:
    return ExperimentConfig.from_dict(
        {
            "data": {
                "data_path": str(data_path),
                "train_size": 5,
                "validation_size": 2,
                "test_size": 2,
                "lazy_load": True,
                "prepared_data_dir": str(tmp_path / "prepared"),
                "memmap_dir": str(tmp_path / "memmap"),
                "download_data": False,
                # warmup_rows defaults to 300, which would exceed this test's
                # tiny train_size=5 and make the cache row-count check trivially
                # pass (expected = max(0, train_size - warmup_rows) = 0).
                "warmup_rows": 0,
            },
            "env": {
                "backend": "tradingenv",
                "price_column": "close",
                "feature_columns": ["feature_signal"],
            },
            "training": {
                "algorithm": "TD3",
                "max_steps": 10,
                "frames_per_batch": 5,
            },
            "evaluation": {
                "eval_steps": 3,
            },
            "logging": {"log_dir": str(tmp_path / "logs")},
        }
    )


def _cache_config_with_sizes(
    tmp_path: Path,
    data_path: Path,
    *,
    train_size: int,
    validation_size: int,
    test_size: int,
) -> ExperimentConfig:
    config = _cache_config(tmp_path, data_path)
    config.data.train_size = train_size
    config.data.validation_size = validation_size
    config.data.test_size = test_size
    return config


def _multi_symbol_cache_config(
    tmp_path: Path,
    train_paths: list[Path],
    val_paths: list[Path],
) -> ExperimentConfig:
    return ExperimentConfig.from_dict(
        {
            "data": {
                "data_path": str(train_paths[0]),
                "data_paths": [str(path) for path in train_paths],
                "val_data_paths": [str(path) for path in val_paths],
                "train_size": 5,
                "validation_size": 50,
                "test_size": 50,
                "lazy_load": True,
                "prepared_data_dir": str(tmp_path / "prepared_multi"),
                "memmap_dir": str(tmp_path / "memmap_multi"),
                "download_data": False,
            },
            "env": {
                "backend": "tradingenv",
                "mode": "hft",
                "price_column": "close",
                "feature_columns": ["feature_signal"],
            },
            "training": {
                "algorithm": "TD3",
                "max_steps": 10,
                "frames_per_batch": 5,
            },
            "evaluation": {
                "eval_steps": 3,
            },
            "logging": {"log_dir": str(tmp_path / "logs")},
        }
    )


def test_prepared_cache_rejects_insufficient_memmap_rows(tmp_path: Path) -> None:
    """A memmap with fewer rows than requested must be rejected."""
    data_path = tmp_path / "raw.parquet"
    _priced_frame(12).to_parquet(data_path)
    config = _cache_config(tmp_path, data_path)
    prepared_dir = Path(config.data.prepared_data_dir)
    memmap_dir = Path(config.data.memmap_dir)

    save_prepared_splits(
        _priced_frame(5),
        _priced_frame(2),
        _priced_frame(2),
        prepared_dir,
    )
    save_symbol_memmap(_priced_frame(3), memmap_dir, prefix="0")  # 3 < train_size=5

    assert not _prepared_cache_compatible(
        config,
        prepared_dir,
        memmap_dir,
        logger=logging.getLogger(__name__),
    )


def test_prepared_cache_accepts_larger_memmap_rows(tmp_path: Path) -> None:
    """A memmap with more rows than requested is valid; caller slices to train_size."""
    data_path = tmp_path / "raw.parquet"
    _priced_frame(12).to_parquet(data_path)
    config = _cache_config(tmp_path, data_path)
    prepared_dir = Path(config.data.prepared_data_dir)
    memmap_dir = Path(config.data.memmap_dir)

    save_prepared_splits(
        _priced_frame(5),
        _priced_frame(2),
        _priced_frame(2),
        prepared_dir,
    )
    save_symbol_memmap(_priced_frame(10), memmap_dir, prefix="0")  # 10 >= train_size=5

    assert _prepared_cache_compatible(
        config,
        prepared_dir,
        memmap_dir,
        logger=logging.getLogger(__name__),
    )


def test_prepared_cache_metadata_accepts_smaller_requested_splits(
    tmp_path: Path,
) -> None:
    """A larger compatible cache can be reused for a smaller split request."""
    data_path = tmp_path / "raw.parquet"
    _priced_frame(20).to_parquet(data_path)
    large_config = _cache_config_with_sizes(
        tmp_path,
        data_path,
        train_size=10,
        validation_size=4,
        test_size=4,
    )
    small_config = _cache_config_with_sizes(
        tmp_path,
        data_path,
        train_size=5,
        validation_size=2,
        test_size=2,
    )
    prepared_dir = Path(large_config.data.prepared_data_dir)
    memmap_dir = Path(large_config.data.memmap_dir)
    train_df = _priced_frame(10)
    val_df = _priced_frame(4)
    test_df = _priced_frame(4)

    save_prepared_splits(train_df, val_df, test_df, prepared_dir)
    memmap_paths = [save_symbol_memmap(train_df, memmap_dir, prefix="0")]
    _write_prepared_cache_metadata(
        large_config,
        prepared_dir,
        train_df,
        val_df,
        test_df,
        memmap_paths,
    )

    assert _prepared_cache_compatible(
        small_config,
        prepared_dir,
        memmap_dir,
        logger=logging.getLogger(__name__),
    )


def test_prepared_cache_metadata_rejects_changed_data_source(tmp_path: Path) -> None:
    """Metadata must invalidate a cache if the raw data source changes."""
    old_data_path = tmp_path / "old.parquet"
    new_data_path = tmp_path / "new.parquet"
    _priced_frame(12).to_parquet(old_data_path)
    _priced_frame(12).to_parquet(new_data_path)
    old_config = _cache_config(tmp_path, old_data_path)
    new_config = _cache_config(tmp_path, new_data_path)
    prepared_dir = Path(old_config.data.prepared_data_dir)
    memmap_dir = Path(old_config.data.memmap_dir)
    train_df = _priced_frame(5)
    val_df = _priced_frame(2)
    test_df = _priced_frame(2)

    save_prepared_splits(train_df, val_df, test_df, prepared_dir)
    memmap_paths = [save_symbol_memmap(train_df, memmap_dir, prefix="0")]
    _write_prepared_cache_metadata(
        old_config,
        prepared_dir,
        train_df,
        val_df,
        test_df,
        memmap_paths,
    )

    assert not _prepared_cache_compatible(
        new_config,
        prepared_dir,
        memmap_dir,
        logger=logging.getLogger(__name__),
    )


def test_prepared_cache_metadata_rejects_changed_filter_lob_levels(
    tmp_path: Path,
) -> None:
    """filter_lob_levels changes which rows are prepared, so it must
    invalidate a cache built under a different value."""
    data_path = tmp_path / "raw.parquet"
    _priced_frame(12).to_parquet(data_path)
    old_config = _cache_config(tmp_path, data_path)
    old_config.data.filter_lob_levels = None
    new_config = _cache_config(tmp_path, data_path)
    new_config.data.filter_lob_levels = 5
    prepared_dir = Path(old_config.data.prepared_data_dir)
    memmap_dir = Path(old_config.data.memmap_dir)
    train_df = _priced_frame(5)
    val_df = _priced_frame(2)
    test_df = _priced_frame(2)

    save_prepared_splits(train_df, val_df, test_df, prepared_dir)
    memmap_paths = [save_symbol_memmap(train_df, memmap_dir, prefix="0")]
    _write_prepared_cache_metadata(
        old_config,
        prepared_dir,
        train_df,
        val_df,
        test_df,
        memmap_paths,
    )

    assert not _prepared_cache_compatible(
        new_config,
        prepared_dir,
        memmap_dir,
        logger=logging.getLogger(__name__),
    )


def test_prepared_cache_metadata_rejects_changed_warmup_rows(tmp_path: Path) -> None:
    """warmup_rows drops early normalized training rows, so it must
    invalidate a cache built under a different value."""
    data_path = tmp_path / "raw.parquet"
    _priced_frame(12).to_parquet(data_path)
    old_config = _cache_config(tmp_path, data_path)
    old_config.data.warmup_rows = 300
    new_config = _cache_config(tmp_path, data_path)
    new_config.data.warmup_rows = 0
    prepared_dir = Path(old_config.data.prepared_data_dir)
    memmap_dir = Path(old_config.data.memmap_dir)
    train_df = _priced_frame(5)
    val_df = _priced_frame(2)
    test_df = _priced_frame(2)

    save_prepared_splits(train_df, val_df, test_df, prepared_dir)
    memmap_paths = [save_symbol_memmap(train_df, memmap_dir, prefix="0")]
    _write_prepared_cache_metadata(
        old_config,
        prepared_dir,
        train_df,
        val_df,
        test_df,
        memmap_paths,
    )

    assert not _prepared_cache_compatible(
        new_config,
        prepared_dir,
        memmap_dir,
        logger=logging.getLogger(__name__),
    )


def test_per_day_cache_does_not_require_uniform_val_test_row_counts(
    tmp_path: Path,
) -> None:
    """Per-day mode validates source/memmap identity but not fixed split row counts."""
    train_paths = [tmp_path / "AAPL_train.parquet", tmp_path / "MSFT_train.parquet"]
    val_paths = [tmp_path / "AAPL_val.parquet", tmp_path / "MSFT_val.parquet"]
    for path in [*train_paths, *val_paths]:
        _priced_frame(8).to_parquet(path)
    config = _multi_symbol_cache_config(tmp_path, train_paths, val_paths)
    prepared_dir = Path(config.data.prepared_data_dir)
    memmap_dir = Path(config.data.memmap_dir)
    train_df = _priced_frame(5)
    val_df = _priced_frame(2)
    test_df = _priced_frame(1)

    save_prepared_splits(train_df, val_df, test_df, prepared_dir)
    memmap_paths = [
        save_symbol_memmap(_priced_frame(5), memmap_dir, prefix="0", symbol="AAPL"),
        save_symbol_memmap(_priced_frame(5), memmap_dir, prefix="1", symbol="MSFT"),
    ]
    _write_prepared_cache_metadata(
        config,
        prepared_dir,
        train_df,
        val_df,
        test_df,
        memmap_paths,
    )

    assert _prepared_cache_compatible(
        config,
        prepared_dir,
        memmap_dir,
        logger=logging.getLogger(__name__),
    )


def test_evaluation_context_disables_training_memmaps(monkeypatch) -> None:
    calls = []

    class BuilderProbe:
        def create(self, df, config, *, use_memmap=True):
            calls.append(use_memmap)
            return object()

    monkeypatch.setattr(
        evaluation_module, "AlgorithmicEnvironmentBuilder", BuilderProbe
    )
    config = ExperimentConfig()
    config.evaluation.eval_steps = 3

    ctx = evaluation_module.build_evaluation_context_for_split(
        split="val",
        df=_priced_frame(5),
        config=config,
    )

    assert calls == [False]
    assert ctx.max_steps == 3


def test_prepared_pipeline_state_round_trips(tmp_path: Path) -> None:
    state = {"feature_x": {"mean": 1.0, "var": 2.0}}
    _write_prepared_pipeline_state(tmp_path, state)

    loaded = _load_prepared_pipeline_state(tmp_path, logging.getLogger(__name__))

    assert loaded == state


def test_prepared_pipeline_state_rejects_tampered_checksum(tmp_path: Path) -> None:
    _write_prepared_pipeline_state(tmp_path, {"feature_x": {"mean": 1.0}})
    (tmp_path / "pipeline_state.pkl.sha256").write_text("0" * 64)

    with pytest.raises(RuntimeError, match="invalid checksum"):
        _load_prepared_pipeline_state(tmp_path, logging.getLogger(__name__))


def test_prepared_pipeline_state_write_none_clears_existing_files(
    tmp_path: Path,
) -> None:
    _write_prepared_pipeline_state(tmp_path, {"feature_x": {"mean": 1.0}})
    _write_prepared_pipeline_state(tmp_path, None)

    assert _load_prepared_pipeline_state(tmp_path, logging.getLogger(__name__)) is None
    assert not (tmp_path / "pipeline_state.pkl").exists()
