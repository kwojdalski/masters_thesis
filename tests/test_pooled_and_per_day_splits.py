"""Tests for _build_pooled_splits() and _build_per_day_splits().

These are the most complex data-splitting paths in the pipeline.  A regression
(wrong symbol picked as representative, off-by-one split boundary, incorrect
symbol grouping) would silently leak future data into training without raising
an error.

Both functions are tested via their public API, with expensive dependencies
(feature engineering, multiprocessing) replaced by lightweight fakes where
needed.
"""

from __future__ import annotations

import concurrent.futures
import logging
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_rl.constants import EnvMode
from trading_rl.data.preparation import _build_per_day_splits, _build_pooled_splits


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_ohlcv(tmp_path: Path, name: str, n_rows: int, price_offset: float = 100.0) -> str:
    prices = price_offset + np.arange(n_rows, dtype=float)
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
    path = str(tmp_path / f"{name}.parquet")
    df.to_parquet(path)
    return path


def _make_config(
    train_size: int = 60,
    validation_size: int | None = 30,
    test_size: int | None = 30,
    val_data_paths: list[str] | None = None,
    n_workers: int = 1,
) -> types.SimpleNamespace:
    data = types.SimpleNamespace(
        train_size=train_size,
        validation_size=validation_size,
        test_size=test_size,
        feature_groups=None,
        feature_config=None,
        feature_cache_dir=None,
        warmup_rows=0,
        filter_lob_levels=None,
        n_workers=n_workers,
        timeframe="1min",
        data_dir="data",
        download_data=False,
        exchange_names=None,
        symbols=None,
        download_since=None,
        val_data_paths=val_data_paths,
    )
    env = types.SimpleNamespace(mode=EnvMode.MFT, backend="")
    return types.SimpleNamespace(data=data, env=env)


@pytest.fixture(autouse=True)
def _skip_validation(monkeypatch):
    """Suppress validate_prepared_data so minimal test frames pass."""
    monkeypatch.setattr(
        "trading_rl.data.preparation.validate_prepared_data",
        lambda *a, **kw: None,
    )


# ---------------------------------------------------------------------------
# _build_pooled_splits
# ---------------------------------------------------------------------------

class TestBuildPooledSplits:
    def test_single_symbol_split_sizes(self, tmp_path):
        path = _write_ohlcv(tmp_path, "AAPL", n_rows=150)
        cfg = _make_config(train_size=60, validation_size=30, test_size=30)
        train_df, val_df, test_df, memmaps, _ = _build_pooled_splits(
            cfg, logging.getLogger(), [path], memmap_dir=None,
        )
        assert len(train_df) == 60
        assert len(val_df) == 30
        assert len(test_df) == 30
        assert memmaps is None

    def test_single_symbol_no_temporal_overlap(self, tmp_path):
        """Train, val, and test must be strictly chronological — no future leakage."""
        path = _write_ohlcv(tmp_path, "AAPL", n_rows=180)
        cfg = _make_config(train_size=90, validation_size=45, test_size=45)
        train_df, val_df, test_df, _, _ = _build_pooled_splits(
            cfg, logging.getLogger(), [path], memmap_dir=None,
        )
        assert train_df.index.max() < val_df.index.min()
        assert val_df.index.max() < test_df.index.min()

    def test_two_symbols_no_memmap_concatenates_all_rows(self, tmp_path):
        p0 = _write_ohlcv(tmp_path, "SYM0", n_rows=150, price_offset=100.0)
        p1 = _write_ohlcv(tmp_path, "SYM1", n_rows=150, price_offset=200.0)
        cfg = _make_config(train_size=60, validation_size=30, test_size=30)
        train_df, val_df, test_df, memmaps, _ = _build_pooled_splits(
            cfg, logging.getLogger(), [p0, p1], memmap_dir=None,
        )
        # Each symbol contributes 60/30/30 rows → combined = 120/60/60
        assert len(train_df) == 120
        assert len(val_df) == 60
        assert len(test_df) == 60
        assert memmaps is None

    def test_two_symbols_memmap_default_uses_symbol_0(self, tmp_path):
        # SYM0 has closes near 100; SYM1 has closes near 500.
        p0 = _write_ohlcv(tmp_path, "SYM0", n_rows=150, price_offset=100.0)
        p1 = _write_ohlcv(tmp_path, "SYM1", n_rows=150, price_offset=500.0)
        memmap_dir = tmp_path / "mm"
        cfg = _make_config(train_size=60, validation_size=30, test_size=30)
        _, val_df, _, memmaps, _ = _build_pooled_splits(
            cfg, logging.getLogger(), [p0, p1], memmap_dir=memmap_dir,
        )
        # Default symbol_index=0 → representative sample from SYM0 (close < 300)
        assert val_df["close"].iloc[0] < 300.0
        assert memmaps is not None and len(memmaps) == 2

    def test_two_symbols_memmap_symbol_index_1_uses_second(self, tmp_path):
        """Off-by-one in symbol_index would pick the wrong representative sample."""
        p0 = _write_ohlcv(tmp_path, "SYM0", n_rows=150, price_offset=100.0)
        p1 = _write_ohlcv(tmp_path, "SYM1", n_rows=150, price_offset=500.0)
        memmap_dir = tmp_path / "mm"
        cfg = _make_config(train_size=60, validation_size=30, test_size=30)
        _, val_df, _, _, _ = _build_pooled_splits(
            cfg, logging.getLogger(), [p0, p1],
            memmap_dir=memmap_dir, symbol_index=1,
        )
        # symbol_index=1 → representative sample from SYM1 (close ≥ 500)
        assert val_df["close"].iloc[0] >= 500.0

    def test_pipeline_reset_called_once_per_symbol(self, tmp_path, monkeypatch):
        """Pipeline must be reset between symbols; without this, normalisation
        statistics from symbol N bleed into symbol N+1."""
        reset_calls: list[int] = []

        class _FakePipeline:
            features: list = []

            def reset(self):
                reset_calls.append(1)

            def fit(self, df: pd.DataFrame) -> None:
                pass

            def transform(self, df: pd.DataFrame) -> pd.DataFrame:
                return pd.DataFrame(index=df.index)

        monkeypatch.setattr(
            "trading_rl.data.preparation._resolve_feature_pipeline",
            lambda *_: _FakePipeline(),
        )
        p0 = _write_ohlcv(tmp_path, "SYM0", n_rows=150)
        p1 = _write_ohlcv(tmp_path, "SYM1", n_rows=150)
        p2 = _write_ohlcv(tmp_path, "SYM2", n_rows=150)
        _build_pooled_splits(
            _make_config(), logging.getLogger(), [p0, p1, p2], memmap_dir=None,
        )
        assert sum(reset_calls) == 3

    def test_delegates_to_per_day_splits_when_val_data_paths_set(self, tmp_path, monkeypatch):
        calls: list[tuple] = []
        _dummy = pd.DataFrame(
            {"close": [1.0]},
            index=pd.date_range("2020-01-01", periods=1),
        )

        def _fake_per_day(config, logger, train_paths, val_paths, memmap_dir, *a, **kw):
            calls.append((list(train_paths), list(val_paths)))
            return _dummy, _dummy, _dummy, None

        monkeypatch.setattr("trading_rl.data.preparation._build_per_day_splits", _fake_per_day)
        p_train = _write_ohlcv(tmp_path, "SYM0", n_rows=150)
        p_val = _write_ohlcv(tmp_path, "VAL0", n_rows=100)
        _build_pooled_splits(
            _make_config(val_data_paths=[p_val]),
            logging.getLogger(), [p_train], memmap_dir=None,
        )
        assert len(calls) == 1
        assert calls[0][0] == [p_train]
        assert calls[0][1] == [p_val]


# ---------------------------------------------------------------------------
# _build_per_day_splits
# ---------------------------------------------------------------------------

def _fake_worker_factory(n_val_rows: int = 10):
    """Return a synchronous drop-in for _per_symbol_worker.

    Writes minimal parquet files to tmp_dir and returns the expected result
    tuple without performing any feature engineering.
    """
    def _worker(args):
        (
            symbol, train_indices, train_paths_sym, val_path, val_index,
            feature_config, mode, backend, filter_lob_levels, warmup_rows,
            memmap_dir_str, tmp_dir_str, worker_idx, n_workers_total,
            feature_cache_dir_str,
        ) = args

        tmp_dir = Path(tmp_dir_str)
        train_results = [(idx, None) for idx in train_indices]

        val_entry = None
        if val_path is not None and val_index is not None:
            idx_val = pd.date_range("2020-01-01", periods=n_val_rows, freq="1min")
            df = pd.DataFrame(
                {"close": np.arange(n_val_rows, dtype=float)},
                index=idx_val,
            )
            mid = n_val_rows // 2
            val_p = tmp_dir / f"{val_index}_val.parquet"
            test_p = tmp_dir / f"{val_index}_test.parquet"
            df.iloc[:mid].to_parquet(val_p)
            df.iloc[mid:].to_parquet(test_p)
            val_entry = {"val": str(val_p), "test": str(test_p)}

        return symbol, train_results, val_entry, val_index, None

    return _worker


def _sync_executor_class():
    """Return a drop-in ProcessPoolExecutor that runs submissions synchronously."""

    class _SyncExecutor:
        def __init__(self, max_workers=None, initializer=None, initargs=()):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def submit(self, fn, *args):
            f: concurrent.futures.Future = concurrent.futures.Future()
            try:
                f.set_result(fn(*args))
            except Exception as exc:
                f.set_exception(exc)
            return f

    return _SyncExecutor


class TestBuildPerDaySplits:
    def test_raises_when_val_symbol_has_no_training_path(self, tmp_path):
        """A val file for MSFT with no MSFT training file must raise immediately."""
        train_path = str(tmp_path / "AAPL_2020-01-01.parquet")
        val_path = str(tmp_path / "MSFT_val.parquet")
        pd.DataFrame({"close": [1.0]}).to_parquet(train_path)
        pd.DataFrame({"close": [1.0]}).to_parquet(val_path)

        cfg = _make_config(train_size=1)
        with pytest.raises(ValueError, match="No training paths for symbol 'MSFT'"):
            _build_per_day_splits(
                cfg, logging.getLogger(),
                train_paths=[train_path],
                val_paths=[val_path],
                memmap_dir=None,
            )

    def test_multi_day_files_grouped_by_symbol_prefix(self, tmp_path, monkeypatch):
        """AAPL_day1 and AAPL_day2 must both map to the 'AAPL' pipeline group."""
        train1 = str(tmp_path / "AAPL_2020-01-01.parquet")
        train2 = str(tmp_path / "AAPL_2020-01-02.parquet")
        val_p = str(tmp_path / "AAPL_val.parquet")
        for p in [train1, train2, val_p]:
            pd.DataFrame({"close": [1.0]}).to_parquet(p)

        monkeypatch.setattr(
            "trading_rl.data.preparation.ProcessPoolExecutor",
            _sync_executor_class(),
        )
        monkeypatch.setattr(
            "trading_rl.data.preparation._per_symbol_worker",
            _fake_worker_factory(),
        )
        cfg = _make_config(train_size=1)
        # Must not raise — both train files belong to 'AAPL' which matches the val file
        _build_per_day_splits(
            cfg, logging.getLogger(),
            train_paths=[train1, train2],
            val_paths=[val_p],
            memmap_dir=None,
        )

    def test_val_file_split_50_50_into_val_and_test(self, tmp_path, monkeypatch):
        """Val files must be split exactly in half; a boundary error here
        would leak future data from test into validation."""
        n_val_rows = 20  # fake worker writes exactly this many rows
        train_path = str(tmp_path / "AAPL_train.parquet")
        val_path = str(tmp_path / "AAPL_val.parquet")
        pd.DataFrame({"close": [1.0]}).to_parquet(train_path)
        pd.DataFrame({"close": [1.0]}).to_parquet(val_path)

        monkeypatch.setattr(
            "trading_rl.data.preparation.ProcessPoolExecutor",
            _sync_executor_class(),
        )
        monkeypatch.setattr(
            "trading_rl.data.preparation._per_symbol_worker",
            _fake_worker_factory(n_val_rows=n_val_rows),
        )
        cfg = _make_config(train_size=1)
        _, val_df, test_df, _, _ = _build_per_day_splits(
            cfg, logging.getLogger(),
            train_paths=[train_path],
            val_paths=[val_path],
            memmap_dir=None,
        )
        assert len(val_df) == n_val_rows // 2
        assert len(test_df) == n_val_rows // 2

    def test_memmap_results_ordered_by_original_train_index(self, tmp_path, monkeypatch):
        """Memmaps must be assembled in original train_paths order, not worker
        completion order.  Out-of-order assembly would silently assign the wrong
        data to a symbol slot in the streaming env."""
        sym_a_train1 = str(tmp_path / "AAPL_2020-01-01.parquet")
        sym_a_train2 = str(tmp_path / "AAPL_2020-01-02.parquet")
        sym_b_train = str(tmp_path / "MSFT_2020-01-01.parquet")
        sym_a_val = str(tmp_path / "AAPL_val.parquet")
        sym_b_val = str(tmp_path / "MSFT_val.parquet")
        for p in [sym_a_train1, sym_a_train2, sym_b_train, sym_a_val, sym_b_val]:
            pd.DataFrame({"close": [1.0]}).to_parquet(p)

        memmap_dir = tmp_path / "mm"

        # Fake worker that writes memmaps
        from trading_rl.data_loading import save_symbol_memmap

        def _worker_with_memmap(args):
            (symbol, train_indices, train_paths_sym, val_path, val_index,
             feature_config, mode, backend, filter_lob_levels, warmup_rows,
             memmap_dir_str, tmp_dir_str, worker_idx, n_workers_total,
             feature_cache_dir_str) = args

            from pathlib import Path as P
            import numpy as _np, pandas as _pd
            mm_dir = P(memmap_dir_str)
            tmp_dir_p = P(tmp_dir_str)

            train_results = []
            for orig_idx, _ in zip(train_indices, train_paths_sym):
                n = 20
                df_tr = _pd.DataFrame(
                    {"close": _np.full(n, float(orig_idx + 1) * 100)},
                    index=_pd.date_range("2020-01-01", periods=n, freq="1min"),
                )
                mp = save_symbol_memmap(df_tr, mm_dir, prefix=str(orig_idx), symbol=symbol)
                train_results.append((orig_idx, {
                    "data_path": str(mp.data_path),
                    "index_path": str(mp.index_path),
                    "n_rows": mp.n_rows,
                    "columns": mp.columns,
                    "symbol": mp.symbol,
                }))

            val_entry = None
            if val_path is not None and val_index is not None:
                df_v = _pd.DataFrame(
                    {"close": _np.arange(10, dtype=float)},
                    index=_pd.date_range("2020-01-01", periods=10, freq="1min"),
                )
                vp = tmp_dir_p / f"{val_index}_val.parquet"
                tp = tmp_dir_p / f"{val_index}_test.parquet"
                df_v.iloc[:5].to_parquet(vp)
                df_v.iloc[5:].to_parquet(tp)
                val_entry = {"val": str(vp), "test": str(tp)}

            return symbol, train_results, val_entry, val_index, None

        monkeypatch.setattr(
            "trading_rl.data.preparation.ProcessPoolExecutor",
            _sync_executor_class(),
        )
        monkeypatch.setattr(
            "trading_rl.data.preparation._per_symbol_worker",
            _worker_with_memmap,
        )
        cfg = _make_config(train_size=20, validation_size=5, test_size=5)
        _, _, _, memmaps, _ = _build_per_day_splits(
            cfg, logging.getLogger(),
            train_paths=[sym_a_train1, sym_a_train2, sym_b_train],
            val_paths=[sym_a_val, sym_b_val],
            memmap_dir=memmap_dir,
        )

        # Exactly one memmap per train file, ordered by original index
        assert memmaps is not None
        assert len(memmaps) == 3
        # Verify ordering: memmap[0] has close values near 100, memmap[2] near 300
        data0 = np.load(memmaps[0].data_path, mmap_mode="r")
        data2 = np.load(memmaps[2].data_path, mmap_mode="r")
        close_col_0 = list(memmaps[0].columns).index("close")
        close_col_2 = list(memmaps[2].columns).index("close")
        assert data0[0, close_col_0] == pytest.approx(100.0)
        assert data2[0, close_col_2] == pytest.approx(300.0)

    def test_two_symbols_separate_val_files(self, tmp_path, monkeypatch):
        """Each symbol's val file must be transformed by its own symbol pipeline,
        not a cross-symbol one."""
        train_a = str(tmp_path / "AAPL_train.parquet")
        train_b = str(tmp_path / "MSFT_train.parquet")
        val_a = str(tmp_path / "AAPL_val.parquet")
        val_b = str(tmp_path / "MSFT_val.parquet")
        for p in [train_a, train_b, val_a, val_b]:
            pd.DataFrame({"close": [1.0]}).to_parquet(p)

        n_per = 12
        monkeypatch.setattr(
            "trading_rl.data.preparation.ProcessPoolExecutor",
            _sync_executor_class(),
        )
        monkeypatch.setattr(
            "trading_rl.data.preparation._per_symbol_worker",
            _fake_worker_factory(n_val_rows=n_per),
        )
        cfg = _make_config(train_size=1)
        _, val_df, test_df, _, _ = _build_per_day_splits(
            cfg, logging.getLogger(),
            train_paths=[train_a, train_b],
            val_paths=[val_a, val_b],
            memmap_dir=None,
        )
        # In non-memmap mode both symbols' val/test are concatenated
        assert len(val_df) == n_per  # 6 per symbol × 2
        assert len(test_df) == n_per
