"""Tests for StreamingTradingEnv._load_window() and _next_symbol_idx().

Uses object.__new__ to bypass the TradingEnv constructor so we can test
the two private methods in isolation without launching a full RL environment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_rl.data_loading import MemmapPaths, save_symbol_memmap
from trading_rl.envs.streaming_env import StreamingTradingEnv


def _make_memmap(
    tmp_path,
    n_rows: int,
    n_cols: int = 3,
    prefix: str = "0",
    fill_value: float | None = None,
) -> MemmapPaths:
    """Write a memmap with data[i, j] = i*n_cols + j (or fill_value if given)."""
    if fill_value is not None:
        data = np.full((n_rows, n_cols), fill_value, dtype=np.float32)
    else:
        data = np.arange(n_rows * n_cols, dtype=np.float32).reshape(n_rows, n_cols)
    df = pd.DataFrame(
        data,
        columns=[f"col_{j}" for j in range(n_cols)],
        index=pd.date_range("2024-01-01", periods=n_rows, freq="s"),
    )
    return save_symbol_memmap(df, tmp_path, prefix)


def _bare_env(memmap_paths: list[MemmapPaths], episode_length: int) -> StreamingTradingEnv:
    """Construct StreamingTradingEnv with no TradingEnv constructor side effects."""
    env = object.__new__(StreamingTradingEnv)
    env._memmap_paths = memmap_paths
    env._episode_length = episode_length
    env._symbol_queue: list[int] = []
    env._symbol_rng = np.random.default_rng(0)
    return env


class TestLoadWindow:
    def test_shape(self, tmp_path):
        mp = _make_memmap(tmp_path, n_rows=20, n_cols=3)
        env = _bare_env([mp], episode_length=5)
        df = env._load_window(0, 0)
        assert df.shape == (5, 3)

    def test_column_names(self, tmp_path):
        mp = _make_memmap(tmp_path, n_rows=20, n_cols=3)
        env = _bare_env([mp], episode_length=5)
        df = env._load_window(0, 0)
        assert list(df.columns) == ["col_0", "col_1", "col_2"]

    def test_start_offset_first_row_values(self, tmp_path):
        """Window starting at row 7 must contain values from row 7 onward."""
        n_cols = 3
        mp = _make_memmap(tmp_path, n_rows=20, n_cols=n_cols)
        env = _bare_env([mp], episode_length=5)
        df = env._load_window(0, start=7)
        # row 7: values are 7*n_cols, 7*n_cols+1, 7*n_cols+2
        np.testing.assert_allclose(df.iloc[0].values, [21.0, 22.0, 23.0], atol=1e-6)

    def test_values_match_source(self, tmp_path):
        """All cells in the window must exactly match the original array."""
        n_cols = 3
        mp = _make_memmap(tmp_path, n_rows=20, n_cols=n_cols)
        env = _bare_env([mp], episode_length=5)
        df = env._load_window(0, start=3)
        expected = np.arange(3 * n_cols, 8 * n_cols, dtype=np.float32).reshape(5, n_cols)
        np.testing.assert_allclose(df.values, expected, atol=1e-6)

    def test_two_symbols_load_independently(self, tmp_path):
        """Symbol 0 and symbol 1 must each return their own data."""
        mp0 = _make_memmap(tmp_path, n_rows=20, n_cols=2, prefix="0")
        mp1 = _make_memmap(tmp_path, n_rows=20, n_cols=2, prefix="1", fill_value=99.0)
        env = _bare_env([mp0, mp1], episode_length=5)

        df0 = env._load_window(0, 0)
        df1 = env._load_window(1, 0)

        # Symbol 0 starts at 0, symbol 1 is all 99s
        assert df0.iloc[0, 0] == pytest.approx(0.0)
        np.testing.assert_allclose(df1.values, 99.0, atol=1e-6)

    def test_end_boundary(self, tmp_path):
        """Window that ends exactly at n_rows must have full episode_length rows."""
        n_rows = 10
        episode_length = 5
        mp = _make_memmap(tmp_path, n_rows=n_rows, n_cols=2)
        env = _bare_env([mp], episode_length=episode_length)
        df = env._load_window(0, start=n_rows - episode_length)
        assert len(df) == episode_length


class TestNextSymbolIdx:
    def test_single_symbol_always_returns_zero(self, tmp_path):
        mp = _make_memmap(tmp_path, n_rows=20)
        env = _bare_env([mp], episode_length=5)
        for _ in range(6):
            assert env._next_symbol_idx() == 0

    def test_all_symbols_seen_once_per_cycle(self, tmp_path):
        N = 5
        mps = [_make_memmap(tmp_path, n_rows=20, prefix=str(i)) for i in range(N)]
        env = _bare_env(mps, episode_length=5)
        seen = [env._next_symbol_idx() for _ in range(N)]
        assert sorted(seen) == list(range(N))

    def test_queue_empty_after_one_cycle(self, tmp_path):
        N = 3
        mps = [_make_memmap(tmp_path, n_rows=20, prefix=str(i)) for i in range(N)]
        env = _bare_env(mps, episode_length=5)
        for _ in range(N):
            env._next_symbol_idx()
        assert env._symbol_queue == []

    def test_queue_refills_for_second_cycle(self, tmp_path):
        """After one full cycle all N symbols appear again in the second cycle."""
        N = 4
        mps = [_make_memmap(tmp_path, n_rows=20, prefix=str(i)) for i in range(N)]
        env = _bare_env(mps, episode_length=5)
        first = sorted(env._next_symbol_idx() for _ in range(N))
        second = sorted(env._next_symbol_idx() for _ in range(N))
        assert first == list(range(N))
        assert second == list(range(N))

    def test_two_cycles_cover_all_symbols_twice(self, tmp_path):
        N = 3
        mps = [_make_memmap(tmp_path, n_rows=20, prefix=str(i)) for i in range(N)]
        env = _bare_env(mps, episode_length=5)
        all_draws = [env._next_symbol_idx() for _ in range(2 * N)]
        from collections import Counter
        counts = Counter(all_draws)
        for sym in range(N):
            assert counts[sym] == 2, f"symbol {sym} appeared {counts[sym]} times"
