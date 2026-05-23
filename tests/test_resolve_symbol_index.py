"""Tests for _resolve_symbol_index in data/preparation.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from trading_rl.data.preparation import _resolve_symbol_index


class TestStrategyFirst:
    def test_always_returns_zero(self):
        assert _resolve_symbol_index("first", n_symbols=5, memmap_dir=None) == 0

    def test_returns_zero_regardless_of_n_symbols(self):
        for n in [1, 3, 10, 100]:
            assert _resolve_symbol_index("first", n_symbols=n, memmap_dir=None) == 0

    def test_unknown_strategy_falls_back_to_first(self):
        assert _resolve_symbol_index("unknown_xyz", n_symbols=5, memmap_dir=None) == 0

    def test_empty_strategy_string_falls_back_to_first(self):
        assert _resolve_symbol_index("", n_symbols=5, memmap_dir=None) == 0


class TestStrategyRandom:
    def test_returns_value_in_valid_range(self):
        for _ in range(20):
            idx = _resolve_symbol_index("random", n_symbols=5, memmap_dir=None)
            assert 0 <= idx < 5

    def test_single_symbol_always_returns_zero(self):
        for _ in range(10):
            assert _resolve_symbol_index("random", n_symbols=1, memmap_dir=None) == 0

    def test_returns_integer(self):
        idx = _resolve_symbol_index("random", n_symbols=4, memmap_dir=None)
        assert isinstance(idx, int)

    def test_does_not_always_return_same_value_over_many_calls(self):
        results = {_resolve_symbol_index("random", n_symbols=100, memmap_dir=None) for _ in range(200)}
        assert len(results) > 1, "expected multiple distinct draws from 100 symbols in 200 calls"


class TestStrategyRotated:
    def test_first_call_returns_zero(self, tmp_path: Path):
        idx = _resolve_symbol_index("rotated", n_symbols=5, memmap_dir=tmp_path)
        assert idx == 0

    def test_second_call_returns_one(self, tmp_path: Path):
        _resolve_symbol_index("rotated", n_symbols=5, memmap_dir=tmp_path)
        idx = _resolve_symbol_index("rotated", n_symbols=5, memmap_dir=tmp_path)
        assert idx == 1

    def test_wraps_around_at_n_symbols(self, tmp_path: Path):
        n = 3
        for _ in range(n):
            _resolve_symbol_index("rotated", n_symbols=n, memmap_dir=tmp_path)
        idx = _resolve_symbol_index("rotated", n_symbols=n, memmap_dir=tmp_path)
        assert idx == 0

    def test_full_rotation_covers_all_symbols(self, tmp_path: Path):
        n = 4
        seen = [_resolve_symbol_index("rotated", n_symbols=n, memmap_dir=tmp_path) for _ in range(n)]
        assert sorted(seen) == list(range(n))

    def test_counter_file_is_created(self, tmp_path: Path):
        _resolve_symbol_index("rotated", n_symbols=3, memmap_dir=tmp_path)
        assert (tmp_path / ".eval_symbol_counter").exists()

    def test_counter_file_increments(self, tmp_path: Path):
        _resolve_symbol_index("rotated", n_symbols=3, memmap_dir=tmp_path)
        _resolve_symbol_index("rotated", n_symbols=3, memmap_dir=tmp_path)
        counter = int((tmp_path / ".eval_symbol_counter").read_text().strip())
        assert counter == 2

    def test_rotated_without_memmap_dir_returns_zero(self):
        idx = _resolve_symbol_index("rotated", n_symbols=5, memmap_dir=None)
        assert idx == 0

    def test_corrupt_counter_file_is_treated_as_zero(self, tmp_path: Path):
        (tmp_path / ".eval_symbol_counter").write_text("not_an_int")
        idx = _resolve_symbol_index("rotated", n_symbols=5, memmap_dir=tmp_path)
        assert idx == 0
