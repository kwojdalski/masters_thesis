"""Tests for _resolve_symbol_index in data/preparation.py."""

from __future__ import annotations

from trading_rl.data.preparation import _resolve_symbol_index


class TestStrategyFirst:
    def test_always_returns_zero(self):
        assert _resolve_symbol_index("first", n_symbols=5) == 0

    def test_returns_zero_regardless_of_n_symbols(self):
        for n in [1, 3, 10, 100]:
            assert _resolve_symbol_index("first", n_symbols=n) == 0

    def test_unknown_strategy_falls_back_to_first(self):
        assert _resolve_symbol_index("unknown_xyz", n_symbols=5) == 0

    def test_empty_strategy_string_falls_back_to_first(self):
        assert _resolve_symbol_index("", n_symbols=5) == 0


class TestStrategyRandom:
    def test_returns_value_in_valid_range(self):
        for _ in range(20):
            idx = _resolve_symbol_index("random", n_symbols=5)
            assert 0 <= idx < 5

    def test_single_symbol_always_returns_zero(self):
        for _ in range(10):
            assert _resolve_symbol_index("random", n_symbols=1) == 0

    def test_returns_integer(self):
        idx = _resolve_symbol_index("random", n_symbols=4)
        assert isinstance(idx, int)

    def test_same_seed_reproduces_the_same_draw(self):
        first = _resolve_symbol_index("random", n_symbols=100, seed=42)
        assert all(
            _resolve_symbol_index("random", n_symbols=100, seed=42) == first
            for _ in range(10)
        )

    def test_does_not_always_return_same_value_over_many_calls(self):
        results = {_resolve_symbol_index("random", n_symbols=100) for _ in range(200)}
        assert len(results) > 1, (
            "expected multiple distinct draws from 100 symbols in 200 calls"
        )


class TestStrategyRotated:
    """Rotation is a pure function of (n_symbols, seed).

    It used to read and increment a counter file shared by every pooled
    scenario, so the agents compared inside one hypothesis each drew a
    different symbol and no rerun reproduced the assignment (#518).
    """

    def test_index_is_seed_modulo_n_symbols(self):
        for seed, expected in [(0, 0), (1, 1), (5, 5), (6, 0), (43, 1)]:
            assert _resolve_symbol_index("rotated", n_symbols=6, seed=seed) == expected

    def test_repeated_calls_with_same_seed_are_stable(self):
        """The regression guard: resolving must not mutate shared state."""
        indices = [
            _resolve_symbol_index("rotated", n_symbols=6, seed=42) for _ in range(10)
        ]
        assert indices == [42 % 6] * 10

    def test_scenarios_sharing_a_seed_agree(self):
        """H1/H2/H3 compare algorithms and must hold the eval symbol fixed."""
        td3 = _resolve_symbol_index("rotated", n_symbols=6, seed=42)
        ddpg = _resolve_symbol_index("rotated", n_symbols=6, seed=42)
        ppo = _resolve_symbol_index("rotated", n_symbols=6, seed=42)
        random_baseline = _resolve_symbol_index("rotated", n_symbols=6, seed=42)
        assert td3 == ddpg == ppo == random_baseline

    def test_differing_seeds_cover_distinct_symbols(self):
        n = 4
        seen = [_resolve_symbol_index("rotated", n_symbols=n, seed=s) for s in range(n)]
        assert sorted(seen) == list(range(n))

    def test_missing_seed_returns_zero(self):
        assert _resolve_symbol_index("rotated", n_symbols=5, seed=None) == 0

    def test_index_stays_in_range_for_large_seeds(self):
        for seed in [7, 99, 100_000]:
            assert 0 <= _resolve_symbol_index("rotated", n_symbols=6, seed=seed) < 6
