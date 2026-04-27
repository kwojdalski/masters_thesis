"""StreamingTradingEnv: episode-window loading from numpy memmap files.

Instead of holding the full training DataFrame in memory, each reset() picks a
random symbol file and a random start position, then loads exactly
``episode_length`` rows via numpy memmap.  Peak memory is therefore proportional
to the episode length, not the total dataset size.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from gym_trading_env.environments import TradingEnv

from logger import get_logger
from trading_rl.data_loading import MemmapPaths

logger = get_logger(__name__)


class StreamingTradingEnv(TradingEnv):
    """TradingEnv that streams episode windows from per-symbol numpy memmap files.

    On each ``reset()`` call the environment:
    1. Picks one symbol file at random from ``memmap_paths``.
    2. Picks a random start row such that ``start + episode_length <= n_rows``.
    3. Reads that slice from the memmap (tiny copy; the rest stays on disk).
    4. Reconstructs a DataFrame and calls ``_set_df`` to update internal arrays.

    The parent ``TradingEnv`` then runs its normal reset logic on the small
    window with ``max_episode_duration='max'`` so it walks the whole window.

    Args:
        memmap_paths: One :class:`~trading_rl.data_loading.MemmapPaths` per
            symbol, produced by :func:`~trading_rl.data_loading.save_symbol_memmap`.
        episode_length: Number of rows per episode window.
        **gym_kwargs: Forwarded to :class:`gym_trading_env.environments.TradingEnv`
            (positions, trading_fees, reward_function, etc.).
            Do *not* pass ``df`` or ``max_episode_duration``.
    """

    def __init__(
        self,
        memmap_paths: list[MemmapPaths],
        episode_length: int,
        **gym_kwargs,
    ) -> None:
        if not memmap_paths:
            raise ValueError("memmap_paths must contain at least one entry")

        self._memmap_paths = memmap_paths
        self._episode_length = episode_length

        # Bootstrap with a concrete window so the parent can set observation_space.
        bootstrap_df = self._load_window(0, 0)
        super().__init__(df=bootstrap_df, max_episode_duration="max", **gym_kwargs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_window(self, file_idx: int, start: int) -> pd.DataFrame:
        """Load ``episode_length`` rows starting at ``start`` from symbol ``file_idx``."""
        mp = self._memmap_paths[file_idx]
        end = start + self._episode_length

        # mmap_mode='r' keeps the file on disk; slicing forces only the window
        # into RAM as a contiguous array.
        data_mm = np.load(mp.data_path, mmap_mode="r")
        index_mm = np.load(mp.index_path, mmap_mode="r")

        window_data = np.array(data_mm[start:end], dtype=np.float32)
        window_index_ns = np.array(index_mm[start:end])

        try:
            index = pd.DatetimeIndex(window_index_ns)
        except Exception:
            index = pd.RangeIndex(len(window_data))

        return pd.DataFrame(window_data, columns=mp.columns, index=index)

    # ------------------------------------------------------------------
    # Override reset to swap in a fresh window each episode
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None, **kwargs):
        rng = np.random.default_rng(seed)
        file_idx = int(rng.integers(0, len(self._memmap_paths)))
        mp = self._memmap_paths[file_idx]
        max_start = mp.n_rows - self._episode_length
        start = int(rng.integers(0, max(1, max_start)))

        window_df = self._load_window(file_idx, start)
        self._set_df(window_df)

        logger.debug(
            "StreamingTradingEnv reset: symbol=%d start=%d length=%d",
            file_idx,
            start,
            self._episode_length,
        )
        return super().reset(seed=seed, options=options, **kwargs)
