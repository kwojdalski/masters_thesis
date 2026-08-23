from __future__ import annotations

import pandas as pd
import pytest

from trading_rl.data.lob_filters import (
    filter_active_lob,
    filter_unchanged_lob,
    filter_valid_lob,
    get_lob_change_stats,
)


def _lob_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "bid_px_00": [99.0, 99.0, 99.5, 101.0],
            "ask_px_00": [101.0, 101.0, 101.5, 100.0],
            "bid_sz_00": [10.0, 10.0, 12.0, 10.0],
            "ask_sz_00": [11.0, 11.0, 11.0, 10.0],
            "bid_px_01": [98.5, 98.5, 99.0, 100.5],
            "ask_px_01": [101.5, 101.5, 102.0, 100.5],
            "bid_sz_01": [9.0, 9.0, 9.0, 10.0],
            "ask_sz_01": [8.0, 8.0, 8.0, 10.0],
        },
        index=pd.date_range("2024-01-01", periods=4, freq="s"),
    )


def test_filter_unchanged_lob_returns_empty_copy_for_empty_input() -> None:
    df = _lob_frame().iloc[:0]

    filtered = filter_unchanged_lob(df, levels=1)

    assert filtered.empty
    assert filtered is not df


def test_filter_unchanged_lob_removes_duplicate_snapshot() -> None:
    filtered = filter_unchanged_lob(_lob_frame(), levels=1)

    assert len(filtered) == 3
    assert filtered.index.tolist() == [
        pd.Timestamp("2024-01-01 00:00:00"),
        pd.Timestamp("2024-01-01 00:00:02"),
        pd.Timestamp("2024-01-01 00:00:03"),
    ]


def test_filter_unchanged_lob_keep_first_false_drops_first_row() -> None:
    filtered = filter_unchanged_lob(_lob_frame(), levels=1, keep_first=False)

    assert len(filtered) == 2
    assert filtered.index.tolist() == [
        pd.Timestamp("2024-01-01 00:00:02"),
        pd.Timestamp("2024-01-01 00:00:03"),
    ]


def test_filter_unchanged_lob_raises_for_missing_required_columns() -> None:
    df = _lob_frame().drop(columns=["ask_sz_00"])

    with pytest.raises(ValueError, match="Missing required LOB columns"):
        filter_unchanged_lob(df, levels=1)


def test_filter_valid_lob_removes_crossed_book() -> None:
    filtered = filter_valid_lob(
        _lob_frame(),
        levels=1,
        min_spread_bps=0.0,
        max_spread_bps=500.0,
    )

    assert len(filtered) == 3
    assert filtered.index.max() == pd.Timestamp("2024-01-01 00:00:02")


def test_filter_valid_lob_removes_too_wide_spread() -> None:
    filtered = filter_valid_lob(
        _lob_frame().iloc[:1],
        levels=1,
        min_spread_bps=0.0,
        max_spread_bps=50.0,
    )

    assert filtered.empty


def test_filter_valid_lob_removes_zero_or_negative_sizes() -> None:
    df = _lob_frame().iloc[:2].copy()
    df.loc[df.index[1], "ask_sz_00"] = 0.0

    filtered = filter_valid_lob(
        df,
        levels=1,
        min_spread_bps=0.0,
        max_spread_bps=500.0,
        min_size=0.0,
    )

    assert filtered.index.tolist() == [df.index[0]]


def test_filter_valid_lob_removes_misordered_deep_levels() -> None:
    df = _lob_frame().iloc[:1].copy()
    df.loc[df.index[0], "bid_px_01"] = 99.5

    filtered = filter_valid_lob(df, levels=2, min_spread_bps=0.0)

    assert filtered.empty


def test_filter_active_lob_validates_then_removes_stale_rows() -> None:
    filtered = filter_active_lob(
        _lob_frame(),
        levels=1,
        remove_unchanged=True,
        validate=True,
    )

    assert filtered.empty


def test_get_lob_change_stats_reports_per_column_changes() -> None:
    stats = get_lob_change_stats(_lob_frame(), levels=1).set_index("column")

    assert stats.loc["bid_px_00", "n_changes"] == 2
    assert stats.loc["ask_sz_00", "n_changes"] == 1
    assert stats.loc["bid_sz_00", "mean_abs_change"] == pytest.approx(2.0)


def test_lob_filters_support_custom_column_prefixes() -> None:
    df = pd.DataFrame(
        {
            "bp00": [10.0, 10.1],
            "ap00": [10.2, 10.2],
            "bs00": [1.0, 1.0],
            "as00": [1.0, 2.0],
        }
    )

    filtered = filter_unchanged_lob(
        df,
        levels=1,
        bid_px_prefix="bp",
        ask_px_prefix="ap",
        bid_sz_prefix="bs",
        ask_sz_prefix="as",
    )

    assert len(filtered) == 2
