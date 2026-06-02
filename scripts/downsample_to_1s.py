#!/usr/bin/env python
"""Downsample raw MBP-10 tick parquet files to 1-second LOB bars.

Usage:
    uv run python scripts/downsample_to_1s.py [--src DIR] [--dst DIR] [--freq 1s]

Defaults:
    --src  data/raw/stocks/daily
    --dst  data/raw/stocks_1s/daily
    --freq 1s

Output files mirror the source tree:
    data/raw/stocks_1s/daily/{SYMBOL}/{SYMBOL}_{DATE}_raw_mbp-10_us_hours.parquet

Aggregation logic
-----------------
LOB snapshot columns (bid/ask price, size, count for all levels):
    .last()  — end-of-second LOB state

tick-level metadata (rtype, publisher_id, instrument_id, flags, ts_in_delta,
                     sequence, ts_event, price, depth, side):
    .last()

action:
    synthetic: 'T' if any trade occurred in the second, else last action

size:
    .last()  (per-event size is not meaningful after aggregation; use trade_volume)

trade_volume:
    sum of size[action=='T'] — total traded size in the bar (used by VWAP benchmark)

volume (raw field, equals size for every event type in MBP-10):
    .sum()  — approximation; trade_volume is preferred

symbol:
    .last()  (constant per file, just carry through)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd


# Columns that carry LOB snapshot state — use end-of-period value.
_LOB_SNAPSHOT_COLS = [
    "bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00", "bid_ct_00", "ask_ct_00",
    "bid_px_01", "ask_px_01", "bid_sz_01", "ask_sz_01", "bid_ct_01", "ask_ct_01",
    "bid_px_02", "ask_px_02", "bid_sz_02", "ask_sz_02", "bid_ct_02", "ask_ct_02",
    "bid_px_03", "ask_px_03", "bid_sz_03", "ask_sz_03", "bid_ct_03", "ask_ct_03",
    "bid_px_04", "ask_px_04", "bid_sz_04", "ask_sz_04", "bid_ct_04", "ask_ct_04",
    "bid_px_05", "ask_px_05", "bid_sz_05", "ask_sz_05", "bid_ct_05", "ask_ct_05",
    "bid_px_06", "ask_px_06", "bid_sz_06", "ask_sz_06", "bid_ct_06", "ask_ct_06",
    "bid_px_07", "ask_px_07", "bid_sz_07", "ask_sz_07", "bid_ct_07", "ask_ct_07",
    "bid_px_08", "ask_px_08", "bid_sz_08", "ask_sz_08", "bid_ct_08", "ask_ct_08",
    "bid_px_09", "ask_px_09", "bid_sz_09", "ask_sz_09", "bid_ct_09", "ask_ct_09",
]

_LAST_COLS = [
    "ts_event", "rtype", "publisher_id", "instrument_id",
    "action", "side", "depth", "price", "size",
    "flags", "ts_in_delta", "sequence", "symbol",
]


def _resample_file(src: Path, dst: Path, freq: str) -> dict:
    """Resample a single tick parquet file to *freq* bars.

    Returns a dict with stats (input_rows, output_rows, elapsed_s).
    """
    t0 = time.perf_counter()
    df = pd.read_parquet(src)

    if df.empty:
        df.to_parquet(dst)
        return {"input_rows": 0, "output_rows": 0, "elapsed_s": time.perf_counter() - t0}

    # --- trade_volume: total traded size per bar (preferred VWAP weight)
    if "action" in df.columns and "size" in df.columns:
        trade_mask = df["action"].astype(str) == "T"
        trade_vol_series = df["size"].where(trade_mask, other=0).resample(freq).sum()
    else:
        trade_vol_series = None

    # --- volume: raw column sum (MBP-10: volume == size per event)
    sum_cols: dict[str, pd.Series] = {}
    if "volume" in df.columns:
        sum_cols["volume"] = df["volume"].resample(freq).sum()

    # --- all LOB snapshot + metadata columns: take last value in period
    last_cols_present = [c for c in _LOB_SNAPSHOT_COLS + _LAST_COLS if c in df.columns]
    resampled_last = df[last_cols_present].resample(freq).last()

    # --- assemble
    out = resampled_last.copy()
    for col, series in sum_cols.items():
        out[col] = series
    if trade_vol_series is not None:
        out["trade_volume"] = trade_vol_series

    # Drop bars where the LOB top-of-book is entirely NaN (no events in that second)
    lob_anchor = [c for c in ("bid_px_00", "ask_px_00") if c in out.columns]
    if lob_anchor:
        out = out.dropna(subset=lob_anchor, how="all")

    # Keep all numeric columns as float64. resample().last() already produces
    # float64 for integer-origin columns; converting to pandas Int64 (nullable)
    # would break numpy ufuncs (isinf, isfinite) used in the feature pipeline.

    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dst)
    elapsed = time.perf_counter() - t0
    return {"input_rows": len(df), "output_rows": len(out), "elapsed_s": elapsed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", default="data/raw/stocks/daily", help="Source directory tree root")
    parser.add_argument("--dst", default="data/raw/stocks_1s/daily", help="Destination directory tree root")
    parser.add_argument("--freq", default="1s", help="Resample frequency (pandas offset alias, e.g. 1s, 5s, 1min)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    args = parser.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)
    freq = args.freq

    files = sorted(src_root.rglob("*_raw_mbp-10_us_hours.parquet"))
    if not files:
        print(f"No parquet files found under {src_root}")
        return

    print(f"Resampling {len(files)} files  {src_root} -> {dst_root}  freq={freq}")
    total_in = total_out = 0
    for i, src_path in enumerate(files, 1):
        rel = src_path.relative_to(src_root)
        dst_path = dst_root / rel
        if dst_path.exists() and not args.overwrite:
            print(f"  [{i}/{len(files)}] SKIP (exists) {rel}")
            continue
        stats = _resample_file(src_path, dst_path, freq)
        total_in += stats["input_rows"]
        total_out += stats["output_rows"]
        compression = 100.0 * (1.0 - stats["output_rows"] / max(stats["input_rows"], 1))
        print(
            f"  [{i}/{len(files)}] {rel}  "
            f"{stats['input_rows']:,} -> {stats['output_rows']:,} rows  "
            f"({compression:.1f}% reduction)  "
            f"{stats['elapsed_s']:.1f}s"
        )

    print(f"\nDone. Total: {total_in:,} -> {total_out:,} rows  ({100.0*(1-total_out/max(total_in,1)):.1f}% reduction)")


if __name__ == "__main__":
    main()
