"""Shared results.json split resolution for the hypothesis report scripts.

Every scenario writes one results.json. Its layout depends on how the evaluate
step ran: a pooled run stores a single ``<split>`` entry, while a --per-symbol
run stores one ``<split>_<SYMBOL>`` entry per instrument and no pooled entry.

The report scripts previously resolved a split by trying the exact key and then
falling through to the first ``<split>_*`` key in JSON order. With --per-symbol
runs that silently returned AAPL alone, labelled as the scenario result, and it
mixed bases inside one table whenever only some arms had a pooled entry (#668).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, NamedTuple


class SplitEntry(NamedTuple):
    """One split's results entry plus the provenance it was derived from.

    ``kind`` is "pooled", "per_symbol", or "" when nothing was found.
    ``symbols`` lists the instruments averaged for a per-symbol entry.
    """

    entry: dict[str, Any]
    kind: str
    symbols: list[str]

    @property
    def label(self) -> str:
        """Short provenance label for a table cell."""
        if self.kind == "pooled":
            return "pooled"
        if self.kind == "per_symbol":
            if len(self.symbols) == 1:
                return self.symbols[0]
            return f"mean({len(self.symbols)})"
        return "—"

    @property
    def key(self) -> str:
        """Comparison key: two rows are comparable only when these match."""
        if self.kind == "per_symbol":
            return "per_symbol:" + ",".join(self.symbols)
        return self.kind


def _mean_numeric(entries: list[Any]) -> Any:
    """Recursively average a list of identically-shaped metric structures.

    Numeric leaves are averaged over the finite values; non-numeric leaves
    (labels, provenance strings) keep the first non-null value. Nested dicts
    are walked so an entry's ``benchmarks`` sub-tree is averaged alongside its
    ``metrics``.
    """
    if not entries:
        return None
    if any(isinstance(e, dict) for e in entries):
        dicts = [e for e in entries if isinstance(e, dict)]
        merged: dict[str, Any] = {}
        for key in {k for d in dicts for k in d}:
            value = _mean_numeric([d[key] for d in dicts if key in d])
            if value is not None:
                merged[key] = value
        return merged
    numbers = [
        float(e)
        for e in entries
        if isinstance(e, int | float) and not isinstance(e, bool) and math.isfinite(e)
    ]
    if numbers:
        return sum(numbers) / len(numbers)
    non_null = [e for e in entries if e is not None]
    return non_null[0] if non_null else None


def load_split_entry(results_json: Path, split: str) -> SplitEntry:
    """Resolve one split from a results.json, averaging per-symbol entries.

    A pooled ``<split>`` entry wins when present. Otherwise every
    ``<split>_<SYMBOL>`` entry is averaged equally, which is the right weighting
    for an agent trained on the pooled symbol set. Picking whichever symbol
    happened to be serialised first is never correct: on the H3 arms the
    per-symbol total returns straddle zero, so that choice decided the sign of
    the reported result.
    """
    if not results_json.exists():
        return SplitEntry({}, "", [])
    try:
        with results_json.open() as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return SplitEntry({}, "", [])

    pooled = data.get(split)
    if pooled:
        return SplitEntry(pooled, "pooled", [])

    prefix = f"{split}_"
    per_symbol = [
        (key[len(prefix) :], entry)
        for key, entry in data.items()
        if key.startswith(prefix) and entry
    ]
    if not per_symbol:
        return SplitEntry({}, "", [])

    symbols = sorted(symbol for symbol, _ in per_symbol)
    if len(per_symbol) == 1:
        return SplitEntry(per_symbol[0][1], "per_symbol", symbols)
    averaged = _mean_numeric([entry for _, entry in per_symbol])
    return SplitEntry(averaged or {}, "per_symbol", symbols)


def basis_warning(entries: list[SplitEntry]) -> str:
    """Return a warning when the rows in one table are not on a common basis.

    Empty string when every populated row shares a basis (or there is nothing
    to compare). A mixed table is the failure worth shouting about: the H5
    ladder had four arms reported on AAPL alone against one pooled arm, an
    apples-to-oranges comparison with nothing on screen to reveal it.
    """
    keys = {e.key for e in entries if e.kind}
    if len(keys) <= 1:
        return ""
    described = ", ".join(f"{e.label} ({e.key})" for e in entries if e.kind)
    return (
        "rows are not on a common basis and are NOT comparable: "
        f"{described}. Re-run the evaluate step so every scenario writes the "
        "same split layout."
    )
