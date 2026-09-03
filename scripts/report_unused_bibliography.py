#!/usr/bin/env python3
"""Report bibliography entries never cited anywhere in the thesis.

Report-only: prints a list, exits 0 regardless. This is deliberately not a
CI gate, unlike the equation/section/algorithm checks in
thesis/qmd/src/crossref_hardcoding.py. Those catch a reference drifting away
from content that still exists; this instead answers a curation question --
does every entry in library.bib belong in this specific document -- and at
last count 105 of 176 entries (60%) were unused, reading as a broad personal
reference library (CAPM/EMH/factor-model classics: Markowitz1952, Sharpe1964,
Fama1970, Lintner1965, Carhart1997) rather than a bug. Failing a build on that
scale would be noise, not signal, until someone has actually decided whether
to prune library.bib or leave it as a broader personal collection.

A "used" entry is one cited by @Key in the rendered prose of any included
.qmd file, OR listed in masters-thesis.qmd's `nocite` block (the mechanism
already used to pull in sources the hand-typed feature table cites but
Pandoc's @key syntax cannot reach -- see thesis_qmd_src/feature_table_citations.py).

Usage:
    uv run python scripts/report_unused_bibliography.py
"""

from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "thesis" / "qmd" / "src"
_BIB = _ROOT / "thesis" / "bibliography" / "library.bib"
_MASTERS = _SRC / "masters-thesis.qmd"

_BIB_KEY_RE = re.compile(r"^@\w+\{([^,]+),", re.M)
_CITE_RE = re.compile(r"@([A-Za-z][A-Za-z0-9]*)")
_NOCITE_BLOCK_RE = re.compile(r"nocite:\s*\|\n((?:^ {2}.*\n?)+)", re.M)
_FENCE_RE = re.compile(r"```.*?```", re.S)
_INCLUDE_RE = re.compile(r"\{\{<\s*include\s+\./(\S+?\.qmd)\s*>\}\}")


def _bib_keys() -> set[str]:
    return set(_BIB_KEY_RE.findall(_BIB.read_text()))


def _nocite_keys() -> set[str]:
    text = _MASTERS.read_text()
    m = _NOCITE_BLOCK_RE.search(text)
    if not m:
        return set()
    return set(_CITE_RE.findall(m.group(1)))


def _cited_keys(bib_keys: set[str]) -> set[str]:
    cited: set[str] = set()
    include_order = _INCLUDE_RE.findall(_MASTERS.read_text())
    for fname in include_order:
        path = _SRC / fname
        if not path.exists():
            continue
        text = _FENCE_RE.sub("", path.read_text())
        cited |= {k for k in _CITE_RE.findall(text) if k in bib_keys}
    return cited


def find_unused() -> list[str]:
    bib_keys = _bib_keys()
    used = _cited_keys(bib_keys) | _nocite_keys()
    return sorted(bib_keys - used)


def main() -> int:
    bib_keys = _bib_keys()
    unused = find_unused()
    print(
        f"report_unused_bibliography: {len(unused)} of {len(bib_keys)} entries "
        f"in library.bib are never cited in prose or nocite.\n"
    )
    for key in unused:
        print(f"  {key}")
    if unused:
        print(
            "\nThis is a report, not a failure -- see this script's docstring for why. "
            "Prune from library.bib or cite in prose; no action required by default."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
