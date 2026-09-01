"""Derive the appendix feature table's Citation column from library.bib.

The feature registry table in 99-appendix.qmd is a hand-built LaTeX longtable
(landscape, repeating headers, per-column widths). Pandoc does not process
Pandoc citations inside a raw LaTeX block, so its Citation column cannot use
`@key` syntax and was previously typed by hand. That drifted: the table printed
"Cao et al. (2004)" while the bibliography entry, and the paper, say 2009, and
two cited works had no bibliography entry at all and so never reached the
reference list.

This module keeps the table's LaTeX intact but makes the citation column a
function of the bibliography. Each row maps to an ordered list of BibTeX keys;
the displayed author-year text is generated from those entries at render time.
A key that is missing, or an entry without an author or year, raises rather
than silently printing an empty or stale citation.

`CITATION_KEYS` also drives the `nocite` list, so a source cited only in this
table still appears in the reference list.
"""

from __future__ import annotations

import re
from pathlib import Path

_BIB = Path(__file__).resolve().parents[2] / "bibliography" / "library.bib"

# Ordered BibTeX keys per distinct citation cell in the feature table. The key
# is the cell's rendered text; regenerating from these is what prevents the
# hand-typed years from drifting away from the bibliography.
CITATION_KEYS: dict[str, list[str]] = {
    "book_pressure": ["Cao2004", "Biais1995"],
    "order_book_imbalance": ["Cont2014"],
    "order_count_imbalance": ["Biais1995"],
    "microprice": ["Stoikov2018"],
    "microprice_divergence": [],
    "vwmp_skew": [],
    "price_vamp": [],
    "depth_ratio": ["Bouchaud2002"],
    "spread_bps": ["Kyle1985", "GlostenMilgrom1985"],
    "spread_ratio": ["Easley2012"],
    "bid_convexity": ["Bouchaud2002"],
    "ask_convexity": ["Bouchaud2002"],
    "bid_slope": ["Bouchaud2002", "Kyle1985"],
    "ask_slope": ["Bouchaud2002", "Kyle1985"],
    "ofi": ["Cont2014"],
    "ofi_rolling": ["Cont2014"],
    "ofi_multilevel": ["Cont2014"],
    "bid_queue_depletion": ["Foucault2005"],
    "ask_queue_depletion": ["Foucault2005"],
    "signed_trade_flow": ["LeeReady1991"],
    "odd_lot_trade_ratio": ["Boehmer2021"],
    "odd_lot_imbalance": ["Boehmer2021"],
    "vpin": ["Easley2012"],
    "cancel_to_trade": ["Foucault2005"],
    "trade_arrival_rate": ["Engle2000"],
    "large_trade_ratio": [],
    "ofi_autocorr": ["Cont2014"],
    "inter_event_time": ["Engle2000"],
    "mid_price_acceleration": ["Abergel2016"],
    "hour_sin": ["Wood1985"],
    "hour_cos": ["Wood1985"],
}


def _entries() -> dict[str, dict[str, str]]:
    """Parse author and year for every entry in library.bib."""
    text = _BIB.read_text()
    out: dict[str, dict[str, str]] = {}
    for match in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", text, re.S):
        key, body = match.group(2).strip(), match.group(3)

        def field(name: str, _body: str = body) -> str:
            # The final field in an entry has no trailing comma and is followed
            # by the closing brace rather than a newline, so the terminator has
            # to accept end-of-string as well.
            found = re.search(
                rf'(?:^|\n)\s*{name}\s*=\s*[{{"](.+?)[}}"]\s*,?\s*(?=\n|$)',
                _body,
                re.S | re.I,
            )
            return re.sub(r"\s+", " ", found.group(1)).strip() if found else ""

        out[key] = {"author": field("author"), "year": field("year")}
    return out


def _surname(author: str) -> str:
    return author.split(",")[0].strip() if "," in author else author.split()[-1]


def cite(*keys: str) -> str:
    """Render one Citation cell as LaTeX author-year text drawn from the bib.

    Raises KeyError for an unknown key and ValueError for an entry missing the
    author or year, so a citation can never render blank or stale.
    """
    if not keys:
        return "---"
    entries = _entries()
    parts = []
    for key in keys:
        if key not in entries:
            raise KeyError(
                f"{key} is cited in the feature table but not in {_BIB.name}"
            )
        author, year = entries[key]["author"], entries[key]["year"]
        if not author or not year:
            raise ValueError(f"{key} is missing an author or year in {_BIB.name}")
        names = [a.strip() for a in author.split(" and ")]
        if len(names) == 1:
            label = _surname(names[0])
        elif len(names) == 2:
            label = rf"{_surname(names[0])} \& {_surname(names[1])}"
        else:
            # Trailing "\\" makes LaTeX's inter-word space, so the period in
            # "et al." is not treated as sentence-ending.
            label = f"{_surname(names[0])} et al." + "\\"
        parts.append(f"{label} ({year})")
    return "; ".join(parts)


def nocite_keys() -> list[str]:
    """Every key the feature table cites, for the document's nocite list."""
    seen: list[str] = []
    for keys in CITATION_KEYS.values():
        for key in keys:
            if key not in seen:
                seen.append(key)
    return seen


def validate_table(qmd_path: str | Path) -> list[str]:
    """Check the feature table's Citation column against the bibliography.

    Reads the longtable out of the given .qmd, and for every row compares the
    hand-written Citation cell with the text `cite()` derives from the keys in
    CITATION_KEYS. Returns a list of human-readable problems; an empty list
    means the table and the bibliography agree.

    Callers should raise on a non-empty result so the render fails rather than
    shipping a citation that has drifted, lost its year, or lost its entry.
    """
    text = Path(qmd_path).read_text()
    table = re.search(r"\\begin\{longtable\}.*?\\end\{longtable\}", text, re.S)
    if table is None:
        return [f"no longtable found in {Path(qmd_path).name}"]
    body = table.group(0).split(r"\endlastfoot")[1].split(r"\end{longtable}")[0]
    rows = [r.strip() for r in body.split("\\\\\n") if r.strip() and "\\texttt" in r]

    problems: list[str] = []
    seen: set[str] = set()
    for row in rows:
        cols = [c.strip() for c in re.split(r"(?<!\\)&", row)]
        name = re.search(r"\\texttt\{hft\\_([a-z_\\]+)\}", cols[0])
        if name is None:
            problems.append(f"could not read a feature name from row: {row[:60]}")
            continue
        feature = name.group(1).replace("\\", "")
        seen.add(feature)
        if feature not in CITATION_KEYS:
            problems.append(f"{feature}: present in the table but absent from CITATION_KEYS")
            continue
        expected = cite(*CITATION_KEYS[feature])
        if cols[5] != expected:
            problems.append(
                f"{feature}: table shows {cols[5]!r}, bibliography gives {expected!r}"
            )
    for feature in CITATION_KEYS:
        if feature not in seen:
            problems.append(f"{feature}: in CITATION_KEYS but no matching table row")
    return problems
