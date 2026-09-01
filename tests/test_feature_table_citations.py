"""Guards for the appendix feature table's Citation column.

The table is a raw LaTeX longtable, so Pandoc cannot process `@key` citations
inside it and the column is written by hand. That is how "Cao et al. (2004)"
reached the document while the bibliography entry, and the paper, say 2009, and
how two cited works ended up with no bibliography entry at all.

`feature_table_citations` derives the column's text from library.bib; these
tests check that the derivation still matches the table and that the drift it
exists to prevent is actually detected.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "thesis" / "qmd" / "src"
sys.path.insert(0, str(_SRC))

from feature_table_citations import (  # noqa: E402
    CITATION_KEYS,
    cite,
    nocite_keys,
    validate_table,
)

_APPENDIX = _SRC / "99-appendix.qmd"


def test_table_citations_agree_with_the_bibliography() -> None:
    """The shipped table must match what library.bib implies, cell for cell."""
    problems = validate_table(_APPENDIX)
    assert problems == [], "\n".join(problems)


def test_validator_detects_a_drifted_year(tmp_path: Path) -> None:
    """The Cao 2004/2009 drift is the bug this guard exists for."""
    drifted = tmp_path / "99-appendix.qmd"
    drifted.write_text(
        _APPENDIX.read_text().replace(r"Cao et al.\ (2009)", r"Cao et al.\ (2004)")
    )

    problems = validate_table(drifted)

    assert len(problems) == 1
    assert "book_pressure" in problems[0]
    assert "2004" in problems[0] and "2009" in problems[0]


def test_cite_renders_one_two_and_three_author_works() -> None:
    # Single author: surname only.
    assert cite("Kyle1985") == "Kyle (1985)"
    # Two authors: joined with an escaped ampersand for LaTeX.
    assert cite("LeeReady1991") == r"Lee \& Ready (1991)"
    # Three or more: "et al." with a trailing control space so LaTeX does not
    # read the period as sentence-ending.
    assert cite("Cont2014") == "Cont et al.\\ (2014)"


def test_cite_joins_multiple_sources_in_the_given_order() -> None:
    assert cite("Kyle1985", "GlostenMilgrom1985") == (
        r"Kyle (1985); Glosten \& Milgrom (1985)"
    )


def test_cite_renders_an_uncited_row_as_an_em_dash() -> None:
    assert cite() == "---"


def test_cite_rejects_a_key_absent_from_the_bibliography() -> None:
    with pytest.raises(KeyError, match="NotARealKey2099"):
        cite("NotARealKey2099")


def test_every_cited_key_resolves() -> None:
    """A key in CITATION_KEYS with no bib entry must fail loudly, not render blank."""
    for keys in CITATION_KEYS.values():
        for key in keys:
            assert cite(key), key


def test_nocite_covers_every_key_the_table_cites() -> None:
    """Sources cited only in the table reach the reference list through nocite."""
    expected = {key for keys in CITATION_KEYS.values() for key in keys}
    assert set(nocite_keys()) == expected


def test_nocite_list_in_the_root_document_is_in_step() -> None:
    """The front-matter nocite must list every key the table cites."""
    front_matter = (_SRC / "masters-thesis.qmd").read_text()
    missing = [k for k in nocite_keys() if f"@{k}" not in front_matter]
    assert missing == [], (
        "masters-thesis.qmd nocite is missing: "
        + ", ".join(missing)
        + " (regenerate with feature_table_citations.nocite_keys())"
    )
