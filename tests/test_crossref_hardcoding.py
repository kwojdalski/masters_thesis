"""Guards against hardcoded chapter/section/algorithm/appendix references drifting.

Quarto's crossref system already validates `@sec-...`/`@alg-...` references at
build time (an unresolved label renders as a literal `?@sec-foo`). It has no
way to check a hardcoded reference -- the literal words "Section 5.1" or
"Algorithm 0" typed into a sentence -- because that is indistinguishable from
ordinary prose to every tool in the pipeline. Two confirmed instances reached
the document exactly this way: six "Section 5.1"/"Section 5.2" references
drifted two sections early when a chapter opener grew extra `##` sections
ahead of them, and six "Algorithm 0" references survived a format-unification
commit that removed the (already inconsistent) label they were copied from.

`crossref_hardcoding` builds the document's true numbering by walking the same
include order Quarto renders and checks every hardcoded reference against it.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "thesis" / "qmd" / "src"
sys.path.insert(0, str(_SRC))

from crossref_hardcoding import find_violations  # noqa: E402


def test_no_hardcoded_crossref_drift() -> None:
    """Every hardcoded Section/Algorithm/Appendix reference must resolve.

    This cannot catch a reference that resolves to an existing number but
    points at the wrong content (that needs the thesis-crossref-auditor agent
    or a human read) -- only a reference to a number that does not exist at
    all. That is still strictly more than nothing caught before: both
    confirmed real-world instances of this bug class (#776, #777) resolved to
    no existing number and would have failed this check immediately.
    """
    violations = find_violations()
    assert violations == [], "\n" + "\n".join(str(v) for v in violations)
