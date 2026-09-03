"""Guards that every figure and table carries a Source line, per WNE UW rules 10/11.

`source_attribution` treats every numbered float in the document -- a
markdown image, a Pandoc pipe-table caption, a Python-generated table
(`display_df`), and a Python-generated figure (`show_plot`) -- as one kind of
object with the same obligation and checks each for a `**Source:**` line (or
the `table_note(source=...)` call that emits one) attached to it. Algorithm
blocks are tracked but never asserted on here: WNE UW's formal rules do not
cover LaTeX algorithm floats, and this document's own convention is to
attribute an algorithm through inline prose citation instead, so an
"algorithm" finding is informational, never a violation.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "thesis" / "qmd" / "src"
sys.path.insert(0, str(_SRC))

from source_attribution import find_violations  # noqa: E402


def test_every_figure_and_table_has_a_source_line() -> None:
    """Every markdown image, Pandoc table, display_df, and show_plot needs a Source.

    Developed against two real bugs in the checker itself, not just the
    thesis: an undersized lookahead window and a heading-stop regex that
    could not distinguish a Markdown heading from a Python `#`-comment both
    produced false positives on thesis/qmd/src/06-03-performance-
    evaluation.qmd's fig-eval-equity and fig-eval-positions figures, which
    were independently confirmed by hand to already carry a Source line.
    Fixed and re-verified with a deliberate negative test (temporarily
    deleting a real Source block and confirming it was caught) before this
    test was trusted to gate anything.
    """
    violations = find_violations()
    assert violations == [], "\n" + "\n".join(str(v) for v in violations)
