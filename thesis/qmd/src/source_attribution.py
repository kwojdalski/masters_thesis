"""Guard that every figure, table, and algorithm carries a Source attribution.

WNE UW formal requirements 10 and 11 require every table and figure to carry
a source citation (table 10: "titled at left margin... with a title and a
reference to the source"; figure 11: "10 pt source citation"). This module
treats every numbered float in the document -- a markdown image, a Pandoc
pipe-table caption, a Python-generated table (`display_df`), a Python-
generated figure (`show_plot`), and a pseudocode algorithm block -- as one
kind of object with the same obligation, per the user's framing, and checks
each one for a `**Source:**` line (or the `table_note(source=...)` call that
emits one) within the same block.

Findings are split into two tiers rather than a single pass/fail:

- Figures and tables are a formal WNE UW requirement. A missing Source here
  is a compliance defect, not a judgment call, so these are collected into
  `find_violations()` and are meant to be enforced.
- Algorithms are NOT covered by any WNE UW rule (Annex D's template predates
  LaTeX algorithm floats entirely) and this document's own convention is to
  attribute an algorithm through inline prose citations to the paper it
  implements (e.g. "TD3 was introduced... [@Fujimoto2018]") rather than a
  formal Source line under the pseudocode. Flagging all four algorithm
  blocks as violations would be manufacturing a rule the thesis never had,
  the same mistake this project already made and reversed once for orphaned
  equations and unused bibliography entries. Algorithm attribution status is
  reported separately, informationally, never as a violation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_SRC = Path(__file__).resolve().parent
_MASTERS = _SRC / "masters-thesis.qmd"

_INCLUDE_RE = re.compile(r"\{\{<\s*include\s+\./(\S+?\.qmd)\s*>\}\}")

# A float's attribution is considered "attached" if a Source line appears
# before the next heading, the next float of any kind, or within this many
# lines -- whichever comes first. The line cap is a runaway-scan safety
# valve, not the real precision mechanism -- that is the early stop at the
# next heading/float below. It must stay generous: a `show_plot()` call is
# routinely followed by an `else: missing_data_notice(...)` fallback branch,
# a cell-fence close, a fresh `#|`-option cell, and several lines of comment
# explaining the note's wording before the `table_note(...)` call itself --
# confirmed up to 29 lines apart in this document (06-03-performance-
# evaluation.qmd's fig-eval-positions). A tight cap here produced two false
# positives during development on floats independently verified to have a
# Source line; this is set well above that observed maximum instead of
# exactly at it.
_LOOKAHEAD_LINES = 60

_SOURCE_MARKERS = (
    re.compile(r"\*\*Source:\*\*"),
    re.compile(r"table_note\s*\("),
)
_HEADING_RE = re.compile(r"^#{1,3}\s")

_FIGURE_MD_RE = re.compile(r"^!\[.*?\]\(")
_TABLE_CAPTION_RE = re.compile(r"^:\s.*\{#tbl-")
_TABLE_PYCALL_RE = re.compile(r"\bdisplay_df\s*\(")
_FIGURE_PYCALL_RE = re.compile(r"\bshow_plot\s*\(")
_ALGORITHM_RE = re.compile(r"\\begin\{algorithm\}")
_MISSING_DATA_RE = re.compile(r"\bmissing_data_notice\s*\(")


def _include_order() -> list[str]:
    return _INCLUDE_RE.findall(_MASTERS.read_text())


@dataclass
class Finding:
    file: str
    line: int
    kind: str
    detail: str
    is_violation: bool

    def __str__(self) -> str:
        tag = "VIOLATION" if self.is_violation else "info"
        return f"{self.file}:{self.line}: [{tag}] {self.kind} -- {self.detail}"


_FENCE_START_RE = re.compile(r"^```")


def _has_nearby_source(
    lines: list[str], start_idx: int, *, starts_in_fence: bool
) -> bool:
    """Look forward from a float's definition line for a Source marker.

    Stops early at the next markdown heading (OUTSIDE a fenced code block --
    a Python `#`-comment such as `# States the omission...` is
    indistinguishable from a Markdown heading by regex alone, and treating
    every `#`-comment as a heading stop caused this checker to report the
    fig-eval-equity/fig-eval-positions figures as missing a Source that is
    genuinely a few lines further inside their own Python cell; confirmed by
    stepping through the scan by hand) or the next float-starting line, so a
    Source that belongs to a *different*, later float is never credited to
    this one.

    `starts_in_fence` must reflect whether the triggering line itself sits
    inside an open fence: `show_plot(`/`display_df(`/`\\begin{algorithm}`
    always do (they are Python/pseudocode cell content), while a markdown
    image or a Pandoc `: caption {#tbl-...}` line never does. Getting this
    wrong desynchronises every subsequent open/close toggle for the rest of
    the scan, not just the first one.
    """
    in_fence = starts_in_fence
    for offset in range(1, _LOOKAHEAD_LINES + 1):
        idx = start_idx + offset
        if idx >= len(lines):
            break
        line = lines[idx]
        if _FENCE_START_RE.match(line):
            in_fence = not in_fence
            continue
        if any(marker.search(line) for marker in _SOURCE_MARKERS):
            return True
        if not in_fence and _HEADING_RE.match(line):
            break
        if offset > 1 and (
            _FIGURE_MD_RE.match(line)
            or _TABLE_CAPTION_RE.match(line)
            or _TABLE_PYCALL_RE.search(line)
            or _FIGURE_PYCALL_RE.search(line)
            or _ALGORITHM_RE.search(line)
        ):
            break
    return False


def find_findings() -> list[Finding]:
    findings: list[Finding] = []

    for fname in _include_order():
        path = _SRC / fname
        if not path.exists():
            continue
        lines = path.read_text().splitlines()

        for i, line in enumerate(lines):
            if _FIGURE_MD_RE.match(line):
                ok = _has_nearby_source(lines, i, starts_in_fence=False)
                findings.append(
                    Finding(
                        fname,
                        i + 1,
                        "figure (markdown image)",
                        line.strip()[:88],
                        is_violation=not ok,
                    )
                )
            elif _TABLE_CAPTION_RE.match(line):
                ok = _has_nearby_source(lines, i, starts_in_fence=False)
                findings.append(
                    Finding(
                        fname,
                        i + 1,
                        "table (pandoc caption)",
                        line.strip()[:88],
                        is_violation=not ok,
                    )
                )
            elif _TABLE_PYCALL_RE.search(line):
                # A display_df() call inside an `if ...: ... else:
                # missing_data_notice(...)` branch has no rendered table on
                # this path when data is absent, so only the branch that
                # actually calls display_df is checked -- the fallback branch
                # is exempt by construction (missing_data_notice explains the
                # gap itself, which is thesis-data-auditor's territory).
                ok = _has_nearby_source(lines, i, starts_in_fence=True)
                findings.append(
                    Finding(
                        fname,
                        i + 1,
                        "table (display_df)",
                        line.strip()[:88],
                        is_violation=not ok,
                    )
                )
            elif _FIGURE_PYCALL_RE.search(line):
                ok = _has_nearby_source(lines, i, starts_in_fence=True)
                findings.append(
                    Finding(
                        fname,
                        i + 1,
                        "figure (show_plot)",
                        line.strip()[:88],
                        is_violation=not ok,
                    )
                )
            elif _ALGORITHM_RE.search(line):
                ok = _has_nearby_source(lines, i, starts_in_fence=True)
                findings.append(
                    Finding(
                        fname,
                        i + 1,
                        "algorithm",
                        "not a WNE UW formal requirement; attributed via inline "
                        "prose citation instead, if at all -- informational only",
                        is_violation=False,
                    )
                )

    return findings


def find_violations() -> list[Finding]:
    """Figures and tables only -- the subset this document is formally required to fix."""
    return [f for f in find_findings() if f.is_violation]


def main() -> int:
    findings = find_findings()
    violations = [f for f in findings if f.is_violation]
    algorithms = [f for f in findings if f.kind == "algorithm"]
    other_info = [f for f in findings if not f.is_violation and f.kind != "algorithm"]

    if violations:
        print(f"source_attribution: {len(violations)} figure/table violation(s):\n")
        for v in violations:
            print(f"  {v}")
    else:
        print("source_attribution: every figure and table has a Source line.")

    if algorithms:
        print(
            f"\nsource_attribution: {len(algorithms)} algorithm(s) found, informational "
            "only (not a WNE UW requirement):"
        )
        for a in algorithms:
            print(f"  {a}")

    if other_info:
        print(
            f"\nsource_attribution: {len(other_info)} other float(s) already compliant."
        )

    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
