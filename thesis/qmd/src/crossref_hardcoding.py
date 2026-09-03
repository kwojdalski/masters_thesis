"""Guard against hardcoded chapter/section/algorithm/appendix references drifting.

Quarto's crossref system (`@sec-...`, `@tbl-...`, `@fig-...`, `@alg-...`)
already validates itself at build time: an unresolved label renders as a
literal `?@sec-foo` rather than silently pointing at the wrong thing. That
guarantee only covers references written in crossref syntax.

A hardcoded reference -- the literal words "Section 5.1" or "Algorithm 0"
typed into a sentence -- is invisible to that mechanism. It looks identical
to ordinary prose, so nothing in the toolchain notices when a later edit
changes the heading structure underneath it and the sentence starts pointing
at the wrong content. Two confirmed instances reached the document this way:
six "Section 5.1"/"Section 5.2" references drifted two sections early when
05-00-implementation.qmd grew two `##` sections ahead of 05-01, and six
"Algorithm 0" references survived a format-unification commit that removed
the (already inconsistent) hardcoded HTML label they were copied from.

This module builds the document's true chapter/section numbering and
algorithm count by walking the same include order Quarto renders, then checks
every hardcoded reference against it. It cannot verify that a reference which
resolves to an *existing* number also points at the *right* content -- that
still needs a human or `thesis-crossref-auditor` -- but it turns "the number
doesn't exist at all" and "a new hardcoded reference was added instead of a
crossref" into a build-time failure instead of a silent drift.

"Chapter N" (57 instances, Roman numerals: `pracamgrwne.cls` renders body
chapters as "CHAPTER I".."CHAPTER VII") gets the weaker existence-only check,
not the Section/Algorithm/Appendix treatment of "convert to `@sec-...` so the
whole class of bug is impossible." That conversion was tried and reverted:
`\thesection` bakes the literal word "CHAPTER" into the chapter number, and
Quarto's crossref system independently, unconditionally prepends its own
"Section" prefix to any `@sec-...` reference. The two collide -- a rendered
test came back as "see Section CHAPTER I for background" -- and fixing it
needs custom crossref-prefix engineering in the class file, not a text
substitution. Existence-checking 57 references for free is still strictly
better than checking none, which was the state before this function existed.

`find_orphaned_equations` answers a related but different question: which
`{#eq-...}`-labelled equations are never cited via `@eq-...` anywhere. This is
deliberately NOT wired into `find_violations`/the CI gate, and it is not a
"the number is wrong" bug the way the checks above are. Most equations in a
thesis are displayed once, exactly where they are defined and discussed, and
have no reason for anything elsewhere to point back to them -- an uncited
equation is normal, not broken. What the list is actually useful for is the
same thing `thesis-condenser` sessions already used it for by hand: a
zero-citation equation is a candidate for "does this need its own display
treatment, or is it restating something already shown" (this is exactly how
`eq-tw-mean`/`eq-tw-var` and `eq-microprice-ch2` were identified and cut --
see docs/masters_thesis/length_reduction_plan.md items 8-9). Treat the output
as an editorial list, not a pass/fail signal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

_SRC = Path(__file__).resolve().parent
_MASTERS = _SRC / "masters-thesis.qmd"

_INCLUDE_RE = re.compile(r"\{\{<\s*include\s+\./(\S+?\.qmd)\s*>\}\}")
_FENCE_RE = re.compile(r"```([^\n]*)\n(.*?)```", re.S)
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
_LATEX_COMMENT_LINE_RE = re.compile(r"(?m)^\s*%.*$")
_FENCE_OPTION_LINE_RE = re.compile(r"(?m)^#\|.*$")

# Body chapters render as Arabic numbers in prose ("Section 5.2") even though
# the chapter heading itself displays as a Roman numeral ("CHAPTER V.") -- the
# two numbering schemes are independent, so this module only needs the plain
# chapter ordinal.
_HEADING_RE = re.compile(r"(?m)^(#{1,3})\s+(.+?)\s*(\{#([a-zA-Z0-9_-]+)\})?\s*$")

_APPENDIX_MARKER = "99-appendix.qmd"
_BODY_END_MARKERS = {
    "98-bibliography.qmd",
    "99-glossary.qmd",
    "99-appendix.qmd",
    "99-audit-log.qmd",
}


def _strip_fence(match: re.Match[str]) -> str:
    """Drop Python/LaTeX/HTML code cells; keep pseudocode captions scannable.

    The algorithm captions and \\REQUIRE lines that actually carry hardcoded
    "Algorithm N" references live inside ```pseudocode fences, so those fences
    cannot be discarded as code the way ```{python}/```{=latex} cells are --
    doing so would blind this checker to the exact bug class it exists to
    catch (confirmed by testing against the known pre-fix "Algorithm 0" state,
    which produced zero violations until this exception was added).
    """
    info, body = match.group(1).strip(), match.group(2)
    if info.startswith("pseudocode"):
        return _FENCE_OPTION_LINE_RE.sub("", body)
    return "\n" * match.group(0).count("\n")


def _strip_non_prose(text: str) -> str:
    """Remove non-pseudocode fenced blocks and comments so they cannot match as prose."""
    text = _FENCE_RE.sub(_strip_fence, text)
    text = _HTML_COMMENT_RE.sub(lambda m: "\n" * m.group(0).count("\n"), text)
    text = _LATEX_COMMENT_LINE_RE.sub("", text)
    return text


def _include_order() -> list[str]:
    text = _MASTERS.read_text()
    return _INCLUDE_RE.findall(text)


@dataclass
class HeadingMap:
    """chapter.section.subsection -> (file, heading text) for the body chapters."""

    sections: dict[tuple[int, ...], tuple[str, str]] = field(default_factory=dict)
    max_chapter: int = 0

    def exists(self, numbers: tuple[int, ...]) -> bool:
        return numbers in self.sections


def build_heading_map() -> HeadingMap:
    hmap = HeadingMap()
    chapter = section = subsection = 0
    for fname in _include_order():
        if fname in _BODY_END_MARKERS:
            break
        raw = (_SRC / fname).read_text()
        prose = _strip_non_prose(raw)
        for match in _HEADING_RE.finditer(prose):
            level = len(match.group(1))
            title = match.group(2).strip()
            if level == 1:
                chapter += 1
                section = 0
                subsection = 0
                hmap.sections[(chapter,)] = (fname, title)
            elif level == 2:
                section += 1
                subsection = 0
                hmap.sections[(chapter, section)] = (fname, title)
            elif level == 3:
                subsection += 1
                hmap.sections[(chapter, section, subsection)] = (fname, title)
    hmap.max_chapter = chapter
    return hmap


def count_algorithms() -> int:
    """Count `\\begin{algorithm}` floats across the document in render order.

    The pseudocode extension numbers algorithms as a single flat LaTeX float
    counter (like `table`/`figure`), not scoped per chapter, so a document-wide
    count in include order is the ground truth -- confirmed against the
    rendered PDF, which shows Algorithm 1-4 with no `\\setcounter{algorithm}`
    anywhere in the sources.
    """
    count = 0
    for fname in _include_order():
        raw = (_SRC / fname).read_text()
        count += raw.count(r"\begin{algorithm}")
    return count


def count_appendix_letters() -> int:
    """Count `# `-level headings inside 99-appendix.qmd (each becomes A, B, C, ...)."""
    raw = (_SRC / _APPENDIX_MARKER).read_text()
    prose = _strip_non_prose(raw)
    return len(re.findall(r"(?m)^# ", prose))


_SECTION_REF_RE = re.compile(r"\bSection\s+(\d+)\.(\d+)(?:\.(\d+))?\b")
_ALGORITHM_REF_RE = re.compile(r"\bAlgorithm\s+(\d+)\b")
_APPENDIX_REF_RE = re.compile(r"\bAppendix\s+([A-Z])\b")
_CHAPTER_REF_RE = re.compile(r"\bChapters?\s+([IVXLCDM]+)(?:\s+and\s+([IVXLCDM]+))?\b")

_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def _roman_to_int(s: str) -> int:
    total = 0
    prev = 0
    for ch in reversed(s.upper()):
        value = _ROMAN_VALUES.get(ch)
        if value is None:
            return -1  # not a valid Roman numeral at all
        total += value if value >= prev else -value
        prev = max(prev, value)
    return total


_ROMAN_NUMERALS = [
    (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
    (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
    (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
]  # fmt: skip


def _int_to_roman(n: int) -> str:
    out = []
    for value, symbol in _ROMAN_NUMERALS:
        count, n = divmod(n, value)
        out.append(symbol * count)
    return "".join(out)


_EQ_LABEL_DEF_RE = re.compile(r"\{#(eq-[a-zA-Z0-9_-]+)\}")
_EQ_LABEL_REF_RE = re.compile(r"@(eq-[a-zA-Z0-9_-]+)")


def find_orphaned_equations() -> list[Violation]:
    """Every `{#eq-...}` label must be cited at least once via `@eq-...`.

    A hardcoded "Equation N"/"Eq. N" reference does not count as a citation:
    if one exists, the equation is still not protected by the crossref
    mechanism, and #776/#777 are the reason hardcoded numeric references are
    treated as their own violation elsewhere in this module, not as coverage.
    """
    defined: dict[str, tuple[str, int]] = {}
    referenced: set[str] = set()

    for fname in _include_order():
        raw = (_SRC / fname).read_text()
        prose = _strip_non_prose(raw)
        for lineno, line in enumerate(prose.splitlines(), start=1):
            for m in _EQ_LABEL_DEF_RE.finditer(line):
                defined.setdefault(m.group(1), (fname, lineno))
            for m in _EQ_LABEL_REF_RE.finditer(line):
                referenced.add(m.group(1))

    violations = []
    for label, (fname, lineno) in sorted(defined.items(), key=lambda kv: kv[1]):
        if label not in referenced:
            violations.append(
                Violation(
                    fname,
                    lineno,
                    "orphaned-equation",
                    label,
                    "labelled but never cited with @"
                    + label
                    + " anywhere -- either reference it in prose or drop the label",
                )
            )
    return violations


@dataclass
class Violation:
    file: str
    line: int
    kind: str
    text: str
    reason: str

    def __str__(self) -> str:
        return f"{self.file}:{self.line}: [{self.kind}] {self.text!r} -- {self.reason}"


def find_violations() -> list[Violation]:
    hmap = build_heading_map()
    n_algorithms = count_algorithms()
    n_appendix_letters = count_appendix_letters()
    violations: list[Violation] = []

    for fname in _include_order():
        path = _SRC / fname
        raw = path.read_text()
        prose = _strip_non_prose(raw)
        for lineno, line in enumerate(prose.splitlines(), start=1):
            for m in _SECTION_REF_RE.finditer(line):
                nums = tuple(int(g) for g in m.groups() if g is not None)
                if not hmap.exists(nums):
                    violations.append(
                        Violation(
                            fname,
                            lineno,
                            "hardcoded-section",
                            m.group(0),
                            f"no heading numbered {'.'.join(map(str, nums))} exists in the "
                            "document's rendered structure -- prefer @sec-... so this is "
                            "validated at build time instead of drifting silently",
                        )
                    )
            for m in _ALGORITHM_REF_RE.finditer(line):
                n = int(m.group(1))
                if not (1 <= n <= n_algorithms):
                    violations.append(
                        Violation(
                            fname,
                            lineno,
                            "hardcoded-algorithm",
                            m.group(0),
                            f"only {n_algorithms} algorithm(s) exist in the document "
                            f"(numbered 1..{n_algorithms}) -- prefer @alg-... instead of a "
                            "literal number",
                        )
                    )
            for m in _APPENDIX_REF_RE.finditer(line):
                letter = m.group(1)
                idx = ord(letter) - ord("A") + 1
                if not (1 <= idx <= n_appendix_letters):
                    violations.append(
                        Violation(
                            fname,
                            lineno,
                            "hardcoded-appendix",
                            m.group(0),
                            f"only {n_appendix_letters} appendix section(s) exist "
                            f"(A..{chr(ord('A') + n_appendix_letters - 1)})",
                        )
                    )
            for m in _CHAPTER_REF_RE.finditer(line):
                for numeral in (g for g in m.groups() if g is not None):
                    n = _roman_to_int(numeral)
                    if not (1 <= n <= hmap.max_chapter):
                        violations.append(
                            Violation(
                                fname,
                                lineno,
                                "hardcoded-chapter",
                                numeral,
                                f"only {hmap.max_chapter} chapter(s) exist "
                                f"(I..{_int_to_roman(hmap.max_chapter)}) -- "
                                "'Chapter N' cannot be converted to @sec-... here: "
                                "pracamgrwne.cls bakes the word CHAPTER into "
                                "\\thesection, and Quarto's crossref system always "
                                "prepends its own 'Section' prefix to @sec-..., so "
                                "the two collide and render as 'Section CHAPTER "
                                "I' (confirmed by rendering a test reference) -- "
                                "this checks existence only, the same as the "
                                "other hardcoded reference types, without "
                                "attempting that conversion",
                            )
                        )
    return violations


def main() -> int:
    violations = find_violations()
    if not violations:
        print("crossref_hardcoding: no violations found.")
    else:
        print(f"crossref_hardcoding: {len(violations)} violation(s) found:\n")
        for v in violations:
            print(f"  {v}")

    orphans = find_orphaned_equations()
    print(
        f"\ncrossref_hardcoding: {len(orphans)} equation(s) defined but never cited "
        "(report only, not a failure -- see module docstring):\n"
    )
    for v in orphans:
        print(f"  {v}")

    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
