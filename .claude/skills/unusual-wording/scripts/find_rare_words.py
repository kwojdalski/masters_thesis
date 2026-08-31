#!/usr/bin/env python3
"""Find candidate unusual/rare words in the thesis prose.

This is a *candidate generator*, not a verdict. It strips markup, counts
word frequency across the whole thesis body, and filters out (a) closed-class
function words, (b) a bundled ~50k-word common-English frequency list, and
(c) terms already defined in the glossary. What's left is still noisy —
academic prose has a long tail of ordinary words that happen to appear once,
and the common-word list is not lemma-aware, so some ordinary inflected forms
("achievable", "accumulates") will slip through. The unusual-wording skill's
next step (an LLM triage pass) is what actually separates genuine findings
from this noise; do not present this raw list to the user as findings.

Usage: python3 find_rare_words.py [--max-freq N] [--min-len N]
"""

from __future__ import annotations

import argparse
import glob
import re
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
COMMON_WORDS_PATH = SCRIPT_DIR.parent / "references" / "common_words_en50k.txt"

# Closed-class function words: articles, pronouns, prepositions, conjunctions,
# auxiliaries/modals, common determiners/quantifiers. Safe to enumerate
# exhaustively (unlike open-class content words, which is what the common-word
# list and the LLM triage step are for).
STOPWORDS = set(
    """
a an the this that these those is are was were am be been being have has had
do does did will would shall should may might must can could ought
of in on at to for with by from as into onto upon over under above below
between among through during before after since until while
and or but nor so yet although though because if unless whether
not no none nothing nobody neither either both all any some each every
much many more most less least few fewer several own such other another same
it its he she they we you i him her them us me his their our your my mine
yours ours theirs who whom whose what which where when why how
here there now then again further once also too very just only even still
already vs etc eg ie
""".split()
)

# Structural markdown/LaTeX bits to strip before tokenizing, applied in order.
# Each entry carries its own correct flags -- DOTALL and MULTILINE do not mix
# safely for a greedy `.*` (DOTALL makes `.` cross lines, so a MULTILINE `$`
# anchor no longer stops it at end-of-line, and `.*` swallows the whole rest
# of the file).
_STRIP_PATTERNS = [
    (r"^---.*?---\n", "", re.DOTALL),  # YAML frontmatter
    (r"```\{[^}]*\}.*?```", " ", re.DOTALL),  # python/latex code chunks
    (r"```.*?```", " ", re.DOTALL),  # any remaining fenced block
    (r"\$\$.*?\$\$", " ", re.DOTALL),  # display math
    (r"\$[^$]+\$", " ", 0),  # inline math
    # citation keys / cross-refs: @Foo2020, and hyphenated-slug forms
    # @sec-x, @tbl-x, @eq-x, @fig-x (the hyphen must be in the class or
    # e.g. "@sec-feature-normalization-and-causality-preservation" only
    # loses its "@sec" prefix, leaking the rest as a fake "word").
    (r"@[A-Za-z][A-Za-z0-9-]*", " ", 0),
    (r"\{[^}]*\}", " ", 0),  # {#sec-...}, {.class}, quarto attrs
    (r"<!--.*?-->", " ", re.DOTALL),  # HTML comments
    (r"!?\[([^\]]*)\]\([^)]*\)", r"\1 ", 0),  # markdown links/images -> keep link text
    (r"`[^`]+`", " ", 0),  # inline code
    (r"\^\[.*?\]", " ", re.DOTALL),  # inline-caret footnotes: text^[note]
    # Pandoc reference-style footnotes: a `[^label]: text...` definition
    # line, and standalone `[^label]` markers elsewhere in prose. Without
    # this, only the definition's footnote body is kept out of the corpus
    # (via the pattern above), but the `[^label]` marker itself survives
    # and its content is included -- rare across this thesis (2 uses) but
    # cheap to handle correctly.
    (r"^\[\^[^\]]+\]:.*$", " ", re.MULTILINE),
    (r"\[\^[^\]]+\]", " ", 0),
    (r"^#+\s.*$", " ", re.MULTILINE),  # heading lines (titles, not prose)
]


def _clean(text: str) -> str:
    for pattern, repl, flags in _STRIP_PATTERNS:
        text = re.sub(pattern, repl, text, flags=flags)
    return text


def _load_glossary_words(repo_root: Path) -> set[str]:
    gloss_path = repo_root / "thesis/qmd/src/99-glossary.qmd"
    if not gloss_path.exists():
        return set()
    text = gloss_path.read_text()
    terms = re.findall(r"\*\*([^*]+)\*\*", text)
    words: set[str] = set()
    for term in terms:
        for w in re.findall(r"[A-Za-z][A-Za-z']*", term):
            words.add(w.lower())
    return words


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-freq", type=int, default=2)
    parser.add_argument("--min-len", type=int, default=5)
    parser.add_argument(
        "--glob",
        default="thesis/qmd/src/0[1-7]-*.qmd",
        help="Which files to scan (default: body chapters 01-07 only)",
    )
    args = parser.parse_args()

    repo_root = Path.cwd()
    common_words = {
        w.strip().lower()
        for w in COMMON_WORDS_PATH.read_text().splitlines()
        if w.strip()
    }
    glossary_words = _load_glossary_words(repo_root)

    files = sorted(glob.glob(args.glob))
    if not files:
        raise SystemExit(f"No files matched {args.glob!r} from {repo_root}")

    word_freq: Counter[str] = Counter()
    first_seen: dict[str, tuple[str, str]] = {}
    hyphen_freq: Counter[str] = Counter()
    hyphen_first_seen: dict[str, tuple[str, str]] = {}

    for fpath in files:
        raw = Path(fpath).read_text()
        text = _clean(raw)
        fname = Path(fpath).name
        for sent in re.split(r"(?<=[.!?])\s+", text):
            # Non-hyphenated words: the main candidate pool.
            for w in re.findall(r"\b[A-Za-z][A-Za-z']*\b", sent):
                wl = w.lower()
                if wl in STOPWORDS or wl in common_words or len(wl) < args.min_len:
                    continue
                word_freq[wl] += 1
                first_seen.setdefault(wl, (fname, sent.strip()[:160]))
            # Hyphenated compounds: tracked separately (near-always rare by
            # construction, so freq alone is a weak signal for these -- see
            # SKILL.md's "Category 4" for how these get a lighter touch).
            for w in re.findall(r"\b[A-Za-z]+(?:-[A-Za-z]+)+\b", sent):
                wl = w.lower()
                hyphen_freq[wl] += 1
                hyphen_first_seen.setdefault(wl, (fname, sent.strip()[:160]))

    candidates = sorted(
        (w, c)
        for w, c in word_freq.items()
        if c <= args.max_freq and w not in glossary_words
    )
    hyphen_candidates = sorted(
        (w, c)
        for w, c in hyphen_freq.items()
        if c <= args.max_freq and w not in glossary_words
    )

    print(f"# Scanned {len(files)} files matching {args.glob!r}")
    print(
        f"# Non-hyphenated candidates (freq<={args.max_freq}, len>={args.min_len}): "
        f"{len(candidates)}"
    )
    print(
        f"# Hyphenated-compound candidates (freq<={args.max_freq}): "
        f"{len(hyphen_candidates)}"
    )
    print()
    print("## Non-hyphenated candidates")
    for w, c in candidates:
        fname, ctx = first_seen[w]
        print(f"{c}\t{w}\t{fname}\t{ctx}")
    print()
    print("## Hyphenated-compound candidates (lower priority -- usually legitimate)")
    for w, c in hyphen_candidates:
        fname, ctx = hyphen_first_seen[w]
        print(f"{c}\t{w}\t{fname}\t{ctx}")


if __name__ == "__main__":
    main()
