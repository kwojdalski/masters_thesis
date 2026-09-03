"""Guards the local patch to the vendored leovan/pseudocode extension.

A crossref written inside a ```pseudocode fence -- in `\\caption{}` or on a
`\\REQUIRE` line -- is part of the CodeBlock's opaque text, so Pandoc never
parses it into a Cite element and Quarto's crossref pass cannot rewrite it. The
extension passes that text through verbatim, so an unresolved `@alg-setup`
reaches the rendered page as literal text in both PDF and HTML. Six references
in 05-02-code.qmd and 99-appendix.qmd shipped that way.

`thesis/qmd/src/_extensions/leovan/pseudocode/pseudocode.lua` carries a local
patch resolving those references in both the HTML and LaTeX handlers;
`_extensions/README.md` documents it. `quarto update` overwrites vendored files
wholesale, so this test exists to fail loudly if an extension upgrade drops the
patch -- the failure mode is otherwise silent, since the document still renders
and `crossref_hardcoding` cannot see a non-rendering `@alg-...` inside a fence.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "thesis" / "qmd" / "src"
_PSEUDOCODE_LUA = _SRC / "_extensions" / "leovan" / "pseudocode" / "pseudocode.lua"
_EXTENSIONS_README = _SRC / "_extensions" / "README.md"

# Every .qmd carrying a ```pseudocode fence.
_PSEUDOCODE_QMDS = ("05-02-code.qmd", "99-appendix.qmd")

_FENCE_RE = re.compile(r"^```pseudocode\n(.*?)^```", re.MULTILINE | re.DOTALL)


def test_patch_is_present_in_both_render_paths() -> None:
    lua = _PSEUDOCODE_LUA.read_text(encoding="utf-8")

    assert "PSEUDOCODE_REF_PATTERN" in lua, (
        "the local in-fence crossref patch is missing from pseudocode.lua -- an "
        "extension upgrade probably overwrote it; see _extensions/README.md"
    )
    # LaTeX path substitutes a real \ref; HTML path looks the number up.
    assert "\\\\ref{%1}" in lua
    assert "html_identifier_number_mapping[ref]" in lua
    assert lua.count("-- LOCAL PATCH") >= 3


def test_patch_is_documented() -> None:
    assert _EXTENSIONS_README.is_file(), (
        "_extensions/README.md records why pseudocode.lua is patched and how to "
        "re-apply it after an upgrade"
    )
    assert "LOCAL PATCH" in _EXTENSIONS_README.read_text(encoding="utf-8")


def test_in_fence_references_use_crossref_syntax_not_literal_numbers() -> None:
    """The patch's reason for existing: these stay crossrefs, never numbers."""
    literal_ref = re.compile(r"\bAlgorithm\s+\d+\b")

    for name in _PSEUDOCODE_QMDS:
        text = (_SRC / name).read_text(encoding="utf-8")
        for fence in _FENCE_RE.findall(text):
            assert not literal_ref.search(fence), (
                f"{name}: a ```pseudocode fence hardcodes an algorithm number; "
                "use @alg-... so it cannot drift (the patch renders it correctly)"
            )


def test_every_in_fence_reference_resolves_to_a_defined_label() -> None:
    """An in-fence @alg-... pointing at no label renders as literal text."""
    labels: set[str] = set()
    references: list[tuple[str, str]] = []
    ref_re = re.compile(r"@(alg[o]?-[\w-]+)")
    label_re = re.compile(r"^#\|\s*label:\s*(\S+)", re.MULTILINE)

    for name in _PSEUDOCODE_QMDS:
        text = (_SRC / name).read_text(encoding="utf-8")
        for fence in _FENCE_RE.findall(text):
            labels.update(label_re.findall(fence))
            references.extend((name, ref) for ref in ref_re.findall(fence))

    assert labels, "expected labelled ```pseudocode fences in the thesis"
    for name, ref in references:
        assert ref in labels, f"{name}: @{ref} matches no ```pseudocode label"


def test_html_path_only_needs_backward_references() -> None:
    """The HTML lookup resolves earlier algorithms only, so order matters.

    Every in-fence reference must point at a fence that appears before it in
    render order (masters-thesis.qmd's include order, which is alphabetical by
    filename for the files involved here).
    """
    seen: set[str] = set()
    label_re = re.compile(r"^#\|\s*label:\s*(\S+)", re.MULTILINE)
    ref_re = re.compile(r"@(alg[o]?-[\w-]+)")

    for name in _PSEUDOCODE_QMDS:
        text = (_SRC / name).read_text(encoding="utf-8")
        for fence in _FENCE_RE.findall(text):
            for ref in ref_re.findall(fence):
                assert ref in seen, (
                    f"{name}: @{ref} is a forward reference; the HTML render "
                    "resolves in-fence references only against already-numbered "
                    "algorithms and would leave this as literal text"
                )
            seen.update(label_re.findall(fence))
