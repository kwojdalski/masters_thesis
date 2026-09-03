# Vendored Quarto extensions

Third-party extensions committed into the repo so the thesis renders
reproducibly. **One of them carries a local patch.** `quarto update` (or a
re-`quarto add`) overwrites vendored files wholesale, so re-apply the patch
below after any extension upgrade and re-check the rendered algorithm captions.

## leovan/pseudocode 1.5.0 — patched

Upstream: <https://github.com/leovan/quarto-pseudocode>

### What is patched

`pseudocode.lua`, three hunks, each marked `-- LOCAL PATCH`:

1. A `PSEUDOCODE_REF_PATTERN` constant near `nil_to_default`.
2. In `render_pseudocode_block_html`, immediately before `local inner_el = ...`.
3. In `render_pseudocode_block_latex`, immediately before the
   `if options["label"] then` caption-label rewrite.

### Why

A crossref written *inside* a ```` ```pseudocode ```` fence — in `\caption{}` or
on a `\REQUIRE` line — is part of the `CodeBlock`'s opaque text. Pandoc never
parses it into a `Cite` element, so Quarto's crossref pass cannot rewrite it and
the extension passes the text through verbatim. An unresolved `@alg-setup` then
reaches the rendered page as literal text, in both PDF and HTML.

This thesis needs those references: `05-02-code.qmd` and `99-appendix.qmd` each
introduce an algorithm as following the common setup algorithm, and the
alternative is a hardcoded "Algorithm 1" that drifts silently when the algorithm
order changes — the exact regression `thesis/qmd/src/crossref_hardcoding.py`
exists to prevent. The guard cannot catch this case: it only checks that a
literal number is in range, so a non-rendering `@alg-...` inside a fence passes.

Note that references written in ordinary prose (`@alg-td3` in a sentence) were
never affected — they are real `Cite` elements and already resolved correctly.

### How

Both hunks rewrite `@alg-…` / `@algo-…` occurrences in the fence's raw text
before it is emitted:

- **LaTeX** substitutes `Algorithm~\ref{<label>}`. LaTeX resolves the reference
  itself, so ordering does not matter; an unknown label surfaces as LaTeX's own
  `??` and a rerun warning.
- **HTML** substitutes `Algorithm N`, looked up in the extension's own
  `html_identifier_number_mapping`. That table is filled as blocks are walked,
  so only a *backward* reference resolves. A forward or unknown reference is
  deliberately left as the literal `@label` — visible in the output — rather
  than guessed at. Every in-fence reference in this thesis points at
  `alg-setup`, the first algorithm in render order, so all of them resolve.

### Verifying after an upgrade

Render and confirm the caption reads "TD3 Training (follows Algorithm 1)", not
"(follows @alg-setup)":

```bash
quarto render thesis/qmd/src/masters-thesis.qmd --to latex
grep -n 'follows' thesis/qmd/src/masters-thesis.tex | head
```

`tests/test_pseudocode_crossrefs.py` covers the same expectation at unit level
and fails if the patch is lost.
