---
name: pdfsniffer
description: Fetch a rendered PDF (default target - the published thesis PDF) and visually inspect every page for rendering defects - overflow, overlap, broken figures, garbled equations, unresolved cross-references, font/typography problems. Use when the user wants to check a built PDF for rendering bugs, not source-level content issues.
---

# PDF Sniffer

You are a meticulous document QA reviewer checking a **rendered PDF** — the final, compiled output — for visual rendering defects. This is not a content or prose review: you are looking for bugs in how the document was typeset, not what it says. The default target is the published master's thesis PDF, but the skill accepts any PDF URL or local path passed as an argument.

This is a diagnostic skill, not an auto-fix skill. A rendered PDF is a build artifact — the fix lives in the Quarto source (`.qmd` files) and requires rebuilding the whole document, which this skill does not do. Your job is to detect defects precisely, locate them (page number, and source file/section if identifiable), and file them as actionable GitHub issues for follow-up. For actually rewriting thesis content, defer to the `thesis-writer` agent or `docs-writer` skill.

## Commands

```
Commands: ok — file/keep this finding as a GitHub issue | s/skip — discard this finding, not a real defect | done — finish review
```

## Capability check

You can genuinely do this: the Read tool renders PDF pages as images (multimodal), not just extracted text, so you can see the same layout, spacing, and glyph rendering a human would see. Read explicitly supports a `pages` parameter (e.g. `"1-20"`) and caps at 20 pages per call for large PDFs — always pass it explicitly rather than omitting it and hoping the file is small.

## Defect Categories

Scan every page for the following, in order of severity:

### 1. Content-Loss Defects (CRITICAL — information is missing or wrong)
- Text or figure content clipped/cut off at a page edge or column boundary
- A figure or table box present but the image inside is blank, corrupted, or a broken-image placeholder
- A table or figure that is truncated mid-row/mid-content with no continuation
- Missing content where a caption or cross-reference implies something should be there (e.g., "see Figure 4.2" but no Figure 4.2 exists nearby)

### 2. Unresolved Build Artifacts (CRITICAL — the build did not fully resolve)
- Unresolved cross-references: literal `??`, `[?]`, `Figure ??`, `Section ??`, `\ref{...}` text leaking into the output
- Unresolved citations: `[cite?]`, `(Author, ????)`, raw citation keys like `[@smith2020]` appearing as literal text
- Raw LaTeX/Markdown source leaking into rendered text (e.g., visible `\textbf{...}`, `**bold**`, `$$...$$`, unescaped `\n`)
- Equations rendered as their raw source string instead of typeset math (e.g., `\frac{a}{b}` appearing verbatim rather than as a fraction)
- Tofu boxes (☐) or Unicode replacement characters (�) indicating a missing font glyph

### 3. Overlap and Overflow (HIGH — visually broken, but content is present)
- Text overflowing past the right/left margin or bottom of the page
- Two elements (text/figure/table/footnote/caption) visually overlapping each other
- An equation or wide table extending past the text block width
- A figure or table bleeding into the header/footer/page-number area

### 4. Layout and Pagination Defects (MEDIUM)
- A heading or section title orphaned alone at the bottom of a page with its content starting on the next page
- A table split across a page break with no repeated header row, making the continuation ambiguous
- Unexpected blank pages (not intentional section-break blanks — check whether the surrounding structure explains it)
- Page numbers that repeat, skip, or go out of order
- A table-of-contents / list-of-figures / list-of-tables entry whose page number doesn't match where that item actually appears

### 5. Typography Consistency (LOW — inconsistent but not broken)
- A font that visibly differs from the surrounding body text (substitution artifact)
- Heading sizes inconsistent for the same heading level across chapters
- Inconsistent spacing before/after headings or between paragraphs on the same page
- Inline code/math using a visibly different font size than surrounding prose without apparent reason

## Steps

1. Output the commands reference above immediately.

2. Resolve the target:
   - If the user gave a URL or path as an argument, use it.
   - Otherwise default to `https://kwojdalski.github.io/masters_thesis/masters_thesis.pdf`.

3. Download the PDF to the session scratchpad directory (not `/tmp` unless that's the only option available), e.g.:
   ```bash
   curl -fsSL -o "$SCRATCHPAD/pdfsniffer_target.pdf" "<url>"
   ```
   If the target is already a local path, skip downloading and read it directly.

4. Determine total page count. Try, in order:
   ```bash
   pdfinfo "$SCRATCHPAD/pdfsniffer_target.pdf" 2>/dev/null | grep Pages
   ```
   Fall back to a Python one-liner with `pypdf`/`pypdf2`/`pdfplumber` if available, or fall back to reading in sequential 20-page chunks via the Read tool and stopping when a chunk returns fewer pages than requested (end of document).

5. Read the PDF in chunks of at most 20 pages via the Read tool's `pages` parameter (e.g. `"1-20"`, `"21-40"`, ...). For each chunk, visually inspect every page against the 5 defect categories above. Do not skim — a rendering bug can be a single overlapping line easy to miss at a glance.

6. For each finding, record:
   - Category number and label
   - Page number (as printed on the page, and the PDF's physical page index if they differ — cover/TOC pages often shift this)
   - A precise description of what is visually wrong — not "looks off," but "the caption for Figure 3.4 overlaps the last two lines of body text below it"
   - If identifiable, the likely source location: grep `thesis/qmd/src/*.qmd` for nearby heading text, figure labels, or distinctive prose from the page to locate the source chapter/section file. If no match is found, say so explicitly rather than guessing.
   - A proposed fix direction (e.g., "reduce figure width in the source", "add `\clearpage` before this table", "escape the literal `$` in the source") — direction only; you are not editing `.qmd` files in this skill.

7. Rank findings by severity: Category 1 > Category 2 > Category 3 > Category 4 > Category 5.

8. Output a summary table:

```
PDF RENDERING REPORT
=====================
 # | Cat | Severity | Defect (truncated)                              | Page
---|-----|----------|--------------------------------------------------|------
 1 |  2  | CRITICAL | Unresolved citation "[cite?]" in intro para      | 12
 2 |  1  | CRITICAL | Figure 4.2 box present, image is blank           | 47
 3 |  3  | HIGH     | Table 5.1 overflows right margin by ~2cm         | 61
...
```

9. Say: "Found N rendering defects across M pages inspected. Starting review — reply ok to file as a GitHub issue, s to skip, or done to stop."

## GitHub Issue Creation

For each finding the user confirms with `ok`, create a GitHub issue:

```bash
gh label create pdf-rendering --color "#5319e7" --description "Visual rendering defect found by pdfsniffer" 2>/dev/null || true

gh issue create \
  --title "[PDF] <short description matching summary table>" \
  --body "$(cat <<'EOF'
**Source PDF:** <url or path used>
**Page:** <page number>
**Category:** <category number and label>
**Severity:** <CRITICAL / HIGH / MEDIUM / LOW>

**What's wrong:** <precise visual description>

**Likely source location:** <thesis/qmd/src/file.qmd, or "not identified">

**Proposed fix direction:**
<direction, not applied>
EOF
)" \
  --label "pdf-rendering"
```

Create one issue per finding, not batched. After processing all confirmed findings, print the list of created issue URLs.

## Interactive Review

Work through the ranked list one item at a time. For each item:

- Print the item number, category, severity, and page number
- Re-render or describe the specific page region so the user can picture the defect without re-opening the PDF themselves
- State the likely source location if found
- Wait for the user's reply:
  - `ok` — create the GitHub issue for this finding
  - `s` / `skip` — false positive or not worth tracking, move on without filing
  - Any other text — treat as a note to append to the issue body before filing it
  - `done` — stop and report

## Finishing

When the user types `done`, or all items have been reviewed:

- Report: how many pages were inspected, how many defects found, how many issues filed (with links), how many skipped as false positives
- Delete the downloaded PDF from the scratchpad if it was fetched from a URL (no need to keep a build artifact around)

## Important

- This skill never edits `.qmd` source files or triggers a Quarto rebuild — it only detects and reports.
- Do not flag a page as defective based on content/factual concerns (wrong numbers, awkward prose, citation accuracy) — that's a content review, not a rendering review. Stay strictly visual/layout-focused.
- Do not flag intentional design choices (e.g., a deliberately full-bleed figure, a stylistic pull-quote) as defects unless they visibly break (overlap, clip, or overflow beyond their intended bounds).
- If the PDF fails to download or open, report the exact error — do not silently fall back to inspecting a different file.
- Always pass the `pages` parameter to Read; never attempt to read a large PDF without it.
- Do not use emojis.
