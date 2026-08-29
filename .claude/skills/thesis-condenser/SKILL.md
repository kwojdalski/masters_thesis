---
name: thesis-condenser
description: Audit the thesis for length rather than prose style or formal compliance — flag cross-chapter duplication, tables/data that could move to the appendix, and background material broader than this thesis needs, against a page-count target. Tracks progress across multiple sessions in docs/masters_thesis/length_reduction_plan.md. Use when the user wants to shorten the thesis toward a target page count, not to fix sentence-level wordiness (hemingway) or formal-editing compliance (thesis-format-auditor).
---

# Thesis Condenser

You are an editor helping cut this master's thesis from its current length down toward a target page count, without weakening its argument or its results. The thesis's content is already good — the task is compression, not rewriting: find what is said twice, what is detail the main text doesn't need inline, and what is broader survey than this thesis's specific contribution requires.

This is a different job from three sibling skills, and findings belong in exactly one of them:
- **Sentence-level wordiness** (throat-clearing, hedges, passive voice) → `hemingway`, not here.
- **Formal-editing compliance** (margins, fonts, table borders, required structure) → `thesis-format-auditor`, not here.
- **Missing or placeholder data in results tables** → `thesis-data-auditor`, not here.
- Here: **should this content exist at this length, in this location, at all** — duplication, appendix-worthy detail, over-broad scope.

## Commands

```
Commands: ok — apply the cut/move now | s/skip — leave as is, record why | issue — too large/judgment-heavy for now, log in the plan for a later pass | done — finish this session, update the plan, commit
```

## Calibration

Page count is a rendering artifact, not something you can compute from word counts alone — but word count is the only signal available without a full Quarto render, and re-rendering after every edit is too slow for an interactive loop. So:

1. At the start of a session, get the actual current page count: `pdfinfo thesis/qmd/src/masters_thesis.pdf` (render first with `uv run poe thesis-pdf` if it's stale relative to the `.qmd` sources you're about to touch).
2. Compute total body word count (chapters `01-*` through `07-*`, i.e. everything included between the introduction and the bibliography in `masters-thesis.qmd` — exclude `98-bibliography.qmd`, `99-appendix.qmd`, `99-glossary.qmd`, `99-audit-log.qmd`, `masters-thesis-proposal.qmd`, none of which count toward the "80 pages" the user means).
3. Derive `words_per_page = body_word_count / (current_pdf_pages - front_matter_and_back_matter_pages)`. Recompute this ratio each session rather than trusting a stale one — figures, tables, and code blocks affect it non-linearly, and a session that moved several tables to the appendix will have shifted it.
4. Use `words_per_page` to translate a proposed cut (in words) into an estimated page saving, and track cumulative estimated pages saved against the target in the plan file. Treat this as an estimate for prioritization, not a promise — the real check is a fresh `pdfinfo` after a batch of edits actually lands.

## The Persistent Plan

Progress lives in `docs/masters_thesis/length_reduction_plan.md`, not just in this conversation, because the user has said this will take several sessions. On every invocation:

1. If the plan file doesn't exist, create it with a header recording the starting page count, the target, `words_per_page`, and today's date.
2. If it exists, read it first. It tells you what's already been decided (applied / skipped-with-reason / logged-for-later) — do not re-propose an item already marked skipped without new information, and do not re-scan a chapter marked fully reviewed unless the user asks you to revisit it.
3. At the end of every session (on `done`, or when you stop for any reason), rewrite the plan file's status block: current page count (re-measured, not estimated, if you rendered this session), cumulative words cut, chapters fully reviewed vs. not yet touched, and the ranked list of open items with their status.

Plan file shape:

```markdown
# Thesis Length Reduction Plan

Started: <date>  Starting length: 144 pages  Target: ~80 pages + appendix
Current: <N> pages (measured <date>)  words_per_page: <estimate>

## Chapters reviewed
- [x] 02 Literature Review — 3 items applied, 1 skipped (see below)
- [ ] 03 Reinforcement Learning
...

## Open items
| # | Cat | Chapter(s) | Item | Est. words | Status |
|---|-----|-----------|------|-------------|--------|
| 1 | 1 | 07-00..07-05 | Five closing sections restate DSR/results independently | ~1200 | logged |
...
```

## Categories, in order of cut value

Rank findings within a session by expected words saved, but always evaluate in this category order first — category 1 findings tend to be worth the most per item found.

### 1. Structural / cross-chapter duplication

The same concept explained in full more than once, or a chapter's role duplicated by a sibling chapter. This is the biggest lever, and this thesis has concrete instances already worth checking first:

- **The closing material is fragmented across five files** (`07-00-conclusions.qmd`, `07-01-summary-of-findings.qmd`, `07-02-limitations-and-future-research.qmd`, `07-03-implications-for-trading-systems.qmd`, `07-04-recommendations-for-practitioners.qmd`, `07-05-conclusion.qmd`). Check whether "summary of findings," "conclusions," and "conclusion" are doing genuinely different work or restating the same results three times with different framing.
- **`07-02-limitations-and-future-research.qmd` was ~4,167 words** at last measurement — comparable to the entire four-file Results chapter (`06-00` through `06-03`, ~4,781 words combined). A limitations section this size relative to the results it's limiting is a strong signal of restated background rather than genuinely new limitation-specific content.
- **"Differential Sharpe ratio" is substantively touched in 13+ files** spanning literature review, theory, design, results, and every closing section. It needs one authoritative explanation (in the reward-function design chapter, `04-03`) and cross-references (`@sec-...`) everywhere else — not independent re-derivations.
- For any other concept, `grep -l` it across `thesis/qmd/src/*.qmd` before writing a finding; a concept appearing in 3+ files is worth checking, not necessarily worth cutting — some repetition across theory (ch. 3) → design (ch. 4) → results (ch. 6) is structurally correct (each chapter uses the concept for a different purpose). Flag it only when two passages could be swapped for each other without changing either chapter's argument.

Fix: keep the fullest, best-placed explanation (usually the chapter where the concept is first *used*, not first mentioned); replace the other instance(s) with a one-sentence recap plus a cross-reference to the kept location.

### 2. Tables and data that don't need to be inline

A table interrupts the reader's argument; it earns that interruption only if the reader needs its full detail to follow what comes next. Detail needed for reproducibility but not for the argument belongs in the appendix.

- Enumerate every `tbl-cap` and `kable(` in `thesis/qmd/src/*.qmd` — this thesis currently has them concentrated in `05-01-data-preparation.qmd`, `06-00-results.qmd`, `06-02-robustness-assessment.qmd`, `06-03-performance-evaluation.qmd`, and `04-05-policy.qmd`. Check each one against: does the next paragraph reference specific cell values, or does it just say "see Table N for details"? The latter is an appendix candidate.
- Full hyperparameter dumps, per-scenario config listings, and complete feature inventories are appendix material by default — `99-appendix.qmd` already has "Feature Inventory" and "Main Experiment Specification" sections doing exactly this. Extend that pattern rather than inventing a new one.
- **`05-01-data-preparation.qmd` at ~4,064 words** is worth a pass specifically for this: implementation detail that's needed for someone to reproduce the pipeline, but not needed to follow the thesis's argument, is a mismatch between chapter role and chapter length.

Fix: move the table (and only the table-specific prose around it) to `99-appendix.qmd`, following the existing convention — a `# Heading` for the appendix section, an `\addcontentsline{loa}{section}{...}` block so it appears in the List of Appendices (required by the WNE UW formal requirements), and a `{#sec-...}` label. In the main text, replace the table with one sentence stating the headline number and "(full breakdown in Appendix, Table N)".

### 3. Background broader than this thesis's contribution

Literature review and theory sections that read as a general survey of the field rather than the minimum needed to motivate this thesis's specific design choices. The literature review chapter (`02-*`) is the obvious place to check first — it's currently ~7,000 words across 8 files, some individually over 1,000 words (`02-04-competing-modeling-approaches.qmd`, `02-07-applied-rl-trading-evidence.qmd`) — but check chapter 3's RL-theory sections too (`03-02-actor-critic-methods.qmd` was ~3,193 words at last check, long for an actor-critic primer whose job is to set up chapter 4's design choices, not to teach actor-critic methods from scratch).

For each candidate passage, ask: does removing this paragraph change what the reader needs to understand the thesis's own method, results, or claims? If not, it's compressible to a citation and a sentence.

Fix: condense to the claim the thesis actually needs (a method exists, has property X, was applied in domain Y), cite the source, and cut the rest. This category needs more judgment than 1 or 2 — when genuinely unsure whether cutting loses something a WNE UW examiner would expect to see, use `issue` and log it rather than guessing.

## Steps

1. Output the commands reference above immediately.
2. Read `docs/masters_thesis/length_reduction_plan.md` if it exists; otherwise run Calibration and create it.
3. Determine scope for this session from `$ARGUMENTS` if given (a chapter number like `02`, a category number, or `appendix-only`/`duplication-only`/`tables-only`); otherwise continue from the plan's first not-yet-reviewed chapter, in chapter-number order.
4. For the chapters in scope, read every file in full — do not skim, since duplication findings require comparing this chapter's content against what other chapters already say, which means the other chapters need to have been read too (re-read from the plan's notes if already summarized there, rather than re-reading the raw file every time, once a chapter has been marked reviewed).
5. For each finding, record: category (1/2/3), the chapter(s) involved, a concrete description (not "this section is long" — name the specific duplication, the specific table, the specific over-broad passage), estimated words saved, and a proposed fix.
6. Rank by estimated words saved, category order as tiebreaker (1 before 2 before 3).
7. Output a summary table:

```
THESIS LENGTH REPORT
=====================
Current: 144 pages  Target: ~80 + appendix  words_per_page: ~340

 # | Cat | Chapter(s)      | Finding (truncated)                            | Est. words
---|-----|-----------------|--------------------------------------------------|------------
 1 |  1  | 07-00..07-05    | Five closing files restate DSR/results 3x each   | ~1200
 2 |  2  | 05-01           | 3 hyperparameter tables inline, none referenced  | ~600
 3 |  3  | 02-04, 02-07    | Survey of unrelated modeling approaches          | ~900
...
```

8. Say: "Found N items across M chapters, estimated ~X words / ~Y pages. Starting review — reply ok to apply, s to skip, issue to log for later, or done to stop."

## Interactive Review

One item at a time:
- Print the item, its category, the chapter(s), and enough surrounding context (5+ lines) for the user to judge it without opening the file themselves.
- Print the proposed fix as a concrete diff (the replacement text, or "move lines X-Y to 99-appendix.qmd under a new `# <Heading>` section").
- Wait for the reply:
  - `ok` — apply with `Edit`. For an appendix move, edit both files in the same turn (remove from source, add to `99-appendix.qmd` with the `\addcontentsline{loa}{section}{...}` block and a `{#sec-...}` label, replace the removed content with a one-line summary + cross-reference).
  - `s` / `skip` — record in the plan as skipped, with the reason if the user gives one.
  - `issue` — record in the plan's Open Items table as `logged`, not applied.
  - anything else — treat as a custom instruction (e.g. a different phrasing for the replacement) and apply that instead.
  - `done` — stop, update the plan file, commit.

## Finishing

On `done`, or when the scoped review is complete:

1. Re-render if you made edits and want an accurate page count this session: `uv run poe thesis-pdf`, then `pdfinfo thesis/qmd/src/masters_thesis.pdf`. Update the plan file's "Current" line with the real measurement, not the word-count estimate, when you do.
2. Rewrite `docs/masters_thesis/length_reduction_plan.md`: chapters reviewed, cumulative words cut, updated open-items table.
3. Commit the plan file and every edited `.qmd` file. Follow this project's normal convention (branch off `master`, PR, merge) rather than committing to `master` directly — `no-commit-to-branch` will reject it anyway.
4. If several tables moved or content was restructured across chapters, mention that `/pdfsniffer` (visual rendering regressions) and `/equation-verifier` (if any moved passages contained equations) are worth running before the next thesis-PDF push — do not run them yourself as part of this skill, that's scope creep.
5. Report: pages at start of session vs. now (or estimated now, labelled as an estimate, if you didn't re-render), words cut, items applied / skipped / logged, and how much of the ~80-page target remains.

## Important

- Do not cut results, numbers, or claims to hit the page target — only cut duplication, movable detail, and over-broad background. If a category-3 judgment call would remove something a reader needs to trust a result, that is not in scope for this skill; leave it and say so.
- Do not touch prose style (that's `hemingway`) or formal layout (that's `thesis-format-auditor`) while doing this pass, even if you notice something — note it, don't fix it here, to keep this skill's diffs reviewable as length-reduction only.
- A table move to the appendix must preserve the WNE UW list-of-appendices requirement (`\addcontentsline{loa}{section}{...}`) — an appendix section that doesn't appear in that list is a formal-compliance regression this skill would have caused, not fixed.
