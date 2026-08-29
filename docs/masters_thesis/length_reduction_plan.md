# Thesis Length Reduction Plan

Started: 2026-08-29  Target: ~80 pages + appendix

**The starting baseline was wrong.** The 144 pages measured at the start of
session 1 was an *incomplete* render: `masters-thesis.qmd` did not include
`07-01` through `07-05` (Summary of Findings, Limitations, Implications and
Recommendations, Conclusion) at all — confirmed via `pdftotext -layout`
page-boundary inspection and `git diff HEAD` (this was `master`'s actual
committed state, not a stray edit). ~6,900 words of finished content were
absent from the PDF. That was fixed this session (see item 0 below), so the
real starting point for the 80-page target is the post-fix count, not 144.

Current: **150 pages** (measured 2026-08-29, fresh render after the session-2
tables pass)
Body word count (chapters 01-07): **33,464 words**
Body page range: printed pages 8-116 (physical PDF pages 9-117), **109 body
pages**; Bibliography begins on printed page 117
words_per_page: ~307 (33,464 / 109 measured body pages)
Cumulative body reduction from the complete pre-condensation draft: **5,137
words**. Session 2 alone removed 710 body words and reduced the measured body
from 111 to 109 pages; the total PDF fell from 151 to 150 pages because one
body table moved into the appendix.

Session 1 scope note: started with chapter 7 (closing material) rather than
chapter-number order, because initial reconnaissance (word counts across all
`.qmd` files) showed it was the single largest concentration of duplication
in the thesis. This uncovered the missing-includes bug (item 0), which
changed the session's actual outcome from "cut N pages" to "fix a
correctness bug, then cut duplication from the now-complete chapter" — net
page count went *up* (144 -> 151), which is correct and expected: the
99-page target was never meaningful against an incomplete document.
Subsequent sessions should resume in chapter-number order (02, 03, 04, 05,
06) unless a similarly strong signal justifies reordering again.

## Chapters reviewed

- [ ] 01 Introduction — not yet reviewed
- [ ] 02 Literature Review — not yet reviewed (word-count recon only: ~7,046
      words across 8 files; `02-04-competing-modeling-approaches.qmd` ~1,486
      words and `02-07-applied-rl-trading-evidence.qmd` ~1,568 words are the
      two most likely category-3 candidates, not yet confirmed by a full read)
- [ ] 03 Reinforcement Learning — `03-02-actor-critic-methods.qmd` fully read
      and reviewed for tables; its redundant algorithm-comparison table was
      removed (item 12). The broader category-3 review is still pending.
- [ ] 04 Design of the Trading Agent — table-bearing files `04-02` and `04-05`
      fully read and reviewed for tables. The duplicated feature-formula and
      actor-architecture tables were removed (items 14-15); the controlled
      policy-comparison table was kept because it is the single implementation-
      level comparison used to frame Chapter 6.
- [ ] 05 Implementation — `05-01-data-preparation.qmd` fully read and reviewed
      for tables. Three table reductions/moves applied across sessions (items 4
      and 13); the transformed-event, feature-correlation, and three-row split
      tables were kept because surrounding prose uses their specific values.
- [ ] 06 Results — all result tables in `06-00`, `06-02`, and `06-03` fully
      read and cleared in the tables-only pass. Kept because the interpretation
      cites specific cells and removing them would remove empirical evidence.
- [x] 07 Conclusions and Future Work — fully read (all 5 files) and reviewed
      interactively with the user. 1 structural bug fixed (item 0), 3
      duplication items applied (items 1-3), 1 over-aggressive compression
      caught by the user and corrected (item 2a). Chapter word count:
      6,901 as originally drafted (0 of it rendering) -> 2,685 now rendering
      (measured) — the ~4,200-word net cut from deduplication is why total
      page count only grew 144 -> 151 instead of the ~19 pages that adding
      6,901 raw words back would otherwise have cost. Considered fully
      reviewed for this pass; a second look after other chapters are done
      would be reasonable but not urgent.

## Open items

| # | Cat | Chapter(s) | Item | Words | Status |
|---|-----|-----------|------|-------|--------|
| 0 | - | masters-thesis.qmd | `07-01`-`07-05` were entirely missing from the include list — a correctness bug, not a length finding. Fixed: includes restored, `07-04` removed (content merged into `07-03` by item 3, so keeping both would have meant real duplication in the rendered PDF). | +6,900 (restored) | applied |
| 1 | 1 | 07-00, 07-05 | Chapter-opener (838 words) pre-empted and near-duplicated the closing "Conclusion" subsection verbatim in its last sentence. Trimmed to a 65-word roadmap paragraph; `07-05` kept as the chapter's single closing synthesis. | -773 | applied |
| 2 | 1 | 07-02 | Limitations section (12 subsections, 4,167 words) restated ch3 TD3 theory, duplicated 07-00's future-work list, and gave every future-work idea a full subsection instead of a compact list. | -2,994 net | applied |
| 2a | - | 07-02 | Correction to item 2, made after the user flagged that compression likely dropped real content — verified true for 5 of the 12 original subsections. Restored as compact additions rather than full subsections: extrapolation error (confirmed genuinely distinct from ch3's overestimation-bias coverage via `grep` — 0 matches for extrapolation/OOD/out-of-distribution in `03-02`), the dual-class-share pairs-trading idea, architecture-search specifics (GELU/ELU, critic layer norm), the longer-horizon predictive-power point, and FX/futures/crypto as the specific cross-asset test set. | +305 (868 -> 1,173) | applied |
| 3 | 1 | 07-03, 07-04 | "Implications for Trading Systems" and "Recommendations for Practitioners" restated 3 of 4 points each, differently framed. Merged into one subsection, each point once, plus both files' unique points preserved (continuous-action rationale from 07-03; chronological-evaluation protocol from 07-04). | -450 (1,010 -> 560, measured) | applied |
| 4 | 2 | 05-01-data-preparation.qmd | Read in full: 7 tables, not 3 (grep for "tbl-cap" missed `output: asis` cells). `tbl-input-schema` and `tbl-raw-sample` did the same "show what raw data looks like" job and were merged. `tbl-raw-file-inventory` moved to `#sec-appendix-raw-inventory`. `tbl-transformed-features` and `cell-feature-correlations` were kept because the prose references specific cells. Remaining decisions are recorded in 4a/4b. | -252 net (schema merge -90, appendix move -162 from body, +226 to appendix) | applied |
| 4a | 2 | 05-01, 99-appendix | Moved `cell-feature-stats` and its detailed distributional explanation to new listed appendix section `#sec-appendix-feature-statistics`. Chapter 5 retains the normalization, OFI heavy-tail, pre-clipping extreme, and observation-clipping conclusions. | -267 body; +263 appendix | applied |
| 4b | 1/2 | 05-01 | `tbl-dataset-splits` is only three rows and gives the post-filter counts used by the experiments; the appendix inventory instead gives per-symbol pre-filter counts. The distinction is load-bearing and consolidation would make the compact body table harder to read. | 0 | cleared — kept |
| 5 | 3 | 02-04, 02-07 | Literature review breadth — flagged by word count only, not yet read in full | TBD | not reviewed |
| 6 | 3 | 03-02-actor-critic-methods.qmd | ~3,193 words for an actor-critic primer whose job is to set up ch4's design choices — flagged by word count only, not yet read for category-3 purposes (was read for the ch7 cross-check, but not evaluated as a cut candidate itself) | TBD | not reviewed |
| 7 | 2 | 06-00, 06-02, 06-03 | Reviewed all H1-H4 and benchmark tables. Their interpretation cites specific returns, drawdowns, turnover, exposure, and trial statistics, so moving or deleting them would separate claims from evidence. | 0 | cleared — kept |
| 8 | 3 | 05-01 (`sec-feature-normalization-and-causality-preservation`) | User flagged: full Welford's-algorithm derivation (2 numbered equations, itemized symbol definitions) for a 60-year-old, off-the-shelf online mean/variance algorithm — inconsistent with how the thesis treats other standard techniques (e.g. ReLU gets a one-line citation, no derivation, in 04-05-policy.qmd). Verified via grep that `eq-welford-mean`/`eq-welford-var` are never cross-referenced anywhere else in the thesis before cutting them. Also found while investigating: the kept equation (`eq-running-normalize`, cited later at line ~253 so it had to stay) redefined $\bar{x}_t$/$\sigma_t^2$/$\varepsilon$ that ch4's `eq-z-score` (04-02-state-space.qmd:147) already fully defines — trimmed to note the correspondence instead of re-itemizing. Causal-normalization reasoning (genuinely thesis-specific: why global normalization would leak future information) kept in full. | -115 (3812 -> 3697) | applied |
| 9 | 3 | Whole thesis | User asked for a systematic sweep for the same over-derivation pattern. Inventoried all 35 numbered equations, cross-reference-counted each (`@eq-...` citations elsewhere), then read every zero/low-citation candidate to separate "thesis's own design, correctly proportionate" (eq-huber-loss, eq-target-actor-update, eq-obs-space, eq-action-space, eq-transaction-fee — all checked and cleared, one equation + few symbols + thesis-specific justification each) from genuine over-derivation. Two confirmed and applied: (a) `eq-tw-mean`/`eq-tw-var` (05-01) — two full equations for a time-weighted normalization variant the text itself says was never used in the main experiments ("Sensitivity analysis... a direction for future work"); compressed to one sentence. (b) `eq-microprice-ch2` (02-02, literature review) — genuine cross-chapter duplication, not just over-derivation: the identical formula with the same citation (@Stoikov2018) already exists in the appendix's Feature Inventory table (99-appendix.qmd:65), and ch4's own body just says "the microprice [@Stoikov2018] is the central construct" without re-deriving it. Cut the display equation and symbol list, kept the intuition-building prose (why microprice beats mid-price when queues are imbalanced — not duplicated anywhere), added a pointer to Appendix A for the exact formula. | -48 (05-01: 3697->3649) + -33 (02-02: 1193->1160) | applied |
| 10 | 3 | 02-04-competing-modeling-approaches.qmd | Found during the item-9 sweep but dropped from the final report until the user asked to look for more instances. `eq-imitation-learning`: generic empirical-risk-minimization loss formula for imitation learning (one of several "competing approaches" surveyed, never this thesis's own method), 6-symbol itemized list, 0 cross-references anywhere. Same shape as items 8/9. Cut the equation, kept the surrounding prose describing behavior cloning in one clause. | -67 (1486 -> 1419) | applied |
| 11 | 3 | 02-07-applied-rl-trading-evidence.qmd | User asked for "too broad process / textbook description" more generally. Read chapter 3 (03-00, 03-01) and 02-04 in full: both look survey-like on the surface but every subsection ties back to this thesis's specific design choice — checked and cleared, not candidates. One genuine finding, flagged as judgment-call rather than mechanical: ~15-20 individually-surveyed papers is closer to PhD-survey breadth than masters-thesis "representative sample," though the content quality is good (every study gets a real methodological caveat, not passive summary). Proposed: merge the "early/proof-of-concept, mixed evidence" cluster (Neuneier1998, Lee2007, Gold2003, Dempster2002 — explicitly framed in the text as making the same point) into one paragraph; keep Majidi2024/Kabbani2022/FinRL2020 individually treated since the closing synthesis names them specifically. Not yet applied — awaiting go-ahead. | ~150-200 est. | not reviewed |
| 12 | 1 | 03-02, 04-05 | Removed the nine-row algorithm-properties table from Chapter 3: its four preceding paragraphs already explain every comparison, and Chapter 4 retains the single implementation-level policy table. Added a cross-reference to `@tbl-policy-comparison`. | -153 body | applied |
| 13 | 2 | 05-01, 99-appendix | See item 4a: moved the full feature-statistics table to the appendix while preserving all Chapter 5 conclusions. | included in 4a | applied |
| 14 | 2 | 04-02 | Removed the six-row key-feature formula table because Appendix A already contains the same formulas plus parameters, citations, and inclusion status. Kept the complete feature-selection rationale inline. | -146 body | applied |
| 15 | 2 | 04-05 | Removed the generated actor-architecture table because the immediately preceding network equation gives every layer and activation, while Appendix B records the exported widths. | -144 body | applied |

## Session log

**2026-08-29, session 1:** Started with chapter 7 on the strength of
word-count recon, not chapter order. Found and fixed a correctness bug
(item 0: `07-01`-`07-05` not rendering at all) before the length-reduction
findings (items 1-3) were meaningful. First compression pass on item 2 was
too aggressive — caught by the user, audited against the original 12
subsections, and corrected (item 2a) by restoring 5 confirmed-distinct ideas
as compact additions rather than leaving them cut. Net result for chapter 7:
structurally sound (no duplicate closing statements, no restated theory, no
overlapping practitioner-advice subsections) and complete (nothing that was
in the original 5 files' distinct content is missing from the compressed
version). Page count moved 144 -> 151 (up, because the fix restored more
content than the cuts removed) — this is the correct outcome given the
starting number was measuring an incomplete document, not a regression
against the 80-page target. Chapters 01-06 not yet reviewed beyond recon
word counts and the one cross-reference read of `03-02`.

**Lesson for next session:** when a compression estimate ("X words saved")
is based on a subsection count or word count alone rather than an idea-by-
idea audit, verify duplication claims per distinct idea, not per whole
subsection — a subsection can be 80% duplicated and 20% novel, and cutting
the whole thing loses the 20%. This is now standard practice for this plan,
not just a one-off fix.

**2026-08-29, session 2 (tables-only):** Read every table-bearing chapter file
and the appendix in full. Removed three body tables whose information was
already present in adjacent prose or the appendix (algorithm properties, key
feature formulas, actor architecture), and moved the full feature-distribution
table to a new listed appendix section while retaining its conclusions inline.
All result tables were kept because their surrounding interpretation cites
specific values. Fresh render succeeded: total PDF 151 -> 150 pages; body 111
-> 109 pages; body words 34,174 immediately before this pass -> 33,464. The
appendix gained one page, which is why the two-page body reduction appears as a
one-page reduction in the total PDF.
