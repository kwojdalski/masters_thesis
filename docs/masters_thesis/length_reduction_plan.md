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

Current: **148 pages** (measured 2026-08-29, fresh render after item 16)
Body word count (chapters 01-07): **32,392 words**
Body page range: printed pages 8-114 (physical PDF pages 9-115), **107 body
pages**; Bibliography begins on printed page 115
words_per_page: ~303 (32,392 / 107 measured body pages)
Cumulative body reduction from the complete pre-condensation draft: **6,209
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

- [x] 01 Introduction — fully read and reviewed. The hypotheses and chapter
      roadmap are proportionate; the novelty/objectives material and the end of
      the introduction repeat the same scope and limitations (item 18).
- [ ] 02 Literature Review — partially reviewed: `02-04` and `02-07` have now
      both been read in full. `02-04` was cleared after item 10; `02-07` is a
      confirmed breadth candidate (item 11). The remaining six files still need
      a category-3 pass.
- [ ] 03 Reinforcement Learning — `03-02-actor-critic-methods.qmd` fully read
      and now reviewed for both tables and category-3 breadth. Its redundant
      table was removed (item 12), and its repeated algorithm-selection material
      was consolidated (item 16). Other Chapter 3 files still need review.
- [ ] 04 Design of the Trading Agent — `04-01`, `04-02`, and `04-05` fully
      read. The duplicated feature-formula and actor-architecture tables were
      removed (items 14-15); the controlled policy-comparison table was kept
      because it is the single implementation-level comparison used to frame
      Chapter 6. Repeated exploration derivations were consolidated (item 17).
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
| 5 | 3 | 02-04, 02-07 | Initial literature-review breadth flag. Both files were later read in full: `02-04` was cleared after item 10, while the confirmed `02-07` proposal is tracked as item 11. | 0 | superseded by items 10-11 |
| 6 | 3 | 03-02-actor-critic-methods.qmd | Initial actor-critic-primer breadth flag. The file was subsequently reviewed in full: duplicated comparison material was removed through items 12 and 16, while the core theoretical lineage and equations were retained. | 0 | superseded by items 12 and 16 |
| 7 | 2 | 06-00, 06-02, 06-03 | Reviewed all H1-H4 and benchmark tables. Their interpretation cites specific returns, drawdowns, turnover, exposure, and trial statistics, so moving or deleting them would separate claims from evidence. | 0 | cleared — kept |
| 8 | 3 | 05-01 (`sec-feature-normalization-and-causality-preservation`) | User flagged: full Welford's-algorithm derivation (2 numbered equations, itemized symbol definitions) for a 60-year-old, off-the-shelf online mean/variance algorithm — inconsistent with how the thesis treats other standard techniques (e.g. ReLU gets a one-line citation, no derivation, in 04-05-policy.qmd). Verified via grep that `eq-welford-mean`/`eq-welford-var` are never cross-referenced anywhere else in the thesis before cutting them. Also found while investigating: the kept equation (`eq-running-normalize`, cited later at line ~253 so it had to stay) redefined $\bar{x}_t$/$\sigma_t^2$/$\varepsilon$ that ch4's `eq-z-score` (04-02-state-space.qmd:147) already fully defines — trimmed to note the correspondence instead of re-itemizing. Causal-normalization reasoning (genuinely thesis-specific: why global normalization would leak future information) kept in full. | -115 (3812 -> 3697) | applied |
| 9 | 3 | Whole thesis | User asked for a systematic sweep for the same over-derivation pattern. Inventoried all 35 numbered equations, cross-reference-counted each (`@eq-...` citations elsewhere), then read every zero/low-citation candidate to separate "thesis's own design, correctly proportionate" (eq-huber-loss, eq-target-actor-update, eq-obs-space, eq-action-space, eq-transaction-fee — all checked and cleared, one equation + few symbols + thesis-specific justification each) from genuine over-derivation. Two confirmed and applied: (a) `eq-tw-mean`/`eq-tw-var` (05-01) — two full equations for a time-weighted normalization variant the text itself says was never used in the main experiments ("Sensitivity analysis... a direction for future work"); compressed to one sentence. (b) `eq-microprice-ch2` (02-02, literature review) — genuine cross-chapter duplication, not just over-derivation: the identical formula with the same citation (@Stoikov2018) already exists in the appendix's Feature Inventory table (99-appendix.qmd:65), and ch4's own body just says "the microprice [@Stoikov2018] is the central construct" without re-deriving it. Cut the display equation and symbol list, kept the intuition-building prose (why microprice beats mid-price when queues are imbalanced — not duplicated anywhere), added a pointer to Appendix A for the exact formula. | -48 (05-01: 3697->3649) + -33 (02-02: 1193->1160) | applied |
| 10 | 3 | 02-04-competing-modeling-approaches.qmd | Found during the item-9 sweep but dropped from the final report until the user asked to look for more instances. `eq-imitation-learning`: generic empirical-risk-minimization loss formula for imitation learning (one of several "competing approaches" surveyed, never this thesis's own method), 6-symbol itemized list, 0 cross-references anywhere. Same shape as items 8/9. Cut the equation, kept the surrounding prose describing behavior cloning in one clause. | -67 (1486 -> 1419) | applied |
| 11 | 3 | 02-07-applied-rl-trading-evidence.qmd | Fully read. The ~15-20 individually-surveyed papers are closer to PhD-survey breadth than a representative master's-thesis sample, although the methodological criticism is good. Merge the early mixed-evidence cluster (Neuneier1998, Lee2007, Gold2003, Dempster2002) into one paragraph and combine the Yang2020/AlphaStock boundary cases into one paragraph explaining why bar-level/cross-asset evidence does not establish LOB-level performance. Keep Majidi2024, Kabbani2022, and FinRL2020 individually treated because they bear directly on the thesis design and closing synthesis. | ~230-300 est. | reviewed — awaiting go-ahead |
| 12 | 1 | 03-02, 04-05 | Removed the nine-row algorithm-properties table from Chapter 3: its four preceding paragraphs already explain every comparison, and Chapter 4 retains the single implementation-level policy table. Added a cross-reference to `@tbl-policy-comparison`. | -153 body | applied |
| 13 | 2 | 05-01, 99-appendix | See item 4a: moved the full feature-statistics table to the appendix while preserving all Chapter 5 conclusions. | included in 4a | applied |
| 14 | 2 | 04-02 | Removed the six-row key-feature formula table because Appendix A already contains the same formulas plus parameters, citations, and inclusion status. Kept the complete feature-selection rationale inline. | -146 body | applied |
| 15 | 2 | 04-05 | Removed the generated actor-architecture table because the immediately preceding network equation gives every layer and activation, while Appendix B records the exported widths. | -144 body | applied |
| 16 | 1/3 | 03-02 | Consolidated the repeated algorithm-selection material. The final 549-word PPO/DDPG/TD3 comparison is now a short bridge to Chapter 4's implementation table, and the selection rationale now states each constraint once. Retained all core equations, TD3's three mechanisms, the SAC scope rationale, the contextual-not-general-superiority caveat, and the market-design limitations. | -672 body (measured) | applied (`95ac3f99`) |
| 17 | 1 | 03-02, 04-01, 04-05 | Gaussian exploration was explained generally in `04-01`, then re-derived separately for DDPG and TD3 in `04-05`; evaluation-time noise removal and TD3 target smoothing were each stated multiple times. Kept `eq-exploration-noise` as the authoritative equation, replaced the two policy subsections with one compact DDPG/TD3 comparison, retained the distinct target-smoothing parameters and purpose, and repaired Chapter 3's PPO cross-reference. | -400 body (measured) | applied |
| 18 | 1/3 | 01-00, 01-01 | The introduction already states the TD3/HFT/LOB/continuous-control contribution, then "Novelty" states it again. The objectives list also restates the hypotheses, while the introduction's final three paragraphs preview execution and evaluation limitations treated fully in Chapters 5 and 7. Keep the hypotheses and one concise boundary paragraph; compress novelty and objectives around them. | ~220-300 est. | reviewed — awaiting go-ahead |
| 19 | 1 | 04-01, 05-01 | The blue-chip short-borrow justification appears twice with different benchmark rates. Keep the quantitative modelling justification in the action-space section, where the cost-symmetry assumption is defined; reduce Chapter 5's asset-selection treatment to a cross-reference. | ~80-100 est. | reviewed — awaiting go-ahead |

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

**2026-08-29, session 3 (reconnaissance):** Fully reviewed Chapter 1,
`02-07`, `03-02`, `04-01`, and `04-05` for the next prose-structure pass.
Confirmed five candidates (items 11 and 16-19) with an estimated combined
reduction of roughly 1,530-1,880 body words, or about 5-6 body pages at the
current measured density. No thesis prose was changed in this reconnaissance
step; each item remains available for interactive approval.

**2026-08-29, session 4 (item 17):** Consolidated the repeated exploration
material across Chapters 3 and 4. Removed two duplicate equations and repeated
symbol definitions while retaining the authoritative exploration equation,
evaluation behavior, and TD3's distinct target-policy-smoothing parameters and
purpose. The cut removed 400 body words. A fresh Quarto render succeeded with
all cross-references resolved; the PDF remains 150 total / 109 body pages
because the cut did not cross a page boundary.

**2026-08-29, session 5 (item 16):** Consolidated Chapter 3's repeated
algorithm-selection and comparison material. The safer final version preserves
the SAC exclusion and all thesis-specific caveats, so the measured saving is 672
words rather than the 750-850 estimate. Commit `95ac3f99` contains the Chapter 3
change. A fresh Quarto render succeeded: total PDF 150 -> 148 pages and measured
body 109 -> 107 pages. The build again reported existing missing-evaluation-data
warnings for seven scenarios, but they did not prevent rendering.
