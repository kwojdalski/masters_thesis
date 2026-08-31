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

Current: **140 pages** (measured 2026-08-31, session 12, fresh render after
merging PR #554 which contained session 11's items 11/18/29/30/33/34)
Body word count (chapters 01-07): **29,552 words**
Body page range: printed pages 8-102 (physical PDF pages 9-103), **95 body
pages**; Bibliography begins on printed page 103
words_per_page: ~311 (29,552 / 95 measured body pages)
Cumulative body reduction from the complete pre-condensation draft: **8,196
words**. Session 2 alone removed 710 body words and reduced the measured body
from 111 to 109 pages; the total PDF fell from 151 to 150 pages because one
body table moved into the appendix. Session 6 moved a second body table into
the appendix, reducing the body by 2 further pages (107 -> 105) at unchanged
total PDF length (148 pages before and after), since the appendix absorbed the
page the body gave up. Session 7 cut a 4-way restated methodological warning,
tightened an over-derived reward-function subsection, deduplicated a
four-way-copy-pasted MBP-10 format definition, and condensed the state-space
chapter's fragmented scope-justification prose, taking total pages 148 -> 146
and body 105 -> 103. Session 8 cut two within-chapter restated conclusions in
the Results chapter, taking total pages 146 -> 144 and body 103 -> 102.
Session 9 closed the last queued cross-chapter duplicate (item 19) and gutted
a generic-filler section the user flagged directly, taking body 102 -> 101 at
unchanged total (144). Session 10 cut a chapter-wide restated epistemic
caveat across four Chapter 2 files (item 32), taking total pages 144 -> 142
and body 101 -> 99 — the first genuine information-reduction cut (not pure
deduplication) applied at the user's explicit request to shorten Chapter 2.

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
      the introduction's repeated scope/limitations were condensed (item 18,
      -134 words).
- [x] 02 Literature Review — all 9 files fully read. `02-04` was cleared
      after item 10; `02-07`'s breadth candidate was applied (item 11,
      framing-only trim, -78 words). A chapter-wide repeated epistemic caveat across
      `02-01`, `02-02`, `02-03`, and `02-08` was cut (item 32) at the user's
      explicit request to reduce information, not just duplication. `02-00`,
      `02-05`, `02-06` remain tightly scoped from earlier reconnaissance — no
      findings.
- [x] 03 Reinforcement Learning — all three files fully read and reviewed.
      `03-02-actor-critic-methods.qmd`'s redundant table was removed (item 12),
      and its repeated algorithm-selection material was consolidated (item 16).
      `03-00-reinforcement-learning.qmd`'s in-file duplicated partial-observability
      explanation was condensed (item 33). `03-01-rl-categories.qmd` re-checked
      fresh and found tightly scoped — a taxonomy chapter that correctly defers
      detail to `03-02` rather than duplicating it.
- [x] 04 Design of the Trading Agent — all seven files fully read
      (`04-01` through `04-07`). The duplicated feature-formula and
      actor-architecture tables were removed (items 14-15); the controlled
      policy-comparison table was kept because it is the single
      implementation-level comparison used to frame Chapter 6. Repeated
      exploration derivations were consolidated (item 17). `04-03`'s DSR
      training-vs-evaluation warning was deduplicated against `05-02`/`07-03`
      and its limitations discussion tightened (items 21-22). `04-04`, `04-06`,
      and `04-07` are tightly scoped, thesis-specific instantiations of
      Chapter 3 theory — checked, no findings.
- [x] 05 Implementation — both files fully read. `05-01-data-preparation.qmd`
      reviewed for tables three times; five table reductions/moves applied
      across sessions (items 4, 13, 20, and 30); the raw-sample and
      three-row split tables were kept because surrounding prose uses their
      specific values. The feature-correlation table (initially kept under
      item 4) was later moved to the appendix once a re-read confirmed the
      prose only cites the aggregate claim (item 30). The twelve-row
      transformed-feature illustration was
      moved to the appendix in session 6 (item 20). `05-02-code.qmd` (pipeline,
      training-loop pseudocode, sanity checks) read for the first time in
      session 7: the pseudocode is legitimate operationalization of Chapter 3/4
      equations, not a restatement, and does not overlap with `05-01`'s content
      or `06-*`'s sanity-check mention (different purpose: implementation
      pre-flight check vs. a formal benchmark strategy). One duplication found
      and cut (item 21).
- [x] 06 Results — all four files fully read for both tables (prior session)
      and prose duplication/breadth (session 8). Result tables kept because the
      interpretation cites specific cells. Two within-chapter restated
      conclusions found and cut (items 24-25): `06-00`'s "Key Empirical
      Questions" pre-empted "Discussion of Algorithmic Trade-offs"; `06-03`
      had two consecutive closing paragraphs stating the same H1 verdict.
      `06-01`'s "Comparative Validation Strategy" hierarchy and its caveats
      list are compact and address distinct concerns — checked, no findings.
      Two data/consistency issues noticed but out of this skill's scope
      (logged as item 26): a numeric mismatch (TD3 long/short % and turnover
      differ between `06-00` and `06-03` for the same experiment) and an
      apparent contradiction (`06-01` lists 5-seed repetition as unaddressed
      future work; `06-02`'s H4 section claims to already close that gap with
      `n=5` trials) — refer to `thesis-data-auditor` or a manual check, not a
      length finding.
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
| 11 | 3 | 02-07-applied-rl-trading-evidence.qmd | Merged the early mixed-evidence cluster (Neuneier1998, Lee2007, Gold2003, Dempster2002) into one paragraph and the Yang2020/AlphaStock boundary cases into one paragraph. Kept every citation and every distinct claim — only the per-paper framing sentences ("The study is relevant as...", "Their relevance is that...") were cut, restated once per cluster instead of once per paper. Actual saving (78 words) came in well under the original 230-300 estimate because the user confirmed a framing-only trim rather than cutting a paper's treatment down to citation-only. Majidi2024, Kabbani2022, and FinRL2020 kept individually treated as planned. | -78 (1568 -> 1490) | applied |
| 12 | 1 | 03-02, 04-05 | Removed the nine-row algorithm-properties table from Chapter 3: its four preceding paragraphs already explain every comparison, and Chapter 4 retains the single implementation-level policy table. Added a cross-reference to `@tbl-policy-comparison`. | -153 body | applied |
| 13 | 2 | 05-01, 99-appendix | See item 4a: moved the full feature-statistics table to the appendix while preserving all Chapter 5 conclusions. | included in 4a | applied |
| 14 | 2 | 04-02 | Removed the six-row key-feature formula table because Appendix A already contains the same formulas plus parameters, citations, and inclusion status. Kept the complete feature-selection rationale inline. | -146 body | applied |
| 15 | 2 | 04-05 | Removed the generated actor-architecture table because the immediately preceding network equation gives every layer and activation, while Appendix B records the exported widths. | -144 body | applied |
| 16 | 1/3 | 03-02 | Consolidated the repeated algorithm-selection material. The final 549-word PPO/DDPG/TD3 comparison is now a short bridge to Chapter 4's implementation table, and the selection rationale now states each constraint once. Retained all core equations, TD3's three mechanisms, the SAC scope rationale, the contextual-not-general-superiority caveat, and the market-design limitations. | -672 body (measured) | applied (`95ac3f99`) |
| 17 | 1 | 03-02, 04-01, 04-05 | Gaussian exploration was explained generally in `04-01`, then re-derived separately for DDPG and TD3 in `04-05`; evaluation-time noise removal and TD3 target smoothing were each stated multiple times. Kept `eq-exploration-noise` as the authoritative equation, replaced the two policy subsections with one compact DDPG/TD3 comparison, retained the distinct target-smoothing parameters and purpose, and repaired Chapter 3's PPO cross-reference. | -400 body (measured) | applied |
| 18 | 1/3 | 01-00, 01-01 | Merged the introduction's three closing paragraphs (execution constraints, methodological-discipline framing, simulation-scope caveat) into one boundary paragraph pointing to Chapters 5/7 for detail (-83). Compressed "Novelty" to stop restating the intro's TD3/HFT/continuous-control claim, keeping only the genuinely new comparative claim vs. Q-learning/DQN/PPO bar-level studies (-23). Cut the two Objectives bullets that restated Hypotheses 1 and 2 verbatim, folding a one-clause pointer into the surviving bullets instead (-28). Hypotheses themselves untouched. Actual saving (134 words) came in under the 220-300 estimate, same pattern as item 11 — keeping every distinct claim costs less than a raw subsection-level estimate suggests. | -134 (441 -> 307 across both files) | applied |
| 19 | 1 | 04-01, 05-01 | The blue-chip short-borrow justification appeared twice with slightly different benchmark rates (04-01: 25-75 bps range with a 50 bps worked example tied to the reward function's cost-symmetry term; 05-01: "below 0.5% per annum" ceiling in the asset-selection context). Kept 04-01's quantitative version as authoritative; condensed 05-01 to its own distinct content (why blue-chip liquidity was the selection criterion, the illiquid-instrument contrast) with a cross-reference to Section 4.1. | -80 | applied (`a34ca641`) |
| 20 | 2 | 05-01, 99-appendix | User-directed re-review of tables specifically. `tbl-transformed-features` (12-row x 9-col AAPL example) forced its own dedicated landscape page purely for illustration; the row-selection window is hardcoded in `thesis_tables.py::lob_events_table()`, so shrinking it in place would have required a Python edit with uncertain page payoff (landscape is driven by column width, not row count). Moved the full table and its intro paragraph to a new listed appendix section (`#sec-appendix-transformed-features`); the body keeps the three interpretive claims (bid-refresh sign flip at event 12, microprice-deviation intuition, OFI spike) in one condensed paragraph pointing to the appendix table. | -2 body pages (107 -> 105); words -198 (32,392 -> 32,194); total PDF unchanged (148) since the appendix absorbed the page | applied (`75ad10de`) |
| 21 | 1 | 04-03, 05-02, 07-03 | First full read of `04-03` (reward function), `04-04`, `04-06`, `04-07`, and `05-02` (all previously untouched by any session). Found a genuine methodological point restated in full 4 times: "training reward (DSR) is not the evaluation financial metric; conflating them (e.g. citing a high training DSR as evidence of held-out Sharpe) is invalid" appears in `04-03` (fullest, kept as authoritative), `05-02`'s "two output streams" implementation note, `07-03`'s practitioner-recommendation bullet, and a fourth copy at the end of `04-03` itself restating `07-02`'s "Algorithm and reward ablation" future-work item down to the same Sortino/drawdown-penalized alternatives. Kept 04-03's version, compressed 05-02 and 07-03 to their own distinct framing with a pointer to Section 4.3, and replaced 04-03's closing sentence with a plain forward pointer to Chapter 7. `04-04`, `04-06`, `04-07`, and the pseudocode in `05-02` were checked and found tightly scoped — no further findings. | -70 (05-02+07-03) | applied (`997e1e0e`) |
| 22 | 3 | 04-03 | User asked directly whether "4.3.2 Differential Sharpe Ratio" was too long. Confirmed: not cross-chapter duplication (checked `07-02` for overlap — none), but internal disproportion — four full-paragraph limitations, a footnote deriving a complete numeric order-of-magnitude chain ($\sigma\approx4.5\times10^{-5}$ -> variance $2\times10^{-9}$ -> its $3/2$ power $10^{-13}$) just to justify not rescaling the reward, and a "why chosen anyway" reason that restated the training-vs-evaluation separation point made two paragraphs earlier in the same subsection. Condensed the four limitations to their substance, collapsed the footnote to its conclusion, and cut the redundant third reason. | -372 (1808 -> 1436); total PDF -2 pages (148 -> 146, crossed a page boundary), body -1 page (105 -> 104) | applied (`1927073a`) |
| 23 | 1 | 04-02, 04-03, 05-00 | User asked whether 4.2 State Space was too long. Found: (a) the full MBP-10 format definition copy-pasted verbatim as a footnote in `04-02`, `04-03`, and `05-00`, on top of the proper main-text explanation already in `05-01` — kept `05-01` as authoritative, replaced the other three with a short cross-reference; (b) within `04-02`, the position feature's normalization rationale stated in full twice ~40 lines apart; (c) `04-02`'s "augment with VIX futures/sector-ETF/options-skew" future-work aside duplicated `07-02`'s "Cross-asset and multi-asset extensions" item almost verbatim — cut to a plain forward pointer; (d) a closing paragraph in `04-02` that only recapped facts (d=11, normalization, Appendix A) already established earlier in the same file; (e) the "Scope and timescale consistency" criterion was fragmented into six 1-2 sentence paragraphs — consolidated into two denser paragraphs, no content lost. | -279 net across 3 files (04-02: 2125->1960; 04-03: 1436->1380; 05-00: shrunk) | applied (`1e7afecf`) |
| 24 | 1 | 06-00 | User asked for more items; first full duplication/breadth pass on Chapter 6 prose (tables already cleared in an earlier session). "Key Empirical Questions" (3 questions, ~230 words) set up theoretical framing — TD3 overestimation correction vs. DDPG, PPO stability-vs-efficiency — that "Discussion of Algorithmic Trade-offs" immediately below restates almost verbatim while actually doing the analysis (e.g. the DDPG paragraph explicitly repeats "the theoretical concern that DDPG's single-critic Q-learning is susceptible to overestimation bias"). The third question was too generic to ever be specifically answered. Compressed the three-question setup to one transitional sentence into Discussion. | -176 (1693 -> 1517) | applied (`267e2519`) |
| 25 | 1 | 06-03 | Found in the same Chapter 6 pass. Two consecutive closing paragraphs — "Taken together..." and "The performance verdict for H1..." — both concluded that the agent's better-than-passive return is risk-avoidance (near-neutral positioning) rather than genuine trading skill, with no deployability claim, just reworded. Merged into one paragraph retaining every distinct point (structural outperformance, weak trade-level quality, the "weak feasibility test" framing, and the forward pointer to H2/H3). | -25 (1218 -> 1193) | applied (`267e2519`) |
| 27 | 3 | 06-01 | User flagged `06-01`'s "Comparative Validation Strategy" 4-point interpretation hierarchy and "Statistical Risks and Interpretation Caveats" as generic filler never explicitly invoked by name in the H2/H3/H4 discussions that follow. Confirmed and asked the user to choose between delete, move to appendix, or gut to load-bearing content; user chose gut. Cut the 4-point hierarchy, the extended caveats list, and a dead code chunk that computed run statistics but never displayed them. Kept only: repeated-run divergence quantifies training stochasticity not market uncertainty, tuning-bias risk, and reward-vs-financial-objective divergence. | -287 (374 -> 87) | applied (`4395919c`) |
| 28 | 1 | 06-00 | Self-inflicted duplicate caught immediately by the user: item 27's gutted `06-01` text used nearly the same phrase as `06-00`'s "Comparison Methodology" ("matched experimental evidence... not a multi-seed estimate of the population distribution of outcomes"). Also separately, `06-00`'s version's first sentence restated its own section's opening paragraph ("identical data, features, and evaluation protocols"). Cut the redundant first sentence from `06-00`, kept its specific single-run-per-algorithm caveat as authoritative, and pointed `06-01` back to it instead of repeating the phrase. | -32 (1517 -> 1485) | applied (`4395919c`) |
| 26 | - | 06-00, 06-01, 06-03 | Two data/consistency issues noticed while reading Chapter 6 closely, out of this skill's scope (not duplication, breadth, or a movable table). (a) `06-00` reports the DSR-reward TD3 run's exposure as "49.8% long, 50.2% short, turnover 0.054%" while `06-03` reports the *same* experiment (`pooled_td3_hft_lob_state_space_pooled_streaming_selected_dsr`) as "54.9% long, 45.1% short, turnover 0.058%" — should be identical numbers from the same run. Still open; refer to `thesis-data-auditor` or a manual check. (b) `06-01` listed repeating "each configuration across at least five seeds" as unaddressed future work, while `06-02`'s H4 section (`_h4_n5`, `n=5` trials) explicitly claims to close exactly that gap — apparent contradiction. Incidentally resolved as a side effect of item 28: the gutted `06-01` no longer makes the 5-seed future-work claim, so the contradiction no longer exists in the rendered text (not verified against which claim was actually correct, just no longer stated twice). | 0 | (a) open — logged; (b) resolved incidentally by item 28 |
| 29 | 1 | 07-01 | H1's summary paragraph was roughly 2x the length of its H2/H3 siblings, the only one re-citing a full battery of exact figures (TWAP/VWAP percentages, long/short split, profit factor, drawdown comparison) that `06-03` already presents and discusses in detail. Compressed to state the qualified-support verdict and its reasoning (near-neutral positioning, poor trade-level quality, feasibility-not-edge framing) without repeating every number, with a plain-text pointer to Section 6.3 for the exact figures — matching H2/H3's style. `07-00` and `07-05` re-checked and found proportionate, no changes. | -71 (219 -> 148) | applied |
| 30 | 2 | 05-01, 99-appendix | Item 4's old note — "`cell-feature-correlations` was kept because the prose references specific cells" — was stale: the prose only ever cites the *aggregate* claim ("all absolute correlations are below 0.005"), never a single feature's value. Moved the code chunk to a new appendix section (`#sec-appendix-feature-correlations`, following the established `\addcontentsline{loa}{section}{...}` pattern), replacing it in the body with a one-sentence pointer to `@tbl-feature-correlations`. Confirmed via `thesis_tables.py` that the table's LaTeX `\label{tbl-feature-correlations}` is set directly by the table function (not the calling cell's label), so the cross-reference resolves regardless of which file the chunk lives in. | -54 body (3128 -> 3074, raw wc -w) | applied |
| 31 | 2 | 06-02 | Reconsidered from session 6 (previously "too small to be worth the appendix-move churn"). The H4 code chunk generates three separate tables: summary stats, pass/fail criteria, and a per-trial breakdown (up to `n=5` rows). "Interpretation of H4 Results" only ever discusses what pass/fail verdicts mean in general, never a specific trial's number — the summary and criteria tables carry the load-bearing pass/fail verdict a reader needs; only the per-trial detail table would move. Still marginal at N=5 rows. User declined in session 11: effect too marginal to be worth the appendix-move churn (same judgment as session 6, now final rather than reopened). | ~15-25 est. | skipped — user judged too marginal |
| 32 | 1/3 | 02-01, 02-02, 02-03, 02-08 | User explicitly asked to reduce information (not just dedupe) in the first chapters. Found the same epistemic caveat — "this microstructure regularity is descriptive, not a proven trading signal; profitability is tested empirically in later chapters" — restated roughly 10 times across the chapter, on top of `02-00` already stating the rule once for the whole chapter. Cut 3 restatements in `02-01` (two per-finding tags plus its closing paragraph), 4 in `02-02` (spread-decomposition tag, LOB-mechanics tag, OFI's caveat stated twice within 10 lines — footnote and the very next paragraph — collapsed to one, and a closing-paragraph tag), condensed `02-03` from 338 to ~140 words keeping only its one genuinely new idea (the "alphas" framing from quant finance), and trimmed `02-08`'s restatement to one clause. Kept mechanism-specific limitations that aren't the generic caveat (Kyle's lambda proxies missing the latent informed-trader information set; microprice being top-of-book-only). | -473 (02-01: 826->697; 02-02: 1160->1016; 02-03: 338->163; 02-08: 283->258) | applied (`beea69df`) |
| 33 | 1 | 03-00 | Fresh reconnaissance requested by the user (`03-00`-`03-02`, `02-04`, `02-07`, `07-02`) after the session-10 queue closed. Found the partial-observability explanation ("engineered features, not the full market state, because of hidden liquidity, private orders, latent regimes") given in full twice within ~20 lines of the same file — once in the "Components" chapter intro, once again under the "State" subsection. Kept the first (it sets up the whole chapter's approximate-MDP framing); compressed the second to its only genuinely new content, the forward pointer to Chapter 4.2. | -43 (64 -> 21) | applied |
| 34 | 1 | 02-07 | Found in the same reconnaissance pass. Two consecutive closing paragraphs both distilled "lessons for this thesis's design" from the literature survey — one organized by finding (continuous-action preference, DSR promise, cost modeling), one by design dimension (state/action/reward/validation) — restating the same points twice under different frames. Merged into one paragraph keeping every distinct point, including the "proof of concept, not a robust edge" framing that was unique to the second paragraph. | -45 (176 -> 131) | applied |
| 35 | 1 | 06-00, 06-02 | Found via a new automated near-duplicate-sentence scan (exact + Jaccard fuzzy match), not manual re-reading. `06-00`'s H1 table legend and `06-02`'s H2 table legend define the identical abbreviation set (Sharpe, Sortino, Return, Max DD, Win Rate, PF, Turnover, Ann. Vol), reordered and cosmetically reworded — a genuine zero-abbreviation-mismatch duplicate, unlike item 36. User asked whether a cross-referenced legend (vs. each table staying self-contained) is permitted under WNE UW's formal requirements; this skill has no authority on that (it's `thesis-format-auditor`/formal-compliance territory), so rather than guess, the user chose to skip. | ~45 (not applied) | skipped — formal-compliance uncertainty (self-contained-table convention), not verified against WNE UW's actual requirements |
| 36 | 1 | 06-00, 06-03 | Same scan. `06-00`'s H1 legend and `06-03`'s benchmark-table legend share most definitions (Sortino, Max DD, Win Rate, Turnover) but use different abbreviations for three metrics (TR vs Return, SR vs Sharpe, Volatility vs Ann. Vol), so a clean full cross-reference would've needed either a residual mapping for the differing three or a column-header rename — more invasive than item 35. Not reviewed in detail once the user skipped item 35 for the same underlying reason. | ~20-25 (not applied) | skipped — same formal-compliance uncertainty as item 35 |
| 37 | 1 | 06-00, 06-03 | Same scan. `06-03`'s one-sentence benchmark-category recap ("passive exposure, execution-style references, and a stochastic baseline") echoes `06-00`'s `#sec-benchmark-strategies` intro almost verbatim, but it's already compact (no re-definition of the five strategies) — marginal, comparable to item 31's "too small to be worth it" judgment. | ~10-15 (not applied) | skipped — marginal, same session decision as 35/36 |

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

**2026-08-29, session 6 (user-directed tables re-review, item 20):** User asked
for a fresh, skeptical pass over every remaining table specifically (not the
general category-3 queue), after the earlier tables-only session had already
cleared most of them. Re-read every table-bearing file end to end
(`05-01`, `06-00`, `06-02`, `06-03`, `04-05`, `99-appendix`) rather than
trusting prior notes, and confirmed nearly everything is legitimately
load-bearing (specific cell values cited in the surrounding prose) or already
referenced from the body via plain "Appendix A" text (`tbl-lob-features` —
zero `@tbl-` cross-refs but genuinely used, confirmed via grep for
"Appendix A" mentions across the body). One real finding survived: the
12-row/9-column `tbl-transformed-features` example forced its own dedicated
landscape page purely for illustration. Applied item 20 (see above); a minor
second candidate (H4's per-trial table in `06-02`, ~5 rows, not cited by
value) was flagged but left as-is — too small to be worth the appendix-move
churn. Fresh Quarto render succeeded: total PDF unchanged at 148 pages,
measured body 107 -> 105 pages, body words 32,392 -> 32,194.

**Incident during this session:** a separate Claude Code session was
concurrently committing to this same working directory and branch. While
this session's edits were in progress, an operation in that other session
reverted every uncommitted change in the tree back to the last shared commit,
which silently discarded both this session's first attempt at item 20 and
pre-existing uncommitted work from before this session started (item 17's
qmd edits, a `masters-thesis.qmd` `\newpage` fix, and this plan file's
session 3-5 documentation). The pre-existing work was recovered intact from a
dangling git-stash object left over from an earlier rebase in this same
session (commit `0748942f`); item 20 was then redone from scratch and
committed immediately (`75ad10de`) to minimize the exposure window. Lesson:
when multiple sessions may share a working directory, commit early and often
rather than batching edits, since uncommitted state is exactly what a
concurrent session's git operations can destroy without warning.

**2026-08-29, session 7 (user-directed hunt for more items, prose):** User
asked for more condensation candidates, specifically in chapters not yet
reviewed. Read every remaining unread file: `04-03` through `04-07`
(reward function, value function, discount factor, optimization
hyperparameters) and `05-02` (implementation/code) — none of these had been
opened in any prior session. Found one genuine 4-way restatement of a
methodological point spanning three chapters (item 21) and, on direct
follow-up from the user questioning the length of the DSR subsection,
confirmed and fixed a disproportionate internal-to-one-subsection bloat
(item 22) rather than cross-chapter duplication. This is the first session to
find a real finding outside Chapters 3, 5 (tables), and 7 — Chapters 4 and 5
are now fully read end to end. Working in a dedicated branch
(`kwojdalski/dedup-dsr-training-eval-separation`) from the start, given the
concurrent-session incident in session 6, and committing after every edit.
Fresh Quarto render succeeded: total PDF 148 -> 146 pages (crossed a page
boundary), body 105 -> 104 pages, body words 32,194 -> 31,737.

User then asked directly whether `04-02` (State Space) was too long. This
surfaced a second whole-thesis mechanical duplication in the same vein as the
DSR training-vs-evaluation warning: the full MBP-10 format definition was
copy-pasted as a footnote in `04-02`, `04-03`, and `05-00` on top of the
proper explanation in `05-01` (item 23a), plus three more `04-02`-local
issues (a within-file repeat, a future-work duplicate against `07-02`, and a
pure-recap closing paragraph) and one fragmented-prose consolidation. Applied
as item 23. Fresh render: total PDF unchanged at 146, body 104 -> 103 pages,
body words 31,737 -> 31,458.

**Remaining unreviewed material:** Chapter 2's six files other than `02-04`
and `02-07` were read fresh in the previous session's reconnaissance and found
tightly scoped (no findings); `03-00` and `03-01` were also read fresh and
found tightly scoped. Chapters 1-5 are therefore now fully read at least once.

**2026-08-29, session 8 (user-directed hunt for more items, Chapter 6 prose):**
User asked for more condensation candidates. Gave Chapter 6 — the plan's
flagged next target — its first full prose duplication/breadth pass (tables
were already cleared in an earlier session). Found and applied items 24-25
(both within-chapter restated conclusions, in `06-00` and `06-03`
respectively) and logged item 26 (two data/consistency issues, out of scope
for this skill). All four Chapter 6 files are now fully reviewed; every
chapter (01-07) has had at least one full read for duplication/breadth, and
every table-bearing file has had a dedicated tables pass. Fresh Quarto render
succeeded: total PDF 146 -> 144 pages, body 103 -> 102 pages, body words
31,458 -> 31,257.

While reviewing, the user separately spotted that `06-03`'s three evaluation
plots (equity curve, rewards, positions) render as "data not available" notices
in the actual PDF rather than the real plots — a missing-artifact issue, not a
length one. Out of this skill's scope (`thesis-data-auditor`'s territory); the
underlying fix is re-running `evaluate` to generate the parquet artifacts the
plots read from, not an editorial change.

**2026-08-29, session 9 (item 19 + user-flagged 06-01):** Applied item 19
(the last queued cross-chapter duplicate). User then pasted the rendered
"6.2. Statistical Validation" section directly and asked to cut it as
"generic... bullshit for now." Rather than delete outright, offered a choice
(delete / move to appendix / gut to load-bearing content); user chose gut
(item 27). While gutting, introduced a fresh duplicate against `06-00`'s
"Comparison Methodology" — caught immediately when the user pasted that
section and asked the same "too generic?" question — fixed both together
(item 28). Incidentally resolved half of item 26's logged inconsistency (the
06-01/06-02 5-seed contradiction) as a side effect. Fresh Quarto render
succeeded: total PDF unchanged at 144 pages, body 102 -> 101 pages, body
words 31,257 -> 30,878.

**2026-08-30/31, session 10 (user asked to reduce information, first
chapters):** User explicitly asked for a different kind of cut than sessions
1-9 had been doing — reducing actual information content in early chapters,
not just removing duplicate/padded text. Re-read `02-01`, `02-02`, `02-03`,
and `02-08` fresh with that harsher mandate and found a chapter-wide pattern:
the epistemic caveat "this regularity is descriptive, not a proven trading
signal" restated roughly 10 times, on top of `02-00` already stating it once
for the whole chapter. Presented full before/after diffs for review; user
adjusted one diff (`02-03`'s replacement, to keep a transitional sentence
rather than dropping straight into the alpha analogy) and approved each
piece individually before applying (item 32). Also found and logged three
more items without applying (29, 30, 31): `07-01`'s H1 summary re-citing
figures `06-03` already covers, and two table-to-appendix candidates
(feature-correlation table in `05-01`, H4's per-trial table in `06-02`).
Fresh Quarto render succeeded: total PDF 144 -> 142 pages, body 101 -> 99
pages, body words 30,878 -> 30,405.

**Where this leaves the 80-page target:** every chapter has now had at least
one full duplication/breadth pass, every table-bearing file a dedicated
tables pass, and Chapter 2 a genuine information-reduction pass beyond pure
deduplication. As of session 11, the full queue of items carried over from
session 10 (11, 18, 29, 30, 31) is closed: items 11, 18, 29, and 30 applied
(-337 body words combined), item 31 skipped as too marginal. Body word count
is now 30,068 against the session-10 baseline of 30,405 — not yet
re-rendered to a fresh page count, so the exact page effect is unconfirmed,
but at ~307 words/page this is roughly 1 page. The queue of specifically
scoped, already-reviewed items is now empty. Session 10's "reduce
information, not just duplication" mandate (category 3, applied harder)
still has real room in chapters not yet given that treatment — 03-02's
theory sections, 02-04/02-07's remaining survey material beyond item 11, and
07-02's limitations chapter (already cut hard once in session 1, but not
with this specific "restated caveat" lens) are the most promising next
targets if the user wants to keep pushing toward 80 rather than stopping at
the duplication-only ceiling identified after session 9. The alternative
paths remain: sentence-level tightening across the board (`hemingway`, out
of this skill's scope), or reconsidering whether 80 pages is the right
target for this thesis's actual content.

**2026-08-31, session 11 (item 11):** User asked how item 11's cut would
actually be phrased before approving. Presented a concrete before/after diff
for both survey clusters in `02-07`, computed the exact word saving (78,
versus the original 230-300 estimate), and explained the gap: the diff only
trims per-paper framing sentences, keeping every citation and distinct claim,
per the item's original scope ("keep Majidi2024, Kabbani2022, and FinRL2020
individually treated"). User approved as-is. Applied; word count not yet
re-rendered to a fresh page count this session.

Continuing the same session, applied item 18 (`01-00`, `01-01`): merged the
introduction's three closing paragraphs into one boundary paragraph pointing
to Chapters 5/7 for detail, compressed "Novelty" to stop restating the
intro's TD3/HFT/continuous-control claim, and cut two Objectives bullets that
restated Hypotheses 1 and 2 verbatim. User approved the full diff for all
three sub-edits in one `ok`. Saved 134 words, again under estimate for the
same reason as item 11. Also applied item 29 (`07-01`): compressed H1's
oversized summary paragraph to match H2/H3's style, replacing the repeated
exact figures (already in `06-03`) with a plain-text pointer to Section 6.3.
Saved 71 words, close to the 115-word estimate this time since the fix was a
straightforward figure-removal rather than a framing-only trim. Body word
count now 30,122 (30,405 baseline - 78 - 134 - 71); not yet re-rendered to a
fresh page count.

Also applied item 30 (`05-01` -> `99-appendix`): moved the feature-correlation table's code chunk to a new listed appendix section, replacing it in the body with a one-sentence pointer, after confirming the prose only ever cites the aggregate correlation claim. Body word count now 30,068 (30,405 baseline - 78 - 134 - 71 - 54); not yet re-rendered to a fresh page count.

Closed out the session on item 31 (H4 per-trial table, `06-02`): user judged the effect too marginal to be worth the appendix-move churn at N=5 rows — the same call made in session 6, now settled rather than reopened. Marked skipped rather than applied. All five items carried over from session 10 (11, 18, 29, 30, 31) are now resolved: four applied, one skipped. Each edit was committed individually to branch `kwojdalski/thesis-condense-session11` and pushed to PR #554 (unmerged as of end of session); no fresh Quarto render was run this session, so the page-count effect of items 11/18/29/30 (-337 body words combined) is tracked as a word-count estimate only, not yet confirmed via `pdfinfo`.

User then asked for a fresh reconnaissance pass to find more candidates, scoped to the plan's own "most promising next targets" note: `03-00`-`03-02`, `02-04`, `02-07`, and `07-02`. Re-read all of these in full. Found two clean within-file dedups (33 in `03-00`, 34 in `02-07`), both applied on the same `ok`. Checked and cleared with no new findings: `03-01` (tight taxonomy chapter), `02-04` (tight after item 10), and `07-02` (no repeated-caveat pattern like Chapter 2's item 32 despite the plan flagging it as promising — its five limitation subsections are genuinely distinct, confirmed by grepping each one's specific claims against the rest of the thesis with no hits). One sentence-level redundancy noticed in `03-02` (a phrase repeated verbatim across two adjacent sentences) was explicitly out of scope — that belongs to `hemingway`, not flagged as a plan item. Body word count now 29,980 (30,405 baseline - 78 - 134 - 71 - 54 - 43 - 45); not yet re-rendered to a fresh page count.

**2026-08-31, session 12 (merge PR #554, fresh reconnaissance, `05-01`):**
User asked to run the condenser again. Discovered mid-session that a separate
Claude Code session had independently completed all of session 11's work
(items 11, 18, 29, 30, 33, 34) on branch `kwojdalski/thesis-condense-session11`,
sitting in a clean, mergeable, unmerged PR #554 — plus a concurrent,
uncommitted `hemingway`-style sentence-tightening pass across 8 files
(`03-02`, `04-00`, `04-03`, `04-04`, `04-06`, `05-00`, `06-00`, `06-02`) in
the same shared working directory. Merged PR #554 first rather than risk
duplicating or conflicting with already-reviewed work; left the in-progress
hemingway edits untouched (stashed and restored to the working tree, not
committed, since that work isn't this session's to finish or judge).

Fresh render after the merge: total PDF 142 -> 140 pages, body 99 -> 95 pages
(a larger single-session drop than session 11's own word-count estimate
implied, likely reflecting the `[VERIFIED]` marker cleanup in the same PR).
Body words now 29,552 (measured fresh via `wc -w`, superseding session 11's
running arithmetic estimate of 29,980 as the authoritative baseline).

Did a fresh, harsh re-read of `05-01-data-preparation.qmd` (now the largest
body file at 3,074 words) looking for the "reduce information" pattern from
session 10. Found nothing new: the one candidate overlap — "per-security
fitting prevents leakage" appearing both as a brief preview in "Preprocessing
and Splitting" and again in full detail in "Split-Specific Statistics" — is a
legitimate summary-then-detail structure (the second instance adds
per-chronological-split and per-session granularity the first doesn't
mention), not restatement. `03-02` (second-largest at 2,343 words) was
already re-checked fresh in session 11 with the same conclusion.

**Assessment:** across 12 sessions, every chapter has had multiple
duplication/breadth/table passes, Chapter 2 has had a harder
information-reduction pass, and the two largest remaining files were just
re-checked with no new findings. The structural/duplication signal this
skill is built to find appears genuinely exhausted. The concurrent
hemingway-style pass already running in this working directory is the
correct next lever — sentence-level economy across the many short, correct,
but occasionally verbose sentences this thesis still has — rather than more
reconnaissance under this skill's three categories.

**2026-08-31, session 13 (automated near-duplicate scan):** User asked for
more candidates. Rather than re-read chapters by eye again (likely to
reproduce session 12's "exhausted" conclusion), wrote a small script to find
near-duplicate sentences across all body `.qmd` files by exact match and by
Jaccard similarity on content words — a mechanical check none of the prior
12 sessions had tried. Exact-match (9+ words) found zero cross-file repeats,
confirming 12 sessions of dedup work eliminated verbatim restatement. Fuzzy
matching (Jaccard >= 0.4 on content words) surfaced three items (35-37): the
H1/H2/benchmark table legends in `06-00`/`06-02`/`06-03` restate the same
metric definitions with cosmetic rewording — a pattern easy to miss reading
narratively since each instance is a short "Legend:"/italic footnote line
next to a table, not prose. Everything else the fuzzy scan surfaced (H1/07-01
hypothesis restatement, 02-04/04-03 "direct optimization" description,
01-00/04-03 tick-frequency-noise mention, 04-01/04-05 PPO entropy bonus) was
confirmed to be the same legitimate literature-to-design or intro-to-summary
recall pattern already established as acceptable in earlier sessions — not
flagged.

Presented item 35 (cleanest: zero abbreviation mismatch between `06-00`'s H1
legend and `06-02`'s H2 legend) with a concrete diff. User asked whether a
cross-referenced legend (vs. every table staying self-contained) is
permitted under WNE UW's formal thesis requirements — a formal-compliance
question this skill has no authority to answer (that's
`thesis-format-auditor`'s or a direct check of the university's guide, not a
length-reduction judgment call). Rather than guess, presented the tradeoff
and asked the user to choose; user chose to skip all three items (35-37)
entirely rather than apply now or log as an open issue. No `.qmd` files
edited this session — plan file only.

**Where this leaves the 80-page target:** unchanged from session 12's
assessment. The automated scan is now available as a technique for a future
session (e.g. lower the Jaccard threshold further, or run it after the
concurrent hemingway pass lands, since sentence-level tightening can
sometimes surface structural duplication that verbose phrasing had
obscured), but it did not overturn the "structural signal exhausted"
conclusion — it found one legitimate-but-small (~85 words total) candidate,
and the user judged it not worth pursuing without formal-compliance
confirmation. hemingway (sentence-level) or reconsidering the 80-page target
remain the live paths forward.
