---
name: unusual-wording
description: Scan the thesis for unusual, rare, or inconsistent word choices — vocabulary that doesn't fit the academic finance/RL register, terms that drift across chapters for the same concept, or words rare enough they may be typos or markdown-stripping artifacts. Use when the user wants a lexical consistency pass, not sentence-level bloat (hemingway), math notation (equation-verifier), or citation accuracy (literature-verifier).
---

# Unusual Wording Scan

You are a copy editor checking a master's thesis on algorithmic trading and reinforcement learning for words that break the academic-technical register: needlessly obscure vocabulary, likely wrong-word slips, inconsistent naming for the same concept, and the occasional artifact where two words got concatenated by a markdown-stripping bug rather than actually mistyped by the author.

This is a different job from three sibling skills:
- **Sentence-level bloat** (throat-clearing, hedges, passive voice, redundant phrasing) → `hemingway`, not here.
- **Mathematical notation** (symbol consistency, LaTeX errors) → `equation-verifier`, not here.
- **Citation accuracy** (does the source say what's claimed) → `literature-verifier`, not here.
- Here: **is this the right word** — rare enough to be worth a second look, and either wrong, inconsistent, or tonally out of place.

## Why this needs a script first

"Unusual" is a frequency question, and academic prose has a long tail: a raw scan of this thesis found roughly 1000+ words that appear only once — the overwhelming majority (`accurate`, `advanced`, `beginning`, `handle`, `purpose`, …) are completely ordinary. Reading all of those individually would be slower and noisier than reading the thesis itself. The fix is a two-stage funnel:

1. **Script** (`scripts/find_rare_words.py`) strips markup/math/citations, counts word frequency across the whole body, and filters out closed-class function words, a bundled ~50k-word common-English list (`references/common_words_en50k.txt`), and terms already in the glossary. This still leaves ~300-400 candidates — not because they're all findings, but because a fixed word list isn't lemma-aware (`achievable`, `accumulates` slip through even though `achieve`/`accumulate` are ordinary) and academic vocabulary skews rarer than casual English.
2. **Your own triage pass** over that candidate list is what actually does the work — reading each candidate word with its one-line context and discarding the ones that are unremarkable (which is most of them) using your own knowledge of English and of finance/RL/ML jargon. Only the survivors become findings that go to the user.

Do not show the raw script output to the user. It is scaffolding for your own read-through, not a deliverable.

## Commands

```
Commands: ok/replace — apply suggested replacement | s/skip — skip this entry | <your text> — use custom replacement | done — finish and commit all applied changes
```

## Categories, in order of severity

### 1. Likely wrong word
The word doesn't mean what the sentence needs — a malapropism, an autocomplete-shaped slip (e.g. a word that's one edit away from the evidently intended one), or a term whose actual definition contradicts the surrounding claim. This is the highest-severity category because it's a correctness risk, not just a style one.

### 2. Inconsistent terminology
The same concept is named differently in different places without a stated reason — e.g. one chapter calls it "position churn" and another calls the same thing "turnover," or a metric gets a new name in Chapter 6 that Chapter 4 never introduced. Cross-check candidates against `99-glossary.qmd` and against how the concept is named elsewhere in the document (a quick `grep` for the concept's other names) before flagging — some variation is legitimate (e.g., a general term in theory chapters vs. a specific implementation name in later chapters).

### 3. Register mismatch
A word that's correct and even fairly common, but tonally wrong for this document's plain, direct, technical voice — needlessly literary, archaic, or ornate where a simpler word would say the same thing with the same precision (e.g. "notwithstanding," "myriad," "auspicious," "eschew" used decoratively rather than because no plainer word fits).

### 4. Awkward coinage or compound
A hyphenated compound or improvised term that reads oddly even by this thesis's own habit of forming compound modifiers (`order-book-derived`, `microstructure-aware` are fine and frequent; something like `retail-identification-like` straining past that pattern is not). The script's hyphenated-candidate list is lower-priority specifically because most hyphenated compounds here are legitimate — only flag ones that a careful author would visibly wince at.

### 5. Possible processing artifact, not a word choice at all
Occasionally a candidate is not a real word: two words got concatenated because a stripped markdown/LaTeX element ate the space between them (e.g. a citation or cross-reference marker glued to the following word), or a raw identifier leaked into prose. These aren't style findings — they're rendering/content bugs, and worth flagging distinctly since the fix is different (repair the source markup, not choose a better word). Recognizable because the "word" won't parse as English at all under any reading.

## Steps

1. Output the commands reference above immediately.

2. Run the candidate generator:
   ```
   python3 .claude/skills/unusual-wording/scripts/find_rare_words.py
   ```
   Optional flags: `--max-freq N` (default 2) to loosen/tighten the frequency cutoff, `--min-len N` (default 5) to change the minimum word length, `--glob PATTERN` to scope to specific files (e.g. a single chapter's files) instead of the full body.

3. Read through the full candidate output (both the non-hyphenated and hyphenated sections). For each candidate, using the one-line context the script prints:
   - Silently discard anything that's ordinary academic English or legitimate finance/RL/ML/stats jargon (the large majority — do not report these, do not explain why for each one).
   - Keep anything matching categories 1-5 above. Record: category, the word, the file, and enough surrounding context (re-read the source file around that line if the script's one-line snippet isn't enough to judge fairly).
   - While reading, also watch for concept names that recur across 3+ files with slightly different wording each time — that pattern is category 2 even if no single instance looked unusual in isolation.

4. Rank findings: category 1 first, then 2, then 3, then 4, then 5 (artifacts are worth fixing but aren't urgent the way a wrong word or inconsistency is). Within a category, order by file reading order.

5. Output a summary table:

```
UNUSUAL WORDING REPORT
=======================
Candidates generated: N  |  Findings after triage: M

 # | Cat | Word / Phrase        | File            | Note (truncated)
---|-----|----------------------|-----------------|--------------------------------
 1 |  1  | "eschews"            | 04-03           | means avoids-on-principle; context wants "omits"
 2 |  2  | "position churn"     | 06-01           | rest of doc calls this "turnover"
 3 |  5  | "nqtvitchspecification" | 05-01        | concatenation artifact, not a word
...
```

6. Say: "Found N candidates, M survived triage as findings. Starting review — reply ok to apply suggestion, s to skip, or type your own replacement. Type 'done' at any time to finish and commit."

## Interactive Review

Work through the ranked findings one at a time. For each:

- Print the finding number, category, file, and at least 5 lines of surrounding context (not just the one-line snippet from the script).
- Explain briefly why it was flagged.
- Print a proposed fix: a specific replacement word/phrase (categories 1-4) or a description of the markup repair needed (category 5).
- Wait for the user's reply:
  - `ok` / `replace` / `yes` — apply the fix to the source file using the Edit tool
  - `s` / `skip` — move to the next item without editing
  - Any other text — treat as a custom replacement and apply that instead
  - `done` — stop and proceed to commit

## Finishing

When the user types `done`, or all findings have been reviewed:

- Apply any pending edits.
- Create a single git commit: `Fix unusual/inconsistent wording in thesis prose`
- Report: how many candidates were generated, how many survived triage, how many were fixed vs. skipped, which files changed.

## Important

- The glossary (`99-glossary.qmd`) is a floor, not a ceiling — plenty of legitimate domain jargon isn't listed there (the glossary itself says it's selective). Don't flag a word just because it's missing from the glossary; use your own judgment of whether it's real finance/RL/ML/stats terminology.
- Do not touch mathematical notation, equations, code blocks, citation keys, or captions.
- Do not flag a word solely because it's long or Latinate — technical writing legitimately needs precise multisyllabic terms (`heteroskedasticity`, `nonstationarity`). Flag tone, not length.
- Preserve the author's meaning exactly. A category-1 fix must restore the intended meaning, not just swap in *a* plausible word.
- When proposing a category-2 (inconsistent terminology) fix, pick the more established or more frequently used term as the standard, and note that the other file(s) may need the same fix if the same drift appears more than once.
- Do not use emojis.
