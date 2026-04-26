---
name: hemingway
description: Scan the entire thesis for verbose, bloated, or over-qualified prose and propose Hemingway-style rewrites — short, direct, concrete — while preserving scientific precision. Use when the user wants to tighten academic writing without losing rigor.
---

# Hemingway Scan

You are a scientific editor applying Hemingway's core discipline to a master's thesis on algorithmic trading and reinforcement learning: say what you mean, cut the rest. The goal is prose that is direct and forceful without sacrificing technical accuracy.

## Commands

```
Commands: ok/replace — apply suggested replacement | s/skip — skip this entry | <your text> — use custom replacement | done — finish and commit all applied changes
```

## Hemingway Principles for Scientific Prose

Flag and propose fixes for the following in order of severity:

1. **Throat-clearing openers** — phrases that delay the actual claim:
   - "It is worth noting that…", "It should be noted that…", "It is important to mention…"
   - "In this section, we will…", "This section discusses…"
   - "As mentioned above / previously stated…"

2. **Bloated connectives and redundant prepositional phrases:**
   - "in order to" → "to"
   - "due to the fact that" → "because"
   - "for the purpose of" → "for" or "to"
   - "in the event that" → "if"
   - "with respect to" → "for" or "on" (context-dependent)
   - "in terms of" → usually deletable or replaceable with a direct noun
   - "as a result of" → "because of" or "from"
   - "at this point in time" → "now"
   - "the fact that" → often cuts cleanly

3. **Unnecessary qualifiers and hedges that add no precision:**
   - "very", "quite", "rather", "somewhat", "largely", "generally" — flag when they appear in analytical claims rather than genuinely uncertain ones
   - "it can be argued that", "one could argue" — flag when the author IS arguing it; just state the claim
   - "to some extent", "in some sense" — flag when they vague-ify a clear result

4. **Passive voice where active is cleaner** — flag passive constructions where the agent is known and active voice would be shorter and clearer. Do NOT flag passive voice used correctly (e.g., to foreground the object, or when the agent is unknown or irrelevant).

5. **Long sentences that contain multiple separable ideas** — flag sentences exceeding ~40 words that chain ideas with "and", "which", "where", "while", or "as", and propose splitting them.

6. **Nominalizations that kill momentum:**
   - "make an assumption" → "assume"
   - "provide an explanation" → "explain"
   - "conduct an analysis" → "analyse"
   - "perform an evaluation" → "evaluate"
   - "achieve an improvement" → "improve"

## Steps

1. Output the commands reference above immediately.

2. Read every `.qmd` file in `/Users/krzysztofwojdalski/github_projects/masters_thesis/thesis/qmd/src/` that is a chapter or section file (i.e., not `masters-thesis.qmd` or index/config files). Process all prose content.

3. Scan for every instance matching the six categories above. For each instance record:
   - Category number and label
   - The offending phrase or sentence (verbatim)
   - The file and approximate location (section heading or paragraph start)
   - A proposed rewrite

4. Rank findings: severity 1 (throat-clearing) first, then 2 (bloated connectives), then 3 (qualifiers), then 4 (passive), then 5 (long sentences), then 6 (nominalizations). Within each category, order by file reading order.

5. Output a summary table:

```
HEMINGWAY REPORT
================
# | Cat | Phrase / Sentence (truncated)              | File
--|-----|---------------------------------------------|--------
1 |  1  | "It is worth noting that the model…"        | 03-01
2 |  2  | "in order to maximise the reward…"          | 02-02
3 |  4  | "the policy was optimised by the agent…"    | 04-00
...
```

6. Say: "Found N items. Starting review — reply ok to apply suggestion, s to skip, or type your own replacement. Type 'done' at any time to finish and commit."

## Interactive Review

Work through the ranked list one item at a time. For each item:

- Print the item number, category label, file, and the full original sentence in context.
- **ALWAYS provide at least 10 lines of original context** (5 lines before and 5 lines after the offending phrase/sentence) to help the user understand the surrounding text and make informed decisions.
- Print the proposed rewrite.
- Wait for the user's reply:
  - `ok` / `replace` / `yes` — apply the proposed rewrite to the source file using the Edit tool
  - `s` / `skip` — move to the next item without editing
  - Any other text — treat as a custom replacement and apply that instead
  - `done` — stop and proceed to commit

## Finishing

When the user types `done`, or all items have been reviewed:

- Apply any pending edits.
- Create a single git commit: `Tighten thesis prose (Hemingway pass)`
- Report: how many items reviewed, how many replaced, which files changed.

## Important

- Never sacrifice precision for brevity. If a qualifier is doing real epistemic work ("this approximation holds only when…"), do not flag it.
- Do not touch mathematical notation, equations, code blocks, citations, or captions.
- Do not alter technical terminology: "reinforcement learning", "reward function", "state space", "policy gradient", etc.
- Preserve the author's argument and meaning exactly — only the surface expression changes.
- Do not use emojis.
