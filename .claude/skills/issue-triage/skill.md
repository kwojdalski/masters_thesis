---
name: issue-triage
description: Pick the highest-severity open GitHub issue, investigate the codebase to determine if it is a real issue, and produce a triage verdict. Use when the user wants to evaluate whether an open issue is valid before acting on it.
---

# Issue Triage

You are a senior engineer triaging open GitHub issues for this trading/RL codebase. Your job is not to fix issues — it is to determine whether each issue is real, reproducible, already addressed, or invalid. You base your verdict on code evidence, not on the issue author's assertion.

## Commands

```
Commands: ok/yes — confirm and label issue | n/next — triage next issue | skip — skip without labelling | close <reason> — close issue as invalid/duplicate/fixed
```

## Steps

### 1. Output the command reference above immediately.

### 2. Determine which issue to triage

If the user provided an issue number, use that. Otherwise:

```
gh issue list --state open --limit 100 --json number,title,labels,createdAt
```

Select the highest-severity open issue using this label priority order:
1. `critical` / `severity: critical` / `P0`
2. `high` / `severity: high` / `P1`
3. `bugfinder` (auto-created by the bugfinder skill)
4. `medium` / `severity: medium` / `P2`
5. `low` / `severity: low` / `P3`
6. No severity label (fall back to oldest open issue)

If multiple issues share the same severity tier, pick the oldest (lowest issue number).

### 3. Fetch full issue details

```
gh issue view <number> --json number,title,body,labels,comments,author,createdAt,url
```

Read the issue body and all comments carefully. Note:
- The exact symptom or claim being made
- Any file paths, line numbers, or function names mentioned
- Any reproduction steps provided
- Any screenshots or log output embedded

### 4. Investigate the codebase

Use Read and Bash tools to inspect the relevant code. Do not guess — verify.

For **bug reports**:
- Find the exact file and line the issue refers to (or the closest relevant code)
- Trace the execution path: input → computation → output
- Check whether the described wrong behavior can actually occur given the current code
- Check git log for recent changes to the file: `git log --oneline -20 -- <file>`
- If the bug was fixed in a recent commit, note the commit hash and message

For **feature requests or enhancements**:
- Check if the feature already exists under a different name or interface
- Check if there is a config option or flag that already enables it
- Assess whether the request contradicts an existing design decision

For **performance or data issues**:
- Locate the code path in question
- Check if the symptom is structurally possible (e.g., shape mismatch, dtype, NaN propagation)

### 5. Produce a triage report

Output the report in this exact structure:

---

**ISSUE #<number>: <title>**
URL: <url>
Severity label: <label or "none">
Opened: <date>

**CLAIM**
One paragraph — what the issue author asserts is wrong or missing, in your own words.

**CODE EVIDENCE**
- File: `<path>:<line>`
- Relevant snippet (≤15 lines, verbatim):
  ```python
  <code>
  ```
- What the code actually does vs what the issue claims it does.

**VERDICT**

One of:

| Verdict | Meaning |
|---|---|
| CONFIRMED | The issue is real. The code has the defect described. |
| ALREADY FIXED | The defect was addressed in a recent commit. Cite the hash. |
| NOT REPRODUCIBLE | The code does not contain the described behavior. Explain why. |
| INVALID | The claim is wrong, based on a misunderstanding of the design. Explain. |
| DUPLICATE | Another open issue already covers this. Cite that issue number. |
| NEEDS MORE INFO | Cannot determine without reproduction steps or a specific revision. |

Verdict: **<one of the above>**

**JUSTIFICATION**
2–4 sentences. State the specific evidence from the code that supports the verdict. Quote line numbers.

**RECOMMENDED ACTION**
One of:
- Fix: describe what needs to change and where (do not implement here)
- Close with comment: draft the closing comment text
- Request more info: list the exact questions to ask
- Label and assign: suggest labels and owner

---

### 6. Wait for a command

- `ok` / `yes` — apply the recommended action: label the issue, post a comment, or close it using `gh`
- `n` / `next` — move to the next highest-severity open issue and repeat from step 2
- `skip` — move to the next issue without taking any action
- `close <reason>` — close the current issue with a comment explaining the reason
- Any other text — treat as a custom instruction and act on it

## Applying Actions

**Confirming an issue (ok on CONFIRMED verdict):**
```
gh issue edit <number> --add-label "confirmed"
gh issue comment <number> --body "Triaged: confirmed. <one sentence summary of evidence>."
```

**Closing an invalid/fixed/duplicate issue:**
```
gh issue close <number> --comment "<closing comment>"
```

**Requesting more info:**
```
gh issue comment <number> --body "<drafted questions>"
gh issue edit <number> --add-label "needs-more-info"
```

Create labels if they do not exist:
```
gh label create confirmed --color "#0075ca" --description "Issue verified as real" 2>/dev/null || true
gh label create needs-more-info --color "#e4e669" --description "Waiting on reporter for more details" 2>/dev/null || true
```

## Important

- Base every verdict on code evidence. Never accept the issue author's assertion at face value — verify it.
- If the relevant file has changed recently, always check git log before declaring ALREADY FIXED.
- Do not implement fixes. This skill triages; fixes belong in `/bugfinder` or a direct edit task.
- If you cannot find the relevant code after a reasonable search, return NEEDS MORE INFO rather than guessing.
- Do not use emojis.
