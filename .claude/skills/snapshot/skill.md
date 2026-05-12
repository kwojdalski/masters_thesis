---
name: snapshot
description: Create a dated stable-version git tag (stable/YYYY-MM-DD) and summarise what changed since the previous snapshot. Use when the user wants to mark the current state as a stable checkpoint or review daily progress.
---

# Snapshot

You are a release engineer creating a daily stable checkpoint for this Python trading / RL codebase.

## Steps — execute in order

### 1. Determine today's date tag

Today is available in the system context as `currentDate`. Construct the tag name:

```
stable/YYYY-MM-DD
```

### 2. Check whether the tag already exists

Run:
```bash
git tag --list "stable/YYYY-MM-DD"
```

If the tag already exists, skip creation and tell the user. Proceed to step 4 (changelog) anyway.

### 3. Create the tag

Check if there are uncommitted changes first:
```bash
git status --short
```

If uncommitted changes exist, warn the user and ask whether to proceed. If they confirm (or there are no uncommitted changes), create an annotated tag:
```bash
git tag -a stable/YYYY-MM-DD -m "Daily stable snapshot YYYY-MM-DD"
```

Report the tag SHA: `git rev-parse stable/YYYY-MM-DD`

### 4. Find the previous snapshot tag

```bash
git tag --list "stable/*" --sort=-version:refname | sed -n '2p'
```

If no previous tag exists, use the first commit: `git rev-list --max-parents=0 HEAD`

### 5. Summarise changes since the previous snapshot

Run the following three commands and present results:

**Commit list** (one line each):
```bash
git log <prev-tag>..HEAD --oneline
```

**Files changed** (stat summary):
```bash
git diff --stat <prev-tag>..HEAD
```

**High-level grouped summary** — group changed files by directory/module and describe in plain English what areas of the codebase moved. Keep it under 15 lines. Focus on: new files added, deleted files, modules with heavy churn, and any config or dependency changes (`pyproject.toml`, `uv.lock`, `CLAUDE.md`).

### 6. Output format

Present the output as:

```
Snapshot: stable/YYYY-MM-DD  (SHA: <sha>)
Previous: stable/PREV-DATE   (or "first commit")

## Commits since last snapshot (<N> total)
<git log output>

## Files changed
<git diff --stat output>

## Summary
<plain-English paragraph describing what changed and in which modules>
```

Keep the summary factual and concise. No preamble, no sign-off.
