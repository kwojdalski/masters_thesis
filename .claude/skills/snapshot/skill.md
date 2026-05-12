---
name: snapshot
description: Create a dated stable-version git tag (stable/YYYY-MM-DD), push it, create a GitHub Release with an AI-generated changelog from the last day's commits, and summarise changes. Use when the user wants to mark the current state as a stable checkpoint or review daily progress.
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

Push the tag to GitHub:
```bash
git push origin stable/YYYY-MM-DD
```

### 4. Find the previous snapshot tag from GitHub Releases

Query GitHub Releases to find the most recently published daily build (at this point today's release has not been created yet, so the first result is yesterday's):

```bash
gh release list --limit 50 --json tagName --jq '.[].tagName' | grep "^stable/" | head -1
```

If no previous `stable/*` release exists on GitHub, fall back to the first commit:
```bash
git rev-list --max-parents=0 HEAD
```

Store the result as PREV.

### 5. Analyse changes since PREV

Collect the raw material:

**Commit list with full messages** (for analysis):
```bash
git log PREV..HEAD --format="%h %s%n%b"
```

**Files changed** (stat summary):
```bash
git diff --stat PREV..HEAD
```

**Commit count:**
```bash
git log PREV..HEAD --oneline | wc -l
```

Now write the release notes. Structure them as follows:

- **Overview** (2-3 sentences): what was the main thrust of today's work — new features, fixes, experiments, refactoring, or thesis writing? Be specific about which subsystems moved.
- **Changes by area**: group commits by module/directory. For each group, write 1-2 sentences describing what changed and why it matters. Focus on: `src/trading_rl/` (RL core), `src/configs/`, `src/cli/`, `tests/`, `thesis/`, `scripts/`, and config/dependency files (`pyproject.toml`, `uv.lock`). Skip areas with no changes.
- **Files changed**: include the `git diff --stat` summary line (e.g. "12 files changed, 340 insertions(+), 45 deletions(-)").

Keep the notes factual, concise, and under 40 lines total. No preamble, no sign-off.

### 6. Create the GitHub Release

Compose the release body from the notes above, then run:

```bash
gh release create stable/YYYY-MM-DD \
  --title "Daily Build YYYY-MM-DD" \
  --notes "RELEASE_BODY" \
  --target master
```

If the release already exists (exit code non-zero with "already exists" message), skip creation and note it.

Report the release URL returned by `gh release create`.

### 7. Output format

Print to the user:

```
Snapshot: stable/YYYY-MM-DD  (SHA: <sha>)
Previous: stable/PREV-DATE   (or "first commit")
Release:  <GitHub release URL>

## Commits since last snapshot (<N> total)
<git log --oneline output>

## Files changed
<git diff --stat output>

## Release notes
<the notes written in step 5>
```
