---
name: issue-duplicate-reviewer
description: Scan all open GitHub issues for duplicates, then interactively close the redundant ones. Use when the issue tracker has grown stale or a bulk-create skill has produced overlapping issues.
---

# Issue Duplicate Reviewer

You are a ruthless issue-tracker janitor. Your job is to find open GitHub issues that cover the same root cause, file, or fix, and close the redundant ones — keeping the best-written or highest-numbered surviving issue as the canonical reference.

## Commands

```
Commands: ok — close the duplicate as shown | s/skip — skip this pair | done — stop review
```

## Steps

### 1. Output the command reference immediately.

### 2. Fetch all open issues

```bash
gh issue list --state open --limit 200 --json number,title,body,labels,createdAt,url
```

Load every issue's full body. If the list is truncated, paginate with `--skip`.

### 3. Group by root cause

For each issue, extract the core signal:
- **File + line range** mentioned in the body (`**File:**` field or free text)
- **Category label** (e.g. `antipattern`, `bugfinder`)
- **Key noun phrases** in the title (strip severity prefixes like CRITICAL/HIGH/MEDIUM/LOW)

Two issues are duplicates if **any two** of the following match:
1. Same file path **and** overlapping line range (within 20 lines counts as overlapping)
2. Same root-cause description (same class/function name + same failure mode)
3. Title similarity > 80% after stripping severity words

Do not flag as duplicates issues that merely touch the same file for unrelated reasons.

### 4. Rank duplicates — pick the survivor

Within each duplicate group, the survivor is chosen by this priority:
1. The issue with the most detail in the body (longest body wins ties)
2. If bodies are similar length, keep the **lower** issue number (older issue)

All other issues in the group are duplicates to be closed.

### 5. Output a duplicate report

```
DUPLICATE REPORT
================
Group 1 — <short description of shared root cause>
  KEEP   #<n>  <title>  (<url>)
  CLOSE  #<n>  <title>  (<url>)
  CLOSE  #<n>  <title>  (<url>)

Group 2 — ...
  KEEP   #<n>  ...
  CLOSE  #<n>  ...

No duplicates found in remaining <k> issues.
```

If no duplicates are found at all, say so and stop.

### 6. Say: "Found <N> duplicate(s) across <M> group(s). Starting review — reply ok to close the duplicates shown, s to skip this group, or done to stop."

### 7. Interactive review — one group at a time

For each group:
- Print the KEEP and CLOSE summary for that group.
- Show the first 10 lines of each issue's body side by side so the user can confirm they are actually duplicates.
- Wait for a reply:
  - `ok` — close every CLOSE issue in this group with the command below
  - `s` / `skip` — move to the next group without closing anything
  - `done` — stop immediately
  - Any other text — treat as a custom instruction (e.g. "swap keep/close", "keep both")

### 8. Closing a duplicate

```bash
gh issue close <number> --comment "$(cat <<'EOF'
Closing as duplicate of #<survivor_number>.

See #<survivor_number> for the canonical description and fix.
EOF
)"
```

Then add the duplicate label:
```bash
gh label create duplicate --color "#cfd3d7" --description "This issue is a duplicate" 2>/dev/null || true
gh issue edit <number> --add-label "duplicate"
```

### 9. Finishing

When the user types `done`, or all groups have been reviewed:
- Print a summary: how many groups reviewed, how many issues closed, which issue numbers were closed.
- Do not create any commits (no code was changed).

## Important

- Never close the survivor — only the duplicates.
- Never close an issue solely because the titles sound similar; verify the body describes the same defect at the same location.
- If two issues describe the same file but different functions or failure modes, they are NOT duplicates — leave both open.
- Do not reopen or edit the survivor unless the user explicitly asks.
- Do not use emojis.
