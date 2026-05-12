---
name: auto-issue-triage
description: Automatically triage and work on highest-severity GitHub issues
type: code
---

# Auto Issue Triage Skill

Automatically identifies, prioritizes, and begins working on the most severe GitHub issues in your trading/RL codebase.

## Commands

```
/ds-triage                  — Auto-triage and work on highest-severity issues
/ds-triage --dry-run         — Preview actions without making changes
/ds-triage --max-issues <n>     — Process at most N issues (default: 5)
/ds-triage --severity <level>    — Only process issues at or above severity level
/ds-triage --skip-labels         — Skip auto-labeling (use existing labels only)
/ds-triage --confirm              — Require manual confirmation before closing issues
```

## Workflow

### 1. Issue Discovery and Prioritization
```bash
gh issue list --state open --limit 100
```

**Priority Logic:**
1. `critical` → P0 (highest priority)
2. `high` → P1 
3. `bugfinder` → P2 (auto-created triage)
4. `medium` → P3
5. `low` → P4
6. No severity label → P5 (lowest)

**Tie-breaking:**
- If multiple issues share same severity, pick newest issue (lowest issue number)
- Ignore issues >30 days old unless they're critical

### 2. Issue Investigation
```bash
gh issue view <number> --json number,title,body,labels,comments,author,createdAt,url
```

**Investigation Checklist:**
- Read issue body and all comments carefully
- Identify exact file paths, line numbers, or function names mentioned
- Any reproduction steps provided
- Any screenshots or log output embedded

### 3. Codebase Verification

**Real Issues:** Confirmed bugs with code evidence

**For Bug Reports:**
- Find exact file and line: issue refers to
- Trace execution path: input → computation → output
- Check whether described wrong behavior can actually occur given current code
- Check git log for recent changes: `git log --oneline -20 -- <file>`

**Verification Points:**
- Bug claims must match actual code logic
- Wrong line numbers indicate misunderstandings
- Feature requests require checking existing implementations

### 4. Triage Decision

**Real Issues:** Confirmed bugs or valid feature requests  
**Feature Requests:** Already implemented or design conflicts  
**Documentation Issues:** Missing docstrings or unclear behavior

**Verdict Categories:**
- `CONFIRMED` — Real bug, code evidence supports claim, reproducible
- `ALREADY FIXED` — Bug fixed in recent commit
- `NOT REPRODUCIBLE` — Code doesn't contain described behavior
- `INVALID` — Wrong claim, misunderstanding of design
- `DUPLICATE` — Issue already exists
- `NEEDS MORE INFO` — Cannot determine without reproduction steps

### 5. Work Assignment

**Automatic Assignment:**
- Pick top N issues (default: 5)
- Add to current work tasks
- Create dedicated branch per issue if needed
- Link related issues (dependencies, blockers)

### 6. Automatic Updates

**Progress Tracking:** Update issue labels and status
- `CONFIRMED` → Add label: `triaged-confirmed`
- `IN_PROGRESS` → Add label: `in-progress`, assign to current sprint
- `FIXED` → Remove `in-progress`, add `triaged-fixed`
- `BLOCKED` → Add `blocked-by-design`, `needs-investigation`

## Usage Examples

```bash
# Standard triage: Top 5 issues
python auto-issue-triage

# Preview actions first
python auto-issue-triage --dry-run

# Custom limit and severity filtering
python auto-issue-triage --max-issues 10 --severity critical

# Require manual confirmation
python auto-issue-triage --confirm --max-issues 3
```

## Workflow Output

For each issue, the skill generates:

```
## Issue #<number>: <title>

**Status:** CONFIRMED

**Evidence:** [ ] Function exists and behaves as described

**Analysis:** This appears to be a real bug.

**Next Steps:**
1. gh issue edit <number> --add-label "in-progress"
2. git checkout -b fix/issue-<number>
3. Implement fix and test
4. git push origin fix/issue-<number>
```

## Important

- Always verify bug claims against actual code behavior
- Provide file paths and line numbers when claiming bugs
- Never accept issue author's assertions without evidence
- Test all fixes with existing test suite before closing issues
- Respect severity hierarchy when prioritizing
- Use labels like `bugfinder`, `triaged-confirmed`, `in-progress` for tracking progress
