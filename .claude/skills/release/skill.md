---
name: release
description: Release a previously claimed GitHub issue (remove in-progress status). Use when done working on an issue, or when abandoning work.
---

# Issue Release

Release a GitHub issue that was previously claimed, removing the `in-progress` status.

## Usage

```
/release <issue_number> [--status <status>]
```

## Options

- `--status <status>` — Optional final status to report:
  - `resolved` — Issue was fixed (default)
  - `blocked` — Issue is blocked by something else
  - `invalid` — Issue turned out to be invalid
  - `needs-info` — Waiting for more information

## What It Does

1. **Remove `in-progress` label**
   ```bash
   gh issue edit <number> --remove-label "in-progress"
   ```

2. **Add closing comment with completion info**
   ```markdown
   Released by Claude Code session
   Hostname: <hostname>
   Completed at: <ISO timestamp>
   Status: <status>
   Session ID: <session_id>  # matches original claim
   ```

3. **Close the issue** when status is `resolved` or `invalid`
   ```bash
   gh issue close <number> --comment "Resolved by session..."
   ```
   This step is **mandatory** for `resolved` and `invalid` — do not skip it. For `blocked` or `needs-info`, leave the issue open.

## Example

```
> /release 87
Releasing issue #87...
✓ Removed in-progress label
✓ Added release comment
✓ Closed issue (status: resolved)

Issue #87 is now closed.
```

## Session Matching

The release comment includes the same session ID as the original claim comment. This creates a traceable audit trail:

```
# Claim comment
Claimed by Claude Code session
Session ID: a3f7b2c1

# Release comment  
Released by Claude Code session
Status: resolved
Session ID: a3f7b2c1
```

## Labels Modified

- Removes: `in-progress`
- Adds: `in-progress` is removed; no new labels added by default

## Integration

Other skills should call this when done:

```bash
# In auto-issue-triage or issue-triage
/release <issue_number> --status resolved
```

## Important

- Always include the same session ID as the claim for traceability
- If you can't find the original claim comment, generate a new session ID and note "session ID unknown"
- Do not release an issue without removing the `in-progress` label
- Only close issues that are actually resolved or invalid; use `blocked` or `needs-info` for others