---
name: d2-workflow
description: Update D2 diagram files in the project
type: code
---

# D2 Workflow Skill

Updates D2 diagram files (`.d2`) in the project. Finds the relevant diagram, edits it according to the user's instructions, and re-renders it if a D2 renderer is available.

## Steps

1. Locate the relevant `.d2` file(s) in the project (search under `docs/`, `assets/`, or project root).
2. Read the current diagram content.
3. Read the actual source code the diagram is meant to represent — trace the real execution flow: entry points, key function calls, data transformations, control flow branches.
4. Compare the diagram against the code: identify nodes or edges that are missing, stale, mislabelled, or in the wrong order relative to what the code actually does.
5. Apply all necessary updates to the `.d2` file so the diagram accurately reflects the current code workflow.
6. If `d2` CLI is available (`which d2`), render the diagram to SVG/PNG: `d2 <file.d2> <file.svg>`.
7. Report a concise diff of what changed in the diagram and why (which code change drove each update).