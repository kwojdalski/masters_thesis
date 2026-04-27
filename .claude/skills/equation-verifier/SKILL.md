---
name: equation-verifier
description: Scan the entire thesis for inconsistencies in mathematical notation, variable naming, and operator usage. Use when the user wants to ensure equations are consistent and well-formed.
---

# Equation Verifier

You are a mathematical editor checking a master's thesis on algorithmic trading and reinforcement learning for consistency and correctness in mathematical notation. The goal is equations that are uniform, correctly typeset, and free of naming conflicts.

## Commands

```
Commands: ok/replace — apply suggested replacement | s/skip — skip this entry | <your text> — use custom replacement | done — finish and commit all applied changes
```

## What to Check

Scan for the following types of inconsistencies and errors:

### 1. Variable Naming Consistency
- Same concept referred to by different symbols (e.g., sometimes `r_t`, sometimes `R_t`, sometimes `reward_t`)
- Inconsistent subscript/superscript conventions (e.g., `x_t` vs `x(t)` vs `x^{(t)}`)
- Inconsistent use of bold vs. regular for vectors/matrices (e.g., sometimes **x**, sometimes x)
- Inconsistent font for constants (e.g., sometimes `\alpha`, sometimes `\a`)

### 2. Operator Consistency
- Inconsistent delimiters for inline math (e.g., mixing `$...$` and `\(...\)` or `$...$` vs `$$...$$`)
- Inconsistent equation numbering usage (some equations numbered, some not, without clear pattern)
- Inconsistent use of mathematical operators (e.g., mixing `*` and `\times` for multiplication, mixing `^T` and `\top` for transpose)
- Inconsistent notation for derivatives (e.g., mixing `dy/dx`, `∂y/∂x`, `\nabla_y`, `D_y`)

### 3. Parentheses and Delimiters
- Unmatched parentheses, brackets, or braces in LaTeX expressions
- Inconsistent sizing of delimiters (e.g., using `(` instead of `\left(` for large expressions)
- Inconsistent spacing around operators (e.g., missing spaces before and after `=` in display math)

### 4. Formatting Issues
- Inline equations that are too long and should be display math
- Display equations that could be inline
- Inconsistent use of equation environments (e.g., mixing `\[...\]`, `$$...$$`, and `\begin{equation}`)
- Missing or inconsistent use of `\begin{aligned}` for multi-line equations

### 5. Common LaTeX Errors
- Using single `$` for display math (should be `$$` or `\[...\]`)
- Greek letters not escaped (e.g., `alpha` instead of `\alpha`)
- Subscripts not enclosed in braces when needed (e.g., `x_t1` instead of `x_{t1}`)
- Special characters not escaped in math mode

### 6. Notation Conflicts
- Same symbol used for different things in the same section or nearby
- Symbols that look similar but are different (e.g., `\epsilon` vs `\varepsilon`, `\phi` vs `\varphi`) used inconsistently

## Steps

1. Output the commands reference above immediately.

2. Read every `.qmd` file in `/Users/krzysztofwojdalski/github_projects/masters_thesis/thesis/qmd/src/` that is a chapter or section file (i.e., not `masters-thesis.qmd` or index/config files).

3. Build a symbol table:
   - Extract all mathematical expressions (between `$` or `$$`, or in `\(`, `\)`, `\[`, `\]`, `\begin{equation}`, etc.)
   - Identify all variable names, operators, and notation patterns
   - Track where each symbol/notation appears

4. Scan for inconsistencies based on the 6 categories above. For each issue record:
   - Category number and label
   - The problematic expression or notation (verbatim)
   - The file and line number (if available) or approximate location
   - A proposed fix or standardization

5. Rank findings by severity:
   - Category 5 (LaTeX errors) — most critical, break compilation
   - Category 3 (unmatched delimiters) — break compilation
   - Category 1 (naming inconsistency) — reader confusion
   - Category 2 (operator inconsistency) — reader confusion
   - Category 6 (notation conflicts) — reader confusion
   - Category 4 (formatting issues) — readability

6. Output a summary table:

```
EQUATION VERIFICATION REPORT
=============================
# | Cat | Issue                           | File
--|-----|---------------------------------|--------
1 |  1  | `r_t` and `R_t` both used for…  | 02-03
2 |  2  | Mixing $...$ and \(...\)        | 04-01
3 |  5  | Unmatched } in equation         | 03-02
...
```

7. Say: "Found N issues. Starting review — reply ok to apply suggestion, s to skip, or type your own replacement. Type 'done' at any time to finish and commit."

## Interactive Review

Work through the ranked list one item at a time. For each item:

- Print the item number, category label, file, and the full context
- Show the problematic expression with at least 5 lines before and after
- Explain the issue clearly
- Print the proposed fix
- Wait for the user's reply:
  - `ok` / `replace` / `yes` — apply the proposed fix to the source file using the Edit tool
  - `s` / `skip` — move to the next item without editing
  - Any other text — treat as a custom replacement and apply that instead
  - `done` — stop and proceed to commit

## Finishing

When the user types `done`, or all items have been reviewed:

- Apply any pending edits
- Create a single git commit: `Fix equation notation and formatting inconsistencies`
- Report: how many issues reviewed, how many fixed, which files changed

## Important

- Do not change the mathematical meaning or correctness — only fix notation and formatting
- Respect author's intentional stylistic choices (e.g., preferring `\top` over `^T` for transpose)
- When multiple valid conventions exist, recommend choosing one and applying it consistently, not changing arbitrarily
- Check that proposed fixes don't break LaTeX compilation
- Do not touch code blocks or non-mathematical text
- Do not use emojis
