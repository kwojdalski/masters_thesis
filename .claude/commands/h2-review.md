Review **Hypothesis 2** (empirical performance is materially sensitive to the transaction-cost assumption: the learned edge is positive under zero fee but changes sign once the per-trade cost reaches a fraction of a basis point) by reading only the relevant thesis sections and result files.

## Step 1 — Read the thesis prose for H2

Read the H2 section inside:
- `thesis/qmd/src/01-01-scope-and-objectives.qmd` (the formal H2 statement — item 2 of the hypothesis list)
- `thesis/qmd/src/04-03-reward-function.qmd` (the "Execution Model and the Signal–Friction Decomposition" subsection — the half-spread ceiling)
- `thesis/qmd/src/06-02-robustness-assessment.qmd` — read only the `## Hypothesis 2: Transaction-Cost Sensitivity` section (stop at `## Hypothesis 3`)
- `thesis/qmd/src/07-01-summary-of-findings.qmd` — read only the Hypothesis 2 paragraph (stop at Hypothesis 3)

## Step 2 — Read the result data for the fee ladder

The H2 comparison sweeps proportional fee levels on the shared baseline. For each, read its thesis snapshot evaluation report:

```bash
# 0 bp — the fee ladder's own baseline
cat thesis/qmd/results/pooled_td3_h2_fees_0_dsr/latest_finished/evaluation_report.json
# 0.01 bp
cat thesis/qmd/results/pooled_td3_h2_fees_1e6_dsr/latest_finished/evaluation_report.json
# 0.1 bp
cat thesis/qmd/results/pooled_td3_h2_fees_1e5_dsr/latest_finished/evaluation_report.json
# 1 bp
cat thesis/qmd/results/pooled_td3_h2_fees_1e4_dsr/latest_finished/evaluation_report.json
```

These are the four directories `_fees_scenarios` in `06-02-robustness-assessment.qmd`
actually loads, so the review reads what the chapter renders. If a snapshot
directory is missing, run `ls thesis/qmd/results/` and note it as a gap.

## Step 3 — Cross-check numbers

For every specific number cited in the prose (total_return, max_drawdown, win_rate, profit_factor, Sharpe, Sortino, turnover), verify it matches the value in the JSON. Flag any mismatch with: `MISMATCH: prose says X, data says Y`.

Key checks:
- `total_return` at each fee level — is it positive at 0 bp, marginal at 0.01 bp, negative from 0.1 bp?
- `profit_factor` monotonically weakening as the fee rises
- the fee level at which the sign flips (should be between 0.01 and 0.1 bp)

## Step 4 — Evaluate the H2 verdict

Answer these questions explicitly:

1. Is the **transaction-cost sensitivity monotone** (performance degrades as costs increase)?
2. Is the **zero-profit threshold between 0.01 bp and 0.1 bp**, i.e. a fraction of a basis point?
3. Does the prose tie that threshold to the **bid–ask half-spread ceiling** on the microstructure signal (Section 4.3)?
4. Is **H2 (how fast the reported result moves) kept distinct from H1 (existence + magnitude of the edge)**, while making clear the two meet at the same quantity?
5. Is the stated H2 verdict consistent with the fee ladder?

## Step 5 — Create GitHub issues for findings

For each gap, mismatch, or recommendation, create a GitHub issue with the `#masters_thesis` label:

```bash
gh issue create \
  --title "[H2] <brief description of issue>" \
  --body "## Issue
<description>

## Context
Found during H2 review of thesis.

## File
<thesis/qmd/src/file.qmd:line>

## Recommendation
<how to fix>" \
  --label "masters_thesis"
```

Track the issue numbers created.

## Step 6 — Report

```
## H2 Review

### Data available
- [x/missing] pooled_td3_h2_fees_0_dsr (0 bp baseline)
- [x/missing] pooled_td3_h2_fees_1e6_dsr (0.01 bp)
- [x/missing] pooled_td3_h2_fees_1e5_dsr (0.1 bp)
- [x/missing] pooled_td3_h2_fees_1e4_dsr (1 bp)

### Fee ladder (from data)
| Metric        | 0 bp | 0.01 bp | 0.1 bp | 1 bp |
|---------------|------|---------|--------|------|
| total_return  | ...  | ...     | ...    | ...  |
| profit_factor | ...  | ...     | ...    | ...  |

### Number verification
| Claim | Prose value | Data value | Status |
|-------|-------------|------------|--------|

### Verdict assessment
[Is H2 supported as stated? Is the sign-flip location and its link to the half-spread ceiling accurate?]

### Gaps and recommendations
- [List issues created: #N, #N, ...]

### GitHub issues created
- #N — <title>
- #N — <title>
```

## Step 7 — Close resolved issues

After fixes are applied, close the related GitHub issues:

```bash
gh issue close <issue-number> --comment "Fixed during H2 review — resolved mismatch/gap."
```
