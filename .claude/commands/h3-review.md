Review **Hypothesis 3** (empirical performance is materially sensitive to feature specification, reward design, and transaction-cost assumptions) by reading only the relevant thesis sections and result files.

## Step 1 — Read the thesis prose for H3

Read these files:
- `thesis/qmd/src/06-02-robustness-assessment.qmd` — read only the `## H3: Sensitivity Analysis` section (stop at `## H4`)
- `thesis/qmd/src/07-01-summary-of-findings.qmd` — read only the H3 paragraph

## Step 2 — Read the result data for all H3 scenarios

H3 covers two sensitivity axes: reward design and transaction costs.

**Reward design axis:**
```bash
# Log-return baseline
cat thesis/qmd/results/pooled_td3_hft_lob_state_space_pooled_streaming_selected/latest_finished/evaluation_report.json

# Differential Sharpe Ratio (DSR)
cat thesis/qmd/results/pooled_td3_hft_lob_state_space_pooled_streaming_selected_dsr/latest_finished/evaluation_report.json
```

**Transaction-cost axis:**
```bash
# 0.01 bp fees
cat thesis/qmd/results/pooled_td3_h3_fees_1e6/latest_finished/evaluation_report.json

# 0.1 bp fees
cat thesis/qmd/results/pooled_td3_h3_fees_1e5/latest_finished/evaluation_report.json

# 1 bp fees
cat thesis/qmd/results/pooled_td3_h3_fees_1e4/latest_finished/evaluation_report.json
```

If any snapshot directory is missing, run:
```bash
ls thesis/qmd/results/
```
and note missing scenarios as gaps.

## Step 3 — Cross-check numbers

The H3 section cites specific values for each axis comparison. Verify each:

**Reward axis claims to check:**
- DSR vs log-return: cumulative return ($2.28 \times 10^{-7}$ vs $1.29 \times 10^{-7}$)
- DSR vs log-return: profit factor (5.70 vs 2.01)
- DSR vs log-return: max drawdown ($-1.05 \times 10^{-7}$ vs $-1.93 \times 10^{-7}$)
- DSR exposure shift: 49.8% long / 50.2% short vs 86.7% long

**Transaction-cost claims to check:**
- 0.01 bp: return still positive ($2.28 \times 10^{-7}$)
- 0.1 bp: return turns negative ($-7.60 \times 10^{-7}$)
- 1 bp: loss increases to ($-5.89 \times 10^{-6}$)

Flag any mismatch with: `MISMATCH: prose says X, data says Y`.

## Step 4 — Evaluate the H3 verdict

Answer these questions explicitly:

1. Does the **reward function change** (log-return → DSR) produce a materially different policy? Confirm the direction and magnitude.
2. Is the **transaction-cost sensitivity** monotone (performance degrades as costs increase)? Is the zero-profit threshold between 0.01 bp and 0.1 bp?
3. Does feature specification sensitivity (covered in H2) further reinforce H3, or is it handled separately?
4. Does the prose correctly bound the economic claim — i.e., does it state that the agent's edge exists only at near-zero transaction costs?
5. Is the H3 verdict ("supported") consistent with all three sensitivity axes?

## Step 5 — Create GitHub issues for findings

For each gap, mismatch, or recommendation, create a GitHub issue with the `#masters_thesis` label:

```bash
gh issue create \
  --title "[H3] <brief description of issue>" \
  --body "## Issue
<description>

## Context
Found during H3 review of thesis.

## File
<thesis/qmd/src/file.qmd:line>

## Recommendation
<how to fix>" \
  --label "masters_thesis"
```

Track the issue numbers created.

## Step 6 — Report

```
## H3 Review

### Data available
- [x/missing] pooled_td3_hft_lob_state_space_pooled_streaming_selected (log-return baseline)
- [x/missing] pooled_td3_hft_lob_state_space_pooled_streaming_selected_dsr (DSR)
- [x/missing] pooled_td3_h3_fees_1e6 (0.01 bp)
- [x/missing] pooled_td3_h3_fees_1e5 (0.1 bp)
- [x/missing] pooled_td3_h3_fees_1e4 (1 bp)

### Reward axis comparison (from data)
| Metric       | Log-return | DSR  |
|--------------|------------|------|
| total_return | ...        | ...  |
| profit_factor| ...        | ...  |
| max_drawdown | ...        | ...  |
| pct_long     | ...        | ...  |

### Fee axis comparison (from data)
| Metric       | 0 bp | 0.01 bp | 0.1 bp | 1 bp |
|--------------|------|---------|--------|------|
| total_return | ...  | ...     | ...    | ...  |

### Number verification
| Claim | Prose value | Data value | Status |
|-------|-------------|------------|--------|

### Verdict assessment
[Is H3 supported as stated? Is the cost-sensitivity bound accurately characterized?]

### Gaps and recommendations
- [List issues created: #N, #N, ...]

### GitHub issues created
- #N — <title>
- #N — <title>
```

## Step 7 — Close resolved issues

After fixes are applied, close the related GitHub issues:

```bash
gh issue close <issue-number> --comment "Fixed during H3 review — resolved mismatch/gap."
```
