Review **Hypothesis 4** (the reward objective materially shapes the learned policy: replacing a log-return reward with a Differential Sharpe Ratio reward changes both out-of-sample performance and the policy's directional exposure) by reading only the relevant thesis sections and result files.

## Step 1 — Read the thesis prose for H4

Read the H4 section inside:
- `thesis/qmd/src/01-01-scope-and-objectives.qmd` (the formal H4 statement — item 4 of the hypothesis list)
- `thesis/qmd/src/06-02-robustness-assessment.qmd` — read only the `## Hypothesis 4: Reward Function Design` section (stop at the seed-robustness note that closes the chapter)
- `thesis/qmd/src/07-01-summary-of-findings.qmd` — read only the Hypothesis 4 paragraph

## Step 2 — Read the result data for the two H4 scenarios

The H4 comparison holds everything fixed except the reward function. For each, read its thesis snapshot evaluation report:

```bash
# Log-return baseline
cat thesis/qmd/results/pooled_td3_hft_lob_state_space_pooled_streaming_selected/latest_finished/evaluation_report.json
# Differential Sharpe Ratio (DSR)
cat thesis/qmd/results/pooled_td3_hft_lob_state_space_pooled_streaming_selected_dsr/latest_finished/evaluation_report.json
```

If a snapshot directory is missing, run `ls thesis/qmd/results/` and note it as a gap.

## Step 3 — Cross-check numbers

The H4 section cites specific values for the reward comparison. Verify each:

- DSR vs log-return: cumulative return ($2.28 \times 10^{-7}$ vs $1.29 \times 10^{-7}$)
- DSR vs log-return: profit factor (5.70 vs 2.01)
- DSR vs log-return: max drawdown ($-1.05 \times 10^{-7}$ vs $-1.93 \times 10^{-7}$)
- DSR exposure shift: 49.8% long / 50.2% short vs 86.7% long

Flag any mismatch with: `MISMATCH: prose says X, data says Y`.

## Step 4 — Evaluate the H4 verdict

Answer these questions explicitly:

1. Does the **reward function change** (log-return → DSR) produce a materially different policy? Confirm the direction and magnitude on return, profit factor, and drawdown.
2. Does the **directional exposure** shift as the prose claims (long bias → approximately neutral)?
3. Is the comparison genuinely holding everything else fixed (same feature set, algorithm, data, zero fee), or is there config drift the prose should acknowledge?
4. Is the stated H4 verdict ("supported") consistent with the two scenarios?

## Step 5 — Create GitHub issues for findings

For each gap, mismatch, or recommendation, create a GitHub issue with the `#masters_thesis` label:

```bash
gh issue create \
  --title "[H4] <brief description of issue>" \
  --body "## Issue
<description>

## Context
Found during H4 review of thesis.

## File
<thesis/qmd/src/file.qmd:line>

## Recommendation
<how to fix>" \
  --label "masters_thesis"
```

Track the issue numbers created.

## Step 6 — Report

```
## H4 Review

### Data available
- [x/missing] pooled_td3_hft_lob_state_space_pooled_streaming_selected (log-return baseline)
- [x/missing] pooled_td3_hft_lob_state_space_pooled_streaming_selected_dsr (DSR)

### Reward comparison (from data)
| Metric        | Log-return | DSR |
|---------------|------------|-----|
| total_return  | ...        | ... |
| profit_factor | ...        | ... |
| max_drawdown  | ...        | ... |
| pct_long      | ...        | ... |

### Number verification
| Claim | Prose value | Data value | Status |
|-------|-------------|------------|--------|

### Verdict assessment
[Is H4 supported as stated? Is the exposure shift and the "not just scale" framing accurate?]

### Gaps and recommendations
- [List issues created: #N, #N, ...]

### GitHub issues created
- #N — <title>
- #N — <title>
```

## Step 7 — Close resolved issues

After fixes are applied, close the related GitHub issues:

```bash
gh issue close <issue-number> --comment "Fixed during H4 review — resolved mismatch/gap."
```
