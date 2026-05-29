Review **Hypothesis 2** (a broader microstructure-aware feature set improves out-of-sample performance relative to a simpler snapshot-based representation) by reading only the relevant thesis sections and result files.

## Step 1 — Read the thesis prose for H2

Read the H2 section inside:
- `thesis/qmd/src/06-02-robustness-assessment.qmd` — read only the `## H2: Feature Specification Sensitivity` section (stop at `## H3`)
- `thesis/qmd/src/07-01-summary-of-findings.qmd` — read only the H2 paragraph (stop at H3)

## Step 2 — Read the result data for the three H2 scenarios

The H2 comparison uses three feature-set configurations. For each, read its thesis snapshot evaluation report:

```bash
# Minimal (3 features)
cat thesis/qmd/results/pooled_td3_h3_features_minimal/latest_finished/evaluation_report.json

# Selected (10 features) — the baseline
cat thesis/qmd/results/pooled_td3_hft_lob_state_space_pooled_streaming_selected/latest_finished/evaluation_report.json

# Full (33 features)
cat thesis/qmd/results/pooled_td3_h3_features_full/latest_finished/evaluation_report.json
```

If a snapshot directory is missing, run:
```bash
ls thesis/qmd/results/
```
to see what is available, and note any missing scenarios as gaps.

## Step 3 — Cross-check numbers

The H2 section cites specific comparison values. Verify each against the JSON data:

Key metrics to compare across scenarios:
- `total_return`
- `sharpe_ratio`, `sortino_ratio`
- `max_drawdown`
- `win_rate`, `profit_factor`
- `pct_long`, `pct_short`

Flag any mismatch with: `MISMATCH: prose says X, data says Y`.

## Step 4 — Evaluate the H2 verdict

Answer these questions explicitly:

1. Does the **selected (10-feature) set outperform the minimal (3-feature) set** on total return and risk-adjusted metrics?
2. Does the **full (33-feature) set outperform the selected set**? If not, does the prose correctly explain why (feature quality vs quantity)?
3. Is the direction of improvement consistent across metrics, or mixed?
4. Is the stated H2 verdict — "supported for minimal→selected transition, not for every broader set" — justified by the numbers?
5. Are there any metrics where H2 goes in the wrong direction that the prose should acknowledge?

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
- [x/missing] pooled_td3_h3_features_minimal
- [x/missing] pooled_td3_hft_lob_state_space_pooled_streaming_selected
- [x/missing] pooled_td3_h3_features_full

### Comparison table (from data)
| Metric       | Minimal | Selected | Full |
|--------------|---------|----------|------|
| total_return | ...     | ...      | ...  |
| sharpe_ratio | ...     | ...      | ...  |
| ...          |         |          |      |

### Number verification
| Metric | Scenario | Prose value | Data value | Status |
|--------|----------|-------------|------------|--------|

### Verdict assessment
[Is H2 supported as stated? What nuance is missing?]

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
