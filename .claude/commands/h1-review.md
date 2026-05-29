Review **Hypothesis 1** (TD3 outperforms benchmark strategies on a risk-adjusted basis) by reading only the relevant thesis sections and result files. Do not scan the whole thesis.

## Step 1 — Read the thesis prose for H1

Read these two files:
- `thesis/qmd/src/06-03-performance-evaluation.qmd`
- `thesis/qmd/src/07-01-summary-of-findings.qmd` (H1 paragraph only — stop before H2)

## Step 2 — Read the result data

Read these files from the thesis snapshot for the main DSR experiment:

```bash
cat thesis/qmd/results/pooled_td3_hft_lob_state_space_pooled_streaming_selected_dsr/latest_finished/evaluation_report.json
cat thesis/qmd/results/pooled_td3_hft_lob_state_space_pooled_streaming_selected_dsr/latest_finished/statistical_tests.json
```

If a file is missing, note it as a gap.

## Step 3 — Cross-check numbers

For every specific number cited in the prose (total_return, max_drawdown, win_rate, profit_factor, pct_long, pct_short, Sharpe, Sortino, turnover), verify it matches the value in the JSON. Flag any mismatch with: `MISMATCH: prose says X, data says Y`.

Key metrics to verify in `evaluation_report.json`:
- `total_return` — agent and all benchmarks
- `max_drawdown` — agent and buy_and_hold
- `win_rate`, `profit_factor`
- `pct_long`, `pct_short`
- `annualized_volatility`, `turnover`

Key results in `statistical_tests.json` to check:
- `benchmark_comparison_table` rows for strategy vs buy_and_hold, twap, vwap
- Any significance test outcomes

## Step 4 — Evaluate the H1 verdict

Answer these questions explicitly:

1. Does the agent outperform passive benchmarks on **total return**? By how much?
2. Does the agent outperform on **risk-adjusted return** (Sharpe/Sortino)? Are these computable?
3. Is the outperformance from **active trading** or from **positioning** (near-neutral in a down/up market)?
4. Does the prose accurately describe the mechanism (not overstate the RL contribution)?
5. Is the final H1 verdict in 07-01 (`supported` / `partially supported` / `not supported`) consistent with the data?

## Step 5 — Create GitHub issues for findings

For each gap, mismatch, or recommendation, create a GitHub issue with the `#masters_thesis` label:

```bash
gh issue create \
  --title "[H1] <brief description of issue>" \
  --body "## Issue
<description>

## Context
Found during H1 review of thesis.

## File
<thesis/qmd/src/file.qmd:line>

## Recommendation
<how to fix>" \
  --label "masters_thesis"
```

Track the issue numbers created.

## Step 6 — Report

Produce a structured report:

```
## H1 Review

### Data available
- [x/missing] evaluation_report.json
- [x/missing] statistical_tests.json

### Number verification
| Metric | Prose value | Data value | Status |
|--------|------------|------------|--------|
| ...    | ...        | ...        | OK / MISMATCH |

### Verdict assessment
[Your analysis of whether the stated H1 conclusion is justified]

### Gaps and recommendations
- [List issues created: #N, #N, ...]

### GitHub issues created
- #N — <title>
- #N — <title>
```

## Step 7 — Close resolved issues

After fixes are applied, close the related GitHub issues:

```bash
gh issue close <issue-number> --comment "Fixed during H1 review — resolved mismatch/gap."
```
