Review **Hypothesis 1** (a TD3 agent learns a real but transaction-cost-bounded risk-adjusted edge from LOB microstructure — profitable under frictionless mid-price execution, not under realistic spread-crossing cost) by reading only the relevant thesis sections and result files. Do not scan the whole thesis.

## Step 1 — Read the thesis prose for H1

Read these files:
- `thesis/qmd/src/01-01-scope-and-objectives.qmd` (the formal H1 statement)
- `thesis/qmd/src/04-03-reward-function.qmd` (the "Execution Model and the Signal–Friction Decomposition" subsection)
- `thesis/qmd/src/06-00-results.qmd` (the H1 algorithm-comparison verdict)
- `thesis/qmd/src/06-02-robustness-assessment.qmd` (the `## Hypothesis 2: Transaction-Cost Sensitivity` section — the fee sweep is the decisive test of the reworded H1)
- `thesis/qmd/src/06-03-performance-evaluation.qmd`
- `thesis/qmd/src/07-01-summary-of-findings.qmd` (H1 paragraph only — stop before Hypothesis 2)

## Step 2 — Read the result data

Read these files from the thesis snapshots:

```bash
cat thesis/qmd/results/pooled_td3_hft_lob_state_space_pooled_streaming_selected_dsr/latest_finished/evaluation_report.json
cat thesis/qmd/results/pooled_td3_hft_lob_state_space_pooled_streaming_selected_dsr/latest_finished/statistical_tests.json
# the fee sweep — the H1 edge should shrink/flip across these
for f in pooled_td3_h3_fees_1e6 pooled_td3_h3_fees_1e5 pooled_td3_h3_fees_1e4; do
  cat "thesis/qmd/results/$f/latest_finished/evaluation_report.json"
done
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

1. Does the agent outperform passive benchmarks on **total return** under the frictionless setup? By how much?
1b. Do the three learners separate from Random, and does the prose avoid claiming a **ranking among** TD3/DDPG/PPO? (DDPG has edged TD3 on return; PPO leads on per-bar Sharpe. Any "TD3 outperforms DDPG/PPO" wording is a defect.)
2. Does the agent outperform on **risk-adjusted return** (Sharpe/Sortino)? Are these computable?
3. Is the outperformance from **active trading** or from **positioning** (near-neutral in a down/up market)?
4. Is **every H1 performance figure labelled** frictionless / mid-price / zero-fee?
5. Does the prose state `|microprice − mid| ≤ half-spread` and tie the frictionless number to it as a **signal ceiling**, not just "frictions could matter"?
6. Does the fee sweep in 06-02 (`## Hypothesis 2`) show the edge shrinking/flipping at a fraction of a basis point, and is that presented as the test of the reworded H1?
7. Is **H1 (existence + magnitude of the signal) kept distinct from H2 (cost-sensitivity envelope)**?
8. Is the final H1 verdict in 07-01 (`supported as reworded` / `not`) consistent with the data?

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
[Your analysis of whether the reworded H1 conclusion (`supported as reworded` = signal real + learnable + magnitude ≈ half-spread + cost-bounded, or `not`) is justified by the data]

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
