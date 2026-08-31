# Experiment audits

Durable log of `experiment-auditor` runs. Newest first. One row per audit; the
detail lives in the dated file.

A finding that should eventually be written into the thesis gets a pointer
comment at its destination, not just a row here — e.g. the frictionless-execution
mechanism (2026-08-31 follow-up) is queued at the `### Execution Realism` section
of `thesis/qmd/src/07-02-limitations-and-future-research.qmd` via a
`TODO(thesis-writer)` HTML comment. Check the target chapter for such comments
before writing that section from scratch.

| Date | Scope | Command | CRIT/HIGH | Verdict headline | File |
|---|---|---|---|---|---|
| 2026-08-31 | h1 leakage and bias sweep (six bias families) | static audit + 5 numerical checks, no run | 0 / 0 | No leakage. Features exactly prefix-invariant (diff 0.0 on all 10); TWAP causal, VWAP ex-post but already disclosed; selection ran on train days only; test split untouched until final eval. Two thesis claims about normalization are false but harmless. | [2026-08-31-h1-leakage-and-bias-sweep.md](2026-08-31-h1-leakage-and-bias-sweep.md) |
| 2026-08-31 | h1 follow-up: frictionless mid-price mechanism | static audit, no run | 1 / 2 | No look-ahead; the edge is the waived half-spread — spread cost is 10.3x gross PnL. `train_size: null` silently killed 5 guardrail checks. | [2026-08-31-h1-followup-frictionless-mechanism.md](2026-08-31-h1-followup-frictionless-mechanism.md) |
| 2026-08-31 | h1 (100% win rate) | `uv run thesis-experiments h1 --max-train-seconds 240` | 4 / 4 | Win rate mis-measured on compounded bars, on a frictionless env, published from a run that never finished. | [2026-08-31-h1-win-rate.md](2026-08-31-h1-win-rate.md) |
