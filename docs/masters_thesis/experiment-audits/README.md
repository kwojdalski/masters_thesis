# Experiment audits

Durable log of `experiment-auditor` runs. Newest first. One row per audit; the
detail lives in the dated file.

A finding that should eventually be written into the thesis gets a pointer
comment at its destination, not just a row here — e.g. the frictionless-execution
mechanism (2026-08-31 follow-up) is queued at the `### Execution Realism` section
of `thesis/qmd/src/07-02-limitations-and-future-research.qmd` via a
`TODO(thesis-writer)` HTML comment. Check the target chapter for such comments
before writing that section from scratch.

**2026-09-01 — hypothesis numbering changed.** The audits below predate a
renumber. New scheme: **H1** = algorithm/signal (unchanged), **H2** =
transaction-cost sensitivity (was the fee sub-axis of the old H3), **H3** =
feature specification (was H2), **H4** = reward design (was the reward sub-axis
of the old H3). The old H4 "learning progression check" was dropped. The runner
keys `thesis-experiments hN` and `experiment_sets/*.yaml` were remapped to
match; scenario directory names (`td3_h3_features_*`, `td3_h3_fees_*`,
`..._h4_n5`) keep their original tokens. Where an entry below says "h3 fee
sweep" / "h3's reward axis" / "h4", read it against the old numbering.

| Date | Scope | Command | CRIT/HIGH | Verdict headline | File |
|---|---|---|---|---|---|
| 2026-09-02 | quant-bias-auditor follow-up: obs timing, trade-sign, eval-time refit, val/test partition, universe survivorship, H5 sweep | static audit + targeted checks, no run | 0 / 0 | No new leakage. Obs timing clean (tradingenv step order + `split_for_latency` only truncates, never advances). Trade-sign is exchange `side` field, no Lee-Ready code anywhere in the package. One eval-time `pipeline.fit()` found off the default path, confirmed inert (running-normalization never consults fit state). Val/test partition verified: 0 shared rows, 59us gap. Universe criterion confirmed pre-outcome. H5 latency ladder (new since 2026-08-31) has no post-hoc arm-selection logic. | [2026-09-02-quant-bias-followup.md](2026-09-02-quant-bias-followup.md) |
| 2026-08-31 | h1 leakage and bias sweep (six bias families) | static audit + 5 numerical checks, no run | 0 / 0 | No leakage. Features exactly prefix-invariant (diff 0.0 on all 10); TWAP causal, VWAP ex-post but already disclosed; selection ran on train days only; test split untouched until final eval. Two thesis claims about normalization are false but harmless. | [2026-08-31-h1-leakage-and-bias-sweep.md](2026-08-31-h1-leakage-and-bias-sweep.md) |
| 2026-08-31 | h1 follow-up: frictionless mid-price mechanism | static audit, no run | 1 / 2 | No look-ahead; the edge is the waived half-spread — spread cost is 10.3x gross PnL. `train_size: null` silently killed 5 guardrail checks. | [2026-08-31-h1-followup-frictionless-mechanism.md](2026-08-31-h1-followup-frictionless-mechanism.md) |
| 2026-08-31 | h1 (100% win rate) | `uv run thesis-experiments h1 --max-train-seconds 240` | 4 / 4 | Win rate mis-measured on compounded bars, on a frictionless env, published from a run that never finished. | [2026-08-31-h1-win-rate.md](2026-08-31-h1-win-rate.md) |
