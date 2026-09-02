# Experiment audit — quant-bias-auditor follow-up (H1-H5 scope)

**Command audited:** static audit + targeted reads/numerical checks, no training run.
**Date/time:** 2026-09-02 ~11:50 UTC
**Auditor:** quant-bias-auditor (claude-sonnet-5)
**Prior work built on:** `2026-08-31-h1-leakage-and-bias-sweep.md`,
`2026-08-31-h1-followup-frictionless-mechanism.md`, `2026-08-31-h1-win-rate.md`.
The six bias families were already swept once on 2026-08-31; this run
targets items the coordinator flagged as still open after that sweep, plus
the newly-added H5 (execution-latency) scenario ladder, which postdates the
prior audit and was therefore unchecked.

The coordinator independently verified and asked not to be re-derived:
the +305% aggregate arithmetic (bp/step x step-count reconstruction matches
reported cumulative returns for all six symbols, ratio 1.00), DSR causal
correctness (`differential_sharpe.py:205-219`, A/B EMAs from t-1, updated
after use), and that VWAP's ex-post normalization is a disclosed, non-defect
choice. Not re-derived here.

## Verdict

No new leakage or look-ahead defect found. All five items in this run's
scope are either **verified clean** by a concrete check or resolve to a
**LOW**, structurally-inert vector. The H5 latency sweep, unexamined by the
prior audit, is architecturally sound: `split_for_latency` only ever
truncates features from the end, never advances them, so higher latency
cannot manufacture look-ahead, and it is a complete, non-argmax-filtered
ladder like H2/H3/H4.

## Findings

| # | Sev | Item | Finding |
|---|---|---|---|
| 1 | — | Obs. timing beyond reward | Verified clean. `tradingenv.env.step()` rebalances at `self.now()==t`, then `_process_nonlatent_events()` advances to `t+1` before returning state — consistent with the already-established `corr=1.0` lag identification. `split_for_latency` (H5) only truncates from the end. |
| 2 | — | Trade-sign / Lee-Ready | Verified clean, and stronger than "inert" — structurally absent. `split_trade_flow` reads the exchange `side` field directly; zero hits for any tick-rule/quote-rule/Lee-Ready pattern anywhere in `src/trading_rl/`. |
| 3 | LOW | Refit at eval time | `pipeline.fit(df)` at `evaluate_command.py:604` exists in the ad-hoc `--data-path` eval mode only, warns explicitly when triggered, is never on the default prepared-split path h1-h5 use, and is inert regardless because all feature sets use `running` normalization (fit state never consulted — established 2026-08-31). |
| 4 | — | Val/test split correctness | Verified clean with a fresh check (not just re-derived): AAPL val/test = 969,219 rows each, 0 shared timestamps, val's last row precedes test's first row by 59us with no gap, no duplicate. |
| 5 | MEDIUM (unchanged) | Universe survivorship | Confirms, does not elevate, the 2026-08-31 MEDIUM #3: `05-01-data-preparation.qmd:5-25` states a pre-outcome liquidity criterion; six names are single-sector mega-cap tech, overstating independent evidence when pooled. No outcome-conditioned filtering language found. |
| 6 | — | H5 sweep snooping | Verified clean. `_H5_SCENARIOS` / `_H5_EVENT_SCENARIOS` (`experiments.py:128-153`) are complete ladders; `grep -rn "argmax\|idxmax\|best\|winner" src/masters_thesis/*.py` returns zero hits, extending the 2026-08-31 "no post-hoc arm selection in artifacts" finding to H5. |

### 1. Observation timing (item 1) — clean

`tradingenv.env.py:286-328`: `step()` processes the rebalancing request at
the current mark (`self.now()==t`), then calls `_process_nonlatent_events()`
which advances `_now` to `t+1` and re-marks the book **before** the state is
returned to the caller. The state returned after `step(action_t)` is
therefore `state(t+1)`, which becomes the input to the *next* decision
`action_{t+1}` — the standard MDP convention, and exactly the alignment the
2026-08-31 follow-up already pinned by exact lag correlation
(`simple_return[i] = action[i-1] * (close[i]/close[i-1]-1)`, corr=1.000000).

The H5 latency ladder (new since the last audit) is implemented in
`src/trading_rl/envs/latency.py:302-336`, `split_for_latency(df, k)`:

```python
feature_df = df.iloc[: n - k]
price_df = df.iloc[k:].copy()
price_df.index = feature_df.index
```

`feature_df` (what the agent observes) is always a *prefix* truncation;
`price_df` (what the agent is filled at) is always drawn from *later* rows
re-indexed onto the feature timeline. For any `k >= 0` the agent is filled at
a row at or after the row that produced its observation — never before.
`k <= 0` returns `(df, df)` unchanged, which is the H1 baseline (obs and
fill drawn from the identical row, matching the already-documented
frictionless-mid-price mechanism). Both the streaming training env and the
DataFrame eval env call the same function (docstring notes this was
previously implemented independently and evaluation omitted the shift
entirely — a defect the current code fixes), so training and evaluation
cannot disagree on the offset. No look-ahead is possible by construction of
this function; it can only ever look backward (stale features), which is the
point of a latency sweep.

### 2. Trade-sign / Lee-Ready inference (item 2) — clean, structurally

`src/trading_rl/features/lob_common.py:120-137`, `split_trade_flow`:

```python
side = df[side_col].astype(str)
...
buy_vol[is_trade & (side == MBOSide.BID)] = size[...]
sell_vol[is_trade & (side == MBOSide.ASK)] = size[...]
```

The sign comes directly from the exchange-supplied MBP-10 `side` field — the
venue's own record of which side of the book was hit, known at the time of
the trade. No inference step exists to introduce a future-quote dependency.

```
grep -rnin "lee.?ready|tick.rule|quote.rule|infer.*sign|classify.*trade" src/trading_rl/
  → 0 matches
```

This closes the vector definitively rather than leaving it as "checked one
call site" — there is no Lee-Ready/tick-rule/quote-rule code anywhere in the
package for a future audit to find later either.

### 3. Refit at evaluation time (item 3) — LOW, confirmed inert on the default path

`src/cli/commands/evaluate_command.py:598-604`, inside `_prepare_arbitrary_df`
("Arbitrary data preparation" — the ad-hoc `--data-path` evaluation mode):

```python
else:
    self.logger.warning(
        "Pipeline state not available in checkpoint — normalizing eval data with eval statistics. "
        "Metrics may not reflect true out-of-sample performance. Use --data-path mode only for "
        "sanity checks; production eval should use prepared splits from training time."
    )
    pipeline.fit(df)
```

This is a real scaler-fit-on-eval-data code path, but three things confine
it:

- It only fires when `restore_result.restored` is `False`, i.e. the
  checkpoint has no `feature_pipeline_state` — the h1-h5 checkpoints do
  carry it.
- The default evaluation path for every h1-h5 scenario is
  `_resolve_per_symbol_splits` (`evaluate_command.py:624`), whose fast path
  reads `val_{sym}_prepared.parquet` / `test_{sym}_prepared.parquet` written
  at training time — no feature recomputation, no `.fit()` call at all, per
  `preparation.py:433,439,1413` which the coordinator already confirmed
  fit-on-train-only.
- Even in the arbitrary-data-path mode, the fit is inert: all h1-h5 feature
  sets use `normalization_method: running`, and the 2026-08-31 audit already
  proved by direct test (AAPL-fitted vs. TSLA-fitted pipeline, `max|diff| =
  0.0` across all ten features) that the running-normalization code path
  never consults the fitted scaler state at all.

Rated LOW: a real defect in isolation (an eval-time fit is the wrong thing to
do, and the log line correctly says so), reachable only off the default
path, and doubly inert given the normalization method actually in use.

### 4. Val/test split correctness (item 4) — clean, fresh numerical check

`src/trading_rl/data/preparation.py:544-548`:

```python
mid = len(val_df_j) // 2
val_df_j.iloc[:mid].to_parquet(val_p)
val_df_j.iloc[mid:].to_parquet(test_p)
```

Directly inspected the materialized AAPL prepared parquet files rather than
re-deriving from the slicing logic alone:

```
val rows 969219  test rows 969219
index overlap count: 0
val last ts:  2026-03-02 16:53:17.344765418
test first ts: 2026-03-02 16:53:17.344824134   (59us later, no gap, no dup)
```

`[:mid]` / `[mid:]` is an exact partition by construction (every row
appears in exactly one half, `mid` is a valid boundary since Python slicing
is exclusive/inclusive at the same point) — confirmed empirically rather
than just algebraically. No row is shared between val and test.

Restating, not re-deriving, the already-logged MEDIUM point: this is one
continuous trading session cut into two adjacent halves sharing one
normalization warm-up stream, not two independently-sampled evaluation
periods. That finding stands from 2026-08-31 and is not changed by this
check.

### 5. Universe survivorship (item 5) — confirms existing MEDIUM, not elevated

`thesis/qmd/src/05-01-data-preparation.qmd:5-25` states the criterion before
any result exists in the text: "high but not extreme liquidity... to ensure
defensible price-taking assumptions and rich microstructure signals,"
explicitly excluding names "whose order-book event density... would be
disproportionately inflated by factors unrelated to underlying economic
activity," and separately justifying the choice on borrow-cost grounds for
the short-selling symmetry assumption. No language in this section
conditions selection on realized return, volatility, trend, or any other
performance-correlated quantity — the criterion is liquidity and data
quality, both determinable before training.

This matches and closes out the 2026-08-31 MEDIUM #3 finding
(single-sector cluster inflates apparent precision of N=6, not apparent
performance) — a documented methodological choice needing the caveat it
already partially has, not a hidden selection-bias defect. No severity
change.

### 6. H5 sweep — no evidence of post-hoc arm selection

`_H5_SCENARIOS` (0us/10us/100us/1ms/5ms) and `_H5_EVENT_SCENARIOS`
(k1/k2/k4/k8) at `src/masters_thesis/experiments.py:128-153` are complete,
enumerated ladders, matching the H2/H3/H4 pattern the 2026-08-31 audit
already cleared. `grep -rn "argmax|idxmax|best|winner" src/masters_thesis/*.py`
returns zero matches — no report script contains logic that could select and
report only a winning arm. This extends, rather than repeats, the prior
"no evidence of post-hoc arm selection in the artifacts" finding to the
scenario axis that postdated it.

## Not re-derived (per coordinator instruction)

- `+305\%` aggregate arithmetic (coordinator's own reconstruction, ratio
  1.00 across all six symbols).
- DSR causal correctness (`differential_sharpe.py:205-219`).
- VWAP's disclosed ex-post normalization.
- `pipeline.fit()` train-only call sites (`preparation.py:433,439,1413`).
- All-33-features-use-`running`-normalization claim.
- No `center=True` rolling windows.
- Memmap train-only construction.
- `MidPriceFutureVelocityFeature` oracle confinement.
- Checkpoint auto-selection-by-mtime (experiment-auditor's territory).
