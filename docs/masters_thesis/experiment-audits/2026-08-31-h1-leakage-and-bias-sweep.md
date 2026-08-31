# Experiment audit — h1 leakage and bias sweep (six bias families)

**Command audited:** static audit + five targeted numerical checks, no training run.
**Date/time:** 2026-08-31 ~23:15 BST
**Auditor:** quant-bias-auditor (claude-opus-5)
**Scope:** h1, primary scenario
`pooled/td3_hft_lob_state_space_pooled_streaming_selected_dsr`. Feature,
benchmark, normalization and reward code is shared across all scenarios, so
findings in those layers apply to h2/h3/h4 too.
**Prior work built on:** `2026-08-31-h1-win-rate.md`,
`2026-08-31-h1-followup-frictionless-mechanism.md`. The action/return lag
identification for `log_return` and the chronological disjointness of the
splits were taken as established and not re-derived.

## Verdict

**No leakage or selection-bias finding severe enough to invalidate the h1
headline result — beyond the already-known frictionless-execution issue,
which remains the only mechanism that fabricates the result.**

All six bias families were checked. Five are clean, and four of those were
cleared by a positive numerical test rather than by reading a docstring:

- Feature construction is **exactly** causal. All ten h1 features are
  bit-identical whether the pipeline sees 200k rows or 400k rows of the
  evaluation day (`max|full - prefix| = 0.000e+00`, 196,491 rows compared).
  A feature that peeked forward by even one tick could not pass this.
- TWAP is a genuine causal execution schedule. VWAP *is* ex-post, and this is
  already labelled in code, propagated to result metadata, and disclosed in
  `06-00-results.qmd`. Nothing to add.
- Feature **selection** ran only on the 18 training-day files. The evaluation
  day is absent from `src/configs/feature_research/pooled_daily_hft_lob.yaml`.
- Periodic evaluation defaults to `[train, val]` and h1 does not override it.
  There is no best-checkpoint selection logic anywhere; evaluation uses the
  final policy. The test split is touched exactly once, at final reporting.
- DSR is a strictly online function of the NLV path with EMA state from
  `t-1` only. Both registered reward types are now checked; none are left
  unchecked.

The two real findings are **documentation defects, not leakage**: the thesis
describes a normalization mechanism that the code does not implement, and
describes val/test as more separated than they are. Both were verified by
running the code, and neither inflates the result.

One corroboration worth recording for the existing CRITICAL finding: the h3
fee sweep shows the sign flip directly. Test-split total return is
**+2.28e-07 at 1e-6 fees, -7.60e-07 at 1e-5, -5.89e-06 at 1e-4** — the
strategy is profitable only in the fee band below the half-spread, exactly as
the frictionless-mechanism audit predicted.

## Findings

| # | Sev | Family | Finding |
|---|---|---|---|
| 1 | MEDIUM | 3 / docs | Thesis states evaluation-day normalization uses training-period statistics. It does not — the fitted state is discarded and the evaluation day is normalized by its own expanding statistics. |
| 2 | MEDIUM | 3 / docs | Thesis states running statistics are maintained "per chronological split". They are not — val and test are two halves of one session and share one continuous statistic stream. |
| 3 | MEDIUM | 5 | Symbol universe is six mega-cap tech names — documented with a stated criterion, but a single-sector cluster, so N=6 overstates the independent evidence. |
| 4 | LOW | 4 / inference | Bootstrap resamples strategy and benchmark independently rather than as a pair. Structurally wrong, but errs conservative. |
| 5 | LOW | 1 | `close = mid.ffill().bfill()` — `bfill` is a leading-edge look-ahead vector. Measurably inert: 0 affected rows. |

### 1. MEDIUM — the stated normalization mechanism is not the implemented one

`thesis/qmd/src/05-01-data-preparation.qmd:151-158` claims:

> The fitted pipeline is then applied independently to each training file and
> to the evaluation file. This per-security fitting ensures that the
> normalization statistics are computed exclusively from training-period
> observations, with no leakage from the held-out evaluation day.

The fitted statistics are **not** used. All ten h1 features are configured
`normalization_method: running` + `reset_on_session_break: true`
(`src/configs/feature_sets/hft_lob_features_all.yaml`). That routes
`Feature.transform` into `_transform_session_aware`
(`src/trading_rl/features/base.py:670-731`), which swaps in a **throwaway**
`RunningMeanStd` per session (`:714-715`) and normalizes with causal
cumulative sums where "stats at index t use only vals[0..t-1]"
(`_transform_running_session_online`, `:823-849`). The scaler that
`pipeline.fit(train_df)` populated is never consulted.

**Verification method.** Transform the same AAPL evaluation day with two
pipelines — one fitted on AAPL training data, one fitted on *TSLA* training
data. If training statistics mattered at all, the outputs would differ.

```
Eval-day features: AAPL-fitted vs TSLA-fitted pipeline
  all 10 h1 features                            max|diff| = 0.000e+00
```

The fit state is provably irrelevant. Consequences:

- The thesis's stated leakage guarantee is not the guarantee the code
  provides. The code's actual guarantee is *stronger* on the leakage axis
  (evaluation-day normalization is purely intra-day and strictly causal, so
  training data cannot contaminate it because it is not used), but the
  sentence as written is false and would not survive an examiner who read
  `base.py`.
- `pipeline_state.pkl`, `feature_pipeline_state` in the checkpoint dataclass,
  and the `dump_pipeline_state`/`load_state` machinery are dead weight for
  the h1 feature set. They are not wrong, just inert. Worth knowing before
  anyone debugs them.

**Not a leakage finding.** The result is unaffected.

### 2. MEDIUM — val and test are one session, not two chronological splits

`05-01-data-preparation.qmd:205` claims statistics are maintained
"**per chronological split**" and that "later validation or test observations
cannot influence earlier ones". The first half of that is false.

`_per_symbol_worker` transforms the whole March 2 file in one pass and only
*then* cuts it in half (`src/trading_rl/data/preparation.py:543-548`):

```python
mid = len(val_df_j) // 2
val_df_j.iloc[:mid].to_parquet(val_p)
val_df_j.iloc[mid:].to_parquet(test_p)
```

Session-break detection sees one continuous session, so the expanding
statistics used on the test half are warm-started by the val half.

**Verification method.** Compute the test half with the val half in front of
it, versus the test half alone, and diff.

```
Test half computed WITH val half in front vs. test half ALONE:
  feature_hft_signed_trade_flow_50               max|diff| = 9.999e+03
  feature_hft_ofi                                max|diff| = 3.160e+02
  feature_hft_ofi_rolling_50                     max|diff| = 2.013e+02
  feature_hft_book_pressure_l0                   max|diff| = 9.804e+01
  ... (all ten differ)
```

The direction of the dependence is backward, so this is **not** look-ahead —
finding 1's prefix-invariance result proves no test-half value depends on any
later row. But two claims in the thesis need correcting:

- "per chronological split" — the split boundary has no effect on the
  statistics; the *session* boundary does, and val/test share a session.
- Val and test are not independent evaluation samples. They are consecutive
  halves of one trading day, sharing one regime, one spread environment, and
  one normalization warm-up. Any variance estimate treating them as
  independent replicates is optimistic.

The warm start itself is defensible — a live system is also warm by
mid-session — but it should be described as what it is.

### 3. MEDIUM — universe selection is documented, but is a single-sector cluster

`05-01-data-preparation.qmd:7-24` gives a stated, non-outcome-correlated
criterion: "high but not extreme liquidity", chosen so the price-taking
assumption holds and short-borrow costs stay symmetric. The date window is
explicitly flagged as a limitation at `:145-149` and again in
`07-02-limitations-and-future-research.qmd:44`. So this is a **documented
methodological choice, not a hidden survivorship filter** — the rubric's
MEDIUM, not HIGH.

The residual issue is that AAPL, MSFT, TSLA, META, AMZN and AVGO are six
mega-cap US tech names with high cross-sectional return correlation over a
four-day window. Pooling them and reporting statistics over the concatenated
tick series treats them as six independent securities when they are closer to
one factor. This inflates apparent precision rather than apparent
performance, and it compounds with finding 2 (one session) and with the
absence of walk-forward validation.

**Verification method.** Read the stated criterion and confirm it is not
outcome-conditioned; confirm the universe is not filtered on any
performance-correlated quantity. It is not — selection is on liquidity and
data availability, both fixed before any result existed.

### 4. LOW — unpaired bootstrap resampling

`src/trading_rl/evaluation/statistical_test_registry.py:107-113` draws the
strategy sample and the baseline sample independently:

```python
strategy_sample = rng.choice(strategy_returns, size=n_strategy, replace=True)
baseline_sample = rng.choice(baseline_returns, size=n_baseline, replace=True)
```

Strategy and benchmark returns are contemporaneous functions of the same
price path, so the correct scheme resamples the time index once and applies it
to both. Independent draws give
`Var(S_strat - S_bench) = Var(S_strat) + Var(S_bench)` instead of
`... - 2·Cov`. Since the covariance is positive (the agent trades the same
path the benchmark holds), the implemented scheme **overstates** the variance
of the difference. It is conservative, so it cannot manufacture significance.

I also checked whether the iid (non-block) draw understates standard errors
on autocorrelated tick returns, which *would* inflate significance.

**Verification method.** Measure the autocorrelation of h1 test-split tick
returns and compare an iid bootstrap Sharpe SE against a moving-block
bootstrap (block length 1000 ticks), n = 969,218.

```
autocorr lag 1: +0.0130   lag 2: +0.0165   lag 5: -0.0034   lag 50: +0.0053
iid bootstrap    Sharpe SE = 1.020e-03
block(1000) boot Sharpe SE = 1.188e-03
understatement factor      = 1.16x
```

1.16x is negligible against the effect sizes involved. Tick returns here are
close to serially uncorrelated, so the iid assumption is empirically almost
harmless. **LOW, and measured rather than assumed** — I expected this to be
worse than it is.

### 5. LOW — `bfill` in close-price construction, measurably inert

`src/trading_rl/data/hft.py:59`:

```python
result["close"] = mid_price.ffill().bfill()
```

`ffill` handles interior and trailing gaps causally. `bfill` then acts only on
a *leading* NaN run — rows before the first two-sided quote — and fills them
with a future price. That is a genuine look-ahead vector.

**Verification method.** Count the leading-NaN run in the mid-price series for
the evaluation day, per symbol, after the level-0–4 filter.

```
AAPL: rows=1,938,438  total_nan=0  leading_nan (bfill-affected)=0
TSLA: rows=1,678,214  total_nan=0  leading_nan (bfill-affected)=0
META: rows=  273,353  total_nan=0  leading_nan (bfill-affected)=0
```

Zero rows affected. This is the rubric's LOW exactly: a leakage vector that
exists in theory and is measurably inert here. It should still be changed to
`.ffill()` with an explicit drop of any leading-NaN rows, so the property is
guaranteed rather than incidental to this dataset.

## Verified clean (positive checks, not assumptions)

**Family 1 — feature look-ahead: CLEAN.** Prefix-invariance test on real AAPL
March 2 data, pipeline fitted on Feb 25, transform of 400k rows vs 200k rows,
196,491 overlapping rows compared:

```
feature_hft_book_pressure_l0            0.000e+00
feature_hft_order_book_imbalance_3l     0.000e+00
feature_hft_order_count_imbalance_l0    0.000e+00
feature_hft_microprice                  0.000e+00
feature_hft_microprice_divergence       0.000e+00
feature_hft_bid_slope                   0.000e+00
feature_hft_ask_slope                   0.000e+00
feature_hft_ofi                         0.000e+00
feature_hft_ofi_rolling_50              0.000e+00
feature_hft_signed_trade_flow_50        0.000e+00
```

Supporting reads: every rolling window in `lob_flow_features.py` and
`lob_trade_features.py` is trailing with `min_periods=1`; `best_level_ofi`
and `ofi_multilevel` use `shift(1)` only (`lob_common.py:100-103`,
`lob_flow_features.py:140-143`); `RollingWindowScaler` shifts its stats by 1
(`base.py:80-84`). Trade sign comes from the MBP-10 `side` field, not a
Lee-Ready classifier, so there is no future-quote dependency
(`split_trade_flow`, `lob_common.py:120-137`).

The one deliberate look-ahead feature, `mid_price_future_velocity`
(`lob_book_features.py:231-251`, `mid.diff().shift(-1)`), is a labelled
training-loop canary. It is confined to two oracle feature sets
(`hft_lob_oracle_with_market.yaml`, `hft_lob_future_log_return.yaml`) and
guarded by `_ORACLE_FEATURE_TYPES` in `config_guardrails_checks.py:187`. It is
**not** in h1's feature set — h1 uses the ten features listed in
`observation.yaml`, all drawn from `hft_lob_features_all.yaml`.

**Family 2 — benchmark look-ahead: CLEAN, and already disclosed.** TWAP builds
exposure on a constant-weight schedule with `lagged_exposure` — position
during period `t` is `t/n`, established before the return it earns
(`statistical_benchmarks.py:61-91`). The divisor is the window length, known
in advance. VWAP normalizes by the window's total realized volume and *is*
ex-post; this is documented in the function docstring
(`compute_vwap_returns:99-119`), carried as `VWAP_SCHEDULE_LABEL` metadata
into the statistical-test results and MLflow (`benchmarks.py:25-31`,
`:124-137`), logged as a runtime warning, and stated in the thesis at
`06-00-results.qmd:42-43`. Nothing further is needed.

One observation, not a finding: TWAP's average exposure over the window is
~0.5 versus buy-and-hold's 1.0, and its exposure is time-varying, so its
Sharpe is not directly comparable to a constant-exposure benchmark. This is a
benchmark-definition question, not look-ahead.

**Family 3 — cross-split leakage: CLEAN.** Two independent confirmations. The
prefix-invariance result means the position of the train/val/test boundary
cannot change any feature value. The fitted-state test (finding 1) means
training statistics do not reach the evaluation day at all. Pooled preparation
fits per symbol on that symbol's training files only
(`preparation.py:410-441`) and there is no path that fits on val or test.

**Family 4 — model-selection leakage: CLEAN.** `TempEvalConfig.splits`
defaults to `[TRAIN, VAL]` (`config.py:323-325`); h1's `train.yaml` sets only
`temp_eval.interval` and `max_steps`, so the default holds and periodic eval
never builds a test env. `experiment_runner.py:205-240` would honour a `test`
entry if one were configured — it is not, but this is a live footgun worth a
guardrail. Early stopping is driven by position-change and saturation
statistics from the training loop (`es_stale_policy_*`, `es_saturation_*`),
both defaulted to 0 = disabled and both unset in h1. There is no
best-checkpoint selection: `PolicyLoader.from_checkpoint` loads an explicitly
named file and evaluation uses the final policy.

On sweep snooping: `_H2_SCENARIOS` and `_H3_SCENARIOS`
(`src/masters_thesis/experiments.py:72-92`) are enumerated as complete grids,
and the report scripts contain no `argmax`/`best`/`winner` logic — they
tabulate every arm. h1's use of the DSR reward is one arm of h3's reward axis,
which would be a selection concern if it were chosen post-hoc; it is not. DSR
is justified a priori on theoretical grounds in
`04-03-reward-function.qmd:70-163` (Moody & Saffell, recursive online form,
risk-adjustment), and `git log -L 65,71` shows the h1 scenario has named
`_selected_dsr` since the 2026-05-28 migration, predating the results. **No
evidence of post-hoc arm selection** in the artifacts. This is the limit of
what artifacts can show — they cannot rule out selection that left no trace.

**Family 6 — reward causal alignment: CLEAN, both types now checked.** The
registry holds exactly two reward types (`rewards/registry.py:90-110`):
`log_return` (cleared by the prior audit, exact corr 1.0 at the causally
correct lag) and `differential_sharpe`. DSR's `calculate`
(`differential_sharpe.py:169-226`) computes
`D_t = (B_{t-1}·ΔA_t - A_{t-1}·ΔB_t/2) / (max(B_{t-1}-A_{t-1}², 0)^1.5 + ε)`
using the EMA values from `t-1` and only *then* updates them (`:203-222`).
`R_t = log(NLV_t / NLV_{t-1})` reads the broker's mark, whose timing the prior
audit already pinned. Nothing in the reward path reaches forward past the
mark the prior audit identified. **No unchecked reward types remain.**

## Noted in passing (experiment-auditor's territory, not investigated)

- `thesis/qmd/results/pooled_td3_hft_lob_state_space_pooled_streaming_selected_dsr/latest_finished/`
  is empty and nothing under it is tracked in git, so h1's primary scenario
  currently has no exported artifacts. With no artifacts,
  `06-03-performance-evaluation.qmd:44-63` falls back to its hardcoded
  placeholder benchmark table.
- The h3 fee-sweep Sharpe values are absurd in magnitude (`3.1e4`, `-2.3e5`),
  which points at an annualization or aggregation defect in the metric. The
  total-return figures quoted in the verdict are the trustworthy part.
