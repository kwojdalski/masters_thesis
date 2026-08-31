# Experiment audit — h1 follow-up (frictionless mid-price mechanism)

**Command audited:** static audit, no run. Artefacts inspected:
`logs/{td3,ddpg}_hft_lob_state_space_pooled_streaming_selected_dsr/`,
`data/prepared/pooled_daily_6sym_selected/test_AAPL_prepared.parquet`,
`src/trading_rl/envs/tradingenvxy_wrapper.py`, vendored `tradingenv` 0.1.3.
**Date/time:** 2026-08-31 ~21:55 BST
**Auditor:** experiment-auditor (claude-opus-5)
**git HEAD:** 77406e78
**Scope note:** new findings only. Per coordinator instruction the re-verification
of the 15 prior findings from `2026-08-31-h1-win-rate.md` was dropped from this run.

## Verdict

There is **no look-ahead**. The environment's causality is correct and I confirmed
it two ways — by reading the step loop, and by an exact lag identification on the
TD3 test rollout (`corr = 1.000000` at the causally-correct pairing
`simple_return[i] = action[i-1] * (close[i]/close[i-1] - 1)`, i.e. the action taken
at t earns the mid move from t to t+1, nothing else).

The mechanism behind the "great results" is arithmetic, not a bug: the feature set
hands the agent `microprice_divergence(t) = microprice(t) - mid(t)`, which is the
textbook one-step-ahead estimator of the mid move. Because microprice is a convex
combination of bid and ask, `|divergence| <= half-spread` **identically** — I verified
this holds on 100.00% of 2,414,196 AAPL rows. So the entire signal the agent can
express is bounded by the half-spread, and `execution_price: mid` hands it over for
free. Measured on the test window: the spread a crossing environment would have
charged is **12.16 NLV-units against a gross PnL of 1.18 NLV-units — 10.3x the
entire profit**. The strategy is not profitable; it is a precise, well-trained
measurement of the transaction cost the environment declines to charge.

Two genuinely new defects on top of that: (1) `train_size: null` — the fix for prior
finding #12 — silently **disabled five guardrail checks**, two of them FATAL-class,
while the CLI still prints "Guardrails passed"; (2) the rollout parquet stores
`action` and `simple_return` off by one row, so any row-wise pairing of the two is
wrong.

## Mechanism (a): the exact fill

`execution_price: mid` takes the `prices=` branch,
`src/trading_rl/envs/tradingenvxy_wrapper.py:747-749` (streaming) / `:464-468` (batch).
The price frame is `price_df[[price_column]]` with `env.price_column: close`, and
`close` is constructed as `(ask_px_00 + bid_px_00)/2` in
`src/trading_rl/data/hft.py:56-59`. So the traded price is literally the mid.

Per-step order in `tradingenv/env.py:286-328`:

1. `_process_latent_events()` — empty. The wrapper never passes `latency=`, so
   `Transmitter._create_partitions(latency=0)` (`transmitter.py:487-490`) routes every
   `EventNBBO` to the **non**-latent partition.
2. `make_rebalancing_request(action, self.now(), broker)` with `self._now == t`.
   Fill price = the exchange's current mark = `mid(t)` — the same row whose features
   were returned as the observation.
3. `broker.rebalance(...)` with `BrokerFees(proportional=0.0, fixed=0.0)`.
4. `_process_nonlatent_events()` advances `_now` to `t+1`, marks the book at `mid(t+1)`.
5. `reward = self._reward.calculate(env)` → `env.broker.net_liquidation_value()`,
   marked at `mid(t+1)` (`src/trading_rl/rewards/differential_sharpe.py:179`).

Buy and sell fill at the **same** price, `mid(t)`. Zero spread, zero commission,
zero latency (`obs_latency_ticks: 0`, `exec_latency_ticks: 0` — the latency model at
`tradingenvxy_wrapper.py:682-694` exists and is correct, it is just switched off).
There is no market-impact model anywhere in the codebase, and even the "realistic"
`bid_ask` branch hardcodes `bid_size=np.inf, ask_size=np.inf`
(`:455-456`, `:736-737`) — no depth, no queue, no partial fills.

## Mechanism (b): why this is "being paid to see the next tick"

Microprice `= (ask_sz·bid_px + bid_sz·ask_px)/(bid_sz+ask_sz)`
(`src/trading_rl/features/lob_book_features.py:56-68`) is a convex combination of
bid and ask, therefore `microprice ∈ [bid, ask]` and

    |microprice(t) − mid(t)|  ≤  (ask(t) − bid(t))/2   for every t.

Verified empirically on AAPL 2026-03-02: `frac |div| <= half-spread = 1.0`
(2,414,196 rows). And the divergence is predictive:
`E[Δmid | div>0] = +1.76e-04`, `E[Δmid | div<0] = -2.66e-04`, `corr = 0.0858`.

So the maximum expected per-tick edge the feature set can express is *exactly* the
half-spread — the same amount a real fill would cost. The two cancel by construction.
Setting `execution_price: mid` removes the cost and leaves the edge as pure profit.
This is why frictionless mid-price trading looks so good: it is not "trading", it is
collecting `E[mid(t+1)] − mid(t)` with the offsetting liability deleted.

Numerically on the test window (969,218 steps, AAPL, 16:53:17–20:59:59):

| quantity | value |
|---|---|
| gross PnL, `sum(simple_return)` | **+1.1847** NLV |
| gross turnover, `sum(abs(Δaction))` | **289,579** NLV |
| mean relative half-spread | 4.066e-05 (0.407 bp) |
| spread cost a `bid_ask` env would charge | **12.161** NLV |
| cost / gross PnL | **10.3x** |
| per-step gross return | 1.222e-06 |
| per-step spread cost | 1.215e-05 |
| breakeven proportional fee | ≈ 4.1e-06 |

The h3 fee ladder is consistent with that breakeven: `fees_1e6` (below breakeven)
is the only fee scenario with a positive `total_return`; `1e-5` and `1e-4` are
negative. (Those snapshots are pre-metrics-fix and from near-untrained policies, so
they corroborate the sign only, not the magnitude.)

Two controls confirm the reading. `random_actions` — also high turnover, also
frictionless — returns **-0.033%**, so this is not free bid-ask-bounce harvesting
available to any churner (the previous audit's framing was wrong on that point); the
signal is doing real work. And the action is correlated with the **future** move
(`corr(a[t], mid move t→t+1) = +0.176`) and **negatively** with the already-realised
move (`-0.120`) — the profile of a forward predictor, not of a leak.

### Off-by-one check: negative

Lag sweep over `corr(simple_return[i], action[i+la] * g[i+lg])` on the TD3 test
rollout, joined against `test_AAPL_prepared.parquet` (identical 969,219-row index):

    a_lag=-1 g_lag=-1  corr = 1.000000   <- exact
    a_lag=-2 g_lag=-1  corr = 0.742
    a_lag=-3 g_lag=-1  corr = 0.564

`simple_return[i] = action[i-1] · (close[i]/close[i-1] − 1)`. Re-indexed: the action
chosen at row `j` earns the price move `j → j+1`. Causally correct. Feature
normalisation is also causal — `MRUNNING` uses
`running_sum[1:] = np.cumsum(clean[:-1])` (`src/trading_rl/features/base.py:838`),
strictly excluding the current value, reset per session.

## Findings (new, ranked)

| # | Severity | Issue | Where | Why it corrupts the result | Command to verify or fix |
|---|----------|-------|-------|----------------------------|--------------------------|
| 16 | CRITICAL | The whole h1 edge is the waived half-spread. Gross PnL 1.18 NLV vs 12.16 NLV of spread the env declines to charge (10.3x). `\|microprice − mid\| ≤ half-spread` identically, so the feature set's maximum expressible edge equals the cost that is deleted. `execution_price: mid` + `trading_fees: 0.0` + `exec_latency_ticks: 0` is not "an ablation", it is the entire result. | `src/configs/scenarios/pooled/td3_hft_lob_state_space_pooled_streaming_selected_dsr/train.yaml:42,50,53-56`; env `src/trading_rl/envs/tradingenvxy_wrapper.py:747-749`; `src/trading_rl/data/hft.py:56-59`; feature `src/trading_rl/features/lob_book_features.py:78-91` | +227% test return is a measurement of AAPL's half-spread, not of TD3. Under `execution_price: bid_ask` the identical action path returns roughly -10.6 (total wipe-out). Extends prior #4 with the mechanism and the magnitude. | `uv run python -c "import pandas as pd,numpy as np; d=pd.read_parquet('data/prepared/pooled_daily_6sym_selected/test_AAPL_prepared.parquet',columns=['bid_px_00','ask_px_00']); r=pd.read_parquet('logs/td3_hft_lob_state_space_pooled_streaming_selected_dsr/evaluation_data/test_rollout.parquet'); b=d.bid_px_00.values;a=d.ask_px_00.values;h=((a-b)/2/((a+b)/2))[1:]; print((np.abs(np.diff(r.action.values))*h[1:]).sum(), np.nansum(r.simple_return.values))"` |
| 17 | HIGH | `train_size: null` (the fix for prior #12) makes **five** guardrail checks raise `TypeError`. `check_config_guardrails` swallows every exception at `config_guardrails_checks.py:1908-1911` and the CLI still prints "Guardrails passed — no issues found", exit 0. Dead on all four h1 scenarios: `_check_streaming_episode_vs_train_size` (FATAL), `_check_train_size_vs_warmup_rows` (FATAL), `_check_warmup_rows`, `_check_frames_per_batch_vs_train_size`, `_check_streaming_episode_too_long`. | `src/trading_rl/config_guardrails_checks.py:1908-1911`; `:168`, `:473`, `:432`, `:453`; triggered by `train.yaml:31` `train_size: null` (commit 266c7091) | The guardrail floor silently dropped by five checks on the exact scenarios the thesis publishes, and the "passed" banner hides it. A regression introduced by fixing #12. | `uv run python src/cli.py validate guardrails -c pooled/td3_hft_lob_state_space_pooled_streaming_selected_dsr 2>&1 \| grep "failed unexpectedly"` — 5 lines, then "Guardrails passed". Fix: make the `None` branch explicit in each check (resolve `train_size=None` to the actual split length), and let `check_config_guardrails` surface swallowed exceptions as a WARN Finding rather than a bare log line. |
| 18 | HIGH | The guardrail catalogue has no check for the frictionless combination. `_check_trading_fees` only fires when `fees > 0.001` (fees *too high*); nothing looks at `trading_fees == 0`, nothing looks at `execution_price`, nothing looks at `exec_latency_ticks`. There is a FATAL oracle-feature check (`:224-246`) but microprice/OFI are legitimate features, so it correctly does not fire — leaving the actual failure mode uncovered. | `src/trading_rl/config_guardrails_checks.py:635-654` | Nothing in the repo can tell you that a scenario is a tick-harvesting fantasy. Each ingredient is defensible alone; the conjunction on a microstructure feature set is not. | Add `_check_frictionless_microstructure`: FATAL (or WARN + `allow_frictionless` opt-in, mirroring `data.allow_oracle_features`) when `env.trading_fees == 0` **and** `env.execution_price == "mid"` **and** `env.exec_latency_ticks == 0` **and** any of `microprice*`/`ofi*`/`*imbalance*`/`book_pressure*` is in `env.feature_columns`. Verify current gap: `grep -n "execution_price\|exec_latency" src/trading_rl/config_guardrails_checks.py` → no matches. |
| 19 | MEDIUM | The training reward is computed in the same frictionless world it is scored in: `DifferentialSharpe.calculate` reads `env.broker.net_liquidation_value()`, marked at the mid, with `BrokerFees(proportional=0.0)`. The agent was optimised *for* the artifact. | `src/trading_rl/rewards/differential_sharpe.py:179`; `src/trading_rl/envs/tradingenvxy_wrapper.py:714` | Not leakage — train/eval are consistent — but consistently unrealistic. Simply re-scoring the existing checkpoint under `bid_ask` will understate a fee-aware agent; the fee/spread has to be in the *training* env, which is what h3's fee ladder is for. Any h1 claim about "algorithm ranking" is a ranking under a cost model no algorithm was asked to respect. | `uv run python src/cli.py validate config -c pooled/td3_hft_lob_state_space_pooled_streaming_selected_dsr \| grep -A3 "trading_fees\|execution_price"` |
| 20 | MEDIUM | Rollout parquet stores `action` and `simple_return` off by one row: `simple_return[i]` belongs to `action[i-1]` (established by exact lag identification, corr=1.000000). The writer zips `last_positions[:n]` and `simple_returns[:n]` positionally with no shift. The last action is never credited a return; `simple_return[0] == 0.0`. | `src/trading_rl/callbacks/artifacts_evaluation.py:103-114`; series originate at `src/trading_rl/pipeline/evaluation.py:404,519` | Harmless for the current metric set (actions are only used for marginal stats: turnover, holding period, pct_long/short, n_trades). But any action-conditional analysis, plot overlay, or reviewer pairing the two columns row-wise gets a one-tick-shifted result — `corr(action[i], simple_return[i])` reads 0.25 instead of the true 1.00. | `uv run python -c "import pandas as pd,numpy as np;d=pd.read_parquet('data/prepared/pooled_daily_6sym_selected/test_AAPL_prepared.parquet',columns=['close']).close.values;r=pd.read_parquet('logs/td3_hft_lob_state_space_pooled_streaming_selected_dsr/evaluation_data/test_rollout.parquet');g=d[1:]/d[:-1]-1;a=r.action.values;s=r.simple_return.values;print(np.corrcoef(s[1:],a[:-1]*g[:-1])[0,1])"` → 1.0. Fix: shift `last_positions` by one before writing, or rename the column `position_after_step` and document the convention. |
| 21 | MEDIUM | No execution mode in the codebase models finite depth. Even `execution_price: bid_ask` hardcodes `bid_size=np.inf, ask_size=np.inf`, so there is no queue position, no partial fill, no market impact at any setting. With gross turnover 289,579x NLV in one 4h07m session and 450,651 threshold-crossing trades, a $10,000 account (`DEFAULT_INITIAL_PORTFOLIO_VALUE`, `config.py:181`) transacts ~$2.9bn of AAPL — roughly a third of AAPL's entire daily notional. | `src/trading_rl/envs/tradingenvxy_wrapper.py:455-456` and `:736-737`; `src/trading_rl/config.py:181` | Switching to `bid_ask` fixes the spread but still publishes an infinite-capacity result. The thesis needs an explicit capacity/impact caveat, or a size-aware fill. | `grep -n "np.inf" src/trading_rl/envs/tradingenvxy_wrapper.py` |
| 22 | MEDIUM | Every artefact currently in `logs/td3_..._dsr/` was produced at commit `58834b24`, i.e. **before** the metrics fix `f117b210`. `benchmark_tables/test_benchmark_table.json` still carries `Strategy win_rate 1.0, max_drawdown 0.0, profit_factor NaN`. The new stale-export guard in `export_eval_to_thesis.py` compares `results.json` mtime against checkpoints only — it does not compare the artefact's recorded `commit` against HEAD. | `logs/.../benchmark_tables/test_benchmark_table.json.meta.json` (`"commit": "58834b24..."`); `src/trading_rl/evaluation/benchmark_table.py:89-102` (correctly calls `build_metric_report`, so a re-run fixes it) | Exporting today would ship the pre-fix `win_rate = 1.0` into the thesis despite the metrics fix having landed. | `for f in logs/td3_hft_lob_state_space_pooled_streaming_selected_dsr/**/*.meta.json; do python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['commit'])" $f; done \| sort -u` — compare against `git rev-parse HEAD`. Fix: have the exporter refuse when any consumed artefact's `meta.json` `commit` is not an ancestor-or-equal of the commit that last touched `src/trading_rl/evaluation/metrics.py`. |
| 23 | MEDIUM | h1 has no comparable evidence for 3 of 4 arms, and the three chapters that consume h1 now load nothing. `logs/ddpg_..._dsr/evaluation_data/` has train+val only (no test); `logs/ppo_..._dsr/evaluation_data/` is empty; `random` has no log dir at all. `thesis/qmd/results/pooled_td3_..._dsr/` is an empty directory after the #2/#6/#11 cleanup, and `pooled_{ddpg,ppo,random}_..._dsr/` do not exist. | `thesis/qmd/src/06-00-results.qmd:72,96-99`; `06-02-robustness-assessment.qmd:18`; `06-03-performance-evaluation.qmd:23` | Deleting the stale snapshots was right, but nothing was re-exported, so `load_experiment_snapshot("pooled_td3_..._dsr")` has no `manifest.json`. Every h1 table in ch. 6 is now data-less. | `ls thesis/qmd/results/pooled_td3_hft_lob_state_space_pooled_streaming_selected_dsr/` → only an empty `latest_finished/`. Resolve by a full h1 re-run, not by restoring the old snapshots. |
| 24 | LOW | Config drift survives in h1: PPO's `training.loss_function: l2` vs `smooth_l1` for TD3/DDPG. `init_rand_steps: 0` and `save_buffer: false` are on-policy necessities; the critic loss function is not. | `src/configs/scenarios/pooled/ppo_..._dsr/train.yaml:83` vs `td3_..._dsr/train.yaml:83` | h1's invariant is that only `training.algorithm` and its sub-block differ. A different value-loss shape is a second factor in a single-factor comparison. | `diff -u src/configs/scenarios/pooled/td3_hft_lob_state_space_pooled_streaming_selected_dsr/train.yaml src/configs/scenarios/pooled/ppo_hft_lob_state_space_pooled_streaming_selected_dsr/train.yaml` |
| 25 | LOW | Zero-cost execution is a subsidy proportional to turnover, so the benchmark table is structurally biased toward the RL arm. `buy_and_hold`/`twap`/`vwap` have ~zero turnover and gain nothing from `trading_fees: 0.0`; the strategy turns over 289,579x NLV and gains 12.16 NLV. The `alpha ≈ 472` / `information_ratio ≈ 1552` in every benchmark row are downstream of the same subsidy plus the 98,280-ppy annualisation of a 4-hour window. | `logs/.../benchmark_tables/test_benchmark_table.json`; `src/trading_rl/evaluation/benchmark_table.py` | "Strategy beats buy_and_hold" is not a like-for-like statement when the comparison's only asymmetry is the cost the winner does not pay. | Report a turnover column next to every benchmark row, and a net-of-half-spread return. Route the alpha/IR annualisation to `evaluation-metrics`. |

Guardrail floor: `uv run python src/cli.py validate guardrails -c pooled/<arm>_hft_lob_state_space_pooled_streaming_selected_dsr` reports "Guardrails passed — no issues found" for td3/ddpg/ppo and 2 WARNs for random (prior #13). Findings 16-25 are all invisible to the catalogue — and per #17 five checks are not even running.

## Recomputed rollout numbers (job c)

`build_metric_report(simple_return, None, action, periods_per_year=98280)` at HEAD
77406e78, straight off the parquet. Only TD3 has a test rollout; PPO has no
evaluation data at all.

| metric | TD3 test | TD3 val | TD3 train | DDPG train |
|---|---|---|---|---|
| n steps | 969,218 | 969,218 | 49,999 | 49,999 |
| total_return | **+2.2696** | +2.2463 | +0.1306 | +0.1810 |
| win_rate (raw per-step) | **0.2550** | 0.2535 | 0.2944 | 0.2666 |
| lose_rate | 0.1924 | 0.1980 | 0.2184 | 0.1663 |
| flat steps | 55.3% | 54.8% | 48.7% | 56.7% |
| hit rate among non-flat | **0.570** | 0.562 | 0.574 | 0.616 |
| payoff_ratio | 2.504 | 1.934 | 1.613 | 1.743 |
| profit_factor | 3.319 | 2.477 | 2.174 | 2.795 |
| max_drawdown | **-0.00182** | -0.00417 | -0.00083 | -0.00051 |
| sharpe_ratio (per bar) | 2.840 | 2.109 | 1.582 | 2.231 |
| sharpe_ratio_annualized | 45.08 | 33.48 | 25.11 | 35.41 |
| sortino_ratio_annualized | **2386.7** | 319.2 | nan | nan |
| n_trades (\|Δpos\|>1e-2) | 450,651 | 414,971 | 21,397 | 17,753 |
| turnover (mean \|Δpos\|) | 0.2988 | 0.2730 | 0.2775 | 0.3278 |
| average_holding_period | 2.151 | 2.336 | 2.337 | 2.816 |
| pct_long / pct_short | 61.1 / 38.9 | 59.9 / 40.1 | 54.1 / 45.9 | 50.5 / 49.5 |
| expectancy_per_period | 1.22e-06 | 1.21e-06 | 2.46e-06 | 3.33e-06 |

**Does the fix change the picture? No.** The metrics are now honest — `win_rate`
1.00 → 0.255, `max_drawdown` 0.0 → -0.18%, `profit_factor` NaN → 3.32 — and the
57.0% conditional hit rate is a plausible, unremarkable microstructure number. But
the *result* is unchanged and still implausible: **+227% with a 0.18% maximum
drawdown** is a Calmar in the thousands. Removing the win-rate distortion moved the
headline not at all, because the distortion was in the *reporting* of a return
series that was itself the artifact. The remaining implausibility is entirely
finding #16 — a 10.3x cost subsidy — plus prior #10 (a single 4-hour session
annualised at 98,280 ppy). Fixing the metrics was necessary and did not touch the
cause.

## Minimum fix before re-running

1. Repair the five crashing guardrails (#17) and make `check_config_guardrails`
   surface swallowed exceptions instead of logging them past a "passed" banner.
   Everything below is unverifiable until the floor is real again.
2. Decide the cost model (#16). At minimum `execution_price: bid_ask` for the
   headline h1 arms; `trading_fees` is a separate, additive decision. Expect the
   current policy to be deeply unprofitable — that is the honest result, and it is
   publishable as one.
3. Add the frictionless-combination guardrail (#18) with an explicit
   `env.allow_frictionless` opt-in, so a zero-cost ablation stays possible but
   cannot be reported by accident.
4. Retrain under the chosen cost model — do not re-score the existing checkpoint
   (#19). Then re-evaluate all four arms so DDPG/PPO/random actually have test
   rollouts (#23).
5. Fix or rename the rollout `action`/`simple_return` alignment (#20).
6. Teach the exporter to reject artefacts whose `meta.json` `commit` predates the
   last change to `metrics.py` (#22), then re-export; the current `logs/` artefacts
   are all pre-fix.
7. Align PPO's `loss_function` with TD3/DDPG or record it as a declared,
   confounding deviation (#24).
8. Add turnover and net-of-half-spread columns to the benchmark table (#25).
