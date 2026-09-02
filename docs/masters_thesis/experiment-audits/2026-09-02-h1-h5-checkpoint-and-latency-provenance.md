# Experiment audit — H1-H5 checkpoint provenance, H5 latency-fix staleness, drawdown/Sortino collapse

**Command audited:** static audit + targeted reads/queries against `mlflow.db` (read-only),
`logs/`, and `thesis/qmd/results/`. No training or evaluation was run.
**Date/time:** 2026-09-02 ~12:00 UTC
**Auditor:** experiment-auditor (claude-sonnet-5)
**git HEAD:** d71516c7
**Scope:** resumed from a prior session that hit its API session limit before writing up.
Coordinator gave three priority leads plus a budget-permitting recheck of H1-H4 single-factor
invariants. This file covers all four.

## Verdict

All three priority leads confirm real, currently-live defects, not stale artifacts of a
already-fixed problem:

1. **Every exported H5 (execution-latency) snapshot predates the fix that made evaluation
   consult latency at all.** All five `logs/td3_h5_latency_*/results.json` mtimes fall in a
   03:36-09:24 UTC window on 2026-09-02; the fix (`abb158fb`, then refined by `1bbaf94d`)
   landed at 10:25 and 10:49 UTC the same day. Every currently-published H5 number was
   evaluated with `exec_latency_us` silently at zero regardless of the configured value, so
   the ladder measures nothing about latency. It is not yet read by any `.qmd` chapter, so
   nothing in the built PDF is corrupted today, but the artifacts sitting in
   `thesis/qmd/results/pooled_td3_h5_latency_{10us,5ms}_dsr/` are ready to be cited as-is by
   a future writing pass and would be wrong if they were.

2. **`max_drawdown == 0.0` / `sortino_ratio == NaN` in 10 of 12 agent rows is a real,
   traced computation collapse**, not a data artifact: `aggregate_to_reporting_frequency`
   (`metrics.py:50-90`) compounds tick returns into the coarsest bar size that still yields
   ≥50 observations — a fix for Sharpe zero-inflation — and for a strategy whose bar-level
   return is (per the existing frictionless-execution finding) almost always positive, that
   compounding produces a bar series with **zero negative bars**. A monotonic equity curve has
   `max_drawdown` exactly 0.0 by construction, and downside deviation exactly 0.0, which
   `_safe_div` (`metrics.py:233-236`) maps to `NaN` rather than `+inf`. Confirmed on the exact
   run (`b5e952238b5a4d0d84b5cc1bc05dc648`): 10/12 splits flip this way, and the two that don't
   (`val_META`, `test_META`) are exactly the two with the smallest edge.

3. **Checkpoint auto-selection by mtime (`evaluate_command.py:832`) is not a latent risk — it
   has already picked the wrong checkpoint once, on disk, right now**, in a directory
   structurally identical to what a bare re-run of the H1 primary scenario would use. Two
   findings compound: (a) `experiments.py`'s `_evaluate_all` never passes `--checkpoint`
   (`experiments.py:507-536`), so every orchestrated eval — not just manual ones — depends on
   mtime auto-selection; (b) the `logging.log_dir` it points at is a fixed, scenario-keyed path
   with no run/timestamp component (`experiments.py:485,512`), so a second orchestrator
   invocation on the same scenario silently commingles checkpoints from unrelated training runs
   in one directory. This already happened: `logs/td3_hft_lob_state_space_pooled_streaming_selected_dsr/`
   holds a finished 3M-step run from 2026-08-31 *and* a later, separate, 200k-step partial rerun
   from 2026-09-01, and the partial rerun's checkpoint is now the mtime-newest file in the
   directory. The currently-published snapshot is safe only because the human who produced it
   passed `--checkpoint .../checkpoint_step_3000000.pt` explicitly (confirmed via the
   `eval_checkpoint` MLflow param on run `b5e952238b5a4d0d84b5cc1bc05dc648`) — nothing in the
   code enforces that, and a bare `evaluate` on this scenario today would not do it. The default
   (un-overridden) `logging.log_dir` for this scenario is a *third*, even more dangerous
   directory (`logs/pooled_td3_hft_lob_state_space_pooled_streaming_selected_dsr/`) whose
   mtime-newest checkpoint is tied with an **interrupted** run that reached only 703,200 of its
   3,000,000-step target (23%).

Budget-permitting recheck of the H1-H4 single-factor invariants: **H2, H3, H4 are clean** —
each varies only its stated factor, confirmed by empty/expected diffs. **H1 has one residual,
previously-flagged confound** (PPO's `training.loss_function: l2` vs `smooth_l1` for
TD3/DDPG, LOW) and is otherwise clean; three previously-CRITICAL H1 findings (wrong scenario
exported, mismatched test windows across arms, evaluate.yaml benchmark-set drift) are now
fixed and verified by diff/count, not just by re-reading the fix commit.

## Findings

| # | Severity | Issue | Where | Why it corrupts the result | Command to verify or fix |
|---|---|---|---|---|---|
| 1 | CRITICAL | Every exported H5 (execution-latency) snapshot's `results.json` predates the commit that made the evaluation path apply latency at all. `results_file_mtime_utc` for all five arms: `0us` 03:36, `10us` 02:46, `100us` 05:24, `1ms` 07:23, `5ms` 09:24 UTC on 2026-09-02. The fix (`abb158fb` "Apply execution latency in the evaluation path", refined by `1bbaf94d` "Fix latency resolution") landed at 10:25 and 10:49 UTC the same day. | `logs/td3_h5_latency_{0,10us,100us,1ms,5ms}_dsr/results.json` mtimes vs `git show -s --format=%cI abb158fb 1bbaf94d`; commit message `fdf74a72`/`abb158fb`: "Only StreamingTradingEnvXY._build_inner_env (training) implemented the shift ... pipeline/evaluation.py ... never consulted [latency] ... every latency scenario was therefore scored at zero delay" | Training correctly applied per-arm latency (each policy is genuinely trained differently), but evaluation scored every arm at zero latency uniformly — the exported "sensitivity to latency" is actually "policies trained at latency k, scored as if k=0". This is not a magnitude error, it is a different, unlabelled experiment. Only 2 of 5 arms (`10us`, `5ms`) even reached `thesis/qmd/results/` with a `manifest.json`/`run.json`; the other three (`0`, `100us`, `1ms`) have leftover `latest_finished/` files with no `run.json`, i.e. an incomplete prior export attempt. | `stat -f "%Sm %N" logs/td3_h5_latency_*/results.json`; re-run `uv run thesis-experiments h5 --skip-train` (train checkpoints already reflect correct per-arm latency; only eval needs to rerun) once satisfied the fix is stable, then re-export. Not yet cited by any `.qmd` chapter (`grep -rn "h5_latency\|latency_5ms\|latency_10us" thesis/qmd/src/*.qmd` → no hits), so nothing published is wrong today — this is a landmine for the next writing pass, not yet a fired one. |
| 2 | HIGH | `max_drawdown == 0.0` and `sortino_ratio == NaN` in 10 of 12 per-symbol-split `benchmark_comparison_table` "agent" rows for run `b5e952238b5a4d0d84b5cc1bc05dc648` (the currently-exported H1 primary snapshot). Mechanism: `aggregate_to_reporting_frequency` (`metrics.py:50-90`) compounds tick returns to the coarsest bar with ≥50 observations to fix Sharpe zero-inflation; for a strategy whose per-bar drift is almost always positive (the established waived-half-spread mechanism), the compounded series has zero negative bars, so `_max_drawdown` (`statistical_benchmarks.py:164-171`, `running_max == equity` everywhere) returns exactly 0.0, downside deviation is exactly 0.0, and `sortino_raw` → `_safe_div(mu_excess, 0.0)` (`metrics.py:233-236`) explicitly returns `NaN` rather than `+inf`. | `mlruns/21/b5e952238b5a4d0d84b5cc1bc05dc648/artifacts/statistical_tests/{split}/*.json` → `benchmark_comparison_table`; `src/trading_rl/evaluation/statistical_benchmarks.py:164-232`; `src/trading_rl/evaluation/metrics.py:50-90,233-236` | These two fields are not merely NaN/zero by omission — they are computed and wrong. A max drawdown of exactly 0.0 on a strategy trading ~3.9M ticks is not a risk statement, it is an artifact of choosing a bar size for a different metric (Sharpe) and applying it to a metric (drawdown) whose entire content is intra-window variation, which that bar size destroys for this return profile. Any thesis table or prose citing "zero drawdown" as a risk-management strength is citing this collapse, not a property of the strategy. `val_META`/`test_META` are the only two splits where the collapse does not fire — exactly the two with the smallest measured edge, which is consistent with (not independent confirmation of, but consistent with) the mechanism. | `python3 -c "import json; d=json.load(open('mlruns/21/b5e952238b5a4d0d84b5cc1bc05dc648/artifacts/statistical_tests/test_AAPL/<file>.json')); print([r for r in d['benchmark_comparison_table'] if r['strategy']=='agent'])"` (filename is a random `tmp*.json`, glob for it). Fix is `evaluation-metrics` territory: either report drawdown/Sortino on the raw tick series (not the Sharpe-oriented aggregated one), or explicitly gate/annotate when the aggregated series has zero downside observations rather than silently returning a "clean" 0.0/NaN pair that reads as good risk metrics. |
| 3 | CRITICAL | Checkpoint auto-selection (`evaluate_command.py:819-834`, `max(matches, key=mtime)` over `log_dir.rglob("*_checkpoint*.pt")`) has already selected the wrong checkpoint on disk, twice, for the same scenario, and nothing in the orchestrator or the config layer prevents it from happening again on the next re-run. Three compounding facts: **(a)** `experiments.py`'s `_evaluate_all._cmd()` (`experiments.py:507-536`) never constructs a `--checkpoint` flag — every orchestrated `thesis-experiments hN` eval step relies on auto-selection, full stop. **(b)** `_train_all`/`_evaluate_all` set `logging.log_dir=<output_root>/<scenario_name>` (`experiments.py:485,512`) — a fixed, scenario-keyed path with no run-id or timestamp component, so a second orchestrator invocation on the same scenario writes into the identical directory as the first. **(c)** This has already happened for the H1 primary scenario: `logs/td3_hft_lob_state_space_pooled_streaming_selected_dsr/` holds a finished 3,000,000-step run (mtimes ending 2026-08-31 23:08-23:09) *and* a later, separate, incomplete rerun that only reached step 200,000 (mtimes 2026-09-01 09:46-10:05) — and the incomplete rerun's `..._checkpoint_step_100000.pt` is now the mtime-newest file in the directory (2026-09-01 10:05:05, vs the finished run's last file at 23:09:49 the day before). | `src/cli/commands/evaluate_command.py:819-834`; `src/masters_thesis/experiments.py:485,507-536,512`; confirmed with `find logs/td3_hft_lob_state_space_pooled_streaming_selected_dsr -iname "*_checkpoint*.pt" -exec stat -f "%m %Sm %N" {} \; \| sort -n` | A bare `evaluate -c pooled/td3_hft_lob_state_space_pooled_streaming_selected_dsr` (no `--checkpoint`) run today, or an orchestrated `thesis-experiments h1 --skip-train` re-eval, would silently score a checkpoint from a 200k-step (6.7% of budget) aborted run and report it as the H1 result, with no error, no warning distinguishable from a normal run — `_resolve_checkpoint` only prints `[dim]Auto-selected checkpoint: <path>[/dim]`, which is easy to miss in a long log. The currently-published snapshot escaped this only because the eval run that produced it (`b5e952238b5a4d0d84b5cc1bc05dc648`) logged `eval_checkpoint=.../checkpoint_step_3000000.pt` as an MLflow param — i.e., a human passed `--checkpoint` by hand outside the orchestrator. | `sqlite3 "file:mlflow.db?mode=ro" "select value from params where run_uuid='b5e952238b5a4d0d84b5cc1bc05dc648' and key='eval_checkpoint';"` confirms the safe path was used manually. Fix: make `_resolve_checkpoint` prefer the highest `_checkpoint_step_N` over any run-named file when both exist and disagree, or fail loudly when the auto-selected file is not the numerically-highest step available; give `experiments.py` a per-invocation `logging.log_dir` (or move/archive prior checkpoints before a rerun) so two runs of the same scenario cannot commingle. |
| 4 | HIGH | The *default* (un-overridden) `logging.log_dir` for the H1 primary scenario — what any bare CLI invocation without `experiments.py`'s override resolves to — is `logs/pooled_td3_hft_lob_state_space_pooled_streaming_selected_dsr/` (derived by `_apply_derived_defaults`, `src/trading_rl/config.py:65-78`, since `train.yaml`'s `logging:` block sets only `log_level`/`save_plots`, never `log_dir`). This directory's mtime-newest checkpoint is a **tie** between `opal-harbor-papa_checkpoint.pt` and `..._checkpoint_interrupt_step_703200_20260829_112951.pt` (both written at 2026-08-29 12:29:51, same second) — an **interrupted** run that reached only 703,200 of a 3,000,000-step target (23.4%). The same directory also contains a genuinely finished `checkpoint_step_3000000.pt` from an earlier, separate run (2026-08-11), which is far older by mtime and would not be auto-selected. | `src/trading_rl/config.py:45-78` (`_derive_experiment_name`, `_apply_derived_defaults`); `src/configs/scenarios/pooled/td3_hft_lob_state_space_pooled_streaming_selected_dsr/train.yaml:100-102` (`logging:` has no `log_dir` key); `logs/pooled_td3_hft_lob_state_space_pooled_streaming_selected_dsr/` full checkpoint listing (48 files, 2026-07-28 through 2026-08-29) | This is the exact mechanism the coordinator's seed observation named (`opal-harbor-papa_checkpoint.pt` instead of `checkpoint_step_3000000.pt`), now traced to its root cause and shown to point at an *interrupted*, not merely *different*, checkpoint. Python's `max()` with a key function breaks exact-mtime ties by iteration order of `Path.rglob`, which is filesystem-dependent, not a documented guarantee — so which of the two same-second files wins is not even deterministic across filesystems/OS. | `find logs/pooled_td3_hft_lob_state_space_pooled_streaming_selected_dsr -iname "*_checkpoint*.pt" -exec stat -f "%m %Sm %N" {} \; \| sort -n \| tail -3`. Any manual reproduction command that omits both `--checkpoint` and the orchestrator's `logging.log_dir` override lands here. Fix: same as #3, plus consider having `_apply_derived_defaults` warn (or refuse) when a derived `log_dir` already contains checkpoints from more than one distinct MLflow run/experiment_name pairing. |
| 5 | LOW | Confirmed still open from `2026-08-31-h1-followup-frictionless-mechanism.md` finding #24: `training.loss_function: l2` for PPO vs `smooth_l1` for TD3/DDPG in the current H1 scenario set. | `diff -u src/configs/scenarios/pooled/td3_hft_lob_state_space_pooled_streaming_selected_dsr/train.yaml src/configs/scenarios/pooled/ppo_hft_lob_state_space_pooled_streaming_selected_dsr/train.yaml` → `-loss_function: smooth_l1` / `+loss_function: l2`, outside the `td3:`/`ppo:` algo-specific sub-block | H1's stated invariant is "only `training.algorithm` and its algo-specific sub-block may differ." `loss_function` is a top-level `training.*` key, not inside `ppo:`, so it is a second varying factor. Plausibly a legitimate on/off-policy convention (PPO value loss is conventionally MSE-family) rather than an oversight, but it is not documented as a deliberate, accepted deviation anywhere in the scenario or the thesis. | `diff -u` command above. Either set PPO's `loss_function` to `smooth_l1` to match, or add a one-line comment in the scenario (and ideally 06-*.qmd) stating this is an accepted algorithm-appropriate deviation, matching how `init_rand_steps`/`save_buffer`/`tau` are already justified as on-policy necessities in the same file. |
| 6 | MEDIUM | The exporter still has no commit/code-provenance gate (open since `2026-08-31-h1-followup-frictionless-mechanism.md` finding #22: "the current stale-export guard compares `results.json` mtime against checkpoints only — it does not compare the artefact's recorded commit against HEAD"). This is the same class of gap that let finding #1 (H5 pre-fix exports) happen: the exporter has no way to know that `results.json` was produced by code that predates a semantically load-bearing fix. | `scripts/export_eval_to_thesis.py` — no reference to `git rev-parse HEAD` or a commit-range check anywhere in the export path; confirmed by `grep -n "commit\|rev-parse" scripts/export_eval_to_thesis.py` returning no gating logic (only pass-through metadata fields, if any) | Every future fix to `metrics.py`, `statistical_benchmarks.py`, or the evaluation env (latency, fees, execution) is silently invisible to the exporter — a stale `results.json` from before the fix will export and overwrite a snapshot with `status: EXPORTED` and no indication anything is wrong, exactly as happened for H5 here (finding #1) and as documented for the pre-metrics-fix H1 win_rate artifact in the prior audit. | `grep -n "commit\|rev-parse\|HEAD" scripts/export_eval_to_thesis.py`. Fix: record `git rev-parse HEAD` (or the last commit touching `src/trading_rl/evaluation/` and `src/trading_rl/envs/`) into each `results.json`/`run.json`, and have the exporter warn or refuse when that commit is not a descendant of the last change to the code paths that computed the artefact. |
| 7 | MEDIUM (not re-verified this round, carried forward) | Rollout-parquet `action`/`simple_return` off-by-one (`2026-08-31-h1-followup-frictionless-mechanism.md` finding #20) was not re-checked this session — budget was spent on the three priority leads. Flagging so it is not silently dropped from the tracked list. | `src/trading_rl/callbacks/artifacts_evaluation.py:103-114` | Unchanged risk assessment from the prior audit: harmless for the current metric set, wrong for any action-conditional analysis pairing the two columns row-wise. | Re-run the lag-identification check from the prior audit's finding #20 command against a current rollout parquet. |

## Prior findings re-checked

Scope note: the hypothesis renumbering (`43036386`, 2026-09-01) and the scenario rewrite
(`8a8736ef` "Give every hypothesis its own scenarios") mean most of the *scenarios* named in
the two prior `experiment-auditor` files no longer exist under those names. Findings are
re-checked against their **current equivalent** (same role: H1 primary TD3-DSR scenario, or
the corresponding new H2/H3/H4 scenario set) where one exists.

From `2026-08-31-h1-win-rate.md`:

- **#1** (`win_rate` = compounded-bar sign, not hit rate) — **still open at the mechanism
  level**. `aggregate_to_reporting_frequency` remains the underlying cause and now also drives
  this session's finding #2 (max_drawdown/Sortino collapse) — same root cause, two symptoms.
- **#2** (published TD3 arm was the wrong, non-`_dsr` scenario) — **fixed**. `_H1_SCENARIOS`
  and `full.yaml` both list the four `_dsr` scenarios consistently; the exported
  `pooled_td3_hft_lob_state_space_pooled_streaming_selected_dsr` snapshot matches.
- **#3** (three arms evaluated on different test windows/symbols under `eval_symbol_selection:
  rotated`) — **fixed, verified by count**. DDPG, PPO, and Random exported snapshots now carry
  identical `n_steps` for all six symbols (val/test), matching each other exactly (e.g.
  `val_AMZN: 1128425` on all three).
- **#4** (frictionless, zero-latency, zero-fee execution drives the headline result) —
  **unchanged as an environment characteristic; thesis narrative has been corrected to match
  it** (`b2b71527` "Reframe H1 as a cost-bounded signal finding", `1bce42e0` "H1: drop the
  algorithm-ranking claim", `8186b58e` "State that the H2 fee ladder already prices spread
  crossing"). Not this agent's mechanism to re-derive; owned by quant-bias-auditor, which
  re-confirmed it is architecturally sound this session (`2026-09-02-quant-bias-followup.md`).
- **#5** (run aborted mid-training, DDPG crashed on a deleted MLflow experiment) — **fixed**.
  All four H1 arms now have complete, matching per-symbol/per-split exports.
- **#6** (exporter silently falls back to a stale log dir and stamps `FINISHED`) — **still
  open**, and now shown to also apply going forward via commit-provenance (finding #6 above).
- **#7** (`actor_weight_decay` 0.0 for TD3 vs 2.0e-06 for DDPG/PPO) — **fixed**. Current diff
  shows no weight-decay difference between TD3 and DDPG; both carry `2.0e-06`.
- **#8** (TD3's benchmark set in `evaluate.yaml` omits `random_actions`; others include it) —
  **fixed**. `evaluate.yaml` is now byte-identical across all four H1 arms.
- **#9** (`statistical_tests.json` has no actual statistical test; `"stats"` missing from
  `_EVAL_ONLY["h1"]`) — **fixed**, and the fix comment in `experiments.py:166-168` explicitly
  cites this audit finding.
- **#10** (test = afternoon half of the same session as val) — **unchanged**, confirmed
  independently by quant-bias-auditor this session with a fresh row-level check
  (`2026-09-02-quant-bias-followup.md` item 4). Documented limitation, not a leakage defect.
- **#11** (`run.json` timestamp collapse: `start==end==exported_at`) — **improved, not fully
  fixed**. Current `run.json` files carry a real, distinct `end_time` (from `results_file_mtime_utc`)
  separate from `exported_at_utc`; `start_time` remains `null` on every export checked this
  session.
- **#12** (`train_size: 50000` vs val/test ~969k) — **unchanged**, now the accepted, documented
  "cost-bounded signal" framing rather than an unstated defect; not re-litigated.
- **#13** (random-arm guardrail WARN, `init_rand_steps > max_steps`) — **not re-verified this
  session** (LOW, budget-limited).
- **#14** (`n_trades` counts sub-threshold action changes) — **not re-verified this session**
  (evaluation-metrics territory).
- **#15** (`sharpe_ratio` per-bar vs `annualized_volatility` mixed in one table) — **superseded**
  by `ee5d8ab4` "Metrics: gate annualised ratios on window length, explain zero-downside
  Sortino", which introduced the aggregation-to-reporting-frequency mechanism. That mechanism
  fixed the mixed-scale issue but is the direct cause of this session's finding #2 — closing one
  metrics defect opened an adjacent one on the same code path.

From `2026-08-31-h1-followup-frictionless-mechanism.md`:

- **#16** (the whole edge is the waived half-spread) — unchanged mechanism; thesis narrative
  reframed (see #4 above). Owned by quant-bias-auditor.
- **#17** (`train_size: null` silently crashed 5 guardrail checks past a "passed" banner) —
  **fixed**. `train_size: null` is still present in the current scenario, but
  `validate guardrails -c pooled/td3_hft_lob_state_space_pooled_streaming_selected_dsr --verbose`
  now produces zero `"failed unexpectedly"` lines — the individual checks were repaired to
  resolve `train_size=None` rather than crash. The swallow-and-log-past-a-passed-banner pattern
  itself (`config_guardrails_checks.py:1970-1972`) is still present as a design, just no longer
  triggered here.
- **#18** (no guardrail for the frictionless combination) — **fixed**.
  `_check_frictionless_microstructure` is now registered in the guardrail catalogue
  (`config_guardrails_checks.py:1901`), and the fee-ladder scenarios explicitly opt in via
  `env.allow_frictionless: true` with a comment naming the check, exactly as the prior audit
  recommended.
- **#19** (training reward computed in the same frictionless world it is scored in) —
  unchanged, inherent to the current design; not a code defect to "fix" independently of #16.
- **#20** (rollout `action`/`simple_return` off-by-one) — **not re-verified this session**,
  carried forward as finding #7 above.
- **#21** (no market depth/impact model, even `bid_ask` execution) — unchanged, structural
  limitation; not re-litigated.
- **#22** (exported artefacts predate the metrics fix, no commit-gating in the exporter) —
  **still open**, and now shown to be live: it is the exact mechanism behind this session's
  finding #1 (H5 exports predate the latency fix) and finding #6.
- **#23** (H1 has no comparable evidence for 3 of 4 arms) — **fixed**, see #5 above.
- **#24** (PPO `loss_function: l2` vs TD3/DDPG `smooth_l1`) — **still open**, re-verified this
  session as finding #5 above.
- **#25** (zero-cost turnover subsidy biases the benchmark table) — unchanged; evaluation-metrics
  / quant-bias-auditor territory.

## Minimum fix before re-running

1. Do not export or cite any H5 (execution-latency) snapshot until it has been re-evaluated
   after `1bbaf94d` (finding #1). Training checkpoints already reflect correct per-arm latency;
   only the evaluate step needs to rerun.
2. Before any bare (non-orchestrated) `evaluate` call on a scenario whose log directory may
   have been reused across runs, pass `--checkpoint` explicitly and verify it is the
   highest-step, non-interrupt file (findings #3, #4). This applies immediately to the H1
   primary TD3-DSR scenario in both of its candidate directories.
3. Give `_resolve_checkpoint` (`evaluate_command.py:819-834`) a step-aware tiebreak: prefer the
   numerically highest `_checkpoint_step_N` over any run-named or interrupt file when they
   disagree, and warn loudly (not just a `[dim]` print) when the selected file is an interrupt
   checkpoint (finding #3, #4).
4. Give `experiments.py` per-invocation output isolation (timestamped or run-id-suffixed
   `logging.log_dir`) so a second orchestrator run on the same scenario cannot commingle
   checkpoints from an unrelated prior run (finding #3).
5. Add commit-provenance gating to `scripts/export_eval_to_thesis.py` so a `results.json`
   produced before a semantically load-bearing fix (metrics, latency, fees, execution) cannot
   silently export as `FINISHED`/`EXPORTED` (finding #6).
6. Route finding #2 (max_drawdown/Sortino collapse under bar aggregation) to
   `evaluation-metrics` for a formula-level fix or an explicit annotation when the aggregated
   series has zero downside observations.
7. Either align PPO's `loss_function` with TD3/DDPG or document it as an accepted
   algorithm-appropriate deviation (finding #5).
