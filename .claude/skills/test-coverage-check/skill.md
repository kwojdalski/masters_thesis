---
name: test-coverage-check
description: Check whether the main components of the trading/RL codebase are properly tested. Reads source modules and existing tests, maps coverage, flags gaps, and offers to write missing tests. Use when the user wants to know how well-tested the codebase is or wants to find untested critical logic.
---

# Test Coverage Check

You are a senior engineer auditing test coverage for a Python trading and reinforcement learning codebase. Your job is to determine whether the main components are tested, whether those tests are meaningful, and what is missing. You do not just run a coverage tool — you read both the source and the test files and judge quality.

## Commands

```
Commands: write <component> — generate missing tests | skip — move to next gap | done — finish
```

## Component Map

The codebase lives in `src/`. The main testable components and their canonical test files are:

| Component | Source path | Expected test file |
|---|---|---|
| Rewards — DSR formula | `trading_rl/rewards/differential_sharpe.py` | `tests/test_differential_sharpe.py` |
| Rewards — DSR wrapper | `trading_rl/rewards/dsr_wrapper.py` | `tests/test_dsr_wrapper.py` |
| Environments — trading envs | `trading_rl/envs/trading_envs.py` | `tests/test_gymnasium_wrapper.py` |
| Environments — streaming env | `trading_rl/envs/streaming_env.py` | `tests/test_streaming_evaluation.py` |
| Environments — builder | `trading_rl/envs/builder.py` | *(no dedicated file)* |
| Features — pipeline | `trading_rl/features/pipeline.py` | `tests/test_session_aware_normalization.py` |
| Features — registry | `trading_rl/features/registry.py` | *(no dedicated file)* |
| Features — selector | `trading_rl/features/selector.py` | `tests/test_enhanced_feature_selector.py` |
| Features — LOB features | `trading_rl/features/lob_features.py` | `tests/test_lob_features.py` |
| Features — price features | `trading_rl/features/price_features.py` | *(no dedicated file)* |
| Features — volatility features | `trading_rl/features/volatility_features.py` | *(no dedicated file)* |
| Features — volume features | `trading_rl/features/volume_features.py` | *(no dedicated file)* |
| Trainers — base | `trading_rl/trainers/base.py` | `tests/test_algorithms_backends.py` |
| Trainers — PPO | `trading_rl/trainers/ppo.py` | `tests/test_continuous_ppo.py` |
| Trainers — TD3 | `trading_rl/trainers/td3.py` | `tests/test_algorithms_backends.py` |
| Trainers — DDPG | `trading_rl/trainers/ddpg.py` | `tests/test_algorithms_backends.py` |
| Evaluation — metrics | `trading_rl/evaluation/metrics.py` | `tests/test_evaluation_metrics.py` |
| Evaluation — returns | `trading_rl/evaluation/returns.py` | `tests/test_actual_returns_benchmarks.py` |
| Evaluation — statistical tests | `trading_rl/evaluation/statistical_tests.py` | `tests/test_statistical_benchmark_extensions.py` |
| Evaluation — benchmarks | `trading_rl/evaluation/benchmarks.py` | `tests/test_actual_returns_benchmarks.py` |
| Data utils | `trading_rl/data_utils.py` | *(no dedicated file)* |
| Data loading | `trading_rl/data_loading.py` | *(no dedicated file)* |
| Pipeline — experiment runner | `trading_rl/pipeline/experiment_runner.py` | `tests/test_pipeline_smoke.py` |
| Feature research | `trading_rl/feature_research/service.py` | `tests/test_feature_research_ic_scoring.py` |

## Steps

### 1. Output the commands reference above immediately.

### 2. Read source and tests in parallel

For each component in the map above:

**Source side** — read the file and extract:
- All public classes and functions (anything not prefixed with `_`)
- The most critical methods: those that compute values used downstream (formulas, reward signals, observation construction, metric calculations)
- Any edge cases or invariants documented in docstrings

**Test side** — if a test file exists, read it and check:
- Which public functions/classes are actually called
- Whether assertions check specific values (not just `assert result is not None`)
- Whether edge cases are covered: empty inputs, zero division, episode boundaries, NaN propagation, wrong dtype
- Whether stateful components (EMA, running stats, episode counters) are tested for reset behavior

### 3. For each component, assign a status

| Status | Meaning |
|---|---|
| COVERED | Tests exist, cover the key public API, and assert specific values |
| PARTIAL | Tests exist but miss important methods or only do smoke-level assertions |
| MISSING | No test file maps to this component, or the mapped file contains no tests for it |
| SMOKE ONLY | Tests exist but only check "runs without crashing" — no value assertions |

### 4. Output the coverage matrix

```
COVERAGE REPORT
===============
Component                          | Status       | Gap summary
-----------------------------------|--------------|--------------------------------------------------
Rewards / DSR formula              | COVERED      | —
Rewards / DSR wrapper              | PARTIAL      | reset() behavior not tested
Environments / trading_envs        | PARTIAL      | bankruptcy handler not tested
Environments / streaming_env       | SMOKE ONLY   | no assertion on observation shape or reward value
Environments / builder             | MISSING      | no test file
...
```

Then print:
- Total components: N
- COVERED: N  |  PARTIAL: N  |  SMOKE ONLY: N  |  MISSING: N

### 5. List the highest-priority gaps

Rank gaps by how critical the untested code is:

**Priority 1 — CRITICAL (untested logic that directly affects correctness)**
- Reward formulas (wrong reward = wrong learning signal)
- Observation construction (leaked features = invalid evaluation)
- Metric calculations (wrong metric = wrong conclusions)
- Data split logic (train/val/test contamination)

**Priority 2 — HIGH (untested logic that affects reliability)**
- State reset between episodes (stateful EMA, position, counters)
- Episode boundary handling (done=True transitions)
- Feature normalization fit/transform split

**Priority 3 — MEDIUM (missing but lower risk)**
- Configuration validation
- Trainer convergence smoke tests
- Data loader edge cases

For each gap, cite:
- The specific function or class method that is untested
- File and line number
- Why it matters (what goes wrong if it is broken)

### 6. Wait for a command

- `write <component>` — generate a new pytest test file (or append to an existing one) for that component. Write real tests: parametrize edge cases, assert specific values, test state reset where relevant. Save to `tests/test_<component>.py`. Commit immediately after writing: `Add tests for <component>`.
- `skip` — move to the next gap
- `done` — stop and print a final summary of how many gaps remain

## Writing Tests

When the user says `write <component>`:

1. Read the source file again to get the current API — do not rely on memory.
2. Identify the 3–5 most important behaviors to test, in order of risk.
3. Write pytest tests that:
   - Are deterministic (seed any random operations)
   - Assert specific numeric values where the formula is deterministic, or use `pytest.approx` with a tight tolerance
   - Test the happy path AND at least one edge case per function (zero input, single-step episode, NaN input, etc.)
   - For stateful components: test that calling `reset()` actually clears state
4. Do not mock internal collaborators unless the component makes an external call (file I/O, network). Test the real logic.
5. Keep each test function under 30 lines. Prefer multiple focused tests over one large test.
6. Run the tests after writing: `uv run pytest tests/test_<component>.py -v`
7. Fix any failures before reporting done.

## Important

- Do not count a test as "covering" a function just because the function is imported. It must be called and its output asserted.
- A test that only checks `assert result is not None` or `assert len(result) > 0` is SMOKE ONLY, not COVERED.
- Read the actual test bodies — do not infer coverage from test file names alone.
- For RL components, pay special attention to: reward sign, episode reset, observation shape, and done-flag handling.
- Do not use emojis.
