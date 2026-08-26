# The `max_length` / max `step_count` training metric

## What `step_count` is

Every trading environment in this project is wrapped with TorchRL's `StepCounter`
transform, with no `max_steps` argument (so it never forces truncation):

```python
# src/trading_rl/envs/trading_envs.py:134, and identically in
# src/trading_rl/envs/builder.py:365,404,458,509
return TransformedEnv(env, StepCounter())
```

`StepCounter` writes a `("next", "step_count")` entry into every transition: an
integer counter that starts at 0 on `reset()` and increments by one on every
`step()`, until the episode ends (`done`/`terminated`, e.g. reaching the end of
a trading window) and the environment resets. Because no `max_steps` cap is
configured here, it is a pure counter with no truncation side effect — it
exists only to record, for each transition, how far into its episode that
transition occurred.

So `step_count` on any single transition answers: *"how many steps has this
trajectory run for, up to and including this transition?"* The **last**
transition of a completed episode therefore carries that episode's total
length.

## What "max" means

`max_length` (the name used throughout the trainers) is `step_count.max()`
taken over a collection of transitions — the **longest-running episode
represented in that collection**. It is purely a training-progress diagnostic:
it never feeds into a loss, a gradient, or a stopping decision. It exists so a
human watching training logs can see episode-length behavior over time (e.g.
"is the agent surviving longer/shorter trading windows than earlier in
training?").

It shows up in two places in the codebase, each with a different scope:

- **During training**, computed once per collected batch in
  `TrainingLoop.run()` (`src/trading_rl/trainers/training_loop.py`) and passed
  into each trainer's `_optimization_step(batch_idx, max_length, buffer_len)`
  (`td3.py`, `ddpg.py`, `sac.py`, `ppo.py`, `recurrent_ppo.py`). It is only
  ever consumed by `_log_progress(...)`, itself gated by
  `_should_log_step` (`base.py:443-445`, `log_interval=1000` by default) — a
  periodic log line, nothing more.
- **During evaluation**, computed independently from an eval rollout:
  `eval_rollout["step_count"].max().item()`, logged as `eval_step_count`
  (`base.py:557,561`). This is the same underlying signal (longest episode in
  a batch of rollouts) applied to evaluation instead of training data.

## Why the training-time value used to be expensive to compute (and now isn't)

For off-policy algorithms (TD3, DDPG, SAC), training data accumulates in a
persistent `ReplayBuffer` (`base.py:383-388`,
`buffer_size=100_000` by default). The original code recomputed `max_length`
by re-scanning the **entire buffer** on every single collected batch:

```python
max_length = trainer.replay_buffer[:]["next", "step_count"].max()
```

This ran unconditionally on every batch, even though the result is discarded
on all but roughly 1-in-13 to 1-in-20 batches (whenever `_should_log_step`
happens to be false). An investigation into "H4 takes ages / looks like a
memory leak" initially flagged this as the cause, on the theory that indexing
`buffer[:]` forces an O(buffer_size) copy that grows across a run.

That theory did not survive benchmarking. Using the exact TorchRL classes this
project uses (`ReplayBuffer` + `LazyTensorStorage`, torchrl 0.12.0), timing
`buffer[:]["next", "step_count"].max()` while filling a 100,000-transition
buffer in realistic 200-frame batches showed a flat ~0.03-0.045ms per call,
independent of buffer occupancy — because basic tensor/TensorDict slicing
(`storage[:n]`) is a view, not a copy, in PyTorch, and reducing `.max()` over
one narrow `(n, 1)` integer field is cheap at any n up to 100k. Total overhead
across a full 500-batch buffer fill: ~19ms. Negligible next to actual training
cost (network forward/backward passes across `optim_steps_per_batch=50`
updates per batch). It was real, redundant work (~13-20x more calls than the
value is ever used for), but not the explanation for the reported slowness or
memory growth — the real cause of that remains open and needs profiling
(the codebase already has `get_profiler()` wired around each training-loop
stage for exactly this purpose) rather than further guessing.

## The fix: an incrementally maintained running max

The full-buffer scan was still worth removing, since it's pure waste even at
negligible per-call cost, compounded over a full run. The fix maintains a
running max on the trainer instead of recomputing anything from the buffer:

```python
# base.py:398
self._replay_buffer_max_step_count = 0
```

```python
# training_loop.py:71-81
if trainer._use_replay_buffer:
    trainer.replay_buffer.extend(data)
    batch_max_step_count = data["next", "step_count"].max()
    trainer._replay_buffer_max_step_count = max(
        trainer._replay_buffer_max_step_count, batch_max_step_count
    )
    max_length = trainer._replay_buffer_max_step_count
    buffer_len = len(trainer.replay_buffer)
```

This costs O(1) per batch (verified at ~0.008ms, flatter and cheaper than the
original full scan even at low buffer occupancy) instead of O(buffer size).

### A genuine behavior change, not just an optimization

Before the buffer fills to capacity, the running max and the old full-scan
value are identical at every step (verified directly: 0 mismatches across 500
simulated batches filling a 100,000-transition buffer).

Once the buffer wraps — which it does mid-run for H4 (`buffer_size=100_000`,
`max_steps=200_000`) — the two diverge, and this is worth being explicit
about rather than treating as an implementation detail:

- The **old** value was "max `step_count` currently present in the live
  buffer" — a number that can *decrease* if the transitions carrying the
  highest `step_count` get evicted by the circular buffer.
- The **new** value is "max `step_count` ever seen so far in the run" — a
  number that only ever increases.

This was confirmed directly: with a small buffer forced to wrap early, after
a spike of long episodes (`step_count=500`) got evicted by a run of short
ones (`step_count=10`), the old-style scan dropped to 10 while the running
max correctly stayed at 500.

For a progress-log diagnostic, "longest episode ever observed in this run" is
the more useful and more stable of the two — it isn't sensitive to unrelated
circular-buffer eviction timing, which is itself just an artifact of
`buffer_size` rather than anything about the policy or environment. But it is
a different number than the old code would have printed at some points in
training, so it's called out here rather than silently shipped as a "pure"
optimization.

## PPO is unaffected

PPO (and recurrent PPO) are on-policy: `_use_replay_buffer` is not set, so
`TrainingLoop.run()` takes the other branch and always computes `max_length`
from just the freshly collected batch (`data["next", "step_count"].max()`),
which was already correct and cheap — PPO's own buffer
(`ppo.py:228-241`, a fresh `ReplayBuffer` scoped to the current epoch) isn't
touched by any of this.
