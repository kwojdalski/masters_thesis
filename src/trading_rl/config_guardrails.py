"""Pre-flight config sanity checks run once at training start.

Two severity levels:
  FATAL — training will crash or produce meaningless results; raises immediately.
  WARN  — likely misconfiguration; prints a summary and prompts y/N before
          proceeding (skipped when training.skip_guardrail_prompts is True).
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from logger import get_logger

if TYPE_CHECKING:
    from trading_rl.config import ExperimentConfig

logger = get_logger(__name__)


class Severity(str, Enum):
    FATAL = "FATAL"
    WARN  = "WARN"


@dataclass
class Finding:
    severity: Severity
    parameter: str          # dotted config key(s) involved
    message: str            # what is wrong
    suggestion: str         # what to change


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _is_off_policy(algorithm: str) -> bool:
    return algorithm.upper() in {"DDPG", "TD3"}


def _is_ppo(algorithm: str) -> bool:
    return algorithm.upper() == "PPO"


def _check_sample_size_vs_init_rand(config: ExperimentConfig) -> Finding | None:
    """FATAL (off-policy): sample_size > init_rand_steps → first buffer sample crashes."""
    if not _is_off_policy(config.training.algorithm):
        return None
    s = config.training.sample_size
    r = config.training.init_rand_steps
    if s > r:
        return Finding(
            severity=Severity.FATAL,
            parameter="training.sample_size / training.init_rand_steps",
            message=(
                f"sample_size={s} > init_rand_steps={r}: the replay buffer will have "
                f"fewer than {s} transitions when the first sampling attempt is made, "
                "causing an immediate crash."
            ),
            suggestion=f"Set init_rand_steps >= sample_size (e.g. {max(s, 1000):,}).",
        )
    return None


def _check_sample_size_vs_buffer(config: ExperimentConfig) -> Finding | None:
    """FATAL: sample_size > buffer_size → can never draw a full batch."""
    s = config.training.sample_size
    b = config.training.buffer_size
    if s > b:
        return Finding(
            severity=Severity.FATAL,
            parameter="training.sample_size / training.buffer_size",
            message=(
                f"sample_size={s} > buffer_size={b}: the buffer can never hold a full "
                "mini-batch, so every sampling call will fail."
            ),
            suggestion=f"Set buffer_size >= sample_size (e.g. {s * 10:,}).",
        )
    return None


def _check_ppo_minibatch_vs_batch(config: ExperimentConfig) -> Finding | None:
    """FATAL (PPO): sample_size > frames_per_batch → mini-batch larger than rollout."""
    if not _is_ppo(config.training.algorithm):
        return None
    s = config.training.sample_size
    f = config.training.frames_per_batch
    if s > f:
        return Finding(
            severity=Severity.FATAL,
            parameter="training.sample_size / training.frames_per_batch",
            message=(
                f"sample_size={s} > frames_per_batch={f}: PPO mini-batch is larger "
                "than the collected rollout — the dataloader will be empty and training "
                "will crash or silently skip all updates."
            ),
            suggestion=f"Set sample_size <= frames_per_batch (e.g. {max(1, f // 4):,}).",
        )
    return None


def _check_streaming_episode_vs_train_size(config: ExperimentConfig) -> Finding | None:
    """FATAL (streaming): episode_length > train_size → environment runs out of data."""
    if not getattr(config.data, "memmap_dir", None):
        return None
    ep = config.env.streaming_episode_length
    ts = config.data.train_size
    if ep > ts:
        return Finding(
            severity=Severity.FATAL,
            parameter="env.streaming_episode_length / data.train_size",
            message=(
                f"streaming_episode_length={ep:,} > train_size={ts:,}: the streaming "
                "environment will attempt to load more rows than the training split "
                "contains, causing an index error on reset."
            ),
            suggestion=f"Set streaming_episode_length <= train_size (e.g. {ts // 2:,}).",
        )
    return None


def _check_init_rand_overflows_buffer(config: ExperimentConfig) -> Finding | None:
    """WARN (off-policy): init_rand_steps > buffer_size → early transitions evicted."""
    if not _is_off_policy(config.training.algorithm):
        return None
    r = config.training.init_rand_steps
    b = config.training.buffer_size
    if r > b:
        overflow = r - b
        return Finding(
            severity=Severity.WARN,
            parameter="training.init_rand_steps / training.buffer_size",
            message=(
                f"init_rand_steps={r:,} > buffer_size={b:,}: the random warm-up phase "
                f"produces {overflow:,} more transitions than the buffer can hold. "
                "Those early transitions are evicted before any gradient update sees them — "
                "wasted collection time, and the buffer is entirely overwritten by the end "
                "of the warm-up."
            ),
            suggestion=(
                f"Set init_rand_steps <= buffer_size (e.g. {b:,}), or increase "
                f"buffer_size to {r:,}."
            ),
        )
    return None


def _check_init_rand_too_small(config: ExperimentConfig) -> Finding | None:
    """WARN (off-policy): init_rand_steps < 5 % of buffer_size → very sparse warm-up."""
    if not _is_off_policy(config.training.algorithm):
        return None
    r = config.training.init_rand_steps
    b = config.training.buffer_size
    threshold = max(config.training.sample_size, int(b * 0.05))
    if r < threshold:
        pct = r / b * 100
        return Finding(
            severity=Severity.WARN,
            parameter="training.init_rand_steps / training.buffer_size",
            message=(
                f"init_rand_steps={r:,} fills only {pct:.1f}% of buffer_size={b:,}. "
                "Early gradient updates draw from a tiny, highly correlated pool of "
                "random transitions, which can bias the critic and slow convergence."
            ),
            suggestion=(
                f"Set init_rand_steps >= buffer_size * 0.05 = {int(b * 0.05):,}, "
                f"or reduce buffer_size to {r * 10:,}."
            ),
        )
    return None


def _check_frames_per_batch_vs_buffer(config: ExperimentConfig) -> Finding | None:
    """WARN (off-policy): frames_per_batch > 25 % of buffer → high temporal correlation."""
    if not _is_off_policy(config.training.algorithm):
        return None
    f = config.training.frames_per_batch
    b = config.training.buffer_size
    if f > b * 0.25:
        pct = f / b * 100
        return Finding(
            severity=Severity.WARN,
            parameter="training.frames_per_batch / training.buffer_size",
            message=(
                f"frames_per_batch={f:,} is {pct:.0f}% of buffer_size={b:,}. "
                "Each collection step replaces a large fraction of the buffer, so "
                "sampled mini-batches are temporally correlated, reducing the variance "
                "reduction benefit of experience replay."
            ),
            suggestion=(
                f"Increase buffer_size to at least frames_per_batch * 4 = {f * 4:,}, "
                f"or reduce frames_per_batch to {b // 10:,}."
            ),
        )
    return None


def _check_eval_interval(config: ExperimentConfig) -> Finding | None:
    """WARN: fewer than 4 evaluation checkpoints during the full run."""
    ei = config.training.eval_interval
    ms = config.training.max_steps
    if ei > 0 and ms // ei < 4:
        n = ms // ei
        return Finding(
            severity=Severity.WARN,
            parameter="training.eval_interval / training.max_steps",
            message=(
                f"eval_interval={ei:,} against max_steps={ms:,} produces only "
                f"{n} evaluation checkpoint{'s' if n != 1 else ''} — too coarse "
                "to observe a learning curve."
            ),
            suggestion=f"Set eval_interval <= max_steps // 10 = {ms // 10:,}.",
        )
    return None


def _check_dsr_reward_scale(config: ExperimentConfig) -> Finding | None:
    """WARN: DSR + high reward_scale → large reward magnitudes; advice depends on grad clipping."""
    from trading_rl.constants import RewardType
    if config.env.reward_type != RewardType.DIFFERENTIAL_SHARPE:
        return None
    scale = config.env.reward_scale
    if scale <= 100:
        return None

    clip = 10.0  # default clip in DifferentialSharpeRatio
    max_reward = clip * scale
    grad_norm = getattr(config.training, "max_grad_norm", 0.0)
    clipping_active = grad_norm > 0

    if clipping_active:
        # Gradient clipping provides a safety net, but if the norm is clipped on
        # nearly every step the effective learning rate is reduced and training
        # may be sluggish.  This is a softer concern than no clipping at all.
        return Finding(
            severity=Severity.WARN,
            parameter="env.reward_scale / training.max_grad_norm",
            message=(
                f"reward_scale={scale} with DSR clip=±{clip} produces rewards up to "
                f"±{max_reward:,.0f}. Gradient clipping (max_grad_norm={grad_norm}) is "
                "active and will prevent explosions, but if the norm is clipped on most "
                "steps the effective learning rate is reduced and convergence may be slow."
            ),
            suggestion=(
                f"Monitor the fraction of steps where gradients are clipped. "
                f"If it exceeds ~50%, consider reducing reward_scale or increasing "
                f"max_grad_norm (currently {grad_norm})."
            ),
        )
    else:
        # No gradient clipping — large reward magnitudes flow directly into the
        # parameter updates and can cause divergence.
        return Finding(
            severity=Severity.WARN,
            parameter="env.reward_scale / training.actor_lr",
            message=(
                f"reward_scale={scale} with DSR clip=±{clip} produces rewards up to "
                f"±{max_reward:,.0f}. Gradient clipping is disabled (max_grad_norm=0), "
                "so large reward magnitudes flow directly into parameter updates and "
                f"may cause divergence. Current actor_lr={config.training.actor_lr}."
            ),
            suggestion=(
                f"Enable gradient clipping (e.g. training.max_grad_norm=1.0), or "
                f"scale learning rates down proportionally: "
                f"actor_lr={config.training.actor_lr / scale:.2e}, "
                f"value_lr={config.training.value_lr / scale:.2e}. "
                "Or reduce reward_scale."
            ),
        )


def _check_ppo_updates_per_rollout(config: ExperimentConfig) -> Finding | None:
    """WARN (PPO): too many gradient updates per rollout risks policy collapse."""
    if not _is_ppo(config.training.algorithm):
        return None
    f = config.training.frames_per_batch
    s = config.training.sample_size
    epochs = config.training.ppo_epochs
    steps_per_batch = config.training.optim_steps_per_batch
    # Total gradient steps per rollout: epochs * ceil(f / s)
    updates = epochs * math.ceil(f / max(s, 1))
    limit = 100
    if updates > limit:
        return Finding(
            severity=Severity.WARN,
            parameter="training.ppo_epochs / training.frames_per_batch / training.sample_size",
            message=(
                f"PPO performs {updates} gradient updates per rollout "
                f"({epochs} epochs × ⌈{f}/{s}⌉ mini-batches). "
                "Repeatedly updating on the same rollout erodes the PPO clipping "
                "guarantee and can cause policy collapse, especially on short episodes."
            ),
            suggestion=(
                f"Reduce ppo_epochs (currently {epochs}) or increase frames_per_batch "
                f"(currently {f:,}) so that updates per rollout <= {limit}."
            ),
        )
    _ = steps_per_batch  # used for future checks
    return None


def _check_warmup_rows(config: ExperimentConfig) -> Finding | None:
    """WARN: warmup_rows discards a large fraction of the training split."""
    w = config.data.warmup_rows
    ts = config.data.train_size
    if ts > 0 and w / ts >= 0.20:
        pct = w / ts * 100
        return Finding(
            severity=Severity.WARN,
            parameter="data.warmup_rows / data.train_size",
            message=(
                f"warmup_rows={w:,} discards {pct:.0f}% of train_size={ts:,} rows "
                "after feature engineering. The effective training set may be too small."
            ),
            suggestion=(
                f"Increase train_size to at least warmup_rows * 5 = {w * 5:,}, "
                f"or reduce warmup_rows (currently {w:,})."
            ),
        )
    return None


# ---------------------------------------------------------------------------
# Additional checks
# ---------------------------------------------------------------------------

def _check_frames_per_batch_vs_train_size(config: ExperimentConfig) -> Finding | None:
    """WARN: frames_per_batch > train_size → env resets multiple times per batch."""
    f = config.training.frames_per_batch
    ts = config.data.train_size
    if f > ts:
        resets = math.ceil(f / ts)
        return Finding(
            severity=Severity.WARN,
            parameter="training.frames_per_batch / data.train_size",
            message=(
                f"frames_per_batch={f:,} > train_size={ts:,}: the collector will reset "
                f"the environment ~{resets}× per batch. Each reset replays the same "
                "data from the beginning, increasing temporal correlation within the "
                "batch and reducing effective data diversity."
            ),
            suggestion=f"Set frames_per_batch <= train_size (e.g. {max(1, ts // 4):,}).",
        )
    return None


def _check_train_size_vs_warmup_rows(config: ExperimentConfig) -> Finding | None:
    """FATAL: train_size <= warmup_rows → zero effective training rows after warmup."""
    w = config.data.warmup_rows
    ts = config.data.train_size
    if w > 0 and ts <= w:
        return Finding(
            severity=Severity.FATAL,
            parameter="data.train_size / data.warmup_rows",
            message=(
                f"train_size={ts:,} <= warmup_rows={w:,}: after discarding warmup rows "
                "the effective training split is empty — the environment will have no "
                "data to step through."
            ),
            suggestion=(
                f"Increase train_size to at least warmup_rows + frames_per_batch = "
                f"{w + config.training.frames_per_batch:,}, or reduce warmup_rows."
            ),
        )
    return None


def _check_optim_steps_replay_reuse(config: ExperimentConfig) -> Finding | None:
    """WARN (off-policy): optim_steps × sample_size > buffer_size → excessive replay reuse."""
    if not _is_off_policy(config.training.algorithm):
        return None
    total_samples = config.training.optim_steps_per_batch * config.training.sample_size
    b = config.training.buffer_size
    if total_samples > b:
        ratio = total_samples / b
        return Finding(
            severity=Severity.WARN,
            parameter="training.optim_steps_per_batch * training.sample_size / training.buffer_size",
            message=(
                f"optim_steps_per_batch={config.training.optim_steps_per_batch} × "
                f"sample_size={config.training.sample_size} = {total_samples:,} samples "
                f"drawn per collection step, but buffer_size={b:,}. Each transition is "
                f"reused {ratio:.1f}× per batch on average, increasing overfitting risk."
            ),
            suggestion=(
                f"Reduce optim_steps_per_batch (currently {config.training.optim_steps_per_batch}) "
                f"or increase buffer_size to at least {total_samples:,}."
            ),
        )
    return None


def _check_td3_noise_vs_clip(config: ExperimentConfig) -> Finding | None:
    """WARN (TD3): policy_noise > noise_clip → clip is always active; noise setting has no effect."""
    if config.training.algorithm.upper() != "TD3":
        return None
    pn = config.training.policy_noise
    nc = config.training.noise_clip
    if pn > nc:
        return Finding(
            severity=Severity.WARN,
            parameter="training.policy_noise / training.noise_clip",
            message=(
                f"policy_noise={pn} > noise_clip={nc}: TD3 clamps target policy noise "
                f"to [-{nc}, {nc}] before adding it, so the actual noise standard "
                f"deviation is always capped at {nc}. policy_noise={pn} is ignored."
            ),
            suggestion=(
                f"Set policy_noise <= noise_clip (e.g. {nc * 0.5:.2f}), "
                f"or increase noise_clip to {pn}."
            ),
        )
    return None


def _check_ppo_clip_epsilon(config: ExperimentConfig) -> Finding | None:
    """WARN (PPO): clip_epsilon > 0.5 → near-unconstrained policy updates."""
    if not _is_ppo(config.training.algorithm):
        return None
    ce = config.training.clip_epsilon
    if ce > 0.5:
        return Finding(
            severity=Severity.WARN,
            parameter="training.clip_epsilon",
            message=(
                f"clip_epsilon={ce} is unusually large. PPO clips the probability ratio "
                "to [1-ε, 1+ε]; above 0.5 the constraint is so loose that updates can "
                "be as large as unconstrained gradient steps, defeating the purpose of "
                "the clipping objective."
            ),
            suggestion="Typical values are 0.1–0.3. Consider clip_epsilon=0.2.",
        )
    return None


def _check_tau_too_large(config: ExperimentConfig) -> Finding | None:
    """WARN (DDPG/TD3): tau > 0.1 → target network updates nearly as fast as online network."""
    if not _is_off_policy(config.training.algorithm):
        return None
    tau = config.training.tau
    if tau > 0.1:
        return Finding(
            severity=Severity.WARN,
            parameter="training.tau",
            message=(
                f"tau={tau} is high. The target network soft-update θ' ← (1-τ)θ' + τθ "
                "converges to the online network in ~{:.0f} steps. The stabilising "
                "effect of a slowly-changing target is largely lost above τ≈0.1.".format(
                    1 / tau
                )
            ),
            suggestion="Typical values are 0.001–0.01. Consider tau=0.005.",
        )
    return None


def _check_dsr_eta_range(config: ExperimentConfig) -> Finding | None:
    """WARN (DSR): reward_eta outside [1e-4, 0.5] → EMA either barely adapts or forgets instantly."""
    from trading_rl.constants import RewardType
    if config.env.reward_type != RewardType.DIFFERENTIAL_SHARPE:
        return None
    eta = config.env.reward_eta
    if eta > 0.5:
        return Finding(
            severity=Severity.WARN,
            parameter="env.reward_eta",
            message=(
                f"reward_eta={eta}: with η > 0.5 the DSR EMA discards more than half "
                "its history at every step. The resulting signal reflects only the most "
                "recent 1–2 returns and has very high variance."
            ),
            suggestion="Typical DSR values are 0.001–0.1. Consider reward_eta=0.01.",
        )
    if eta < 1e-4:
        half_life = math.log(0.5) / math.log(1 - eta)
        return Finding(
            severity=Severity.WARN,
            parameter="env.reward_eta",
            message=(
                f"reward_eta={eta}: the DSR EMA half-life is ~{half_life:,.0f} steps. "
                "The signal barely moves within a typical episode and will be nearly "
                "constant at the start of training when A_t and B_t are cold."
            ),
            suggestion="Typical DSR values are 0.001–0.1. Consider reward_eta=0.01.",
        )
    return None


def _check_obs_clip_none(config: ExperimentConfig) -> Finding | None:
    """WARN: obs_clip=None → no clipping during running-normalizer cold start."""
    if config.env.obs_clip is not None:
        return None
    return Finding(
        severity=Severity.WARN,
        parameter="env.obs_clip",
        message=(
            "obs_clip=None: observations are not clipped. Running normalizers "
            "(RunningMeanStd) produce extreme values for the first ~warmup_rows steps "
            "before their statistics converge. Without clipping, those spikes pass "
            "directly into the network and can cause gradient explosions early in training."
        ),
        suggestion=(
            "Set obs_clip to a finite value such as 5.0 or 10.0. "
            "Alternatively, ensure warmup_rows is large enough to cover the cold-start period."
        ),
    )


def _check_trading_fees(config: ExperimentConfig) -> Finding | None:
    """WARN: trading_fees > 0.1% per trade — at typical HFT turnover this dominates returns."""
    fee = config.env.trading_fees
    if fee > 0.001:
        pct = fee * 100
        return Finding(
            severity=Severity.WARN,
            parameter="env.trading_fees",
            message=(
                f"trading_fees={fee} ({pct:.3f}% per trade). At high-frequency turnover "
                "fees accumulate rapidly and can easily exceed total gross returns, "
                "making profitability nearly impossible regardless of signal quality."
            ),
            suggestion=(
                f"Verify the fee is intentional. Typical HFT fees are 0.0–0.0002 "
                f"(0.0–0.02%). Current value is {pct / 0.02:.0f}× the typical upper bound."
            ),
        )
    return None


def _check_learning_rates(config: ExperimentConfig) -> Finding | None:
    """WARN: actor_lr or value_lr > 1e-2 → likely to diverge with Adam."""
    findings_parts = []
    if config.training.actor_lr > 1e-2:
        findings_parts.append(f"actor_lr={config.training.actor_lr:.2e}")
    if config.training.value_lr > 1e-2:
        findings_parts.append(f"value_lr={config.training.value_lr:.2e}")
    if not findings_parts:
        return None
    return Finding(
        severity=Severity.WARN,
        parameter="training.actor_lr / training.value_lr",
        message=(
            f"{', '.join(findings_parts)} exceed 1e-2. Adam learning rates above 1e-2 "
            "frequently cause divergence on RL objectives, especially when reward "
            "magnitudes are large or observations are not well-normalised."
        ),
        suggestion="Typical Adam LRs for RL are 1e-4–3e-4 (actor) and 1e-3–3e-3 (critic).",
    )


def _check_seed_none(config: ExperimentConfig) -> Finding | None:
    """WARN: seed=None → non-reproducible run."""
    if config.seed is not None:
        return None
    return Finding(
        severity=Severity.WARN,
        parameter="seed",
        message=(
            "seed=None: this run is not reproducible. Network initialisation, data "
            "shuffling, and random baselines will differ across runs, making it "
            "impossible to attribute result differences to config changes."
        ),
        suggestion="Set seed to a fixed integer (e.g. seed=42).",
    )


def _check_validation_size_vs_eval_steps(config: ExperimentConfig) -> Finding | None:
    """WARN: validation_size < eval_steps → eval silently runs fewer steps than requested."""
    vs = config.data.validation_size
    es = config.evaluation.eval_steps
    if vs is not None and vs < es:
        return Finding(
            severity=Severity.WARN,
            parameter="data.validation_size / evaluation.eval_steps",
            message=(
                f"validation_size={vs:,} < eval_steps={es:,}: the evaluator will run "
                f"for at most {vs:,} steps (capped by available data) without raising "
                "an error, so eval metrics represent a shorter horizon than intended."
            ),
            suggestion=(
                f"Set eval_steps <= validation_size (e.g. {vs:,}), "
                f"or increase validation_size to at least {es:,}."
            ),
        )
    return None


def _check_no_checkpoints_long_run(config: ExperimentConfig) -> Finding | None:
    """WARN: no checkpointing on a long run → a crash loses all progress."""
    if config.training.checkpoint_interval != 0:
        return None
    threshold = 100_000
    if config.training.max_steps > threshold:
        hours_estimate = config.training.max_steps / 10_000  # rough: ~10k steps/min
        return Finding(
            severity=Severity.WARN,
            parameter="training.checkpoint_interval / training.max_steps",
            message=(
                f"checkpoint_interval=0 (disabled) with max_steps={config.training.max_steps:,}. "
                f"A crash or OOM error loses all training progress "
                f"(estimated run time: ~{hours_estimate:.0f}+ minutes)."
            ),
            suggestion=(
                f"Set checkpoint_interval to e.g. {config.training.max_steps // 5:,} "
                "(save 5 checkpoints over the run)."
            ),
        )
    return None


def _check_log_interval_vs_eval_interval(config: ExperimentConfig) -> Finding | None:
    """WARN: log_interval > eval_interval → loss logs are coarser than eval cadence."""
    li = config.training.log_interval
    ei = config.training.eval_interval
    if ei > 0 and li > ei:
        return Finding(
            severity=Severity.WARN,
            parameter="training.log_interval / training.eval_interval",
            message=(
                f"log_interval={li:,} > eval_interval={ei:,}: evaluation fires more "
                "frequently than loss logging. You will see eval snapshots without "
                "any loss metrics between them, making it hard to diagnose whether "
                "a drop in eval performance corresponds to a loss spike."
            ),
            suggestion=f"Set log_interval <= eval_interval (e.g. {ei:,}).",
        )
    return None


def _check_single_position(config: ExperimentConfig) -> Finding | None:
    """WARN: only one position available → agent has no meaningful choice to learn."""
    positions = config.env.positions
    if positions is not None and len(set(positions)) == 1:
        return Finding(
            severity=Severity.WARN,
            parameter="env.positions",
            message=(
                f"env.positions={positions} contains only one unique value. "
                "The agent has no choice to make at each step — every action produces "
                "the same outcome. No policy can be learned."
            ),
            suggestion="Add at least two distinct positions, e.g. positions=[-1, 1].",
        )
    return None


def _check_streaming_episode_vs_warmup_rows(config: ExperimentConfig) -> Finding | None:
    """WARN (streaming): episode_length <= warmup_rows → every episode starts with unconverged stats."""
    if not getattr(config.data, "memmap_dir", None):
        return None
    ep = config.env.streaming_episode_length
    w = config.data.warmup_rows
    if w > 0 and ep <= w:
        return Finding(
            severity=Severity.WARN,
            parameter="env.streaming_episode_length / data.warmup_rows",
            message=(
                f"streaming_episode_length={ep:,} <= warmup_rows={w:,}: every episode "
                "is shorter than the running-normalizer warm-up window. The normalizer "
                "never converges within an episode, so observations are perpetually "
                "in the cold-start regime with high variance."
            ),
            suggestion=(
                f"Set streaming_episode_length > warmup_rows "
                f"(e.g. {w * 5:,}), or reduce warmup_rows."
            ),
        )
    return None


def _check_no_exploration(config: ExperimentConfig) -> Finding | None:
    """WARN (DDPG/TD3): exploration_noise_std=0 and init_rand_steps=0 → no exploration at all."""
    if not _is_off_policy(config.training.algorithm):
        return None
    noise = config.training.exploration_noise_std
    rand_steps = config.training.init_rand_steps
    if noise == 0.0 and rand_steps == 0:
        return Finding(
            severity=Severity.WARN,
            parameter="training.exploration_noise_std / training.init_rand_steps",
            message=(
                "exploration_noise_std=0 and init_rand_steps=0: the agent performs "
                "no exploration whatsoever from step one. The critic is trained "
                "entirely on deterministic trajectories, severely limiting the "
                "state-action coverage in the replay buffer."
            ),
            suggestion=(
                "Set init_rand_steps >= sample_size to seed the buffer with random "
                "transitions, and/or set exploration_noise_std > 0 (e.g. 0.1)."
            ),
        )
    return None


def _check_ppo_entropy_bonus(config: ExperimentConfig) -> Finding | None:
    """WARN (PPO): entropy_bonus > 0.1 → entropy term dominates, policy stays near-uniform."""
    if not _is_ppo(config.training.algorithm):
        return None
    eb = config.training.entropy_bonus
    if eb > 0.1:
        return Finding(
            severity=Severity.WARN,
            parameter="training.entropy_bonus",
            message=(
                f"entropy_bonus={eb}: the entropy regularisation term is unusually "
                "large. Above ~0.1 it dominates the policy gradient loss, pushing the "
                "policy toward a uniform distribution and preventing it from committing "
                "to high-reward actions."
            ),
            suggestion="Typical values are 0.0–0.05. Consider entropy_bonus=0.01.",
        )
    return None


def _check_ppo_vf_coef(config: ExperimentConfig) -> Finding | None:
    """WARN (PPO): vf_coef > 1.0 → value loss dominates, actor barely trained."""
    if not _is_ppo(config.training.algorithm):
        return None
    vc = config.training.vf_coef
    if vc > 1.0:
        return Finding(
            severity=Severity.WARN,
            parameter="training.vf_coef",
            message=(
                f"vf_coef={vc}: the value function loss is weighted {vc}× relative "
                "to the policy gradient. Above 1.0 the value loss dominates the "
                "combined objective and the actor receives proportionally smaller "
                "gradient signal, slowing policy improvement."
            ),
            suggestion="Typical values are 0.25–1.0. Consider vf_coef=0.5.",
        )
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_ALL_CHECKS = [
    # FATAL first
    _check_sample_size_vs_init_rand,
    _check_sample_size_vs_buffer,
    _check_ppo_minibatch_vs_batch,
    _check_streaming_episode_vs_train_size,
    _check_train_size_vs_warmup_rows,
    # WARN
    _check_frames_per_batch_vs_train_size,
    _check_init_rand_overflows_buffer,
    _check_init_rand_too_small,
    _check_frames_per_batch_vs_buffer,
    _check_optim_steps_replay_reuse,
    _check_eval_interval,
    _check_dsr_reward_scale,
    _check_dsr_eta_range,
    _check_ppo_updates_per_rollout,
    _check_ppo_clip_epsilon,
    _check_ppo_entropy_bonus,
    _check_ppo_vf_coef,
    _check_td3_noise_vs_clip,
    _check_tau_too_large,
    _check_obs_clip_none,
    _check_trading_fees,
    _check_learning_rates,
    _check_warmup_rows,
    _check_validation_size_vs_eval_steps,
    _check_no_checkpoints_long_run,
    _check_log_interval_vs_eval_interval,
    _check_single_position,
    _check_streaming_episode_vs_warmup_rows,
    _check_no_exploration,
    _check_seed_none,
]


def check_config_guardrails(config: ExperimentConfig) -> list[Finding]:
    """Run all guardrail checks; return every finding, sorted FATAL first."""
    findings: list[Finding] = []
    for check in _ALL_CHECKS:
        try:
            result = check(config)
            if result is not None:
                findings.append(result)
        except Exception as exc:  # never let a guardrail crash the run
            logger.warning("guardrail check %s failed unexpectedly: %s", check.__name__, exc)
    findings.sort(key=lambda f: 0 if f.severity == Severity.FATAL else 1)
    return findings


def run_guardrail_check(config: ExperimentConfig) -> None:
    """Check config, log findings, raise on FATAL, prompt on WARN.

    Controlled by training.skip_guardrail_prompts:
      False (default) — print WARN findings and ask for y/N confirmation.
      True            — log WARNs but proceed without prompting (for scripts).
    """
    findings = check_config_guardrails(config)
    if not findings:
        logger.info("config guardrail check passed — no issues found")
        return

    skip_prompts = getattr(config.training, "skip_guardrail_prompts", False)

    fatals = [f for f in findings if f.severity == Severity.FATAL]
    warns  = [f for f in findings if f.severity == Severity.WARN]

    # Always surface every finding in the log regardless of skip_prompts.
    for f in findings:
        logger.warning(
            "guardrail %s [%s] %s | suggestion: %s",
            f.severity.value, f.parameter, f.message, f.suggestion,
        )

    if fatals:
        lines = ["\nCONFIG GUARDRAIL — FATAL ERRORS\n" + "=" * 50]
        for i, f in enumerate(fatals, 1):
            lines.append(f"\n[{i}] {f.parameter}")
            lines.append(f"    Problem:    {f.message}")
            lines.append(f"    Fix:        {f.suggestion}")
        lines.append("\nTraining cannot start with these settings.")
        print("\n".join(lines), file=sys.stderr)
        raise ValueError(
            f"Config guardrail check failed with {len(fatals)} fatal error(s). "
            "See output above for details."
        )

    if warns:
        lines = ["\nCONFIG GUARDRAIL — WARNINGS\n" + "=" * 50]
        for i, f in enumerate(warns, 1):
            lines.append(f"\n[{i}] {f.parameter}")
            lines.append(f"    Problem:    {f.message}")
            lines.append(f"    Suggestion: {f.suggestion}")
        print("\n".join(lines))

        if skip_prompts:
            print(
                "\n[guardrail] training.skip_guardrail_prompts=True — proceeding without prompt.\n"
            )
            return

        # Non-interactive stdin (CI, scripts, pytest) — warn and continue.
        if not sys.stdin.isatty():
            logger.warning(
                "guardrail: %d warning(s) found but stdin is non-interactive — "
                "proceeding automatically. Set training.skip_guardrail_prompts=true "
                "to silence this message.",
                len(warns),
            )
            return

        print("\nProceed anyway? [y/N] ", end="", flush=True)
        try:
            answer = input().strip().lower()
        except (EOFError, OSError):
            answer = "n"

        if answer not in {"y", "yes"}:
            raise SystemExit(
                "Aborted by user after config guardrail warnings. "
                "Fix the issues above or set training.skip_guardrail_prompts=true "
                "to suppress the prompt."
            )
