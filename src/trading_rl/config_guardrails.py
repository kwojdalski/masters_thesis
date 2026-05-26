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
    """WARN: DSR + high reward_scale → reward magnitudes may destabilise training."""
    from trading_rl.constants import RewardType
    if config.env.reward_type != RewardType.DIFFERENTIAL_SHARPE:
        return None
    scale = config.env.reward_scale
    if scale > 100:
        clip = 10.0  # default clip in DifferentialSharpeRatio
        max_reward = clip * scale
        return Finding(
            severity=Severity.WARN,
            parameter="env.reward_scale / training.actor_lr",
            message=(
                f"reward_scale={scale} with DSR clip=±{clip} produces rewards up to "
                f"±{max_reward:,.0f}. Actor and critic gradients will be correspondingly "
                f"large. Current actor_lr={config.training.actor_lr} may be too high."
            ),
            suggestion=(
                f"Consider scaling learning rates down proportionally, e.g. "
                f"actor_lr={config.training.actor_lr / scale:.2e}, "
                f"value_lr={config.training.value_lr / scale:.2e}. "
                "Or reduce reward_scale."
            ),
        )
    return None


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
# Public API
# ---------------------------------------------------------------------------

_ALL_CHECKS = [
    # FATAL first
    _check_sample_size_vs_init_rand,
    _check_sample_size_vs_buffer,
    _check_ppo_minibatch_vs_batch,
    _check_streaming_episode_vs_train_size,
    # WARN
    _check_init_rand_overflows_buffer,
    _check_init_rand_too_small,
    _check_frames_per_batch_vs_buffer,
    _check_eval_interval,
    _check_dsr_reward_scale,
    _check_ppo_updates_per_rollout,
    _check_warmup_rows,
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
