"""Pre-flight config sanity check functions.

Each ``_check_*`` function is a pure function ``(ExperimentConfig) -> Finding | None``.
``check_config_guardrails`` runs all of them and returns a sorted list of findings.
The interactive prompt/reporting layer lives in ``config_guardrails_runner.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from logger import get_logger

if TYPE_CHECKING:
    from trading_rl.config import ExperimentConfig

logger = get_logger(__name__)


class Severity(StrEnum):
    FATAL = "FATAL"
    WARN = "WARN"


@dataclass
class Finding:
    severity: Severity
    parameter: str  # dotted config key(s) involved
    message: str  # what is wrong
    suggestion: str  # what to change


# ---------------------------------------------------------------------------
# Algorithm helpers
# ---------------------------------------------------------------------------


def _is_off_policy(algorithm: str) -> bool:
    return algorithm.upper() in {"DDPG", "TD3", "SAC"}


def _is_ppo(algorithm: str) -> bool:
    return algorithm.upper() == "PPO"


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


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
            suggestion=f"Set init_rand_steps >= sample_size (e.g. {max(s, 1000):_}).",
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
            suggestion=f"Set buffer_size >= sample_size (e.g. {s * 10:_}).",
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
            suggestion=f"Set sample_size <= frames_per_batch (e.g. {max(1, f // 4):_}).",
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
                f"streaming_episode_length={ep:_} > train_size={ts:_}: the streaming "
                "environment will attempt to load more rows than the training split "
                "contains, causing an index error on reset."
            ),
            suggestion=f"Set streaming_episode_length <= train_size (e.g. {ts // 2:_}).",
        )
    return None


_ORACLE_FEATURE_TYPES = frozenset({"mid_price_future_velocity"})
_ORACLE_FEATURE_OUTPUTS = frozenset({"feature_future_close_vel"})


def _configured_oracle_features(config: ExperimentConfig) -> list[str]:
    """Return configured oracle feature names/types from env columns and YAML."""
    found: set[str] = set()

    for column in getattr(config.env, "feature_columns", None) or []:
        if str(column) in _ORACLE_FEATURE_OUTPUTS:
            found.add(str(column))

    feature_config = getattr(config.data, "feature_config", None)
    if not feature_config:
        return sorted(found)

    path = Path(feature_config)
    if not path.exists():
        return sorted(found)

    data = yaml.safe_load(path.read_text()) or {}
    for feature in data.get("features", []) or []:
        if not isinstance(feature, dict):
            continue
        feature_type = str(feature.get("feature_type", ""))
        output_name = str(
            feature.get("output_name") or f"feature_{feature.get('name', '')}"
        )
        if (
            feature_type in _ORACLE_FEATURE_TYPES
            or output_name in _ORACLE_FEATURE_OUTPUTS
        ):
            found.add(output_name if output_name else feature_type)

    return sorted(found)


def _check_oracle_features_require_opt_in(config: ExperimentConfig) -> Finding | None:
    """FATAL: future-price oracle features must be explicitly marked as sanity checks."""
    if getattr(config.data, "allow_oracle_features", False):
        return None

    oracle_features = _configured_oracle_features(config)
    if not oracle_features:
        return None

    return Finding(
        severity=Severity.FATAL,
        parameter="data.allow_oracle_features / data.feature_config / env.feature_columns",
        message=(
            "Oracle feature(s) configured without explicit opt-in: "
            f"{oracle_features}. These features encode future market movement and "
            "make training/evaluation results unusable as real trading performance."
        ),
        suggestion=(
            "Remove the oracle feature(s), or set data.allow_oracle_features=true "
            "only for isolated sanity-check scenarios that will not be reported as "
            "production or thesis experiment results."
        ),
    )


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
                f"init_rand_steps={r:_} > buffer_size={b:_}: the random warm-up phase "
                f"produces {overflow:_} more transitions than the buffer can hold. "
                "Those early transitions are evicted before any gradient update sees them — "
                "wasted collection time, and the buffer is entirely overwritten by the end "
                "of the warm-up."
            ),
            suggestion=(
                f"Set init_rand_steps <= buffer_size (e.g. {b:_}), or increase "
                f"buffer_size to {r:_}."
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
                f"init_rand_steps={r:_} fills only {pct:.1f}% of buffer_size={b:_}. "
                "Early gradient updates draw from a tiny, highly correlated pool of "
                "random transitions, which can bias the critic and slow convergence."
            ),
            suggestion=(
                f"Set init_rand_steps >= buffer_size * 0.05 = {int(b * 0.05):_}, "
                f"or reduce buffer_size to {r * 10:_}."
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
                f"frames_per_batch={f:_} is {pct:.0f}% of buffer_size={b:_}. "
                "Each collection step replaces a large fraction of the buffer, so "
                "sampled mini-batches are temporally correlated, reducing the variance "
                "reduction benefit of experience replay."
            ),
            suggestion=(
                f"Increase buffer_size to at least frames_per_batch * 4 = {f * 4:_}, "
                f"or reduce frames_per_batch to {b // 10:_}."
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
                f"eval_interval={ei:_} against max_steps={ms:_} produces only "
                f"{n} evaluation checkpoint{'s' if n != 1 else ''} — too coarse "
                "to observe a learning curve."
            ),
            suggestion=f"Set eval_interval <= max_steps // 10 = {ms // 10:_}.",
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
    epochs = config.training.ppo.epochs
    steps_per_batch = config.training.optim_steps_per_batch
    updates = epochs * math.ceil(f / max(s, 1))
    limit = 100
    if updates > limit:
        return Finding(
            severity=Severity.WARN,
            parameter="training.ppo.epochs / training.frames_per_batch / training.sample_size",
            message=(
                f"PPO performs {updates} gradient updates per rollout "
                f"({epochs} epochs × ⌈{f}/{s}⌉ mini-batches). "
                "Repeatedly updating on the same rollout erodes the PPO clipping "
                "guarantee and can cause policy collapse, especially on short episodes."
            ),
            suggestion=(
                f"Reduce training.ppo.epochs (currently {epochs}) or increase frames_per_batch "
                f"(currently {f:_}) so that updates per rollout <= {limit}."
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
                f"warmup_rows={w:_} discards {pct:.0f}% of train_size={ts:_} rows "
                "after feature engineering. The effective training set may be too small."
            ),
            suggestion=(
                f"Increase train_size to at least warmup_rows * 5 = {w * 5:_}, "
                f"or reduce warmup_rows (currently {w:_})."
            ),
        )
    return None


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
                f"frames_per_batch={f:_} > train_size={ts:_}: the collector will reset "
                f"the environment ~{resets}× per batch. Each reset replays the same "
                "data from the beginning, increasing temporal correlation within the "
                "batch and reducing effective data diversity."
            ),
            suggestion=f"Set frames_per_batch <= train_size (e.g. {max(1, ts // 4):_}).",
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
                f"train_size={ts:_} <= warmup_rows={w:_}: after discarding warmup rows "
                "the effective training split is empty — the environment will have no "
                "data to step through."
            ),
            suggestion=(
                f"Increase train_size to at least warmup_rows + frames_per_batch = "
                f"{w + config.training.frames_per_batch:_}, or reduce warmup_rows."
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
                f"sample_size={config.training.sample_size} = {total_samples:_} samples "
                f"drawn per collection step, but buffer_size={b:_}. Each transition is "
                f"reused {ratio:.1f}× per batch on average, increasing overfitting risk."
            ),
            suggestion=(
                f"Reduce optim_steps_per_batch (currently {config.training.optim_steps_per_batch}) "
                f"or increase buffer_size to at least {total_samples:_}."
            ),
        )
    return None


def _check_td3_noise_vs_clip(config: ExperimentConfig) -> Finding | None:
    """WARN (TD3): policy_noise > noise_clip → clip is always active; noise setting has no effect."""
    if config.training.algorithm.upper() != "TD3":
        return None
    pn = config.training.td3.policy_noise
    nc = config.training.td3.noise_clip
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
    ce = config.training.ppo.clip_epsilon
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
                f"validation_size={vs:_} < eval_steps={es:_}: the evaluator will run "
                f"for at most {vs:_} steps (capped by available data) without raising "
                "an error, so eval metrics represent a shorter horizon than intended."
            ),
            suggestion=(
                f"Set eval_steps <= validation_size (e.g. {vs:_}), "
                f"or increase validation_size to at least {es:_}."
            ),
        )
    return None


def _check_no_checkpoints_long_run(config: ExperimentConfig) -> Finding | None:
    """WARN: no checkpointing on a long run → a crash loses all progress."""
    if config.training.checkpoint_interval != 0:
        return None
    threshold = 100_000
    if config.training.max_steps > threshold:
        hours_estimate = config.training.max_steps / 10_000
        return Finding(
            severity=Severity.WARN,
            parameter="training.checkpoint_interval / training.max_steps",
            message=(
                f"checkpoint_interval=0 (disabled) with max_steps={config.training.max_steps:_}. "
                f"A crash or OOM error loses all training progress "
                f"(estimated run time: ~{hours_estimate:.0f}+ minutes)."
            ),
            suggestion=(
                f"Set checkpoint_interval to e.g. {config.training.max_steps // 5:_} "
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
                f"log_interval={li:_} > eval_interval={ei:_}: evaluation fires more "
                "frequently than loss logging. You will see eval snapshots without "
                "any loss metrics between them, making it hard to diagnose whether "
                "a drop in eval performance corresponds to a loss spike."
            ),
            suggestion=f"Set log_interval <= eval_interval (e.g. {ei:_}).",
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
                f"streaming_episode_length={ep:_} <= warmup_rows={w:_}: every episode "
                "is shorter than the running-normalizer warm-up window. The normalizer "
                "never converges within an episode, so observations are perpetually "
                "in the cold-start regime with high variance."
            ),
            suggestion=(
                f"Set streaming_episode_length > warmup_rows "
                f"(e.g. {w * 5:_}), or reduce warmup_rows."
            ),
        )
    return None


def _check_no_exploration(config: ExperimentConfig) -> Finding | None:
    """WARN (DDPG/TD3): exploration_noise_std=0 and init_rand_steps=0 → no exploration at all."""
    if not _is_off_policy(config.training.algorithm):
        return None
    noise = config.training.td3.exploration_noise_std
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
    eb = config.training.ppo.entropy_bonus
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
    vc = config.training.ppo.vf_coef
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


def _check_buffer_never_full(config: ExperimentConfig) -> Finding | None:
    """WARN (off-policy): buffer_size > total transitions collected → buffer always partially empty."""
    if not _is_off_policy(config.training.algorithm):
        return None
    total = config.training.max_steps
    b = config.training.buffer_size
    if b > total:
        fill_pct = total / b * 100
        return Finding(
            severity=Severity.WARN,
            parameter="training.buffer_size / training.max_steps",
            message=(
                f"buffer_size={b:_} > max_steps={total:_}: the replay buffer can never "
                f"be filled — training ends with only {fill_pct:.0f}% utilisation. "
                "Sampling from a partially filled buffer skews the replay distribution "
                "toward earlier, less-trained transitions and wastes memory."
            ),
            suggestion=(
                f"Reduce buffer_size to at most max_steps={total:_}, "
                f"or increase max_steps to {b:_}."
            ),
        )
    return None


def _check_grad_norm_too_tight(config: ExperimentConfig) -> Finding | None:
    """WARN: max_grad_norm enabled but < 0.01 → gradients nearly zeroed; network stalls."""
    gn = getattr(config.training, "max_grad_norm", 0.0)
    if gn <= 0:
        return None
    if gn < 0.01:
        return Finding(
            severity=Severity.WARN,
            parameter="training.max_grad_norm",
            message=(
                f"max_grad_norm={gn}: gradient clipping is so aggressive that nearly "
                "all parameter updates are scaled to near-zero. The network can receive "
                "gradients orders of magnitude smaller than needed for any meaningful "
                "weight update, causing training to stall."
            ),
            suggestion=(
                "Typical values are 0.5–10.0. A value < 0.01 suggests a misconfiguration. "
                "Consider max_grad_norm=1.0, or set max_grad_norm=0 to disable clipping."
            ),
        )
    return None


def _check_exploration_noise_too_large(config: ExperimentConfig) -> Finding | None:
    """WARN (DDPG/TD3): exploration_noise_std > 1.0 → noise overwhelms policy output."""
    if not _is_off_policy(config.training.algorithm):
        return None
    noise = config.training.td3.exploration_noise_std
    if noise > 1.0:
        return Finding(
            severity=Severity.WARN,
            parameter="training.exploration_noise_std",
            message=(
                f"exploration_noise_std={noise}: the actor output is bounded to [-1, 1] "
                "(via tanh), but the exploration noise standard deviation exceeds that "
                "range. Actions are dominated by noise rather than policy, making the "
                "exploration essentially random regardless of what the network learns."
            ),
            suggestion=(
                "Typical values are 0.05–0.3. With std > 1.0 the effective policy signal "
                "is buried under noise. Consider exploration_noise_std=0.1."
            ),
        )
    return None


def _check_actor_lr_dominates_value_lr(config: ExperimentConfig) -> Finding | None:
    """WARN: actor_lr > value_lr × 10 → unusually large actor step relative to critic."""
    alr = config.training.actor_lr
    vlr = config.training.value_lr
    if alr > vlr * 10:
        ratio = alr / vlr
        return Finding(
            severity=Severity.WARN,
            parameter="training.actor_lr / training.value_lr",
            message=(
                f"actor_lr={alr:.2e} is {ratio:.0f}× value_lr={vlr:.2e}. "
                "In actor-critic methods the critic is the bootstrap target for policy "
                "updates. When the actor learns faster than the critic, the policy "
                "updates chase a rapidly-changing, noisy baseline and training destabilises."
            ),
            suggestion=(
                "Typical practice is value_lr >= actor_lr. Consider "
                f"actor_lr={vlr:.2e} or value_lr={alr:.2e}."
            ),
        )
    return None


def _check_streaming_episode_too_long(config: ExperimentConfig) -> Finding | None:
    """WARN (streaming): episode_length > 50% of train_size → very few distinct episodes."""
    if not getattr(config.data, "memmap_dir", None):
        return None
    ep = config.env.streaming_episode_length
    ts = config.data.train_size
    half = ts // 2
    if ep > half:
        episodes_per_pass = ts / ep
        return Finding(
            severity=Severity.WARN,
            parameter="env.streaming_episode_length / data.train_size",
            message=(
                f"streaming_episode_length={ep:_} is more than half of "
                f"train_size={ts:_} ({ep / ts * 100:.0f}%). Each pass through the "
                f"training data produces only ~{episodes_per_pass:.1f} episodes, giving "
                "the agent very few episode resets and limiting its exposure to diverse "
                "starting conditions within the training window."
            ),
            suggestion=(
                f"Set streaming_episode_length <= train_size // 4 = {ts // 4:_} "
                "for at least 4 starting positions per data pass."
            ),
        )
    return None


def _check_positions_one_sided(config: ExperimentConfig) -> Finding | None:
    """WARN: all positions non-negative or non-positive → agent cannot take the other side."""
    positions = config.env.positions
    if not positions or len(set(positions)) < 2:
        return None
    all_non_neg = all(p >= 0 for p in positions)
    all_non_pos = all(p <= 0 for p in positions)
    if all_non_neg:
        return Finding(
            severity=Severity.WARN,
            parameter="env.positions",
            message=(
                f"env.positions={list(positions)} contains no negative values. "
                "The agent cannot short the asset. In trending or mean-reverting markets "
                "this limits returns to only one market direction."
            ),
            suggestion="Add at least one negative position (e.g. -1) to enable shorting.",
        )
    if all_non_pos:
        return Finding(
            severity=Severity.WARN,
            parameter="env.positions",
            message=(
                f"env.positions={list(positions)} contains no positive values. "
                "The agent cannot go long. Returns are limited to short positions only."
            ),
            suggestion="Add at least one positive position (e.g. 1) to enable going long.",
        )
    return None


def _check_effective_training_steps(config: ExperimentConfig) -> Finding | None:
    """WARN: max_steps - init_rand_steps < 1000 → almost no gradient updates will occur."""
    effective = config.training.max_steps - config.training.init_rand_steps
    if effective < 1000:
        approx_updates = effective // config.training.frames_per_batch
        return Finding(
            severity=Severity.WARN,
            parameter="training.max_steps - training.init_rand_steps",
            message=(
                f"After the random warm-up (init_rand_steps={config.training.init_rand_steps:_}), "
                f"only {effective:_} steps remain for gradient training "
                f"(~{approx_updates} collection iterations). This is unlikely to be "
                "sufficient for any meaningful policy learning."
            ),
            suggestion=(
                f"Increase max_steps to at least init_rand_steps + 10_000 = "
                f"{config.training.init_rand_steps + 10_000:_}."
            ),
        )
    return None


def _check_val_size_much_smaller_than_train(config: ExperimentConfig) -> Finding | None:
    """WARN: validation_size < train_size // 10 → model selection signal is very noisy."""
    vs = config.data.validation_size
    ts = config.data.train_size
    if vs is None:
        return None
    threshold = ts // 10
    if vs < threshold:
        ratio = ts / vs if vs > 0 else float("inf")
        return Finding(
            severity=Severity.WARN,
            parameter="data.validation_size / data.train_size",
            message=(
                f"validation_size={vs:_} is only {vs / ts * 100:.1f}% of "
                f"train_size={ts:_} (ratio {ratio:.0f}:1). Validation metrics computed "
                "on a very small set have high variance, making it unreliable for "
                "checkpoint selection and early stopping."
            ),
            suggestion=(
                f"Set validation_size >= train_size // 10 = {threshold:_} for a "
                "reasonably stable evaluation signal."
            ),
        )
    return None


def _check_log_interval_too_coarse(config: ExperimentConfig) -> Finding | None:
    """WARN: fewer than 4 log entries over the entire run → nearly blind training."""
    li = config.training.log_interval
    ms = config.training.max_steps
    if li > 0 and ms // li < 4:
        n = ms // li
        return Finding(
            severity=Severity.WARN,
            parameter="training.log_interval / training.max_steps",
            message=(
                f"log_interval={li:_} against max_steps={ms:_} produces only "
                f"{n} log entr{'ies' if n != 1 else 'y'} for the entire run. "
                "Without frequent loss logging it is nearly impossible to detect "
                "divergence, instability, or a flat loss early enough to intervene."
            ),
            suggestion=f"Set log_interval <= max_steps // 20 = {ms // 20:_}.",
        )
    return None


def _check_action_thresholds_invalid(config: ExperimentConfig) -> Finding | None:
    """WARN: continuous_action_thresholds not ordered or outside tanh output range."""
    thresholds = config.env.continuous_action_thresholds
    if not thresholds or len(thresholds) < 2:
        return None
    lo, hi = thresholds[0], thresholds[1]
    if lo >= hi:
        return Finding(
            severity=Severity.WARN,
            parameter="env.continuous_action_thresholds",
            message=(
                f"continuous_action_thresholds=[{lo}, {hi}]: lower threshold >= upper "
                f"threshold. The middle bucket (neutral position) has zero width or is "
                "inverted, so the bucketing logic maps most or all actor outputs to the "
                "same position regardless of what the network outputs."
            ),
            suggestion=(
                "Thresholds must satisfy thresholds[0] < thresholds[1]. "
                "Typical values: [-0.33, 0.33]."
            ),
        )
    if hi >= 1.0:
        return Finding(
            severity=Severity.WARN,
            parameter="env.continuous_action_thresholds",
            message=(
                f"continuous_action_thresholds upper bound {hi} >= 1.0. The actor "
                "output passes through tanh and is bounded to (-1, 1), so the upper "
                "threshold is unreachable. The highest position bucket can never be "
                "selected — the agent effectively has only two actions."
            ),
            suggestion=(
                "Set upper threshold < 1.0 (e.g. 0.33). "
                f"Current value {hi} is at or above the tanh ceiling."
            ),
        )
    if lo <= -1.0:
        return Finding(
            severity=Severity.WARN,
            parameter="env.continuous_action_thresholds",
            message=(
                f"continuous_action_thresholds lower bound {lo} <= -1.0. The actor "
                "output is bounded to (-1, 1) via tanh, so the lower threshold is "
                "unreachable. The lowest position bucket can never be selected."
            ),
            suggestion=(
                "Set lower threshold > -1.0 (e.g. -0.33). "
                f"Current value {lo} is at or below the tanh floor."
            ),
        )
    return None


def _check_reward_scale_zero(config: ExperimentConfig) -> Finding | None:
    """FATAL: reward_scale=0 → all rewards are identically zero; no learning signal."""
    if config.env.reward_scale == 0.0:
        return Finding(
            severity=Severity.FATAL,
            parameter="env.reward_scale",
            message=(
                "reward_scale=0: every reward produced by the environment is multiplied "
                "by zero, making the learning signal exactly zero at every step. The "
                "policy gradient / TD error will always be zero and no learning can occur."
            ),
            suggestion="Set reward_scale to a non-zero value (default: 1.0).",
        )
    return None


def _check_eval_steps_too_small(config: ExperimentConfig) -> Finding | None:
    """WARN: eval_steps < 10 → evaluation metrics computed on fewer than 10 steps are noise."""
    es = config.evaluation.eval_steps
    if es < 10:
        return Finding(
            severity=Severity.WARN,
            parameter="evaluation.eval_steps",
            message=(
                f"eval_steps={es}: evaluation metrics (Sharpe, drawdown, win-rate, etc.) "
                "computed over fewer than 10 steps have very high variance and are "
                "essentially random. They cannot reliably distinguish good policies from "
                "bad ones."
            ),
            suggestion="Set eval_steps >= 50 for minimally useful metrics.",
        )
    return None


def _check_ppo_frames_too_small(config: ExperimentConfig) -> Finding | None:
    """WARN (PPO): frames_per_batch < 32 → advantage estimates across too few steps are very noisy."""
    if not _is_ppo(config.training.algorithm):
        return None
    f = config.training.frames_per_batch
    if f < 32:
        return Finding(
            severity=Severity.WARN,
            parameter="training.frames_per_batch",
            message=(
                f"frames_per_batch={f} for PPO: advantage estimates computed over "
                f"only {f} steps have very high variance. PPO's clipped objective "
                "assumes a reasonably large batch to form stable advantage baselines; "
                "with so few steps the update direction is dominated by noise."
            ),
            suggestion="Set frames_per_batch >= 64, ideally 256–2048 for PPO.",
        )
    return None


def _check_actor_network_bottleneck(config: ExperimentConfig) -> Finding | None:
    """WARN: narrowest actor layer < 8 → almost certainly too small to learn non-trivial policies."""
    min_dim = min(config.network.actor_hidden_dims)
    if min_dim < 8:
        return Finding(
            severity=Severity.WARN,
            parameter="network.actor_hidden_dims",
            message=(
                f"network.actor_hidden_dims={config.network.actor_hidden_dims}: "
                f"the narrowest layer has only {min_dim} units. "
                "With fewer than 8 neurons in a layer the actor almost certainly lacks "
                "the representational capacity to learn non-trivial trading policies, "
                "regardless of how long training runs."
            ),
            suggestion=(
                "Use at least 16 units in every hidden layer. "
                "A common baseline for trading is [64, 32] or [128, 64]."
            ),
        )
    return None


def _check_no_neutral_position(config: ExperimentConfig) -> Finding | None:
    """WARN: 0 not in positions → agent is always invested; cannot go flat."""
    positions = config.env.positions
    if positions is None or len(positions) < 2:
        return None
    if 0 not in positions:
        return Finding(
            severity=Severity.WARN,
            parameter="env.positions",
            message=(
                f"env.positions={list(positions)}: the neutral (flat) position 0 is "
                "absent. The agent is always invested and can never exit the market. "
                "This forces the agent to choose between Long and Short at every step, "
                "which may be intentional but eliminates one important degree of freedom "
                "and increases exposure during adverse conditions."
            ),
            suggestion=(
                "Add 0 to positions if a flat position is meaningful for this strategy, "
                "e.g. positions=[-1, 0, 1]."
            ),
        )
    return None


def _check_temp_eval_shorter_than_final(config: ExperimentConfig) -> Finding | None:
    """WARN: temp_eval.max_steps < eval_steps → periodic eval runs fewer steps than final eval."""
    temp_max = config.training.temp_eval.max_steps
    final_steps = config.evaluation.eval_steps
    if temp_max < final_steps:
        return Finding(
            severity=Severity.WARN,
            parameter="training.temp_eval.max_steps / evaluation.eval_steps",
            message=(
                f"temp_eval.max_steps={temp_max:_} < eval_steps={final_steps:_}: "
                "periodic evaluation during training is capped at fewer steps than the "
                "final post-training evaluation. The best-checkpoint selection signal "
                "(from periodic eval) is noisier than the evaluation it is supposed to "
                "predict, making checkpoint selection less reliable."
            ),
            suggestion=(
                f"Set temp_eval.max_steps >= eval_steps={final_steps:_}, "
                "or reduce eval_steps."
            ),
        )
    return None


def _check_value_network_underpowered_ppo(config: ExperimentConfig) -> Finding | None:
    """WARN (PPO): value network narrower than actor → critic can't accurately evaluate what actor learns."""
    if not _is_ppo(config.training.algorithm):
        return None
    min_actor = min(config.network.actor_hidden_dims)
    min_value = min(config.network.value_hidden_dims)
    if min_value < min_actor:
        return Finding(
            severity=Severity.WARN,
            parameter="network.value_hidden_dims / network.actor_hidden_dims",
            message=(
                f"PPO value network bottleneck ({min_value} units) is narrower than "
                f"actor bottleneck ({min_actor} units). "
                "In PPO the value function is the baseline used to compute advantages. "
                "An underpowered critic produces high-variance advantage estimates, "
                "making actor updates noisy even when the actor has sufficient capacity."
            ),
            suggestion=(
                "Set value_hidden_dims so its narrowest layer >= actor's narrowest layer. "
                f"Consider value_hidden_dims with min >= {min_actor}."
            ),
        )
    return None


def _check_val_test_imbalance(config: ExperimentConfig) -> Finding | None:
    """WARN: val_size and test_size both set but differ by more than 5×."""
    vs = config.data.validation_size
    ts = config.data.test_size
    if vs is None or ts is None or ts == 0:
        return None
    ratio = max(vs, ts) / min(vs, ts)
    if ratio > 5:
        bigger, bigger_val, smaller, smaller_val = (
            ("validation_size", vs, "test_size", ts)
            if vs > ts
            else ("test_size", ts, "validation_size", vs)
        )
        return Finding(
            severity=Severity.WARN,
            parameter="data.validation_size / data.test_size",
            message=(
                f"{bigger}={bigger_val:_} is {ratio:.0f}× {smaller}={smaller_val:_}. "
                "A strongly imbalanced val/test split means that either model selection "
                "signal is noisier than holdout evaluation, or the final out-of-sample "
                "estimate is less reliable than the validation score. "
                "Either way, the split does not represent the real deployment scenario."
            ),
            suggestion=(
                "Keep validation_size and test_size within 2–3× of each other. "
                f"Consider {smaller}={bigger_val:_} or {bigger}={smaller_val:_}."
            ),
        )
    return None


def _check_positions_high_leverage(config: ExperimentConfig) -> Finding | None:
    """WARN: any position magnitude > 2 → more than 2× leverage, high bankruptcy risk."""
    positions = config.env.positions
    if not positions:
        return None
    max_lev = max(abs(p) for p in positions)
    if max_lev > 2:
        large = [p for p in positions if abs(p) > 2]
        return Finding(
            severity=Severity.WARN,
            parameter="env.positions",
            message=(
                f"env.positions contains values with |position| > 2: {large}. "
                f"At {max_lev}× leverage an adverse move of {1 / max_lev * 100:.0f}% "
                "is sufficient to wipe the entire portfolio. Combined with trading fees "
                "and realistic market volatility, the agent is at high risk of broker "
                "bankruptcy during evaluation, producing -100% return cliffs in the "
                "equity curve."
            ),
            suggestion=(
                "Use position magnitudes of 1 or less unless leverage is explicitly "
                "supported by the environment and your research goal requires it."
            ),
        )
    return None


def _check_es_stale_policy_config(config: ExperimentConfig) -> Finding | None:
    """WARN: stale-policy early stopping enabled but window is too small to be reliable."""
    ratio = getattr(config.training, "es_stale_policy_min_ratio", 0.0)
    window = getattr(config.training, "es_stale_policy_window", 20)
    if ratio <= 0.0:
        return None
    if window < 5:
        return Finding(
            severity=Severity.WARN,
            parameter="training.es_stale_policy_window",
            message=(
                f"es_stale_policy_window={window} with es_stale_policy_min_ratio={ratio}: "
                "a window smaller than 5 episodes is too noisy — early stopping may fire "
                "spuriously during the random warm-up or after a single degenerate episode."
            ),
            suggestion="Set es_stale_policy_window >= 10 for a stable signal.",
        )
    return None


def _check_es_saturation_config(config: ExperimentConfig) -> Finding | None:
    """WARN: saturation early stopping enabled but window is too small or threshold implausible."""
    rate = getattr(config.training, "es_saturation_max_rate", 0.0)
    window = getattr(config.training, "es_saturation_window", 20)
    if rate <= 0.0:
        return None
    if window < 5:
        return Finding(
            severity=Severity.WARN,
            parameter="training.es_saturation_window",
            message=(
                f"es_saturation_window={window} with es_saturation_max_rate={rate}: "
                "a window smaller than 5 episodes is too noisy — saturation early stopping "
                "may fire spuriously during warm-up."
            ),
            suggestion="Set es_saturation_window >= 10 for a stable signal.",
        )
    if rate < 0.5:
        return Finding(
            severity=Severity.WARN,
            parameter="training.es_saturation_max_rate",
            message=(
                f"es_saturation_max_rate={rate} is very low. The saturation rate measures "
                "the fraction of steps at non-neutral positions. A threshold below 0.5 "
                "would fire even when the agent spends less than half its time invested, "
                "which is probably not the intent."
            ),
            suggestion="Typical values are 0.85–0.99 (fire when agent is almost always invested).",
        )
    return None


def _check_off_policy_too_few_total_updates(config: ExperimentConfig) -> Finding | None:
    """WARN (off-policy): total gradient updates < 500 → too few weight updates for convergence."""
    if not _is_off_policy(config.training.algorithm):
        return None
    effective_steps = config.training.max_steps - config.training.init_rand_steps
    if effective_steps <= 0:
        return None
    batches = effective_steps // max(config.training.frames_per_batch, 1)
    total_updates = batches * config.training.optim_steps_per_batch
    if total_updates < 500:
        return Finding(
            severity=Severity.WARN,
            parameter=(
                "(training.max_steps - training.init_rand_steps) "
                "/ training.frames_per_batch * training.optim_steps_per_batch"
            ),
            message=(
                f"Total gradient updates = {total_updates} "
                f"({batches} batches × {config.training.optim_steps_per_batch} optim steps). "
                "Fewer than 500 weight updates is generally not enough for an off-policy "
                "algorithm to converge on any meaningful signal, even in simple environments."
            ),
            suggestion=(
                f"Increase max_steps or reduce frames_per_batch so that "
                f"(max_steps - init_rand_steps) / frames_per_batch × optim_steps_per_batch >= 500. "
                f"E.g. max_steps >= {config.training.init_rand_steps + 500 * config.training.frames_per_batch // max(config.training.optim_steps_per_batch, 1):_}."
            ),
        )
    return None


def _is_sac(algorithm: str) -> bool:
    """Check if algorithm is SAC (Soft Actor-Critic)."""
    return algorithm.upper() == "SAC"


def _is_td3(algorithm: str) -> bool:
    """Check if algorithm is TD3 (Twin Delayed DDPG)."""
    return algorithm.upper() == "TD3"


# ---------------------------------------------------------------------------
# SAC-specific guardrails
# ---------------------------------------------------------------------------


def _check_sac_initial_alpha_too_large(config: ExperimentConfig) -> Finding | None:
    """WARN (SAC): initial_alpha > 1.0 → entropy term dominates; policy commits to random actions."""
    if not _is_sac(config.training.algorithm):
        return None
    alpha = config.training.sac.initial_alpha
    if alpha > 1.0:
        return Finding(
            severity=Severity.WARN,
            parameter="training.sac.initial_alpha",
            message=(
                f"initial_alpha={alpha}: the entropy coefficient is very large. "
                "In SAC the objective is J = Q(s,a) - α log π(a|s). With α > 1, "
                "the entropy term dominates and the policy is pushed toward maximum "
                "entropy (uniform) regardless of value estimates. The agent effectively "
                "learns to act randomly."
            ),
            suggestion="Typical values are 0.1–0.5. Consider initial_alpha=0.2.",
        )
    return None


def _check_sac_initial_alpha_too_small(config: ExperimentConfig) -> Finding | None:
    """WARN (SAC): initial_alpha < 0.01 → almost no exploration; policy may commit prematurely."""
    if not _is_sac(config.training.algorithm):
        return None
    alpha = config.training.sac.initial_alpha
    if alpha < 0.01:
        return Finding(
            severity=Severity.WARN,
            parameter="training.sac.initial_alpha",
            message=(
                f"initial_alpha={alpha}: the entropy coefficient is near-zero. "
                "SAC relies on entropy regularization for exploration; with such a low "
                "initial value the policy is almost deterministic from the start. "
                "This can cause premature convergence to suboptimal deterministic policies."
            ),
            suggestion="Typical values are 0.1–0.5. Consider initial_alpha=0.2.",
        )
    return None


def _check_sac_target_entropy_implausible(config: ExperimentConfig) -> Finding | None:
    """WARN (SAC): target_entropy outside plausible range for action dimension."""
    if not _is_sac(config.training.algorithm):
        return None
    target = config.training.sac.target_entropy
    if target is None:
        return None  # Auto-set to -dim, which is correct

    # For continuous actions with TanhNormal, max entropy is -dim (uniform distribution)
    # Minimal plausible entropy depends on action space, but -0.1 * dim is a reasonable lower bound
    # We need n_act which depends on env, but we can infer from action bounds if available
    # or just use a generic warning for extreme values
    if target > 0:
        return Finding(
            severity=Severity.WARN,
            parameter="training.sac.target_entropy",
            message=(
                f"target_entropy={target} > 0: target entropy is positive. "
                "Entropy is always ≤ 0 for continuous distributions; a positive target "
                "is meaningless and may cause the alpha optimizer to diverge."
            ),
            suggestion="Set target_entropy to a negative value (e.g. -dim(actions)), or None for auto.",
        )
    if target < -10:
        return Finding(
            severity=Severity.WARN,
            parameter="training.sac.target_entropy",
            message=(
                f"target_entropy={target}: very negative target entropy. "
                "This asks the policy to be extremely deterministic, which may limit "
                "exploration and cause premature convergence. A continuous action's "
                "maximum entropy is approximately -dim(actions)."
            ),
            suggestion="Typical values are in [-dim(actions), -0.5*dim(actions)]. Consider None for auto (-dim).",
        )
    return None


def _check_sac_alpha_lr_imbalance(config: ExperimentConfig) -> Finding | None:
    """WARN (SAC): alpha_lr >> actor_lr → temperature adapts faster than policy, causing instability."""
    if not _is_sac(config.training.algorithm):
        return None
    alpha_lr = config.training.sac.alpha_lr
    actor_lr = config.training.actor_lr
    if alpha_lr > actor_lr * 10:
        ratio = alpha_lr / actor_lr
        return Finding(
            severity=Severity.WARN,
            parameter="training.sac.alpha_lr / training.actor_lr",
            message=(
                f"alpha_lr={alpha_lr:.2e} is {ratio:.0f}× actor_lr={actor_lr:.2e}. "
                "The temperature parameter α will adapt much faster than the policy, "
                "causing the entropy regularization strength to fluctuate wildly during "
                "training. This destabilizes the policy objective J = Q - α log π."
            ),
            suggestion="Keep alpha_lr within 1–10× of actor_lr. Typical: both in [1e-4, 3e-4].",
        )
    if alpha_lr < actor_lr / 10:
        ratio = actor_lr / alpha_lr
        return Finding(
            severity=Severity.WARN,
            parameter="training.sac.alpha_lr / training.actor_lr",
            message=(
                f"alpha_lr={alpha_lr:.2e} is {ratio:.0f}× smaller than actor_lr={actor_lr:.2e}. "
                "The temperature parameter adapts so slowly that α essentially never adjusts "
                "from its initial value. You might as well use a fixed_alpha SAC variant."
            ),
            suggestion="Increase alpha_lr to be within 0.1–1× of actor_lr. Typical: both in [1e-4, 3e-4].",
        )
    return None


# ---------------------------------------------------------------------------
# TD3-specific guardrails
# ---------------------------------------------------------------------------


def _check_td3_policy_delay_too_small(config: ExperimentConfig) -> Finding | None:
    """WARN (TD3): policy_delay < 2 → eliminates TD3's key advantage over DDPG."""
    if not _is_td3(config.training.algorithm):
        return None
    delay = config.training.td3.policy_delay
    if delay < 2:
        return Finding(
            severity=Severity.WARN,
            parameter="training.td3.policy_delay",
            message=(
                f"policy_delay={delay}: TD3 updates the actor every {delay} critic steps, "
                f"but TD3's key innovation is delayed actor updates (typically 2 or more). "
                "With delay < 2, the actor updates every step, which eliminates the "
                "variance reduction benefit and makes TD3 essentially DDPG with twin critics."
            ),
            suggestion="Set policy_delay >= 2. The original TD3 paper uses policy_delay=2.",
        )
    if delay > 10:
        return Finding(
            severity=Severity.WARN,
            parameter="training.td3.policy_delay",
            message=(
                f"policy_delay={delay}: actor updates only every {delay} critic steps. "
                "Such a large delay can slow policy improvement significantly, especially "
                "in the early stages of training when the critic is still learning."
            ),
            suggestion="Typical values are 1–3. Consider policy_delay=2.",
        )
    return None


def _check_td3_exploration_noise_too_large(config: ExperimentConfig) -> Finding | None:
    """WARN (TD3): exploration_noise_std > 1.0 → noise overwhelms policy output."""
    if not _is_td3(config.training.algorithm):
        return None
    noise = config.training.td3.exploration_noise_std
    if noise > 1.0:
        return Finding(
            severity=Severity.WARN,
            parameter="training.td3.exploration_noise_std",
            message=(
                f"exploration_noise_std={noise}: TD3 actor output is bounded to [-1, 1] "
                "(via tanh), but the exploration noise standard deviation exceeds that range. "
                "Actions are dominated by noise rather than policy, making exploration "
                "essentially random regardless of what the network learns."
            ),
            suggestion="Typical values are 0.05–0.3. Consider exploration_noise_std=0.1.",
        )
    return None


# ---------------------------------------------------------------------------
# General learning rate guardrails
# ---------------------------------------------------------------------------


def _check_actor_lr_too_small(config: ExperimentConfig) -> Finding | None:
    """WARN: actor_lr < 1e-5 → policy essentially never updates."""
    lr = config.training.actor_lr
    if lr < 1e-5:
        return Finding(
            severity=Severity.WARN,
            parameter="training.actor_lr",
            message=(
                f"actor_lr={lr:.2e} is extremely small. With Adam, a learning rate "
                "below 1e-5 means parameter updates are so tiny that the policy "
                "effectively never changes, even after many gradient steps. Training "
                "will appear to run but the agent will not learn."
            ),
            suggestion="Typical actor LRs are 1e-4–3e-4. Consider actor_lr=1e-4.",
        )
    return None


def _check_value_lr_too_small(config: ExperimentConfig) -> Finding | None:
    """WARN: value_lr < 1e-5 → critic essentially never learns."""
    lr = config.training.value_lr
    if lr < 1e-5:
        return Finding(
            severity=Severity.WARN,
            parameter="training.value_lr",
            message=(
                f"value_lr={lr:.2e} is extremely small. The critic updates so slowly "
                "that value estimates remain near their initialisation throughout training. "
                "This produces noisy or flat advantages, making policy updates ineffective."
            ),
            suggestion="Typical value LRs are 1e-3–3e-3 (10× actor). Consider value_lr=1e-3.",
        )
    return None


def _check_weight_decay_imbalance(config: ExperimentConfig) -> Finding | None:
    """WARN: actor_weight_decay and value_weight_decay differ by > 100×."""
    awd = config.training.actor_weight_decay
    vwd = config.training.value_weight_decay
    if awd == 0 and vwd == 0:
        return None
    if awd > 0 and vwd > 0:
        ratio = max(awd, vwd) / min(awd, vwd)
        if ratio > 100:
            larger = "actor" if awd > vwd else "value"
            return Finding(
                severity=Severity.WARN,
                parameter="training.actor_weight_decay / training.value_weight_decay",
                message=(
                    f"{larger}_weight_decay is {ratio:.0f}× the other. Such asymmetric "
                    f"L2 regularization heavily penalises one network relative to the other, "
                    "biasing the learning dynamics. The heavily regularized network may "
                    "underfit while the other overfits."
                ),
                suggestion="Keep weight decay ratios within 10–100×. Typical: actor=1e-4, value=1e-2.",
            )
    return None


def _check_sample_size_too_small(config: ExperimentConfig) -> Finding | None:
    """WARN: sample_size < 16 → gradient estimates are very noisy."""
    s = config.training.sample_size
    if s < 16:
        return Finding(
            severity=Severity.WARN,
            parameter="training.sample_size",
            message=(
                f"sample_size={s}: mini-batches smaller than 16 produce very noisy "
                "gradient estimates. High variance updates can cause training instability "
                "and slow convergence, especially for value-based methods."
            ),
            suggestion="Set sample_size >= 32, ideally 64–256 for stable gradients.",
        )
    return None


def _check_optim_steps_excessive(config: ExperimentConfig) -> Finding | None:
    """WARN: optim_steps_per_batch > 100 → risk of overfitting to a single batch."""
    steps = config.training.optim_steps_per_batch
    if steps > 100:
        return Finding(
            severity=Severity.WARN,
            parameter="training.optim_steps_per_batch",
            message=(
                f"optim_steps_per_batch={steps}: performing many gradient updates on "
                "the same collected batch before collecting new data. This risks "
                "overfitting to the current batch, especially early in training when "
                "the replay buffer has high temporal correlation."
            ),
            suggestion="Typical values are 1–10. Consider optim_steps_per_batch <= 10.",
        )
    return None


def _check_frames_per_batch_vs_max_steps(config: ExperimentConfig) -> Finding | None:
    """WARN: frames_per_batch > max_steps // 2 → fewer than 2 gradient updates total."""
    f = config.training.frames_per_batch
    ms = config.training.max_steps
    if f > ms // 2:
        updates = ms // f
        return Finding(
            severity=Severity.WARN,
            parameter="training.frames_per_batch / training.max_steps",
            message=(
                f"frames_per_batch={f:_} allows only ~{updates} collection iteration(s) "
                f"before max_steps={ms:_} is reached. With so few gradient updates, "
                "meaningful learning is unlikely."
            ),
            suggestion=f"Set frames_per_batch <= max_steps // 10 = {ms // 10:_} for ~10 updates.",
        )
    return None


def _check_buffer_size_vs_sample_size(config: ExperimentConfig) -> Finding | None:
    """WARN (off-policy): buffer_size < 10 × sample_size → samples are highly correlated."""
    if not _is_off_policy(config.training.algorithm):
        return None
    b = config.training.buffer_size
    s = config.training.sample_size
    if b < s * 10:
        return Finding(
            severity=Severity.WARN,
            parameter="training.buffer_size / training.sample_size",
            message=(
                f"buffer_size={b:_} is only {b / s:.1f}× sample_size={s}. "
                "With such a small buffer, sampled mini-batches have high temporal "
                "correlation, reducing the variance-reduction benefit of experience replay."
            ),
            suggestion=f"Set buffer_size >= sample_size × 50 = {s * 50:_}.",
        )
    return None


def _check_first_layer_bottleneck(config: ExperimentConfig) -> Finding | None:
    """WARN: first hidden layer narrower than observation dim → input information bottleneck."""
    if not config.network.actor_hidden_dims:
        return None
    first_layer = config.network.actor_hidden_dims[0]
    # Observation dimension varies; we can't get it directly from config
    # But we can warn if it's suspiciously small (less than 16)
    if first_layer < 16:
        return Finding(
            severity=Severity.WARN,
            parameter="network.actor_hidden_dims",
            message=(
                f"actor_hidden_dims first layer={first_layer} is very narrow. "
                "If this is smaller than the observation dimension, it creates an "
                "information bottleneck at the input that prevents the network from "
                "learning useful representations."
            ),
            suggestion="Use at least 32 units in the first layer, ideally 64–128.",
        )
    return None


def _check_network_depth_mismatch(config: ExperimentConfig) -> Finding | None:
    """WARN: actor and value networks have vastly different depths."""
    actor_depth = len(config.network.actor_hidden_dims)
    value_depth = len(config.network.value_hidden_dims)
    if abs(actor_depth - value_depth) >= 3:
        return Finding(
            severity=Severity.WARN,
            parameter="network.actor_hidden_dims / network.value_hidden_dims",
            message=(
                f"Actor has {actor_depth} hidden layers, value has {value_depth}. "
                "Large depth differences mean the two networks have very different "
                "representational capacities and learning speeds. This can cause "
                "training instability, especially when the value function is used "
                "as the bootstrap target."
            ),
            suggestion="Keep layer counts within 1–2 of each other. Typical: both 2–3 layers.",
        )
    return None


def _check_action_bounds_asymmetry(config: ExperimentConfig) -> Finding | None:
    """WARN (continuous): action bounds highly asymmetric → bias toward one side of action space."""
    # This only applies to continuous action environments with custom bounds
    thresholds = config.env.continuous_action_thresholds
    if not thresholds or len(thresholds) < 2:
        return None
    lo, hi = thresholds[0], thresholds[1]
    if abs(lo + hi) > 0.3:  # Not symmetric around 0
        mid = (lo + hi) / 2
        return Finding(
            severity=Severity.WARN,
            parameter="env.continuous_action_thresholds",
            message=(
                f"continuous_action_thresholds=[{lo}, {hi}] are asymmetric around 0 "
                f"(midpoint={mid:.2f}). The tanh output is symmetric, so asymmetric "
                "thresholds create a bias where one extreme position is easier to reach "
                "than the other. The agent may learn to favor one side."
            ),
            suggestion="Use symmetric thresholds around 0, e.g. [-0.33, 0.33].",
        )
    return None


# ---------------------------------------------------------------------------
# Registry and public entry point
# ---------------------------------------------------------------------------

_ALL_CHECKS = [
    # FATAL first
    _check_reward_scale_zero,
    _check_sample_size_vs_init_rand,
    _check_sample_size_vs_buffer,
    _check_ppo_minibatch_vs_batch,
    _check_streaming_episode_vs_train_size,
    _check_oracle_features_require_opt_in,
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
    _check_buffer_never_full,
    _check_grad_norm_too_tight,
    _check_exploration_noise_too_large,
    _check_actor_lr_dominates_value_lr,
    _check_streaming_episode_too_long,
    _check_positions_one_sided,
    _check_effective_training_steps,
    _check_val_size_much_smaller_than_train,
    _check_log_interval_too_coarse,
    _check_action_thresholds_invalid,
    _check_eval_steps_too_small,
    _check_ppo_frames_too_small,
    _check_actor_network_bottleneck,
    _check_no_neutral_position,
    _check_temp_eval_shorter_than_final,
    _check_value_network_underpowered_ppo,
    _check_val_test_imbalance,
    _check_positions_high_leverage,
    _check_off_policy_too_few_total_updates,
    _check_es_stale_policy_config,
    _check_es_saturation_config,
    # New guardrails
    _check_sac_initial_alpha_too_large,
    _check_sac_initial_alpha_too_small,
    _check_sac_target_entropy_implausible,
    _check_sac_alpha_lr_imbalance,
    _check_td3_policy_delay_too_small,
    _check_td3_exploration_noise_too_large,
    _check_actor_lr_too_small,
    _check_value_lr_too_small,
    _check_weight_decay_imbalance,
    _check_sample_size_too_small,
    _check_optim_steps_excessive,
    _check_frames_per_batch_vs_max_steps,
    _check_buffer_size_vs_sample_size,
    _check_first_layer_bottleneck,
    _check_network_depth_mismatch,
    _check_action_bounds_asymmetry,
]


def check_config_guardrails(config: ExperimentConfig) -> list[Finding]:
    """Run all guardrail checks; return every finding, sorted FATAL first.

    Checks listed in ``config.disabled_guardrails`` are skipped.  Each entry
    is matched against the check function name with or without the leading
    underscore, so both ``"_check_td3_policy_delay_too_small"`` and
    ``"check_td3_policy_delay_too_small"`` are accepted.
    """
    skip = {n.lstrip("_") for n in getattr(config, "disabled_guardrails", [])}

    findings: list[Finding] = []
    for check in _ALL_CHECKS:
        if check.__name__.lstrip("_") in skip:
            continue
        try:
            result = check(config)
            if result is not None:
                findings.append(result)
        except Exception as exc:  # never let a guardrail crash the run
            logger.warning(
                "guardrail check {} failed unexpectedly: {}", check.__name__, exc
            )
    findings.sort(key=lambda f: 0 if f.severity == Severity.FATAL else 1)
    return findings
