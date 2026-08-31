"""Trainer-level correctness checks that the algorithm smoke tests do not make.

`test_algorithms_backends.py` / `test_continuous_ppo.py` run a full training job
and only assert it did not raise. That would not catch:

  * a reward wired with the wrong sign, or detached from position/price;
  * training that silently performs zero gradient updates
    (e.g. init_rand_steps >= max_steps, or the issue #356 buffer-gate bug),
    leaving the policy identical to its initialisation.

Both are checked here deterministically -- no dependence on the agent actually
converging, which needs a long run and belongs in the `slow` tier.

The DSR reward's sign is not asserted against price direction here: DSR can
legitimately be negative on a gain when variance dynamics dominate. Its exact
formula is covered by test_dsr_formula_exact.py. These tests use `log_return`,
whose sign is unambiguous.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torchrl.envs.utils import step_mdp

from trading_rl.config import ExperimentConfig
from trading_rl.train_trading_agent import build_training_context

_FEATURE_YAML = (
    "features:\n"
    '  - name: "lag1"\n'
    '    feature_type: "return_lag"\n'
    "    normalize: true\n"
    "    params:\n"
    '      column: "close"\n'
    "      lag: 1\n"
    '  - name: "trend"\n'
    '    feature_type: "trend"\n'
    "    normalize: false\n"
)


def _write_dataset(path: Path, close: np.ndarray) -> Path:
    idx = pd.date_range("2024-01-01", periods=len(close), freq="h")
    df = pd.DataFrame(
        {
            "open": close - 0.05,
            "high": close + 0.10,
            "low": close - 0.10,
            "close": close,
            "volume": 1000.0 + np.arange(len(close)),
        },
        index=idx,
    )
    df.to_parquet(path)
    return path


def _make_config(
    tmp_path: Path,
    close: np.ndarray,
    *,
    algorithm: str = "TD3",
    max_steps: int = 8,
    frames_per_batch: int = 4,
    init_rand_steps: int = 0,
    reward_type: str = "log_return",
) -> ExperimentConfig:
    data_path = _write_dataset(tmp_path / "data.parquet", close)
    feat_path = tmp_path / "features.yaml"
    feat_path.write_text(_FEATURE_YAML, encoding="utf-8")
    n = len(close)
    return ExperimentConfig.from_dict(
        {
            "experiment_name": "trainer_correctness",
            "seed": 0,
            "data": {
                "data_path": str(data_path),
                "train_size": int(n * 0.75),
                "validation_size": int(n * 0.15),
                "download_data": False,
                "feature_config": str(feat_path),
                "warmup_rows": 0,
            },
            "env": {
                "backend": "tradingenv",
                "price_column": "close",
                "feature_columns": ["feature_lag1", "feature_trend"],
                "reward_type": reward_type,
                "streaming_episode_length": 50,
            },
            "training": {
                "algorithm": algorithm,
                "max_steps": max_steps,
                "frames_per_batch": frames_per_batch,
                "sample_size": 32,
                "init_rand_steps": init_rand_steps,
                "buffer_size": 400,
                "optim_steps_per_batch": 2,
                "log_interval": frames_per_batch,
                "gamma": 0.9,
            },
            "evaluation": {"eval_steps": 20},
            "logging": {
                "log_dir": str(tmp_path / "logs"),
                "log_level": "ERROR",
                "save_plots": False,
            },
            "tracking": {"tracking_uri": f"file://{tmp_path / 'mlruns'}"},
        }
    )


def _rollout_fixed_action(env, value: float, n_steps: int = 60) -> float:
    """Sum of step rewards when the agent holds a constant portfolio weight."""
    td = env.reset()
    total = 0.0
    for _ in range(n_steps):
        td.set(
            "action",
            torch.full(
                env.action_spec.shape, float(value), dtype=env.action_spec.dtype
            ),
        )
        td = env.step(td)
        total += float(td["next", "reward"].sum())
        if bool(td["next", "done"].any()):
            break
        td = step_mdp(td)
    return total


def _flat_actor_params(trainer) -> torch.Tensor:
    return torch.cat([p.detach().flatten().clone() for p in trainer.actor.parameters()])


# ---------------------------------------------------------------------------
# Reward wiring: sign follows position * price move
# ---------------------------------------------------------------------------


@pytest.fixture
def _rising(tmp_path: Path):
    close = 100.0 + np.arange(160) * 0.5  # strictly increasing
    return build_training_context(
        config=_make_config(tmp_path, close), create_mlflow_callback=False
    )


@pytest.fixture
def _falling(tmp_path: Path):
    close = 180.0 - np.arange(160) * 0.5  # strictly decreasing, stays > 0
    return build_training_context(
        config=_make_config(tmp_path, close), create_mlflow_callback=False
    )


def test_long_position_earns_in_a_rising_market_and_loses_in_a_falling_one(
    _rising, _falling
):
    up = _rollout_fixed_action(_rising["env"], +1.0)
    down = _rollout_fixed_action(_falling["env"], +1.0)

    assert up > 0.0, f"holding long through a rising market lost money: {up}"
    assert down < 0.0, f"holding long through a falling market made money: {down}"


def test_short_position_sign_is_the_mirror_of_long(_rising, _falling):
    short_up = _rollout_fixed_action(_rising["env"], -1.0)
    short_down = _rollout_fixed_action(_falling["env"], -1.0)

    assert short_up < 0.0, "shorting a rising market made money"
    assert short_down > 0.0, "shorting a falling market lost money"


def test_reward_magnitude_is_near_antisymmetric_in_position_sign(_rising):
    long_r = _rollout_fixed_action(_rising["env"], +1.0)
    short_r = _rollout_fixed_action(_rising["env"], -1.0)

    # Mid-price fills, no fees: long and short over the same path are equal and
    # opposite up to compounding curvature.
    assert long_r > 0.0 and short_r < 0.0
    assert abs(long_r + short_r) < 0.25 * abs(long_r)


# ---------------------------------------------------------------------------
# Training actually performs gradient updates and changes the policy
# ---------------------------------------------------------------------------


def test_training_runs_gradient_updates_and_moves_the_actor(tmp_path: Path):
    rng = np.random.default_rng(0)
    close = np.maximum(100.0 + np.cumsum(0.02 + 0.05 * rng.standard_normal(400)), 1.0)
    ctx = build_training_context(
        config=_make_config(
            tmp_path,
            close,
            algorithm="TD3",
            max_steps=200,
            frames_per_batch=40,
            init_rand_steps=40,
        ),
        create_mlflow_callback=False,
    )
    trainer = ctx["trainer"]

    before = _flat_actor_params(trainer)
    logs = trainer.train(callback=None)
    after = _flat_actor_params(trainer)

    assert trainer.total_count >= 200, "training did not run to max_steps"
    assert len(logs.get("loss_value", [])) > 0, "no critic gradient updates ran"
    assert len(logs.get("loss_actor", [])) > 0, "no actor gradient updates ran"
    assert all(np.isfinite(v) for v in logs["loss_value"]), "non-finite critic loss"
    assert all(np.isfinite(v) for v in logs["loss_actor"]), "non-finite actor loss"
    assert not torch.allclose(before, after), (
        "actor parameters are unchanged after training -- the policy did not learn"
    )


def test_no_gradient_updates_when_warmup_covers_the_whole_run(tmp_path: Path):
    """Companion to the test above: proves its assertions can fail.

    With init_rand_steps >= max_steps every collected step is random warmup, so
    _optimization_step is never reached and the actor is untouched. This is the
    silent no-op condition (issue #356 class) that the smoke tests miss.
    """
    rng = np.random.default_rng(0)
    close = np.maximum(100.0 + np.cumsum(0.02 + 0.05 * rng.standard_normal(400)), 1.0)
    ctx = build_training_context(
        config=_make_config(
            tmp_path,
            close,
            algorithm="TD3",
            max_steps=200,
            frames_per_batch=40,
            init_rand_steps=240,  # > max_steps
        ),
        create_mlflow_callback=False,
    )
    trainer = ctx["trainer"]

    before = _flat_actor_params(trainer)
    logs = trainer.train(callback=None)
    after = _flat_actor_params(trainer)

    assert len(logs.get("loss_actor", [])) == 0
    assert len(logs.get("loss_value", [])) == 0
    assert torch.allclose(before, after)
