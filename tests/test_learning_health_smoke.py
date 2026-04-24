"""Learning health smoke tests.

These tests verify that a minimal optimization step produces numerically sane
gradients, losses, and parameter updates. They are designed to catch silent
failures — cases where training runs without error but the agent is not
actually learning:

  - NaN in rewards or observations that propagates to NaN loss
  - A frozen feature pipeline that feeds constant observations to the network
  - An optimizer that is not attached to the network parameters
  - Gradient explosion caused by a reward scaling change or learning-rate bump
  - A feature/network shape mismatch that causes silent wrong behaviour

Each test uses a tiny configuration (8 collected frames, 1 PPO epoch) and
inspects the internal state of the trainer after a single optimization step.
They run as part of the standard ``smoke`` marker and should complete in a
few seconds each.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from trading_rl.config import ExperimentConfig
from trading_rl.train_trading_agent import build_training_context


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------


def _write_dataset(path: Path, periods: int = 80) -> Path:
    """Write a small parquet file with a noisy upward-trending price series.

    A purely linear series produces constant log-returns, which would make
    the reward-variance check trivially pass for the wrong reason. Adding
    Gaussian noise ensures that rewards genuinely vary across steps.
    """
    idx = pd.date_range("2024-01-01", periods=periods, freq="h")
    rng = np.random.default_rng(42)
    noise = rng.normal(0.0, 0.4, periods).cumsum()
    close = pd.Series(100.0 + noise + np.linspace(0, 8, periods), index=idx)
    df = pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "volume": 1000.0 + close * 2.0,
        },
        index=idx,
    )
    df.to_parquet(path)
    return path


def _write_feature_config(path: Path) -> Path:
    path.write_text(
        "features:\n"
        '  - name: "lag1"\n'
        '    feature_type: "return_lag"\n'
        "    normalize: true\n"
        "    params:\n"
        '      column: "close"\n'
        "      lag: 1\n"
        '  - name: "trend"\n'
        '    feature_type: "trend"\n'
        "    normalize: false\n",
        encoding="utf-8",
    )
    return path


def _make_health_config(tmp_path: Path) -> ExperimentConfig:
    data_path = _write_dataset(tmp_path / "data.parquet")
    feature_config = _write_feature_config(tmp_path / "features.yaml")
    return ExperimentConfig.from_dict(
        {
            "experiment_name": "health_smoke",
            "data": {
                "data_path": str(data_path),
                "train_size": 48,
                "validation_size": 16,
                "download_data": False,
                "feature_config": str(feature_config),
            },
            "env": {
                "backend": "tradingenv",
                "price_column": "close",
                "feature_columns": ["feature_lag1", "feature_trend"],
            },
            "training": {
                "algorithm": "PPO",
                "max_steps": 8,
                "frames_per_batch": 8,
                "sample_size": 4,
                "init_rand_steps": 0,
                "ppo_epochs": 1,
                "eval_steps": 4,
                "log_interval": 1000,
            },
            "logging": {
                "log_dir": str(tmp_path / "logs"),
                "log_level": "WARNING",
                "save_plots": False,
            },
            "tracking": {
                "tracking_uri": f"file://{tmp_path / 'mlruns'}",
            },
        }
    )


def _collect_one_batch(trainer):
    """Collect exactly one batch from the environment without running the full loop."""
    return next(iter(trainer.collector))


def _run_one_optimization_step(trainer) -> None:
    """Collect one batch and run one PPO optimization step.

    This replicates what ``_run_training_loop`` does for each collected batch,
    but without the surrounding logging and checkpoint machinery.
    """
    data = _collect_one_batch(trainer)
    trainer._current_batch = data
    trainer.total_count = data.numel()
    max_length = int(data["next", "step_count"].max().item())
    trainer._optimization_step(0, max_length, data.numel())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLearningHealth:
    """Smoke tests for gradient and loss health after a single optimization step.

    A change that breaks learning — e.g. a reward that always returns zero,
    a feature that produces NaN, or a misconfigured optimizer — will cause at
    least one of these tests to fail before any full training run is wasted.
    """

    pytestmark = pytest.mark.smoke

    @pytest.fixture()
    def context(self, tmp_path: Path) -> dict:
        config = _make_health_config(tmp_path)
        return build_training_context(config=config, create_mlflow_callback=False)

    # ------------------------------------------------------------------
    # 1. Shape alignment
    # ------------------------------------------------------------------

    def test_observation_shape_matches_n_obs(self, context: dict) -> None:
        """The observation dimension produced by the environment must equal n_obs.

        A mismatch here means either a feature was added/removed without
        updating the network input size, or the environment is producing extra
        dimensions that the trainer is silently ignoring.
        """
        n_obs = context["n_obs"]
        env = context["env"]

        obs_td = env.reset()
        actual_dim = int(obs_td["observation"].shape[-1])

        assert n_obs == actual_dim, (
            f"n_obs={n_obs} reported by the training context does not match "
            f"the actual observation dimension={actual_dim} produced by the "
            "environment. A feature was added or removed without updating the "
            "network input size, or feature_columns is out of sync with the "
            "feature pipeline output."
        )

    # ------------------------------------------------------------------
    # 2. Reward signal health
    # ------------------------------------------------------------------

    def test_reward_is_finite(self, context: dict) -> None:
        """Every reward in the first collected batch must be a finite number.

        Non-finite rewards (NaN, +/-inf) propagate through the loss function
        and make all parameter gradients NaN, after which no learning occurs.
        Common causes: log(0) in log-return reward when portfolio value hits
        zero; division by zero in the DSR wrapper; incorrect price column.
        """
        trainer = context["trainer"]
        data = _collect_one_batch(trainer)
        rewards = data["next", "reward"].detach().numpy().flatten()

        non_finite = [float(r) for r in rewards if not math.isfinite(float(r))]
        assert not non_finite, (
            f"Batch contains {len(non_finite)} non-finite reward(s): "
            f"{non_finite[:5]}. Check the reward function for division by zero "
            "or log of a non-positive portfolio value."
        )

    def test_reward_is_not_constant(self, context: dict) -> None:
        """Rewards must not all be identical across the first collected batch.

        A constant reward — most commonly all zeros — provides no gradient
        signal. The agent's loss becomes a constant, parameter updates are
        zero or noise, and the policy does not improve.  Common causes:
        reward function short-circuited to 0.0, wrong portfolio valuation
        key, or trading fees set so high that every episode ends at break-even.
        """
        trainer = context["trainer"]
        data = _collect_one_batch(trainer)
        rewards = data["next", "reward"].detach().numpy().flatten()

        reward_std = float(np.std(rewards))
        assert reward_std > 1e-8, (
            f"All rewards in the batch are effectively identical "
            f"(std={reward_std:.2e}). The reward function is not producing a "
            "meaningful learning signal. Verify that the portfolio valuation "
            "changes across steps and that the reward key is correctly wired."
        )

    # ------------------------------------------------------------------
    # 3. Observation health
    # ------------------------------------------------------------------

    def test_observations_are_finite(self, context: dict) -> None:
        """All observations in the first batch must be finite.

        NaN or inf values in observations corrupt every downstream computation:
        the network forward pass produces NaN activations, the loss is NaN,
        and gradients are NaN. The most common cause is a feature that divides
        by zero during normalization (e.g. a constant-valued column with std=0).
        """
        trainer = context["trainer"]
        data = _collect_one_batch(trainer)
        obs = data["observation"].detach().numpy()

        n_bad = int((~np.isfinite(obs)).sum())
        assert n_bad == 0, (
            f"Batch observations contain {n_bad} non-finite value(s). "
            "Check the feature pipeline for division by zero in normalization "
            "or a feature that returns NaN for the given price series."
        )

    def test_each_feature_column_varies_across_batch(self, context: dict) -> None:
        """Every feature column must vary across the collected batch.

        A column that is constant across all steps carries no information and
        indicates a frozen or broken feature (e.g. a lag feature applied to a
        column that never changes, or normalization that clips everything to
        the same value). The network still trains but learns nothing from that
        feature, and the gradient with respect to that input is always zero.
        """
        trainer = context["trainer"]
        data = _collect_one_batch(trainer)
        obs = data["observation"].detach().numpy()

        feature_names = context["prepared_dataset"].feature_columns
        n_features = obs.shape[-1]

        for col_idx in range(n_features):
            col_std = float(np.std(obs[..., col_idx]))
            name = feature_names[col_idx] if col_idx < len(feature_names) else f"col_{col_idx}"
            assert col_std > 1e-8, (
                f"Feature '{name}' (index {col_idx}) is constant across the "
                f"entire batch (std={col_std:.2e}). This feature carries no "
                "information and may indicate a broken feature implementation "
                "or a normalization step that collapses all values."
            )

    # ------------------------------------------------------------------
    # 4. Optimization health
    # ------------------------------------------------------------------

    def test_loss_is_finite_after_one_step(self, context: dict) -> None:
        """Actor and critic losses must be finite numbers after one optimizer step.

        A NaN or inf loss means that a NaN has propagated from the reward or
        observations through the forward pass of the network. Once a NaN loss
        appears, all subsequent gradient updates are NaN and training is
        effectively dead, even though it continues to run.
        """
        trainer = context["trainer"]
        _run_one_optimization_step(trainer)

        assert trainer.logs.get("loss_actor"), (
            "No actor loss was logged after an optimization step. "
            "Check that _optimization_step correctly populates trainer.logs."
        )
        assert trainer.logs.get("loss_value"), (
            "No critic loss was logged after an optimization step."
        )

        for loss_name in ("loss_actor", "loss_value"):
            for v in trainer.logs[loss_name]:
                assert math.isfinite(v), (
                    f"{loss_name}={v} is not finite after one optimization step. "
                    "A NaN has propagated from the reward function, feature "
                    "pipeline, or network initialization through the loss."
                )

    def test_actor_parameters_change_after_update(self, context: dict) -> None:
        """Actor parameters must be updated after one optimization step.

        Parameters that do not change indicate one of: the optimizer is not
        attached to the actor parameters, the learning rate is zero, or the
        gradients are exactly zero (e.g. because all rewards are identical and
        the advantage estimate is zero everywhere).
        """
        trainer = context["trainer"]

        before = {
            name: param.data.clone()
            for name, param in trainer.actor.named_parameters()
        }
        _run_one_optimization_step(trainer)

        any_changed = any(
            not torch.equal(before[name], param.data)
            for name, param in trainer.actor.named_parameters()
        )
        assert any_changed, (
            "No actor parameter changed after one optimization step. "
            "The optimizer may not be attached to the actor network, the "
            "learning rate may be zero, or the gradient signal is degenerate."
        )

    def test_actor_parameters_are_finite_after_update(self, context: dict) -> None:
        """Actor parameters must be finite (no NaN, no inf) after one update.

        Non-finite parameters arise from gradient explosion: the loss gradient
        overflows, the parameter update step sets weights to NaN or inf, and
        all subsequent forward passes produce NaN outputs. This is invisible
        at the training loop level — training continues, loss is logged as NaN,
        but the agent is irrecoverably broken. Common triggers: learning rate
        too high, reward scale change, or removing gradient clipping.
        """
        trainer = context["trainer"]
        _run_one_optimization_step(trainer)

        for name, param in trainer.actor.named_parameters():
            n_bad = int((~torch.isfinite(param.data)).sum().item())
            assert n_bad == 0, (
                f"Actor parameter '{name}' has {n_bad} non-finite value(s) "
                "after one optimization step. Gradient explosion has occurred. "
                "Check the learning rate, reward scaling, and whether gradient "
                "clipping is still active."
            )

    def test_critic_parameters_are_finite_after_update(self, context: dict) -> None:
        """Value network parameters must be finite after one optimization step."""
        trainer = context["trainer"]
        _run_one_optimization_step(trainer)

        for name, param in trainer.value_net.named_parameters():
            n_bad = int((~torch.isfinite(param.data)).sum().item())
            assert n_bad == 0, (
                f"Value network parameter '{name}' has {n_bad} non-finite "
                "value(s) after one optimization step."
            )
