"""Tests for per-trial teardown in run_multiple_experiments (#363)."""

from __future__ import annotations

import contextlib
import gc
from typing import Any

import trading_rl.train_trading_agent as tta


def test_run_multiple_experiments_collects_gc_between_trials(
    monkeypatch, tmp_path
) -> None:
    """Each trial boundary must force gc.collect() (#363)."""
    monkeypatch.setattr(tta, "setup_mlflow_experiment", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        tta.mlflow, "start_run", lambda *args, **kwargs: contextlib.nullcontext()
    )

    trials: list[Any] = []
    monkeypatch.setattr(
        tta,
        "run_single_experiment",
        lambda **kwargs: trials.append(kwargs) or {},
    )

    collected: list[bool] = []
    monkeypatch.setattr(
        gc, "collect", lambda: collected.append(True) or 0, raising=True
    )

    result = tta.run_multiple_experiments(
        n_trials=3,
        base_seed=42,
        show_progress=False,
    )

    assert isinstance(result, str)
    assert len(trials) == 3
    # One forced collection per completed trial, between trials.
    assert len(collected) == 3
