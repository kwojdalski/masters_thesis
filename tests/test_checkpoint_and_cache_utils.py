from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from trading_rl.cache_utils import clear_all_caches, get_cache_info
from trading_rl.pipeline.checkpoint import (
    _create_resumption_callback,
    _get_episode_count_from_trainer,
    _resolve_experiment_name_from_checkpoint,
    _setup_mlflow_tracking_from_checkpoint,
)
from trading_rl.trainers.checkpointing import CheckpointManager


class _Trainer:
    def __init__(self, tmp_path, *, interval: int = 10) -> None:
        self.config = SimpleNamespace(checkpoint_interval=interval, max_steps=100)
        self.checkpoint_dir = tmp_path / "checkpoints"
        self.checkpoint_prefix = "experiment"
        self.total_count = 0
        self.saved_paths: list[str] = []

    def save_checkpoint(self, path: str) -> None:
        self.saved_paths.append(path)
        pd.DataFrame({"x": [1]}).to_parquet(path)


def test_checkpoint_manager_does_not_save_before_interval(tmp_path) -> None:
    trainer = _Trainer(tmp_path, interval=10)
    trainer.total_count = 9
    manager = CheckpointManager(trainer)

    manager.maybe_save_checkpoint()

    assert trainer.saved_paths == []


def test_checkpoint_manager_saves_when_interval_reached(tmp_path) -> None:
    trainer = _Trainer(tmp_path, interval=10)
    trainer.total_count = 10
    manager = CheckpointManager(trainer)

    manager.maybe_save_checkpoint()

    assert len(trainer.saved_paths) == 1
    assert trainer.saved_paths[0].endswith("experiment_checkpoint_step_10.pt")


def test_checkpoint_manager_updates_last_saved_step(tmp_path) -> None:
    trainer = _Trainer(tmp_path, interval=10)
    manager = CheckpointManager(trainer)
    trainer.total_count = 10
    manager.maybe_save_checkpoint()
    trainer.total_count = 15
    manager.maybe_save_checkpoint()

    assert len(trainer.saved_paths) == 1


def test_interrupt_checkpoint_returns_none_without_checkpoint_config(tmp_path) -> None:
    trainer = _Trainer(tmp_path)
    trainer.checkpoint_dir = None
    manager = CheckpointManager(trainer)

    assert manager.save_interrupt_checkpoint() is None


def test_interrupt_checkpoint_saves_timestamped_file(tmp_path) -> None:
    trainer = _Trainer(tmp_path)
    trainer.total_count = 42
    manager = CheckpointManager(trainer)

    path = manager.save_interrupt_checkpoint()

    assert path is not None
    assert "experiment_checkpoint_interrupt_step_42_" in path
    assert trainer.saved_paths == [path]


def test_get_episode_count_prefers_logged_episode_count() -> None:
    trainer = SimpleNamespace(logs={"episode_log_count": [3, 7]}, total_episodes=99)

    assert _get_episode_count_from_trainer(trainer) == 7


def test_get_episode_count_handles_tensor_like_total() -> None:
    trainer = SimpleNamespace(logs={}, total_episodes=np.array(4))

    assert _get_episode_count_from_trainer(trainer) == 4


def test_setup_mlflow_tracking_from_checkpoint_sets_uri(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        "trading_rl.pipeline.checkpoint.mlflow.set_tracking_uri",
        lambda uri: calls.append(uri),
    )

    uri = _setup_mlflow_tracking_from_checkpoint(
        SimpleNamespace(mlflow_tracking_uri="sqlite:///run.db")
    )

    assert uri == "sqlite:///run.db"
    assert calls == ["sqlite:///run.db"]


def test_resolve_experiment_name_updates_config_and_checkpoint_prefix(monkeypatch) -> None:
    monkeypatch.setattr(
        "trading_rl.pipeline.checkpoint.mlflow.get_experiment",
        lambda experiment_id: SimpleNamespace(name="resumed_experiment"),
    )
    config = SimpleNamespace(experiment_name="current")
    trainer = SimpleNamespace(mlflow_experiment_id="12", checkpoint_prefix="current")

    effective = _resolve_experiment_name_from_checkpoint(
        trainer,
        config,
        "current",
        SimpleNamespace(info=lambda *args, **kwargs: None),
    )

    assert effective == "resumed_experiment"
    assert config.experiment_name == "resumed_experiment"
    assert trainer.checkpoint_prefix == "resumed_experiment"


def test_create_resumption_callback_prefers_configured_price_column(monkeypatch) -> None:
    captured = {}

    class CallbackProbe:
        def __init__(self, experiment_name, *, tracking_uri, price_series, start_run):
            captured["experiment_name"] = experiment_name
            captured["tracking_uri"] = tracking_uri
            captured["price_series"] = price_series
            captured["start_run"] = start_run

    monkeypatch.setattr("trading_rl.pipeline.checkpoint.MLflowTrainingCallback", CallbackProbe)
    train_df = pd.DataFrame({"mid": [10.0, 11.0], "close": [100.0, 101.0]})
    config = SimpleNamespace(
        env=SimpleNamespace(price_column="mid"),
        tracking=SimpleNamespace(tracking_uri="sqlite:///fallback.db"),
    )
    trainer = SimpleNamespace(logs={"episode_log_count": [2]}, total_episodes=0)

    callback = _create_resumption_callback(
        trainer,
        config,
        train_df,
        "exp",
        tracking_uri=None,
    )

    assert isinstance(callback, CallbackProbe)
    assert captured["tracking_uri"] == "sqlite:///fallback.db"
    pd.testing.assert_series_equal(captured["price_series"], train_df["mid"])
    assert callback._episode_count == 2


def test_get_cache_info_for_missing_dir(tmp_path) -> None:
    info = get_cache_info(str(tmp_path / "missing"))

    assert info == {"exists": False, "size_mb": 0, "num_files": 0}


def test_get_cache_info_counts_files_and_size(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "a.bin").write_bytes(b"1234")
    (cache_dir / "nested").mkdir()
    (cache_dir / "nested" / "b.bin").write_bytes(b"12")

    info = get_cache_info(str(cache_dir))

    assert info["exists"] is True
    assert info["num_files"] == 3
    assert info["size_mb"] == pytest.approx(6 / (1024 * 1024))


def test_clear_all_caches_removes_existing_cache_dir(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "x.txt").write_text("cached")

    clear_all_caches(str(cache_dir))

    assert not cache_dir.exists()
