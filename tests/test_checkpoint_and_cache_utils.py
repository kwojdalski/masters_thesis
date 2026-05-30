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


from trading_rl.trainers.checkpointing import TrainingCheckpoint


def _manager(tmp_path, *, interval: int = 10) -> CheckpointManager:
    return CheckpointManager(
        checkpoint_dir=tmp_path / "checkpoints",
        checkpoint_prefix="experiment",
        interval=interval,
    )


def _snapshot() -> TrainingCheckpoint:
    return TrainingCheckpoint(
        algorithm="test",
        total_count=0,
        total_episodes=0,
        logs={},
        network_state={"actor_params_state": {}, "value_params_state": {}},
    )


def test_checkpoint_manager_does_not_save_before_interval(tmp_path) -> None:
    manager = _manager(tmp_path, interval=10)
    saved: list[str] = []
    manager.maybe_save(9, _snapshot)
    assert not list((tmp_path / "checkpoints").glob("*.pt")) if (tmp_path / "checkpoints").exists() else True


def test_checkpoint_manager_saves_when_interval_reached(tmp_path, monkeypatch) -> None:
    manager = _manager(tmp_path, interval=10)
    saved: list[str] = []
    monkeypatch.setattr("trading_rl.trainers.checkpointing.torch.save", lambda obj, path: saved.append(str(path)))
    monkeypatch.setattr(
        "trading_rl.trainers.checkpointing.CheckpointManager.save",
        lambda self, path, cp: saved.append(path),
    )

    manager.maybe_save(10, _snapshot)

    assert len(saved) == 1
    assert saved[0].endswith("experiment_checkpoint_step_10.pt")


def test_checkpoint_manager_updates_last_saved_step(tmp_path, monkeypatch) -> None:
    manager = _manager(tmp_path, interval=10)
    saved: list[str] = []
    monkeypatch.setattr(
        "trading_rl.trainers.checkpointing.CheckpointManager.save",
        lambda self, path, cp: saved.append(path),
    )

    manager.maybe_save(10, _snapshot)
    manager.maybe_save(15, _snapshot)

    assert len(saved) == 1


def test_interrupt_checkpoint_returns_none_without_checkpoint_config(tmp_path) -> None:
    manager = CheckpointManager(checkpoint_dir=None, checkpoint_prefix=None, interval=10)
    assert manager.save_interrupt(42, _snapshot) is None


def test_interrupt_checkpoint_saves_timestamped_file(tmp_path, monkeypatch) -> None:
    manager = _manager(tmp_path)
    saved: list[str] = []
    monkeypatch.setattr(
        "trading_rl.trainers.checkpointing.CheckpointManager.save",
        lambda self, path, cp: saved.append(path),
    )

    path = manager.save_interrupt(42, _snapshot)

    assert path is not None
    assert "experiment_checkpoint_interrupt_step_42_" in path
    assert saved == [path]


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
