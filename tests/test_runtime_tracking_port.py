"""Tests for the runtime-hook tracking boundary."""

from types import SimpleNamespace

from trading_rl.callbacks import tracking as tracking_module
from trading_rl.callbacks.tracking import MLflowExperimentTracker
from trading_rl.trainers.runtime_hooks import TrainerRuntimeHooks


class _Tracker:
    def __init__(self, active: bool) -> None:
        self.active = active

    def is_tracking_active(self) -> bool:
        return self.active


def test_runtime_hooks_use_trainer_callback_as_tracking_port() -> None:
    tracker = _Tracker(active=True)
    hooks = TrainerRuntimeHooks(SimpleNamespace(callback=tracker))

    assert hooks._active_tracker() is tracker


def test_runtime_hooks_disable_tracking_without_compatible_callback() -> None:
    hooks = TrainerRuntimeHooks(SimpleNamespace(callback=object()))

    assert hooks._active_tracker() is None


def test_experiment_tracker_delegates_mlflow_setup(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        tracking_module.mlflow,
        "set_tracking_uri",
        lambda uri: calls.append(("uri", uri)),
    )
    monkeypatch.setattr(
        tracking_module.mlflow,
        "set_experiment",
        lambda name: calls.append(("experiment", name)),
    )

    result = MLflowExperimentTracker().configure_experiment(
        "architecture-test", "sqlite:///tracking.db"
    )

    assert result == "architecture-test"
    assert calls == [
        ("uri", "sqlite:///tracking.db"),
        ("experiment", "architecture-test"),
    ]
