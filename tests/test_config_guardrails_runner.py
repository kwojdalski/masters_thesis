from __future__ import annotations

import pytest

from trading_rl.config import ExperimentConfig
from trading_rl.config_guardrails_runner import run_guardrail_check


class _FakeProgress:
    """Records stop/start ordering relative to input() the way Rich's Progress does."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def stop(self) -> None:
        self.calls.append("stop")

    def start(self) -> None:
        self.calls.append("start")


def _warn_config() -> ExperimentConfig:
    # buffer_size > max_steps triggers a WARN-level guardrail finding.
    config = ExperimentConfig()
    config.training.max_steps = 100
    config.training.buffer_size = 100_000
    return config


def test_warn_prompt_stops_and_restarts_progress_bar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    progress = _FakeProgress()
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda: "y")

    run_guardrail_check(_warn_config(), progress_bar=progress)

    assert progress.calls == ["stop", "start"]


def test_warn_prompt_restarts_progress_bar_even_if_declined(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    progress = _FakeProgress()
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda: "n")

    with pytest.raises(SystemExit):
        run_guardrail_check(_warn_config(), progress_bar=progress)

    assert progress.calls == ["stop", "start"]


def test_non_interactive_stdin_skips_prompt_without_touching_progress_bar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    progress = _FakeProgress()
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)

    def _fail_input() -> str:
        raise AssertionError(
            "input() should not be called when stdin is non-interactive"
        )

    monkeypatch.setattr("builtins.input", _fail_input)

    run_guardrail_check(_warn_config(), progress_bar=progress)

    assert progress.calls == []


def test_no_progress_bar_is_a_safe_no_op(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda: "y")

    run_guardrail_check(_warn_config(), progress_bar=None)
