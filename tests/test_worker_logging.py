from __future__ import annotations

import os

from logger import setup_logging
from trading_rl.data import preparation


def test_setup_logging_propagates_defaults_to_spawned_workers(monkeypatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)

    setup_logging(level="WARNING", console_output=False, colored_output=True)

    assert os.environ["LOGURU_LEVEL"] == "WARNING"
    assert os.environ["LOGURU_COLORIZE"] == "YES"


def test_setup_logging_honors_no_color(monkeypatch) -> None:
    monkeypatch.setenv("NO_COLOR", "1")

    setup_logging(console_output=False, colored_output=True)

    assert os.environ["LOGURU_COLORIZE"] == "NO"


def test_worker_initializer_reuses_inherited_logging_policy(monkeypatch) -> None:
    calls: list[dict] = []
    monkeypatch.setenv("LOGURU_LEVEL", "ERROR")
    monkeypatch.setenv("LOGURU_COLORIZE", "YES")
    monkeypatch.setattr(
        preparation, "setup_logging", lambda **kwargs: calls.append(kwargs)
    )

    preparation._worker_log_init()

    assert calls == [{"level": "ERROR", "colored_output": True}]
