"""Lightweight CLI subcommands must not pull in the training/evaluation stack.

Regression guard for the lazy-import layout of ``src/cli.py`` + ``cli.commands``:
command classes and ``trading_rl`` are imported inside each callback, so parsing
args / printing help / running a thin subcommand (dashboard, checkpoints, ...)
stays fast and does not load torch, torchrl, or the statistical-test registry.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_HEAVY = (
    "torch",
    "torchrl",
    "trading_rl.evaluation",
    "trading_rl.train_trading_agent",
)


def test_importing_cli_module_does_not_load_training_stack() -> None:
    code = (
        "import sys, cli\n"
        f"bad = [m for m in {_HEAVY!r} if m in sys.modules]\n"
        "print('LOADED:' + ','.join(bad))\n"
        "sys.exit(1 if bad else 0)\n"
    )
    result = subprocess.run(  # noqa: S603 -- sys.executable + fixed args
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("argv", [["--help"], ["dashboard", "--help"]])
def test_help_does_not_register_statistical_tests(argv: list[str]) -> None:
    result = subprocess.run(  # noqa: S603 -- sys.executable + fixed args
        [sys.executable, "src/cli.py", *argv],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "registered statistical test" not in (result.stdout + result.stderr)
