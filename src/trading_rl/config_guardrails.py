"""Backwards-compatible re-export shim for trading_rl.config_guardrails.

The guardrail logic has been split into focused modules:
  - config_guardrails_checks.py  — Severity, Finding, all check functions,
                                   _ALL_CHECKS registry, check_config_guardrails
  - config_guardrails_runner.py  — run_guardrail_check (stdin/stderr/prompt layer)

Import from those modules directly for new code.
"""

from __future__ import annotations

from trading_rl.config_guardrails_checks import (
    _ALL_CHECKS,
    Finding,
    Severity,
    _is_off_policy,
    _is_ppo,
    check_config_guardrails,
)
from trading_rl.config_guardrails_runner import run_guardrail_check

__all__ = [
    "_ALL_CHECKS",
    "Finding",
    "Severity",
    "_is_off_policy",
    "_is_ppo",
    "check_config_guardrails",
    "run_guardrail_check",
]
