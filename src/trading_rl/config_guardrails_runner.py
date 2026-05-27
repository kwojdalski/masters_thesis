"""Interactive guardrail runner — prints findings, prompts, raises on FATAL.

Separated from the check functions so the validation logic in
``config_guardrails_checks`` can be tested without any I/O.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from logger import get_logger
from trading_rl.config_guardrails_checks import Finding, Severity, check_config_guardrails

if TYPE_CHECKING:
    from trading_rl.config import ExperimentConfig

logger = get_logger(__name__)


def run_guardrail_check(config: ExperimentConfig) -> None:
    """Check config, log findings, raise on FATAL, prompt on WARN.

    Controlled by training.skip_guardrail_prompts:
      False (default) — print WARN findings and ask for y/N confirmation.
      True            — log WARNs but proceed without prompting (for scripts).
    """
    findings = check_config_guardrails(config)
    if not findings:
        logger.info("config guardrail check passed — no issues found")
        return

    skip_prompts = getattr(config.training, "skip_guardrail_prompts", False)

    fatals = [f for f in findings if f.severity == Severity.FATAL]
    warns  = [f for f in findings if f.severity == Severity.WARN]

    # Always surface every finding in the log regardless of skip_prompts.
    for f in findings:
        logger.warning(
            "guardrail %s [%s] %s | suggestion: %s",
            f.severity.value, f.parameter, f.message, f.suggestion,
        )

    if fatals:
        lines = ["\nCONFIG GUARDRAIL — FATAL ERRORS\n" + "=" * 50]
        for i, f in enumerate(fatals, 1):
            lines.append(f"\n[{i}] {f.parameter}")
            lines.append(f"    Problem:    {f.message}")
            lines.append(f"    Fix:        {f.suggestion}")
        lines.append("\nTraining cannot start with these settings.")
        print("\n".join(lines), file=sys.stderr)
        raise ValueError(
            f"Config guardrail check failed with {len(fatals)} fatal error(s). "
            "See output above for details."
        )

    if warns:
        lines = ["\nCONFIG GUARDRAIL — WARNINGS\n" + "=" * 50]
        for i, f in enumerate(warns, 1):
            lines.append(f"\n[{i}] {f.parameter}")
            lines.append(f"    Problem:    {f.message}")
            lines.append(f"    Suggestion: {f.suggestion}")
        print("\n".join(lines))

        if skip_prompts:
            print(
                "\n[guardrail] training.skip_guardrail_prompts=True — proceeding without prompt.\n"
            )
            return

        # Non-interactive stdin (CI, scripts, pytest) — warn and continue.
        if not sys.stdin.isatty():
            logger.warning(
                "guardrail: %d warning(s) found but stdin is non-interactive — "
                "proceeding automatically. Set training.skip_guardrail_prompts=true "
                "to silence this message.",
                len(warns),
            )
            return

        print("\nProceed anyway? [y/N] ", end="", flush=True)
        try:
            answer = input().strip().lower()
        except (EOFError, OSError):
            answer = "n"

        if answer not in {"y", "yes"}:
            raise SystemExit(
                "Aborted by user after config guardrail warnings. "
                "Fix the issues above or set training.skip_guardrail_prompts=true "
                "to suppress the prompt."
            )
