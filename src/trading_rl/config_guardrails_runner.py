"""Interactive guardrail runner — prints findings, prompts, raises on FATAL.

Separated from the check functions so the validation logic in
``config_guardrails_checks`` can be tested without any I/O.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.markup import escape

from logger import get_logger
from trading_rl.config_guardrails_checks import Severity, check_config_guardrails

if TYPE_CHECKING:
    from trading_rl.config import ExperimentConfig

logger = get_logger(__name__)

_console = Console()
_err_console = Console(stderr=True)


def run_guardrail_check(config: ExperimentConfig, progress_bar: Any = None) -> None:
    """Check config, log findings, raise on FATAL, prompt on WARN.

    Controlled by training config flags:
      skip_guardrails=True       — skip the check entirely (dev/smoke runs).
      skip_guardrail_prompts=True — log WARNs but proceed without prompting.

    Args:
        config: Experiment config to validate.
        progress_bar: Active Rich ``Progress``/``Live`` display, if any. When
            given, it is paused around the WARN confirmation prompt so the
            prompt text isn't clobbered by the display's next redraw.
    """
    if getattr(config.training, "skip_guardrails", False):
        logger.debug("config guardrail check skipped (training.skip_guardrails=True)")
        return

    findings = check_config_guardrails(config)
    if not findings:
        logger.info("config guardrail check passed — no issues found")
        return

    skip_prompts = getattr(config.training, "skip_guardrail_prompts", False)
    scenario_name = getattr(config, "experiment_name", None) or ""
    scenario_tag = (
        f"  [dim]{escape(f'[{scenario_name}]')}[/dim]" if scenario_name else ""
    )

    fatals = [f for f in findings if f.severity == Severity.FATAL]
    warns = [f for f in findings if f.severity == Severity.WARN]

    # Always surface every finding in the log regardless of skip_prompts.
    for f in findings:
        logger.warning(
            "guardrail {} [{}] {} | suggestion: {}",
            f.severity.value,
            f.parameter,
            f.message,
            f.suggestion,
        )

    if fatals:
        _err_console.print(
            f"\n[bold red]CONFIG GUARDRAIL — FATAL ERRORS[/bold red]{scenario_tag}"
        )
        _err_console.print("=" * 50)
        for i, f in enumerate(fatals, 1):
            _err_console.print(
                f"\n[bold cyan][{i}][/bold cyan] [yellow]{escape(f.parameter)}[/yellow]"
            )
            _err_console.print(f"    Problem:    {escape(f.message)}")
            _err_console.print(f"    Fix:        [cyan]{escape(f.suggestion)}[/cyan]")
        _err_console.print("\nTraining cannot start with these settings.")
        raise ValueError(
            f"Config guardrail check failed with {len(fatals)} fatal error(s). "
            "See output above for details."
        )

    if warns:
        _console.print(
            f"\n[bold yellow]CONFIG GUARDRAIL — WARNINGS[/bold yellow]{scenario_tag}"
        )
        _console.print("=" * 50)
        for i, f in enumerate(warns, 1):
            _console.print(
                f"\n[bold cyan][{i}][/bold cyan] [yellow]{escape(f.parameter)}[/yellow]"
            )
            _console.print(f"    Problem:    {escape(f.message)}")
            _console.print(f"    Suggestion: [cyan]{escape(f.suggestion)}[/cyan]")

        if skip_prompts:
            _console.print(
                "\n[dim][guardrail] training.skip_guardrail_prompts=True — proceeding without prompt.[/dim]\n"
            )
            return

        # Non-interactive stdin (CI, scripts, pytest) — warn and continue.
        if not sys.stdin.isatty():
            logger.warning(
                "guardrail: {} warning(s) found but stdin is non-interactive — "
                "proceeding automatically. Set training.skip_guardrail_prompts=true "
                "to silence this message.",
                len(warns),
            )
            return

        if progress_bar is not None:
            progress_bar.stop()
        try:
            _console.print("\nProceed anyway? [y/N] ", end="")
            try:
                answer = input().strip().lower()
            except (EOFError, OSError):
                answer = "n"
        finally:
            if progress_bar is not None:
                progress_bar.start()

        if answer not in {"y", "yes"}:
            raise SystemExit(
                "Aborted by user after config guardrail warnings. "
                "Fix the issues above or set training.skip_guardrail_prompts=true "
                "to suppress the prompt."
            )
