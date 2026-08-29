"""
Core logging functionality for the project.

Thin wrapper around loguru that preserves the original public API
(setup_logging, get_logger, configure_logging, …) while delegating all
formatting, coloring, rotation, and serialisation to loguru.
"""

import logging
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger

# ---------------------------------------------------------------------------
# Level-check helper (replaces logging.Logger.isEnabledFor)
# ---------------------------------------------------------------------------

def is_level_enabled(level: str) -> bool:
    """Return True if messages at *level* would reach at least one handler."""
    try:
        return logger.level(level).no >= logger._core.min_level  # type: ignore[attr-defined]
    except Exception:
        return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_DEFAULT_FMT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)

_PLAIN_FMT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
    "{name}:{function}:{line} - {message}"
)

# Matches the value part of key=value tokens in log messages.
# Excludes whitespace and most punctuation used by loguru markup (<>) and
# Python format strings ({}). Commas between digits are allowed so that
# comma-formatted numbers like 11,643 are captured as a single token.
_KV_VALUE_RE = re.compile(r"(?<==)([^\s,\)\]\[|{}<>]+(?:,[0-9]+)*)")

_PATH_EXTENSIONS: frozenset[str] = frozenset({
    ".py", ".yaml", ".yml", ".npy", ".parquet", ".json", ".csv",
    ".txt", ".pkl", ".pt", ".pth", ".log", ".db",
})


def _looks_like_path(val: str) -> bool:
    if val.startswith(("/", "./", "../")):
        return True
    dot = val.rfind(".")
    return dot > 0 and val[dot:] in _PATH_EXTENSIONS


def _highlight_kv(msg: str) -> str:
    """Color key=value tokens by value type:
    - True → green, False → red
    - None → dim
    - paths → bold
    - negative numbers → red, positive numbers → magenta
    - strings → light-cyan
    """
    def _replace(m: re.Match) -> str:
        val = m.group(1)

        if val == "True":
            return f"<green>{val}</green>"
        if val == "False":
            return f"<red>{val}</red>"
        if val == "None":
            return f"<dim>{val}</dim>"
        if _looks_like_path(val):
            return f"<bold>{val}</bold>"

        # Strip trailing % and common unit suffixes for numeric check
        check_val = val.rstrip("%")
        for suffix in ("us/step", "us", "ms"):
            if check_val.endswith(suffix) and len(check_val) > len(suffix):
                check_val = check_val[: -len(suffix)]
                break

        # Handle N/M fraction (e.g., step=96921/969218) and N/s rate
        if "/" in check_val:
            parts = check_val.split("/")
            if len(parts) == 2:
                left, right = parts
                left_ok = left.replace(".", "", 1).replace("-", "", 1).isdigit()
                right_ok = right == "s" or right.replace(".", "", 1).replace("-", "", 1).isdigit()
                if left_ok and right_ok:
                    return f"<magenta>{val}</magenta>"

        try:
            num = float(check_val.replace(",", ""))
            return f"<red>{val}</red>" if num < 0 else f"<magenta>{val}</magenta>"
        except ValueError:
            return f"<light-cyan>{val}</light-cyan>"

    return _KV_VALUE_RE.sub(_replace, msg)


def _make_kv_format(fmt: str) -> Any:
    """Return a loguru format callable that highlights key=value pairs."""
    def _format(record: dict) -> str:
        # Escape bare < so loguru's colorizer does not treat them as color tags.
        # In a format callable, loguru strips the leading \ and renders plain <.
        raw = record["message"].replace("<", r"\<")
        highlighted = _highlight_kv(raw)
        # Escape bare braces so format_map doesn't misinterpret message content.
        safe = highlighted.replace("{", "{{").replace("}", "}}")
        return fmt.replace("{message}", safe, 1) + "\n"
    return _format


def setup_logging(
    level: str = "INFO",
    log_file: str | None = None,
    console_output: bool = True,
    colored_output: bool = True,
    structured_logging: bool = False,
    max_file_size: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
    format_string: str | None = None,
    log_regex: str | None = None,
) -> Any:
    """Configure loguru sinks for the project and return the logger."""
    logger.remove()

    use_color = (
        colored_output and not structured_logging and "NO_COLOR" not in os.environ
    )
    # Spawned workers import Loguru before application initializers run. These
    # defaults make their initial sink match the parent process from first import.
    os.environ["LOGURU_LEVEL"] = level.upper()
    os.environ["LOGURU_COLORIZE"] = "YES" if use_color else "NO"

    if log_regex is None:
        log_regex = os.environ.get("LOG_REGEX")

    compiled_re = None
    if log_regex:
        try:
            compiled_re = re.compile(log_regex)
        except re.error:
            # logger has no sinks yet at this point in setup, so print is the only way out
            print(f"Invalid LOG_REGEX ignored: {log_regex}", file=sys.stderr)  # noqa: T201

    def _filter(record: dict) -> bool:
        if compiled_re is not None:
            return bool(compiled_re.search(record["message"]))
        return True

    if format_string is None:
        fmt = _DEFAULT_FMT if (colored_output and not structured_logging) else _PLAIN_FMT
    else:
        fmt = format_string

    if console_output:
        console_fmt = _make_kv_format(fmt) if use_color else fmt
        logger.add(
            sys.stdout,
            level=level.upper(),
            format=console_fmt,
            colorize=use_color,
            serialize=structured_logging,
            filter=_filter,
        )

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            level=level.upper(),
            rotation=max_file_size,
            retention=backup_count,
            serialize=structured_logging,
            encoding="utf-8",
        )

    # Silence noisy third-party stdlib loggers that don't use loguru
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("selectors").setLevel(logging.WARNING)
    logging.getLogger("joblib").setLevel(logging.ERROR)
    logging.getLogger("torchrl").setLevel(logging.WARNING)

    return logger


def get_logger(name: str, extra_config: dict[str, Any] | None = None) -> Any:
    """Return the loguru logger singleton (name is accepted for API compat)."""
    return logger


def configure_logging(
    component: str = "",
    debug: bool = False,
    level: str = "INFO",
    simplified: bool = True,
    log_dir: str | None = None,
    structured_logging: bool = False,
    include_console: bool = True,
    log_regex: str | None = None,
) -> Any:
    """Configure logging for project components."""
    effective_level = "DEBUG" if debug else level

    log_file = None
    if log_dir:
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(UTC).strftime("%Y%m%d")
        log_file = log_dir_path / f"{component}_{timestamp}.log"

    if simplified:
        fmt = "{function}:{line} - {message}"
    else:
        fmt = (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
            f"{component} | {{name}}:{{function}}:{{line}} - {{message}}"
        )

    return setup_logging(
        level=effective_level,
        log_file=str(log_file) if log_file else None,
        console_output=include_console,
        colored_output=not structured_logging,
        structured_logging=structured_logging,
        format_string=fmt if not structured_logging else None,
        log_regex=log_regex,
    )


def setup_component_logger(
    component_name: str,
    log_level: str = "INFO",
    log_to_file: bool = True,
    log_dir: str = "logs",
) -> Any:
    """Return the loguru logger (component setup is done via setup_logging)."""
    return logger


def configure_weles_logging(
    component: str = "weles",
    debug: bool = False,
    level: str = "INFO",
    simplified: bool = True,
    log_dir: str | None = None,
    structured_logging: bool = False,
    include_console: bool = True,
) -> Any:
    """Configure logging for Weles components (compatibility alias)."""
    return configure_logging(
        component=component,
        debug=debug,
        level=level,
        simplified=simplified,
        log_dir=log_dir,
        structured_logging=structured_logging,
        include_console=include_console,
    )
