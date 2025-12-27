"""
Core logging functionality for the project.

This module provides the main logging setup and configuration utilities.
"""

import json
import logging
import os
import re
import sys
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, ClassVar


def _supports_color() -> bool:
    """Check if the current environment supports colored output."""
    # Check for explicit environment variables
    if os.environ.get("FORCE_COLOR") in ("1", "true", "True"):
        return True
    if os.environ.get("NO_COLOR"):
        return False
    
    # Check if we're in IPython/Jupyter
    try:
        # This will succeed if we're in IPython
        get_ipython()  # type: ignore
        return True
    except NameError:
        pass
    
    # Check for common color-supporting terminals
    if os.environ.get("COLORTERM") in ("truecolor", "24bit"):
        return True
    
    term = os.environ.get("TERM", "")
    if any(term_type in term for term_type in ["color", "256", "xterm"]):
        return True
    
    # Fallback to isatty check
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


class ColoredFormatter(logging.Formatter):
    """Colored log formatter for enhanced console readability."""

    COLORS: ClassVar[dict[str, str]] = {
        "DEBUG": "\033[96m",    # Bright cyan
        "INFO": "\033[92m",     # Bright green  
        "WARNING": "\033[93m",  # Bright yellow
        "ERROR": "\033[91m",    # Bright red
        "CRITICAL": "\033[95m", # Bright magenta
        "RESET": "\033[0m",     # Reset
        "BOLD": "\033[1m",      # Bold
        "DIM": "\033[2m",       # Dim
    }

    def format(self, record):
        if not _supports_color():
            return super().format(record)
            
        log_color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
        
        # Color the level name with bold
        colored_levelname = (
            f"{self.COLORS['BOLD']}{log_color}{record.levelname}{self.COLORS['RESET']}"
        )
        
        # Store original levelname and replace temporarily
        original_levelname = record.levelname
        record.levelname = colored_levelname
        
        # Format the message
        formatted = super().format(record)
        
        # Restore original levelname
        record.levelname = original_levelname
        
        return formatted


class StructuredFormatter(logging.Formatter):
    """JSON structured formatter for machine-readable logs."""

    def format(self, record):
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        if hasattr(record, "extra_data"):
            log_entry.update(record.extra_data)

        return json.dumps(log_entry)


def setup_logging(
    level: str = "INFO",
    log_file: str | None = None,
    console_output: bool = True,
    colored_output: bool = True,
    structured_logging: bool = False,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    format_string: str | None = None,
    log_regex: str | None = None,
) -> logging.Logger:
    """
    Set up comprehensive logging configuration for the project.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to write logs to
        console_output: Whether to output logs to console
        colored_output: Whether to use colored console output
        structured_logging: Whether to use JSON structured logging
        max_file_size: Maximum log file size before rotation (bytes)
        backup_count: Number of backup log files to keep
        format_string: Custom format string for log messages

    Returns:
        Configured root logger
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Default format string
    if format_string is None:
        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s "
            + "- %(funcName)s:%(lineno)d - %(message)s"
        )

    # Clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(numeric_level)
    # Silence noisy asyncio selector debug output on macOS
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("selectors").setLevel(logging.WARNING)

    if log_regex is None:
        log_regex = os.environ.get("LOG_REGEX")

    class _RegexFilter(logging.Filter):
        def __init__(self, pattern: str):
            super().__init__()
            self._pattern = re.compile(pattern)

        def filter(self, record: logging.LogRecord) -> bool:
            return bool(self._pattern.search(record.getMessage()))

    regex_filter = None
    if log_regex:
        try:
            regex_filter = _RegexFilter(log_regex)
        except re.error:
            print(
                f"Invalid LOG_REGEX pattern ignored: {log_regex}",
                file=sys.stderr,
            )

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)

        if structured_logging:
            console_formatter = StructuredFormatter()
        elif colored_output and _supports_color():
            console_formatter = ColoredFormatter(format_string)
        else:
            console_formatter = logging.Formatter(format_string)

        console_handler.setFormatter(console_formatter)
        if regex_filter is not None:
            console_handler.addFilter(regex_filter)
        root_logger.addHandler(console_handler)

    # File handler with rotation
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            log_file, maxBytes=max_file_size, backupCount=backup_count, encoding="utf-8"
        )
        file_handler.setLevel(numeric_level)

        if structured_logging:
            file_formatter = StructuredFormatter()
        else:
            file_formatter = logging.Formatter(format_string)

        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    return root_logger


def get_logger(name: str, extra_config: dict[str, Any] | None = None) -> logging.Logger:
    """
    Get a logger instance for the given name with optional extra configuration.

    This follows the standard pattern used throughout the project.

    Args:
        name: Logger name (typically __name__)
        extra_config: Optional extra configuration for the logger

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)

    if extra_config:
        if "level" in extra_config:
            logger.setLevel(getattr(logging, extra_config["level"].upper()))

        if "propagate" in extra_config:
            logger.propagate = extra_config["propagate"]

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
) -> logging.Logger:
    """
    Configure logging for project components.

    This function provides component-specific logging configuration
    that can be shared across different modules.

    Args:
        component: Component name (e.g., 'tardis_downloader', 'market_data_fetcher')
        debug: Enable debug logging
        log_dir: Directory to store log files
        structured_logging: Use JSON structured logging format
        include_console: Include console output

    Returns:
        Configured logger
    """
    level = level
    level = "DEBUG" if debug else level

    log_file = None
    if log_dir:
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(UTC).strftime("%Y%m%d")
        log_file = log_dir_path / f"{component}_{timestamp}.log"
    if simplified:
        format_string = "%(funcName)s:%(lineno)d - %(message)s"
    else:
        format_string = (
            f"%(asctime)s - {component} - %(name)s "
            + "- %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        )

    return setup_logging(
        level=level,
        log_file=str(log_file) if log_file else None,
        console_output=include_console,
        colored_output=not structured_logging,
        structured_logging=structured_logging,
        format_string=format_string,
        log_regex=log_regex,
    )


def setup_component_logger(
    component_name: str,
    log_level: str = "INFO",
    log_to_file: bool = True,
    log_dir: str = "logs",
) -> logging.Logger:
    """
    Set up a logger for a specific component.

    Args:
        component_name: Name of the component (e.g., 'tardis_downloader')
        log_level: Logging level
        log_to_file: Whether to log to file
        log_dir: Directory for log files

    Returns:
        Configured component logger
    """
    logger = get_logger(component_name)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)

    if _supports_color():
        console_formatter = ColoredFormatter(
            f"%(asctime)s - {component_name} - %(levelname)s - %(message)s"
        )
    else:
        console_formatter = logging.Formatter(
            f"%(asctime)s - {component_name} - %(levelname)s - %(message)s"
        )

    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_to_file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            log_path / f"{component_name}.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
        )
        file_handler.setLevel(numeric_level)
        file_formatter = logging.Formatter(
            f"%(asctime)s - {component_name} - %(name)s "
            + "- %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False

    return logger
