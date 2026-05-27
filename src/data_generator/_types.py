from __future__ import annotations

import logging
from enum import StrEnum


class PatternType(StrEnum):
    """Supported synthetic price data generation patterns."""

    SINE_WAVE = "sine_wave"
    UPWARD_DRIFT = "upward_drift"
    MEAN_REVERSION = "mean_reversion"
    TRENDING = "trending"
    HFT_SINE_WAVE_LOB = "hft_sine_wave_lob"


# Default start date for synthetic time indices.  The value is a label only —
# synthetic prices are generated independently of wall-clock time — so this
# never needs to track "today".  One constant means one place to update.
DEFAULT_SYNTHETIC_START_DATE = "2024-01-01"


def _parse_log_level(level: int | str | None) -> int:
    """Convert log level provided as string or int into numeric constant."""
    level_map = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "NOTSET": logging.NOTSET,
    }
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        return level_map.get(level.upper(), logging.INFO)
    return logging.INFO
