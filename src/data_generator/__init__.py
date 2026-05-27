"""Synthetic price data generation package.

Public API mirrors the old data_generator.py module so all existing imports
continue to work without changes:

    from data_generator import PriceDataGenerator, PatternType
"""

from ._generator import PriceDataGenerator
from ._types import DEFAULT_SYNTHETIC_START_DATE, PatternType

__all__ = [
    "DEFAULT_SYNTHETIC_START_DATE",
    "PatternType",
    "PriceDataGenerator",
]
