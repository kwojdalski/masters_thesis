"""Feature engineering module for trading RL.

This module provides a flexible, configurable feature engineering system that:
- Separates train/test normalization (no data leakage)
- Supports config-based feature definitions
- Uses a registry pattern for extensibility
- Properly handles time-series normalization
"""

from trading_rl.features.base import Feature, FeatureConfig
from trading_rl.features.pipeline import FeaturePipeline
from trading_rl.features.registry import FeatureRegistry, register_feature

# Import concrete features to register them
from trading_rl.features.price_features import (
    HighFeature,
    LogReturnFeature,
    LowFeature,
    TrendFeature,
)
from trading_rl.features.volume_features import LogVolumeFeature

__all__ = [
    "Feature",
    "FeatureConfig",
    "FeaturePipeline",
    "FeatureRegistry",
    "register_feature",
    "LogReturnFeature",
    "HighFeature",
    "LowFeature",
    "LogVolumeFeature",
    "TrendFeature",
]
