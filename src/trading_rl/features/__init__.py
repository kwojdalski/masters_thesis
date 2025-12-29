"""Feature engineering module for trading RL.

This module provides a flexible, configurable feature engineering system that:
- Separates train/test normalization (no data leakage)
- Supports config-based feature definitions
- Uses a registry pattern for extensibility
- Properly handles time-series normalization
"""

from trading_rl.features.base import Feature, FeatureConfig
from trading_rl.features.pipeline import FeaturePipeline, create_default_pipeline
from trading_rl.features.registry import FeatureRegistry, register_feature

# Import concrete features to register them
from trading_rl.features.price_features import (
    HighFeature,
    LogReturnFeature,
    LowFeature,
    RSIFeature,
    SimpleReturnFeature,
    TrendFeature,
)
from trading_rl.features.volume_features import (
    LogVolumeFeature,
    VolumeChangeFeature,
    VolumeMAFeature,
)

# Import TA-Lib features to register them
try:
    from trading_rl.features.talib_features import (
        ADFeature,
        ADOSCFeature,
        ADXFeature,
        AROONFeature,
        ATRFeature,
        BBANDSFeature,
        CCIFeature,
        CMOFeature,
        DEMAFeature,
        EMAFeature,
        MACDFeature,
        MOMFeature,
        NATRFeature,
        OBVFeature,
        ROCFeature,
        SMAFeature,
        TALibRSIFeature,
        TEMAFeature,
        WILLRFeature,
        WMAFeature,
    )

    _TALIB_AVAILABLE = True
except ImportError:
    _TALIB_AVAILABLE = False

__all__ = [
    "Feature",
    "FeatureConfig",
    "FeaturePipeline",
    "create_default_pipeline",
    "FeatureRegistry",
    "register_feature",
    # Price features
    "LogReturnFeature",
    "SimpleReturnFeature",
    "HighFeature",
    "LowFeature",
    "TrendFeature",
    "RSIFeature",
    # Volume features
    "LogVolumeFeature",
    "VolumeChangeFeature",
    "VolumeMAFeature",
]

if _TALIB_AVAILABLE:
    __all__ += [
        # TA-Lib features
        "SMAFeature",
        "EMAFeature",
        "WMAFeature",
        "DEMAFeature",
        "TEMAFeature",
        "TALibRSIFeature",
        "MACDFeature",
        "MOMFeature",
        "ROCFeature",
        "CMOFeature",
        "WILLRFeature",
        "CCIFeature",
        "ATRFeature",
        "NATRFeature",
        "BBANDSFeature",
        "OBVFeature",
        "ADFeature",
        "ADOSCFeature",
        "ADXFeature",
        "AROONFeature",
    ]
