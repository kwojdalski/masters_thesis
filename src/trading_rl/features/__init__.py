"""Feature engineering module for trading RL.

This module provides a flexible, configurable feature engineering system that:
- Separates train/test normalization (no data leakage)
- Supports config-based feature definitions
- Uses a registry pattern for extensibility
- Properly handles time-series normalization
"""

from trading_rl.features.base import Feature, FeatureConfig
from trading_rl.features.column_features import ColumnValueFeature
from trading_rl.features.pipeline import FeaturePipeline, create_default_pipeline

# Import concrete features to register them
from trading_rl.features.price_features import (
    HighFeature,
    LogReturnFeature,
    LowFeature,
    ReturnLagFeature,
    RSIFeature,
    SimpleReturnFeature,
    TrendFeature,
)
from trading_rl.features.registry import FeatureRegistry, register_feature
from trading_rl.features.volatility_features import (
    RealizedVolatilityFeature,
    VolatilityRatioFeature,
)
from trading_rl.features.volume_features import (
    LogVolumeFeature,
    VolumeChangeFeature,
    VolumeMAFeature,
)
from trading_rl.features.temporal_features import (
    DayOfWeekCosFeature,
    DayOfWeekSinFeature,
    HourCosFeature,
    HourSinFeature,
    MinuteOfHourCosFeature,
    MinuteOfHourSinFeature,
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
    "FeatureRegistry",
    "ColumnValueFeature",
    "DayOfWeekCosFeature",
    "DayOfWeekSinFeature",
    "HighFeature",
    "HourCosFeature",
    "HourSinFeature",
    "LogReturnFeature",
    "LogVolumeFeature",
    "LowFeature",
    "MinuteOfHourCosFeature",
    "MinuteOfHourSinFeature",
    "RSIFeature",
    "RealizedVolatilityFeature",
    "ReturnLagFeature",
    "SimpleReturnFeature",
    "TrendFeature",
    "VolatilityRatioFeature",
    "VolumeChangeFeature",
    "VolumeMAFeature",
    "create_default_pipeline",
    "register_feature",
]

if _TALIB_AVAILABLE:
    __all__ += [
        "ADFeature",
        "ADOSCFeature",
        "ADXFeature",
        "AROONFeature",
        "ATRFeature",
        "BBANDSFeature",
        "CCIFeature",
        "CMOFeature",
        "DEMAFeature",
        "EMAFeature",
        "MACDFeature",
        "MOMFeature",
        "NATRFeature",
        "OBVFeature",
        "ROCFeature",
        "SMAFeature",
        "TALibRSIFeature",
        "TEMAFeature",
        "WILLRFeature",
        "WMAFeature",
    ]
