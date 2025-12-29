# Feature Engineering Pipeline - User Guide

Complete guide to using the new configurable feature engineering system.

## Table of Contents
1. [Quick Start](#quick-start)
2. [Using Existing Features](#using-existing-features)
3. [Creating Custom Features](#creating-custom-features)
4. [Integration with Training Pipeline](#integration-with-training-pipeline)
5. [Configuration Format](#configuration-format)
6. [Best Practices](#best-practices)

---

## Quick Start

### Basic Usage (Python)

```python
from trading_rl.features import FeaturePipeline, FeatureConfig
import pandas as pd

# Load your data
df = pd.read_parquet("data/btc.parquet")

# Split train/test
train_df = df[:1000]
test_df = df[1000:]

# Define features
configs = [
    FeatureConfig(name="log_return", feature_type="log_return"),
    FeatureConfig(name="high", feature_type="high"),
    FeatureConfig(name="low", feature_type="low"),
]

# Create pipeline
pipeline = FeaturePipeline(configs)

# Fit on training data ONLY (critical!)
pipeline.fit(train_df)

# Transform both train and test
train_features = pipeline.transform(train_df)
test_features = pipeline.transform(test_df)

print(train_features.head())
#    feature_log_return  feature_high  feature_low
# 0           -0.123456      0.234567    -0.345678
# 1            0.456789     -0.123456     0.234567
# ...
```

### From Config Dictionary (YAML)

```python
from trading_rl.features import FeaturePipeline

# This is what you'd load from YAML
feature_configs = [
    {"name": "log_return", "feature_type": "log_return"},
    {"name": "high", "feature_type": "high"},
    {"name": "rsi_14", "feature_type": "rsi", "params": {"period": 14}},
]

pipeline = FeaturePipeline.from_config_dict(feature_configs)
pipeline.fit(train_df)
features = pipeline.transform(df)
```

---

## Using Existing Features

### Available Features

**Price Features:**
```python
# Log return: (close_t / close_t-1) - 1
FeatureConfig("log_return", "log_return")

# Simple return: close.pct_change()
FeatureConfig("simple_return", "simple_return")

# High relative to close
FeatureConfig("high", "high")

# Low relative to close
FeatureConfig("low", "low")

# Cumulative trend (min-max normalized)
FeatureConfig("trend", "trend")

# RSI with custom period
FeatureConfig("rsi_14", "rsi", params={"period": 14})
```

**Volume Features:**
```python
# Log-transformed volume
FeatureConfig("log_vol", "log_volume")

# Volume change
FeatureConfig("vol_change", "volume_change")

# Volume vs moving average
FeatureConfig("vol_ma", "volume_ma_ratio", params={"window": 20})
```

### Feature Parameters

Some features accept parameters:

```python
# RSI with different periods
FeatureConfig("rsi_7", "rsi", params={"period": 7})
FeatureConfig("rsi_14", "rsi", params={"period": 14})
FeatureConfig("rsi_21", "rsi", params={"period": 21})

# Volume MA with different windows
FeatureConfig("vol_ma_10", "volume_ma_ratio", params={"window": 10})
FeatureConfig("vol_ma_20", "volume_ma_ratio", params={"window": 20})
```

### Custom Output Names

```python
# Override the default "feature_X" naming
FeatureConfig(
    name="log_return",
    feature_type="log_return",
    output_name="my_custom_return"  # Instead of "feature_log_return"
)
```

### Disable Normalization

```python
# Some features shouldn't be normalized (like trend)
FeatureConfig(
    name="trend",
    feature_type="trend",
    normalize=False  # Skip z-score normalization
)
```

---

## Creating Custom Features

### Method 1: Simple Feature (Recommended)

```python
from trading_rl.features import Feature, register_feature
import pandas as pd

@register_feature("price_volume_ratio")
class PriceVolumeRatioFeature(Feature):
    """Custom feature: close price * volume."""

    def required_columns(self) -> list[str]:
        """List columns this feature needs."""
        return ["close", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Compute the raw feature values."""
        return df["close"] * df["volume"]

# Now you can use it!
config = FeatureConfig("pv_ratio", "price_volume_ratio")
```

### Method 2: Feature with Parameters

```python
@register_feature("moving_average")
class MovingAverageFeature(Feature):
    """Moving average of close price."""

    def required_columns(self) -> list[str]:
        return ["close"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        # Get window parameter (default to 20)
        window = self.config.params.get("window", 20)
        return df["close"].rolling(window=window, min_periods=1).mean()

# Use with different parameters
ma_20 = FeatureConfig("ma_20", "moving_average", params={"window": 20})
ma_50 = FeatureConfig("ma_50", "moving_average", params={"window": 50})
```

### Method 3: Feature with Custom Normalization

```python
import numpy as np

@register_feature("bollinger_position")
class BollingerPositionFeature(Feature):
    """Position within Bollinger Bands (already 0-1, no z-score needed)."""

    def __init__(self, config):
        # Disable automatic z-score normalization
        config.normalize = False
        super().__init__(config)

    def required_columns(self) -> list[str]:
        return ["close"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        window = self.config.params.get("window", 20)
        std_dev = self.config.params.get("std_dev", 2)

        # Calculate Bollinger Bands
        ma = df["close"].rolling(window=window, min_periods=1).mean()
        std = df["close"].rolling(window=window, min_periods=1).std()

        upper_band = ma + (std * std_dev)
        lower_band = ma - (std * std_dev)

        # Position within bands (0 = lower band, 1 = upper band)
        position = (df["close"] - lower_band) / (upper_band - lower_band + 1e-8)

        return position.clip(0, 1)  # Clamp to [0, 1]
```

### Method 4: Multi-Column Feature

```python
@register_feature("ohlc_range")
class OHLCRangeFeature(Feature):
    """Range across OHLC prices."""

    def required_columns(self) -> list[str]:
        return ["open", "high", "low", "close"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        max_price = df[["open", "high", "low", "close"]].max(axis=1)
        min_price = df[["open", "high", "low", "close"]].min(axis=1)
        return (max_price - min_price) / df["close"]
```

### Complete Custom Feature Example

```python
from trading_rl.features import Feature, register_feature
import pandas as pd
import numpy as np

@register_feature("momentum")
class MomentumFeature(Feature):
    """Rate of change over N periods."""

    def required_columns(self) -> list[str]:
        return ["close"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        # Get period from params (default 10)
        period = self.config.params.get("period", 10)

        # Calculate momentum: (close_t - close_t-n) / close_t-n
        momentum = (df["close"] - df["close"].shift(period)) / df["close"].shift(period)

        return momentum.fillna(0)

# Register happens automatically with @register_feature decorator!

# Use it
config = FeatureConfig("mom_10", "momentum", params={"period": 10})
pipeline = FeaturePipeline([config])
```

---

## Integration with Training Pipeline

### Option 1: Replace `create_features()` in `data_utils.py`

Update `prepare_data()` to use the new pipeline:

```python
from trading_rl.features import FeaturePipeline

def prepare_data_new(
    data_path: str,
    feature_configs: list[dict] | None = None,
    train_size: int | None = None,
    no_features: bool = False,
    **kwargs
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare data with new feature pipeline.

    Returns:
        (train_df, test_df): Both with features already computed
    """
    # Load data
    df = load_trading_data(data_path)

    if no_features:
        return df, pd.DataFrame()

    # Split train/test
    if train_size:
        train_df = df[:train_size]
        test_df = df[train_size:]
    else:
        train_df = df
        test_df = pd.DataFrame()

    # Create feature pipeline
    pipeline = FeaturePipeline.from_config_dict(feature_configs)

    # Fit on train, transform both
    pipeline.fit(train_df)
    train_features = pipeline.transform(train_df)
    test_features = pipeline.transform(test_df) if len(test_df) > 0 else pd.DataFrame()

    # Combine OHLCV + features
    train_result = pd.concat([train_df, train_features], axis=1)
    test_result = pd.concat([test_df, test_features], axis=1) if len(test_df) > 0 else pd.DataFrame()

    return train_result, test_result
```

### Option 2: Use in Existing Workflow

```python
from trading_rl.features import FeaturePipeline

# In your training script
df = prepare_data(data_path, no_features=True)  # Get raw OHLCV only

# Apply new feature pipeline
feature_configs = [
    {"name": "log_return", "feature_type": "log_return"},
    {"name": "high", "feature_type": "high"},
    {"name": "low", "feature_type": "low"},
]

pipeline = FeaturePipeline.from_config_dict(feature_configs)
pipeline.fit(df[:config.data.train_size])

# Get features
df_with_features = pd.concat([
    df,
    pipeline.transform(df)
], axis=1)
```

---

## Configuration Format

### In YAML Config Files

```yaml
# New feature configuration format
features:
  # Basic feature
  - name: "log_return"
    feature_type: "log_return"
    normalize: true  # default

  # Feature with parameters
  - name: "rsi_14"
    feature_type: "rsi"
    params:
      period: 14
    normalize: true

  # Feature with custom output name
  - name: "high"
    feature_type: "high"
    normalize: true
    output_name: "feature_high_custom"

  # Feature without normalization
  - name: "trend"
    feature_type: "trend"
    normalize: false

  # Custom feature (if you registered it)
  - name: "momentum_10"
    feature_type: "momentum"
    params:
      period: 10
```

### Backward Compatibility

You can still specify columns directly (will use old system):

```yaml
# Old format (still works with create_features)
feature_columns: ["feature_log_return", "feature_high", "feature_low"]

# New format (uses FeaturePipeline)
features:
  - name: "log_return"
    feature_type: "log_return"
  - name: "high"
    feature_type: "high"
  - name: "low"
    feature_type: "low"
```

---

## Best Practices

### 1. Always Fit on Training Data Only

```python
# ✓ CORRECT
pipeline.fit(train_df)
train_out = pipeline.transform(train_df)
test_out = pipeline.transform(test_df)

# ✗ WRONG - Data leakage!
pipeline.fit(pd.concat([train_df, test_df]))
```

### 2. Validate Features Early

```python
# Check what features will be created
pipeline = FeaturePipeline.from_config_dict(configs)
print(f"Features: {pipeline.get_feature_names()}")
# ['feature_log_return', 'feature_high', 'feature_low']
```

### 3. Handle Missing Data

```python
# The pipeline drops NaN rows after transformation
rows_before = len(df)
features = pipeline.transform(df)
rows_after = len(features)

if rows_before != rows_after:
    print(f"Warning: Dropped {rows_before - rows_after} rows with NaN")
```

### 4. Save and Load Pipelines

```python
import joblib

# Save fitted pipeline
joblib.dump(pipeline, "feature_pipeline.pkl")

# Load and use
pipeline = joblib.load("feature_pipeline.pkl")
features = pipeline.transform(new_data)
```

### 5. Debug Feature Values

```python
# Check feature statistics
features = pipeline.transform(train_df)
print(features.describe())

# Verify normalization
print(f"Mean: {features['feature_log_return'].mean():.6f}")  # Should be ~0
print(f"Std: {features['feature_log_return'].std():.6f}")    # Should be ~1
```

### 6. Feature Selection

```python
# Easy to experiment with different feature sets
configs_minimal = [
    {"name": "log_return", "feature_type": "log_return"},
]

configs_full = [
    {"name": "log_return", "feature_type": "log_return"},
    {"name": "high", "feature_type": "high"},
    {"name": "low", "feature_type": "low"},
    {"name": "rsi_14", "feature_type": "rsi", "params": {"period": 14}},
    {"name": "log_vol", "feature_type": "log_volume"},
]

# Try different configs easily
for configs in [configs_minimal, configs_full]:
    pipeline = FeaturePipeline.from_config_dict(configs)
    # ... train and evaluate ...
```

---

## Common Patterns

### Pattern 1: Multiple RSI Periods

```python
rsi_configs = [
    {"name": f"rsi_{p}", "feature_type": "rsi", "params": {"period": p}}
    for p in [7, 14, 21]
]
pipeline = FeaturePipeline.from_config_dict(rsi_configs)
```

### Pattern 2: Technical Indicator Suite

```python
technical_features = [
    {"name": "log_return", "feature_type": "log_return"},
    {"name": "rsi_14", "feature_type": "rsi", "params": {"period": 14}},
    {"name": "trend", "feature_type": "trend", "normalize": False},
    {"name": "vol_ma_20", "feature_type": "volume_ma_ratio", "params": {"window": 20}},
]
```

### Pattern 3: Minimal Price Action

```python
price_action = [
    {"name": "log_return", "feature_type": "log_return"},
    {"name": "high", "feature_type": "high"},
    {"name": "low", "feature_type": "low"},
]
```

---

## Troubleshooting

### Error: "Pipeline must be fitted before transform"

```python
# You forgot to fit!
pipeline = FeaturePipeline(configs)
features = pipeline.transform(df)  # ✗ Error

# Fix: Call fit() first
pipeline.fit(train_df)
features = pipeline.transform(df)  # ✓ Works
```

### Error: "DataFrame missing required columns"

```python
# Your data doesn't have the columns needed
# Check what's required:
for feature in pipeline.features:
    print(f"{feature.get_output_name()}: needs {feature.required_columns()}")
```

### Error: "Unknown feature type: X"

```python
# Feature not registered. Check available features:
from trading_rl.features import FeatureRegistry
print(FeatureRegistry.list_features())

# Register your custom feature:
from trading_rl.features import register_feature

@register_feature("my_feature")
class MyFeature(Feature):
    # ...
```

---

## Migration from Old System

### Before (Old `create_features`)

```python
df = load_trading_data("data.parquet")
df = create_features(df)  # ✗ Uses all data for normalization
train_df = df[:1000]
test_df = df[1000:]
```

### After (New Pipeline)

```python
df = load_trading_data("data.parquet")
train_df = df[:1000]
test_df = df[1000:]

configs = [
    {"name": "log_return", "feature_type": "log_return"},
    {"name": "high", "feature_type": "high"},
    {"name": "low", "feature_type": "low"},
]

pipeline = FeaturePipeline.from_config_dict(configs)
pipeline.fit(train_df)  # ✓ Only uses training data

train_features = pipeline.transform(train_df)
test_features = pipeline.transform(test_df)
```

---

## Next Steps

1. **Try the demo**: Run `python examples/feature_engineering_demo.py`
2. **Create custom features**: Add your own technical indicators
3. **Update configs**: Migrate your YAML configs to use the new format
4. **Experiment**: Try different feature combinations easily

## Questions?

- Check `src/trading_rl/features/README.md` for more details
- Look at existing features in `src/trading_rl/features/price_features.py` and `volume_features.py`
- Run the demo: `examples/feature_engineering_demo.py`
