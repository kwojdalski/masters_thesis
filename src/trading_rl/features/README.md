# Feature Engineering System

A flexible, configurable feature engineering system for trading RL experiments.

## Key Features

1. **No Data Leakage**: Proper train/test split handling
   - Normalization statistics computed ONLY on training data
   - Same statistics applied to test data
   - Prevents look-ahead bias

2. **Configuration-Based**: Define features in config files
   - Easy to add/remove features per experiment
   - No code changes needed for different feature sets

3. **Extensible**: Registry pattern for adding new features
   - Simple decorator-based registration
   - Easy to create custom features

4. **Proper Separation of Concerns**:
   - Feature computation separate from normalization
   - Data loading separate from feature engineering

## Quick Start

### Basic Usage

```python
from trading_rl.features import FeaturePipeline, FeatureConfig

# Define features
configs = [
    FeatureConfig(name="log_return", feature_type="log_return"),
    FeatureConfig(name="high", feature_type="high"),
    FeatureConfig(name="low", feature_type="low"),
]

# Create pipeline
pipeline = FeaturePipeline(configs)

# Fit on training data ONLY
pipeline.fit(train_df)

# Transform both train and test
train_features = pipeline.transform(train_df)
test_features = pipeline.transform(test_df)
```

### From Dictionary Config

```python
# Load from config dict (e.g., from YAML)
feature_configs = [
    {"name": "log_return", "feature_type": "log_return"},
    {"name": "rsi_14", "feature_type": "rsi", "params": {"period": 14}},
    {"name": "log_vol", "feature_type": "log_volume"},
]

pipeline = FeaturePipeline.from_config_dict(feature_configs)
```

## Available Features

### Price Features

- **log_return**: Log returns `(close_t / close_t-1) - 1`
- **simple_return**: Simple returns `close.pct_change()`
- **high**: High relative to close `(high / close) - 1`
- **low**: Low relative to close `(low / close) - 1`
- **trend**: Cumulative trend (min-max normalized)
- **rsi**: Relative Strength Index (configurable period)

### Volume Features

- **log_volume**: Log-transformed volume `log1p(volume)`
- **volume_change**: Volume change `(volume_t / volume_t-1) - 1`
- **volume_ma_ratio**: Volume relative to moving average

## Configuration Format

### In YAML Config

```yaml
# New format
features:
  - name: "log_return"
    feature_type: "log_return"
    normalize: true  # Apply z-score normalization (default)

  - name: "rsi_14"
    feature_type: "rsi"
    params:
      period: 14
    normalize: true

  - name: "high"
    feature_type: "high"
    normalize: true

  - name: "low"
    feature_type: "low"
    normalize: true

  - name: "log_vol"
    feature_type: "log_volume"
    normalize: true

# Old format (deprecated but still supported)
feature_columns: ["feature_log_return", "feature_high", "feature_low"]
```

## Creating Custom Features

```python
from trading_rl.features import Feature, register_feature
import pandas as pd

@register_feature("my_custom_feature")
class MyCustomFeature(Feature):
    """Custom feature description."""

    def required_columns(self) -> list[str]:
        return ["close", "volume"]  # Columns needed

    def compute(self, df: pd.DataFrame) -> pd.Series:
        # Your feature logic here
        return df["close"] * df["volume"]
```

## Migration from Old System

### Old System (create_features)

```python
# Old: Hardcoded features, global normalization
df = create_features(df)
# Problems:
# - All features always created
# - Normalization uses entire dataset (data leakage!)
# - Can't configure features per experiment
```

### New System

```python
# New: Configurable, proper train/test split
pipeline = FeaturePipeline.from_config_dict(feature_configs)
pipeline.fit(train_df)  # Fit ONLY on training data
train_features = pipeline.transform(train_df)
test_features = pipeline.transform(test_df)  # Use same normalization
```

## Design Principles

1. **Fit on Train, Transform on Both**
   - Normalization stats computed from training data only
   - Test data transformed using training stats
   - No information leakage from future/test data

2. **Configuration Over Code**
   - Feature selection in config files
   - Easy to experiment with different feature sets
   - No code changes needed

3. **Composability**
   - Features are independent
   - Can mix and match freely
   - Easy to add new features via registry

4. **Explicitness**
   - Feature names reflect their computation
   - `feature_log_return` vs `feature_return`
   - Clear what transformations are applied

## Example: Full Pipeline

```python
import pandas as pd
from trading_rl.features import FeaturePipeline, FeatureConfig

# Load data
df = pd.read_parquet("data.parquet")

# Split data
train_size = 1000
train_df = df[:train_size]
test_df = df[train_size:]

# Define features
features = [
    FeatureConfig("log_return", "log_return"),
    FeatureConfig("high", "high"),
    FeatureConfig("low", "low"),
    FeatureConfig("rsi", "rsi", params={"period": 14}),
]

# Create and fit pipeline
pipeline = FeaturePipeline(features)
pipeline.fit(train_df)

# Transform
train_features = pipeline.transform(train_df)
test_features = pipeline.transform(test_df)

# Add features back to original data
train_result = pd.concat([train_df, train_features], axis=1)
test_result = pd.concat([test_df, test_features], axis=1)
```

## Testing

```python
# Verify no data leakage
pipeline = FeaturePipeline(configs)

# This is correct
pipeline.fit(train_df)
train_out = pipeline.transform(train_df)
test_out = pipeline.transform(test_df)

# This would be WRONG (data leakage!)
# pipeline.fit(pd.concat([train_df, test_df]))  # DON'T DO THIS
```
