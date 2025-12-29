# Feature Engineering - Quick Reference Card

## Basic Usage (3 Steps)

```python
from trading_rl.features import FeaturePipeline

# 1. Define features
configs = [
    {"name": "log_return", "feature_type": "log_return"},
    {"name": "high", "feature_type": "high"},
]

# 2. Create and fit pipeline (TRAIN DATA ONLY!)
pipeline = FeaturePipeline.from_config_dict(configs)
pipeline.fit(train_df)

# 3. Transform both train and test
train_features = pipeline.transform(train_df)
test_features = pipeline.transform(test_df)
```

## Create a Custom Feature (Copy-Paste Template)

```python
from trading_rl.features import Feature, register_feature
import pandas as pd

@register_feature("YOUR_FEATURE_NAME")  # ← Change this
class YourFeature(Feature):
    """Description of your feature."""

    def required_columns(self) -> list[str]:
        return ["close"]  # ← Columns you need

    def compute(self, df: pd.DataFrame) -> pd.Series:
        # ← Your feature logic here
        return df["close"].pct_change().fillna(0)
```

## Available Built-in Features

| Feature Type | Description | Required Columns | Parameters |
|-------------|-------------|------------------|------------|
| `log_return` | Log returns | close | - |
| `simple_return` | Simple returns | close | - |
| `high` | High/close - 1 | high, close | - |
| `low` | Low/close - 1 | low, close | - |
| `trend` | Cumulative trend | close | - |
| `rsi` | RSI indicator | close | period (default: 14) |
| `log_volume` | Log(volume) | volume | - |
| `volume_change` | Volume change | volume | - |
| `volume_ma_ratio` | Volume/MA ratio | volume | window (default: 20) |

## Common Config Patterns

### Minimal Setup
```yaml
features:
  - name: "log_return"
    feature_type: "log_return"
```

### Full Price + Volume
```yaml
features:
  - name: "log_return"
    feature_type: "log_return"
  - name: "high"
    feature_type: "high"
  - name: "low"
    feature_type: "low"
  - name: "log_vol"
    feature_type: "log_volume"
```

### With Parameters
```yaml
features:
  - name: "rsi_14"
    feature_type: "rsi"
    params:
      period: 14
```

## Quick Checks

```python
# List available features
from trading_rl.features import FeatureRegistry
print(FeatureRegistry.list_features())

# Check output names
print(pipeline.get_feature_names())

# Verify normalization
features = pipeline.transform(train_df)
print(features.describe())  # Mean ~0, Std ~1
```

## Common Mistakes

```python
# ✗ WRONG - Data leakage!
pipeline.fit(pd.concat([train_df, test_df]))

# ✓ CORRECT - Fit on train only
pipeline.fit(train_df)
train_out = pipeline.transform(train_df)
test_out = pipeline.transform(test_df)
```

## Running the Demo

```bash
python examples/feature_engineering_demo.py
```
