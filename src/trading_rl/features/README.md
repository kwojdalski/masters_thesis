# Feature Engineering System

A flexible, configurable feature engineering system for trading RL experiments with proper train/test split handling.

## Quick Start

```python
from trading_rl.features import FeaturePipeline

# From dictionary (loaded from YAML)
configs = [
    {"name": "log_return", "feature_type": "log_return"},
    {"name": "rsi_14", "feature_type": "rsi", "params": {"period": 14}},
]

pipeline = FeaturePipeline.from_config_dict(configs)
pipeline.fit(train_df)  # Fit on train only!
features = pipeline.transform(df)
```

## YAML Configuration Support

Yes! The system fully supports YAML configurations:

```yaml
# In your experiment config file
features:
  - name: "log_return"
    feature_type: "log_return"

  - name: "high"
    feature_type: "high"

  - name: "rsi_14"
    feature_type: "rsi"
    params:
      period: 14

  - name: "sma_20"
    feature_type: "sma"
    params:
      period: 20
```

Then load it in Python:

```python
import yaml
from trading_rl.features import FeaturePipeline

# Load config
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Create pipeline
pipeline = FeaturePipeline.from_config_dict(config["features"])
```

## Available Features with Required Columns

### Basic Price Features

| Feature | Type | Required Columns | Parameters | Description |
|---------|------|-----------------|------------|-------------|
| Log Return | `log_return` | `close` | - | (close_t / close_t-1) - 1 |
| Simple Return | `simple_return` | `close` | - | close.pct_change() |
| High | `high` | `high`, `close` | - | (high / close) - 1 |
| Low | `low` | `low`, `close` | - | (low / close) - 1 |
| Trend | `trend` | `close` | - | Cumulative trend (min-max norm) |
| RSI | `rsi` | `close` | period (14) | Custom RSI implementation |

### Volume Features

| Feature | Type | Required Columns | Parameters | Description |
|---------|------|-----------------|------------|-------------|
| Log Volume | `log_volume` | `volume` | - | log1p(volume) |
| Volume Change | `volume_change` | `volume` | - | (volume_t / volume_t-1) - 1 |
| Volume MA Ratio | `volume_ma_ratio` | `volume` | window (20) | Volume / MA ratio |

### TA-Lib: Trend Indicators

| Feature | Type | Required Columns | Parameters | Description |
|---------|------|-----------------|------------|-------------|
| SMA | `sma` | `close` | period (20) | Simple Moving Average |
| EMA | `ema` | `close` | period (20) | Exponential Moving Average |
| WMA | `wma` | `close` | period (20) | Weighted Moving Average |
| DEMA | `dema` | `close` | period (20) | Double Exponential MA |
| TEMA | `tema` | `close` | period (20) | Triple Exponential MA |

### TA-Lib: Momentum Indicators

| Feature | Type | Required Columns | Parameters | Description |
|---------|------|-----------------|------------|-------------|
| RSI (TA-Lib) | `talib_rsi` | `close` | period (14) | Relative Strength Index |
| MACD | `macd` | `close` | fastperiod (12), slowperiod (26), signalperiod (9) | MACD line |
| Momentum | `mom` | `close` | period (10) | Price momentum |
| ROC | `roc` | `close` | period (10) | Rate of Change |
| CMO | `cmo` | `close` | period (14) | Chandra Momentum Oscillator |
| Williams %R | `willr` | `high`, `low`, `close` | period (14) | Williams %R |
| CCI | `cci` | `high`, `low`, `close` | period (14) | Commodity Channel Index |

### TA-Lib: Volatility Indicators

| Feature | Type | Required Columns | Parameters | Description |
|---------|------|-----------------|------------|-------------|
| ATR | `atr` | `high`, `low`, `close` | period (14) | Average True Range |
| NATR | `natr` | `high`, `low`, `close` | period (14) | Normalized ATR |
| Bollinger Bands | `bbands` | `close` | period (20), nbdevup (2), nbdevdn (2) | %B position in bands |

### TA-Lib: Volume Indicators

| Feature | Type | Required Columns | Parameters | Description |
|---------|------|-----------------|------------|-------------|
| OBV | `obv` | `close`, `volume` | - | On Balance Volume (ROC) |
| A/D Line | `ad` | `high`, `low`, `close`, `volume` | - | Accumulation/Distribution (ROC) |
| Chaikin A/D | `adosc` | `high`, `low`, `close`, `volume` | fastperiod (3), slowperiod (10) | Chaikin A/D Oscillator |

### TA-Lib: Trend Strength

| Feature | Type | Required Columns | Parameters | Description |
|---------|------|-----------------|------------|-------------|
| ADX | `adx` | `high`, `low`, `close` | period (14) | Average Directional Index |
| Aroon | `aroon` | `high`, `low` | period (14) | Aroon Oscillator |

## Data Requirements Summary

**Minimal (close only):** 15 features
- `log_return`, `simple_return`, `trend`, `rsi`, `sma`, `ema`, `wma`, `dema`, `tema`, `talib_rsi`, `macd`, `mom`, `roc`, `cmo`, `bbands`

**OHLC (high, low, close):** 5 features
- `high`, `low`, `willr`, `cci`, `atr`, `natr`, `adx`

**With Volume (close + volume):** 3 features
- `log_volume`, `volume_change`, `volume_ma_ratio`, `obv`

**Full OHLCV (high, low, close, volume):** 2 features
- `ad`, `adosc`

**Special (high, low only):** 1 feature
- `aroon`

## Configuration Examples

### Minimal Setup (Close Only)

```yaml
features:
  - name: "log_return"
    feature_type: "log_return"
  - name: "sma_20"
    feature_type: "sma"
    params:
      period: 20
```

### Price Action (OHLC)

```yaml
features:
  - name: "log_return"
    feature_type: "log_return"
  - name: "high"
    feature_type: "high"
  - name: "low"
    feature_type: "low"
  - name: "atr_14"
    feature_type: "atr"
    params:
      period: 14
```

### With Volume (OHLCV)

```yaml
features:
  - name: "log_return"
    feature_type: "log_return"
  - name: "log_vol"
    feature_type: "log_volume"
  - name: "obv"
    feature_type: "obv"
  - name: "ad"
    feature_type: "ad"
```

### Technical Analysis Suite

```yaml
features:
  # Trend
  - name: "sma_20"
    feature_type: "sma"
    params:
      period: 20
  - name: "ema_50"
    feature_type: "ema"
    params:
      period: 50

  # Momentum
  - name: "rsi_14"
    feature_type: "talib_rsi"
    params:
      period: 14
  - name: "macd"
    feature_type: "macd"

  # Volatility
  - name: "bbands_20"
    feature_type: "bbands"
    params:
      period: 20

  # Volume
  - name: "obv"
    feature_type: "obv"
```

### Multiple Periods

```yaml
features:
  # Short-term RSI
  - name: "rsi_7"
    feature_type: "talib_rsi"
    params:
      period: 7

  # Medium-term RSI
  - name: "rsi_14"
    feature_type: "talib_rsi"
    params:
      period: 14

  # Long-term RSI
  - name: "rsi_21"
    feature_type: "talib_rsi"
    params:
      period: 21

  # Moving averages
  - name: "sma_10"
    feature_type: "sma"
    params:
      period: 10
  - name: "sma_20"
    feature_type: "sma"
    params:
      period: 20
  - name: "sma_50"
    feature_type: "sma"
    params:
      period: 50
```

## Custom Features

Create your own features:

```python
from trading_rl.features import Feature, register_feature
import pandas as pd

@register_feature("my_feature")
class MyFeature(Feature):
    """Custom feature description."""

    def required_columns(self) -> list[str]:
        return ["close", "volume"]  # Specify what you need

    def compute(self, df: pd.DataFrame) -> pd.Series:
        # Your logic here
        return df["close"] * df["volume"]
```

Then use in YAML:

```yaml
features:
  - name: "custom"
    feature_type: "my_feature"
```

## Best Practices

1. **Always fit on training data only:**
   ```python
   pipeline.fit(train_df)  # NOT pd.concat([train_df, test_df])!
   ```

2. **Check data requirements:**
   ```python
   # If using volume features, ensure you have volume column
   assert "volume" in df.columns
   ```

3. **Validate features:**
   ```python
   print(pipeline.get_feature_names())
   features = pipeline.transform(train_df)
   print(features.describe())  # Check mean ≈ 0, std ≈ 1
   ```

4. **Match required columns:**
   - Using `atr`? Need: `high`, `low`, `close`
   - Using `obv`? Need: `close`, `volume`
   - Using `ad`? Need: `high`, `low`, `close`, `volume`

## List All Features

```python
from trading_rl.features import FeatureRegistry

# See all available features
features = FeatureRegistry.list_features()
print(f"Available features: {len(features)}")
for f in sorted(features):
    print(f"  - {f}")
```

## Integration Example

```python
import yaml
import pandas as pd
from trading_rl.features import FeaturePipeline

# 1. Load config
with open("experiment.yaml") as f:
    config = yaml.safe_load(f)

# 2. Load data
df = pd.read_parquet("data.parquet")
train_df = df[:1000]
test_df = df[1000:]

# 3. Create pipeline
pipeline = FeaturePipeline.from_config_dict(config["features"])

# 4. Fit on train only
pipeline.fit(train_df)

# 5. Transform both
train_features = pipeline.transform(train_df)
test_features = pipeline.transform(test_df)

# 6. Combine with original data
train_result = pd.concat([train_df, train_features], axis=1)
test_result = pd.concat([test_df, test_features], axis=1)
```
