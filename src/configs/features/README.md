# Feature Configuration Files

Pre-configured feature sets for different trading strategies and use cases.

## Available Configurations

### 1. `minimal.yaml` - Close Only
**Required columns:** `close`
**Features:** 1
**Use for:** Quick experiments, baseline models

```yaml
features:
  - log_return
```

### 2. `price_action.yaml` - OHLC
**Required columns:** `open`, `high`, `low`, `close`
**Features:** 4
**Use for:** Pure price-based strategies

```yaml
features:
  - log_return
  - high (relative to close)
  - low (relative to close)
  - atr_14
```

### 3. `momentum.yaml` - Momentum Focus
**Required columns:** `high`, `low`, `close`
**Features:** 9
**Use for:** Trend-following, momentum strategies

```yaml
features:
  - log_return
  - rsi_7, rsi_14, rsi_21
  - macd
  - roc_10
  - willr_14
  - cci_14
  - adx_14
```

### 4. `volatility.yaml` - Volatility Focus
**Required columns:** `high`, `low`, `close`
**Features:** 7
**Use for:** Volatility-based strategies, risk management

```yaml
features:
  - log_return
  - high, low
  - atr_14, natr_14
  - bbands_20, bbands_10
```

### 5. `volume.yaml` - Volume Focus
**Required columns:** `high`, `low`, `close`, `volume`
**Features:** 7
**Use for:** Volume analysis, market strength

```yaml
features:
  - log_return
  - log_vol, vol_change, vol_ma_20
  - obv, ad, adosc
```

### 6. `moving_averages.yaml` - MA Suite
**Required columns:** `close`
**Features:** 9
**Use for:** Trend identification, crossover strategies

```yaml
features:
  - log_return
  - sma_10, sma_20, sma_50
  - ema_10, ema_20, ema_50
  - dema_20, tema_20
```

### 7. `technical_suite.yaml` - Complete TA
**Required columns:** `high`, `low`, `close`, `volume`
**Features:** 14
**Use for:** Full technical analysis approach

```yaml
features:
  - Price: log_return, high, low
  - Trend: sma_20, ema_20
  - Momentum: rsi_14, macd, willr_14
  - Volatility: atr_14, bbands_20
  - Volume: log_vol, obv
  - Strength: adx_14
```

### 8. `multi_timeframe.yaml` - Multiple Horizons
**Required columns:** `high`, `low`, `close`
**Features:** 11
**Use for:** Capturing multiple time horizons

```yaml
features:
  - Short (7): rsi_7, sma_7, atr_7
  - Medium (14): rsi_14, sma_14, atr_14
  - Long (21): rsi_21, sma_21, atr_21
  - Very Long (50): sma_50
```

## How to Use

### In Your Experiment Config

```yaml
# Option 1: Reference feature config file
feature_config_file: "configs/features/momentum.yaml"

# Option 2: Inline features
features:
  - name: "log_return"
    feature_type: "log_return"
```

### In Python

```python
import yaml
from trading_rl.features import FeaturePipeline

# Load feature config
with open("configs/features/momentum.yaml") as f:
    feature_config = yaml.safe_load(f)

# Create pipeline
pipeline = FeaturePipeline.from_config_dict(feature_config["features"])

# Fit and transform
pipeline.fit(train_df)
features = pipeline.transform(df)
```

## Choosing the Right Config

| Strategy Type | Recommended Config | Data Requirements |
|---------------|-------------------|-------------------|
| Quick baseline | `minimal.yaml` | Close only |
| Day trading | `price_action.yaml` | OHLC |
| Trend following | `momentum.yaml` | OHLC |
| Mean reversion | `volatility.yaml` | OHLC |
| Volume analysis | `volume.yaml` | Full OHLCV |
| Classic TA | `technical_suite.yaml` | Full OHLCV |
| Multi-scale | `multi_timeframe.yaml` | OHLC |

## Creating Custom Configs

1. Copy an existing config
2. Modify feature selection
3. Adjust parameters
4. Save with descriptive name

Example:
```yaml
# my_custom_features.yaml
features:
  - name: "log_return"
    feature_type: "log_return"

  - name: "rsi_custom"
    feature_type: "talib_rsi"
    params:
      period: 9  # Custom period

  - name: "my_sma"
    feature_type: "sma"
    params:
      period: 15  # Custom period
```

## Feature Count vs Performance

- **Too few features:** May miss important patterns
- **Too many features:** Overfitting, slower training
- **Sweet spot:** 5-15 features for most strategies

Start with `minimal.yaml` or `price_action.yaml`, then gradually add features based on performance.
