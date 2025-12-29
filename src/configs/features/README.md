# Feature Configurations - Based on Actual Scenarios

These feature configurations are extracted from actual experiment scenarios in `configs/scenarios/`.

## Available Configurations

### 1. `sine_wave_price_action.yaml` (3 features)
**Used in:**
- `sine_wave_ppo_no_trend_continuous.yaml`
- `sine_wave_ppo_no_trend_tradingenv.yaml`
- `sine_wave_td3_no_trend_tradingenv.yaml`

**Required columns:** `open`, `high`, `low`, `close`

**Pattern:** Mean-reverting sine wave

**Features:**
```yaml
features:
  - log_return  # Price momentum
  - high        # Intraday volatility (upside)
  - low         # Intraday volatility (downside)
```

**Use case:** Cyclical/mean-reverting patterns where intraday volatility helps identify turning points.

---

### 2. `btc_with_volume.yaml` (4 features)
**Used in:**
- `btc_td3_tradingenv.yaml`

**Required columns:** `open`, `high`, `low`, `close`, `volume`

**Pattern:** Real cryptocurrency market data

**Features:**
```yaml
features:
  - log_return   # Price momentum
  - high         # Intraday volatility (upside)
  - low          # Intraday volatility (downside)
  - log_volume   # Trading volume indicator
```

**Use case:** Real market trading where volume provides information about market strength and participation.

---

### 3. `upward_trend_with_return.yaml` (2 features)
**Used in:**
- `upward_trend_ddpg_tradingenv.yaml`

**Required columns:** `close`

**Pattern:** Strong upward drift

**Features:**
```yaml
features:
  - log_return  # Price momentum (z-scored, no trend info)
  - trend       # Cumulative trend (preserves upward drift)
```

**Use case:** Trending markets where both short-term momentum and long-term trend are important.

**Note:** `log_return` is z-score normalized so loses trend signal; `trend` preserves it via min-max normalization.

---

### 4. `upward_trend_only.yaml` (1 feature)
**Used in:**
- `upward_trend_td3_tradingenv.yaml`

**Required columns:** `close`

**Pattern:** Strong upward drift

**Features:**
```yaml
features:
  - trend  # Cumulative trend only
```

**Use case:** Pure trend-following in strong directional markets. Forces agent to learn from cumulative price movement only.

**Note:** This is the minimal signal for trend following - just the cumulative price ratio.

---

## Feature Selection by Market Pattern

| Market Pattern | Recommended Config | Features Count | Rationale |
|----------------|-------------------|----------------|-----------|
| Mean-reverting cycles | `sine_wave_price_action.yaml` | 3 | Intraday volatility helps identify reversals |
| Real market (crypto) | `btc_with_volume.yaml` | 4 | Volume adds market strength signal |
| Trending + momentum | `upward_trend_with_return.yaml` | 2 | Both short-term and long-term signals |
| Pure trend following | `upward_trend_only.yaml` | 1 | Minimal, forces focus on trend |

## How to Use in Your Experiments

### Option 1: Reference in Scenario Config

Update your scenario YAML to reference a feature config:

```yaml
# In your experiment config
env:
  backend: "tradingenv"

  # OLD WAY (deprecated)
  # feature_columns: ["feature_log_return", "feature_high", "feature_low"]

  # NEW WAY - Reference feature config
  feature_config_file: "configs/features/sine_wave_price_action.yaml"
```

### Option 2: Load in Python

```python
import yaml
from trading_rl.features import FeaturePipeline

# Load feature config
with open("configs/features/sine_wave_price_action.yaml") as f:
    feature_config = yaml.safe_load(f)

# Create pipeline
pipeline = FeaturePipeline.from_config_dict(feature_config["features"])

# Fit on training data
pipeline.fit(train_df)

# Transform
features = pipeline.transform(df)
```

### Option 3: Inline in Scenario

You can also inline the features directly:

```yaml
# In your experiment config
env:
  backend: "tradingenv"

  # Inline feature definition
  features:
    - name: "log_return"
      feature_type: "log_return"
    - name: "high"
      feature_type: "high"
    - name: "low"
      feature_type: "low"
```

## Migration from Old Format

### Old Format (feature_columns)
```yaml
env:
  feature_columns: ["feature_log_return", "feature_high", "feature_low"]
```

### New Format (features)
```yaml
env:
  features:
    - name: "log_return"
      feature_type: "log_return"
    - name: "high"
      feature_type: "high"
    - name: "low"
      feature_type: "low"
```

**OR**

```yaml
env:
  feature_config_file: "configs/features/sine_wave_price_action.yaml"
```

## Feature Characteristics

### log_return (Price Returns)
- **Normalization:** Z-score (mean=0, std=1)
- **Stationarity:** Yes
- **Trend preservation:** **No** (z-score removes trend)
- **Use:** Short-term momentum, mean reversion

### high / low (Intraday Volatility)
- **Normalization:** Z-score
- **Stationarity:** Yes
- **Information:** Intraday range relative to close
- **Use:** Volatility signals, reversal detection

### log_volume (Volume)
- **Normalization:** Z-score on log-transformed values
- **Stationarity:** Yes (after log transform)
- **Information:** Trading activity level
- **Use:** Market strength, breakout confirmation

### trend (Cumulative Trend)
- **Normalization:** Min-max to [0, 1]
- **Stationarity:** **No** (cumulative)
- **Trend preservation:** **Yes**
- **Use:** Long-term directional signal

## Best Practices

1. **Match features to pattern:**
   - Mean-reverting → Include intraday volatility (high/low)
   - Trending → Include trend feature
   - Real markets → Add volume

2. **Understand normalization:**
   - Z-score normalization **removes trend**
   - Use `trend` feature to preserve long-term direction
   - Mix both for multi-timeframe strategies

3. **Start simple:**
   - Begin with `upward_trend_only.yaml` (1 feature) for trending
   - Or `sine_wave_price_action.yaml` (3 features) for mean-reverting
   - Add features incrementally

4. **Validate features:**
   ```python
   features = pipeline.transform(train_df)
   print(features.describe())  # Check mean ≈ 0, std ≈ 1
   ```

## Creating Custom Configs

Copy an existing config and modify:

```bash
cp configs/features/sine_wave_price_action.yaml configs/features/my_custom.yaml
# Edit my_custom.yaml with your features
```

Example custom config:
```yaml
# my_custom.yaml
features:
  - name: "log_return"
    feature_type: "log_return"

  - name: "rsi_14"
    feature_type: "talib_rsi"
    params:
      period: 14

  - name: "sma_20"
    feature_type: "sma"
    params:
      period: 20
```
