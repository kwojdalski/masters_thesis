# Feature Pipeline Architecture

## Overview

The feature pipeline transforms raw LOB (Limit Order Book) data into engineered observation vectors for the RL agent. It supports two modes of feature set construction:

1. **Direct YAML config** — manually specify features in a YAML file (legacy)
2. **Group-based composition** — compose features from named groups, optionally with research-based selection

```mermaid
flowchart TD
    RAW["Raw LOB Data<br/>(MBP-10 parquet)"]
    FILTER["LOB Filters<br/>(unchanged, valid, active)"]
    SPLIT["Train / Val / Test Split"]
    FP["FeaturePipeline<br/>(fit on train, transform all)"]
    AGENT["Agent Observation<br/>(float32 vector)"]

    RAW --> FILTER --> SPLIT --> FP --> AGENT

    FP -.->|config path| YML["Feature YAML<br/>(hft_lob_features_state_space.yaml)"]
    FP -.->|group-based| GRP["FeatureGroupResolver<br/>(feature_groups.yaml)"]

    GRP -->|strategy=all| FP
    GRP -->|strategy=research| SEL["FeatureSelector<br/>(score + filter)"]
    SEL --> FP

    style RAW fill:#e1f5fe
    style AGENT fill:#e8f5e9
    style FP fill:#fff3e0
    style SEL fill:#fce4ec
```

## Feature Group Composition

Features are organized into semantic groups matching the thesis state-space design (Chapter 4, Section 4.2):

| Group | Description | Features |
|-------|-------------|----------|
| `imbalance` | Order book imbalance signals | book_pressure (L0-L2), order_book_imbalance (3L), order_count_imbalance |
| `fair_value` | Fair value and microprice estimates | microprice, microprice_divergence, vwmp_skew |
| `spread` | Spread and cost features | spread_bps, spread_ratio, book_convexity (bid/ask), bid_ask_slope (bid/ask) |
| `flow` | Order flow and trade flow features | ofi, ofi_rolling, queue_depletion (bid/ask), signed_trade_flow, odd_lot_trade_ratio, odd_lot_imbalance |
| `regime` | Regime and temporal features | inter_event_time, mid_price_acceleration, hour_sin, hour_cos |

Groups are composed via `FeatureGroupResolver`:

```python
from trading_rl.features import FeatureGroupResolver, FeaturePipeline

resolver = FeatureGroupResolver.from_yaml("src/configs/features/feature_groups.yaml")

# Use specific groups
configs = resolver.resolve(["imbalance", "fair_value", "flow"])
pipeline = FeaturePipeline(configs)

# Use all groups with exclusions
configs = resolver.resolve(
    resolver.list_groups(),
    exclude=["feature_hft_book_pressure_l2"],
)
pipeline = FeaturePipeline(configs)
```

## Feature Selection (Research Strategy)

The `FeatureSelector` applies offline screening to reduce a candidate pool:

```mermaid
flowchart LR
    CAND["Candidate Pool<br/>(25 features)"]
    FIT["Fit Pipeline<br/>on Training Data"]
    TRANS["Transform<br/>Train + Val"]
    SCORE["Score Features<br/>(spearman, pearson, MI, stability)"]
    FILTER["Greedy Redundancy Filter<br/>(corr < threshold)"]
    SEL["Selected Features<br/>(12 features)"]

    CAND --> FIT --> TRANS --> SCORE --> FILTER --> SEL

    style SCORE fill:#fff3e0
    style FILTER fill:#fce4ec
    style SEL fill:#e8f5e9
```

### Scoring

Each feature is scored against a forward log-return proxy target:

| Component | Weight | Description |
|-----------|--------|-------------|
| `|spearman_val|` | 0.45 | Validation set Spearman correlation |
| `|pearson_val|` | 0.35 | Validation set Pearson correlation |
| `mutual_information` | 0.20 | Non-linear dependence (sklearn) |
| `stability_gap` | -0.20 | Penalty for train/validation divergence |

### Redundancy Filter

Features are ranked by score. Each candidate is added only if its absolute correlation with all already-selected features is below `corr_threshold` (default 0.85). This prevents selecting near-duplicate signals.

### Usage

Feature selection is an **offline** step. Run it once to produce a reduced feature YAML, then point the training config at that file.

**Step 1: Run selection offline**

```python
from trading_rl.features import FeatureGroupResolver, FeatureSelector, FeatureSelectorConfig

resolver = FeatureGroupResolver.from_yaml("src/configs/features/feature_groups.yaml")
candidates = resolver.resolve(resolver.list_groups())

selector = FeatureSelector(FeatureSelectorConfig(top_k=12, corr_threshold=0.85))
result = selector.select(candidates, train_df, val_df)

# Write selected features to a YAML for training
FeatureSelector.write_selected_yaml(
    result,
    "src/configs/features/selected_aapl_hft.yaml",
)
```

**Step 2: Train with the selected features**

```yaml
data:
  feature_config: "src/configs/features/selected_aapl_hft.yaml"
```

### In Scenario Config

Feature groups can be used directly in experiment YAML for **manual composition** (no selection, just pick groups):

```yaml
data:
  data_path: "./data/raw/stocks/AAPL_2026-02-25_2026-02-27_raw_mbp-10_us_hours.parquet"
  train_size: 15000
  validation_size: 2500
  feature_groups: "src/configs/features/feature_groups.yaml"
  feature_selection:
    groups: ["imbalance", "fair_value", "spread", "flow", "regime"]
    exclude: ["feature_hft_book_pressure_l2"]
```

For **research-based selection**, run the offline step to produce a reduced YAML, then use `feature_config` for training.

## Data Flow

```mermaid
flowchart TD
    DATA["Raw LOB Data"]
    FILTER["LOB Filters"]
    SPLIT["Chronological Split<br/>train / val / test"]

    DATA --> FILTER --> SPLIT

    SPLIT -->|train| FIT["Fit Pipeline<br/>(z-score params)"]
    SPLIT -->|val| TRANS_V["Transform (val)"]
    SPLIT -->|test| TRANS_T["Transform (test)"]

    FIT --> TRANS_TRAIN["Transform (train)"]
    FIT -.->|scaler params| TRANS_V
    FIT -.->|scaler params| TRANS_T

    TRANS_TRAIN --> CONCAT_T["Concat<br/>raw + features"]
    TRANS_V --> CONCAT_V["Concat<br/>raw + features"]
    TRANS_T --> CONCAT_TE["Concat<br/>raw + features"]

    CONCAT_T --> ENV["TradingEnv<br/>feature_* columns → observation"]
    CONCAT_V --> ENV
    CONCAT_TE --> ENV

    style FIT fill:#fff3e0
    style ENV fill:#e8f5e9
```

Key invariant: normalization parameters (mean, std for z-score) are estimated on the **training split only** and applied unchanged to validation and test splits. This prevents information leakage from future data into the training representation.