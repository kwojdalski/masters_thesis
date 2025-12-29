# Trading RL - Modular Implementation (PPO, DDPG, TD3)

A clean, modular implementation of PPO, DDPG, and TD3 for trading environments
using TorchRL. The package includes environment builders, a feature engineering
pipeline, training loops, and MLflow-compatible experiment utilities.

## Project Structure

```
trading_rl/
├── __init__.py           # Package exports
├── cache_utils.py        # Joblib cache helpers
├── callbacks/            # MLflow and training callbacks
├── config.py             # Configuration dataclasses
├── continuous_action_wrapper.py  # Continuous action helpers
├── data_utils.py         # Data loading and preprocessing
├── docs/                 # Package-specific docs
├── envs/                 # Environment builders/wrappers
├── features/             # Feature engineering pipeline
├── models.py             # Neural network architectures
├── plotting.py           # Plotnine visualizations
├── rewards/              # Reward function implementations
├── train_trading_agent.py # Main training script
├── trainers/             # PPO, DDPG, TD3 trainer implementations
├── training.py           # Training loop and utilities
├── utils.py              # Shared utilities
└── README.md             # This file
```

## Module Overview

### `config.py`
Defines configuration using dataclasses for:
- **DataConfig**: Data loading and preprocessing settings
- **EnvConfig**: Trading environment parameters
- **NetworkConfig**: Neural network architectures
- **TrainingConfig**: Training hyperparameters
- **LoggingConfig**: Logging settings
- **ExperimentConfig**: Combined configuration

**Benefits**: Type-safe, easy to modify, can be serialized/deserialized

### `data_utils.py`
Data handling utilities:
- `download_trading_data()`: Download from exchanges
- `load_trading_data()`: Load from pickle/parquet
- `create_features()`: Default feature engineering (returns, volatility, etc.)
- `reward_function()`: Portfolio valuation reward
- `prepare_data()`: End-to-end data preparation

**Benefits**: Reusable data pipeline, consistent feature engineering. The
hardcoded `create_features()` utility still exists for Gym-compatible datasets,
while `features/` provides a configurable pipeline for experiments that need
train/test-safe normalization.

### `features/`
Feature engineering pipeline with train/test-safe normalization and a registry
for custom features.

**Benefits**: Configurable feature sets, no data leakage, and easy extension.

### `models.py`
Neural network components:
- `DiscreteNet`: Flexible discrete action network
- `create_actor()`: Build probabilistic actor
- `create_value_network()`: Build value network
- `count_parameters()`: Utility for model inspection

**Benefits**: Configurable architectures, easy to experiment

### `training.py`
Training infrastructure:
- `BaseTrainer`: Abstract base class with shared logic
- `DDPGTrainer`: DDPG implementation
- `PPOTrainer`: PPO implementation
- `TD3Trainer`: TD3 implementation
- **Features**:
  - Replay buffer management
  - Data collection
  - Optimization steps
  - Periodic evaluation
  - Checkpoint saving/loading
  - Comprehensive logging

**Benefits**: Encapsulated training logic, reproducible experiments

### `trainers/`
Algorithm-specific trainer implementations (PPO/DDPG/TD3) built on TorchRL.

### `train_trading_agent.py`
Main entry point that:
1. Loads configuration
2. Prepares data
3. Creates environment and models
4. Runs training
5. Saves checkpoints
6. Generates visualizations

**Benefits**: Simple, readable, easy to customize

## Usage

### Basic Training

```python
from trading_rl import ExperimentConfig, run_single_experiment

config = ExperimentConfig()
print(f"Training entrypoint ready: {run_single_experiment.__name__}, seed={config.seed}")
```

### Custom Configuration

```python
from trading_rl import ExperimentConfig, TrainingConfig

# Create custom config
config = ExperimentConfig()
config.training.actor_lr = 5e-5
config.training.max_steps = 1_000_000
config.data.train_size = 2000
config.experiment_name = "my_experiment"

# Modify main() to accept config or use directly
```

### Using Individual Components

```python
from trading_rl import (
    ExperimentConfig,
    create_actor,
    create_value_network,
)
from torchrl.data import OneHotDiscreteTensorSpec

# Create lightweight configuration and models (no data/env required)
config = ExperimentConfig()
n_obs, n_act = 8, 3
action_spec = OneHotDiscreteTensorSpec(n_act)

actor = create_actor(n_obs, n_act, spec=action_spec)
value_net = create_value_network(n_obs, n_act)

print(f"Actor: {actor.__class__.__name__}, ValueNet: {value_net.__class__.__name__}")
print(f"Training max steps: {config.training.max_steps}")
```

## Feature Engineering

The feature pipeline is designed to avoid data leakage by fitting
normalization on training data only. Features are registered by name and
instantiated via configuration, so you can mix and match without editing code.

### Key Ideas

1. **Fit on train, transform on both**: normalization stats come only from
   training data.
2. **Configuration-based**: define features in dicts or YAML.
3. **Registry pattern**: add new features with a decorator.

### Basic Usage

```python
from trading_rl.features import FeaturePipeline, FeatureConfig

configs = [
    FeatureConfig(name="log_return", feature_type="log_return"),
    FeatureConfig(name="high", feature_type="high"),
    FeatureConfig(name="low", feature_type="low"),
]

pipeline = FeaturePipeline(configs)
pipeline.fit(train_df)
train_features = pipeline.transform(train_df)
test_features = pipeline.transform(test_df)
```

### From Dictionary Config

```python
feature_configs = [
    {"name": "log_return", "feature_type": "log_return"},
    {"name": "rsi_14", "feature_type": "rsi", "params": {"period": 14}},
    {"name": "log_vol", "feature_type": "log_volume"},
]

pipeline = FeaturePipeline.from_config_dict(feature_configs)
```

### Available Features

Price features:
- `log_return`: (close_t / close_t-1) - 1
- `simple_return`: close.pct_change()
- `high`: (high / close) - 1
- `low`: (low / close) - 1
- `trend`: min-max normalized cumulative trend
- `rsi`: Relative Strength Index (configurable period)

Volume features:
- `log_volume`: log1p(volume)
- `volume_change`: (volume_t / volume_t-1) - 1
- `volume_ma_ratio`: volume relative to moving average

### Configuration Format

```yaml
features:
  - name: "log_return"
    feature_type: "log_return"
    normalize: true

  - name: "rsi_14"
    feature_type: "rsi"
    params:
      period: 14
    normalize: true

  - name: "high"
    feature_type: "high"
    normalize: true
```

Deprecated (still supported in legacy data flows):

```yaml
feature_columns: ["feature_log_return", "feature_high", "feature_low"]
```

### Creating Custom Features

```python
from trading_rl.features import Feature, register_feature
import pandas as pd

@register_feature("my_custom_feature")
class MyCustomFeature(Feature):
    """Custom feature description."""

    def required_columns(self) -> list[str]:
        return ["close", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        return df["close"] * df["volume"]
```

### Migration from `create_features`

```python
# Old: hardcoded features, global normalization
df = create_features(df)

# New: configurable, train/test-safe
pipeline = FeaturePipeline.from_config_dict(feature_configs)
pipeline.fit(train_df)
train_features = pipeline.transform(train_df)
test_features = pipeline.transform(test_df)
```

## Configuration Examples

### Quick Experiment
```python
from trading_rl import ExperimentConfig

config = ExperimentConfig()
config.training.max_steps = 100_000
config.training.eval_interval = 500
config.data.train_size = 500
```

### Production Training
```python
from trading_rl import ExperimentConfig

config = ExperimentConfig()
config.training.max_steps = 10_000_000
config.training.actor_lr = 3e-5
config.training.value_lr = 1e-3
config.training.buffer_size = 1_000_000
config.data.train_size = 10000
config.network.actor_hidden_dims = [128, 128, 64]
config.network.value_hidden_dims = [128, 128, 64, 32]
```

### Debugging
```python
from trading_rl import ExperimentConfig

config = ExperimentConfig()
config.training.max_steps = 10_000
config.training.log_interval = 100
config.logging.log_level = "DEBUG"
config.data.train_size = 100
```

## Testing Individual Components

```python
from trading_rl import (
    ExperimentConfig,
    create_actor,
    create_value_network,
    prepare_data,
)
from torchrl.data import OneHotDiscreteTensorSpec

config = ExperimentConfig()
action_spec = OneHotDiscreteTensorSpec(2)
actor = create_actor(n_obs=6, n_act=2, spec=action_spec)
value_net = create_value_network(n_obs=6, n_act=2)

try:
    prepare_data("missing_dataset.parquet")
except FileNotFoundError as exc:
    print(f"prepare_data raised as expected: {exc.__class__.__name__}")

print(f"Actor type: {actor.__class__.__name__}")
print(f"Value net type: {value_net.__class__.__name__}")
print(f"Default actor LR: {config.training.actor_lr}")
```
