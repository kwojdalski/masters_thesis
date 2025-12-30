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

Example feature config snippet:
```yaml
features:
  - name: "return_lag_3"
    feature_type: "return_lag"
    params:
      column: "close"
      lag: 3
    normalize: true
```

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
