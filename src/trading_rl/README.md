# Trading RL - Modular Implementation (PPO, DDPG, TD3)

A clean, modular implementation of PPO, DDPG, and TD3 for trading environments using TorchRL.

## Project Structure

```
trading_rl/
├── __init__.py           # Package exports
├── config.py             # Configuration dataclasses
├── data_utils.py         # Data loading and preprocessing
├── models.py             # Neural network architectures
├── training.py           # Training loop and utilities
└── README.md            # This file

../scripts/
└── train_trading_agent.py  # Main training script
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
- `load_trading_data()`: Load from pickle files
- `create_features()`: Feature engineering (returns, volatility, etc.)
- `reward_function()`: Portfolio valuation reward
- `prepare_data()`: End-to-end data preparation

**Benefits**: Reusable data pipeline, consistent feature engineering

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

## Extending the Code

### Adding a New Feature

Edit `data_utils.py`:
```python
import pandas as pd
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    # ... existing features ...

    # Add your feature
    df["feature_my_indicator"] = compute_my_indicator(df)
    df["feature_my_indicator"] = normalize(df["feature_my_indicator"])

    return df
```

### Adding a New Algorithm

Extend `BaseTrainer` in `trading_rl/training.py`:
```python
class MyNewAlgoTrainer(BaseTrainer):
    def __init__(self, actor, value_net, env, config):
        """Initialize algorithm components."""
        super().__init__(env, config)
        # Initialize specific loss modules and optimizers here

    def train_step(self, batch):
        """Implement single training step."""
        # Calculate loss, update networks
        return metrics
```

### Custom Reward Function

Edit `data_utils.py`:
```python
def sharpe_reward_function(history: dict) -> float:
    """Risk-adjusted reward using Sharpe ratio."""
    returns = compute_returns(history)
    return returns / (returns.std() + 1e-8)
```

See `requirements.txt` in project root for dependencies.
