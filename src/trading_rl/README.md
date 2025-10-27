# Trading RL - Modular DDPG Implementation

A clean, modular implementation of DDPG (Deep Deterministic Policy Gradient) for trading environments using TorchRL.

## Project Structure

```
trading_rl/
├── __init__.py           # Package exports
├── config.py             # Configuration dataclasses
├── data_utils.py         # Data loading and preprocessing
├── models.py             # Neural network architectures
├── training.py           # Training loop and utilities
└── README.md            # This file

scripts/
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
- `DDPGTrainer`: Complete training loop with:
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
from trading_rl import ExperimentConfig
from train_trading_agent import main

# Use default configuration
results = main()
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
    prepare_data,
    create_actor,
    create_value_network,
    DDPGTrainer,
    ExperimentConfig,
)

# Load data
config = ExperimentConfig()
df = prepare_data(config.data.data_path)

# Create environment (your code)
env = create_environment(df, config)

# Create models
n_obs, n_act = get_env_dims(env)
actor = create_actor(n_obs, n_act)
value_net = create_value_network(n_obs, n_act)

# Train
trainer = DDPGTrainer(actor, value_net, env, config.training)
logs = trainer.train()
```

## Key Improvements Over Original

### 1. **Separation of Concerns**
- Data processing separate from training
- Model definitions separate from usage
- Configuration separate from code

### 2. **Reusability**
- All components can be imported and reused
- Easy to test individual components
- Can create variations without duplication

### 3. **Maintainability**
- Clear module boundaries
- Type hints throughout
- Comprehensive docstrings
- Logging at appropriate levels

### 4. **Configurability**
- All hyperparameters in config
- Easy to run experiments with different settings
- No hardcoded magic numbers

### 5. **Reproducibility**
- Seed setting
- Checkpoint save/load
- Complete logging of hyperparameters

### 6. **Extensibility**
- Easy to add new features
- Easy to swap components (e.g., different networks)
- Easy to add new algorithms

## Configuration Examples

### Quick Experiment
```python
config = ExperimentConfig()
config.training.max_steps = 100_000
config.training.eval_interval = 500
config.data.train_size = 500
```

### Production Training
```python
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
config = ExperimentConfig()
config.training.max_steps = 10_000
config.training.log_interval = 100
config.logging.log_level = "DEBUG"
config.data.train_size = 100
```

## Testing Individual Components

```python
# Test data loading
from trading_rl import prepare_data
df = prepare_data("path/to/data.pkl")
assert "feature_return" in df.columns

# Test model creation
from trading_rl import create_actor
actor = create_actor(n_obs=10, n_act=3)
assert actor is not None

# Test configuration
from trading_rl import ExperimentConfig
config = ExperimentConfig()
assert config.training.actor_lr == 1e-4
```

## Extending the Code

### Adding a New Feature

Edit `data_utils.py`:
```python
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    # ... existing features ...

    # Add your feature
    df["feature_my_indicator"] = compute_my_indicator(df)
    df["feature_my_indicator"] = normalize(df["feature_my_indicator"])

    return df
```

### Using a Different Algorithm

Create `trading_rl/ppo_training.py`:
```python
class PPOTrainer:
    def __init__(self, actor, critic, env, config):
        # ... setup PPO ...
```

### Custom Reward Function

Edit `data_utils.py`:
```python
def sharpe_reward_function(history: dict) -> float:
    """Risk-adjusted reward using Sharpe ratio."""
    returns = compute_returns(history)
    return returns / (returns.std() + 1e-8)
```

## Tips

1. **Start Small**: Use small train_size and max_steps for debugging
2. **Monitor Logs**: Check loss values and eval metrics regularly
3. **Save Checkpoints**: Enable checkpointing for long runs
4. **Experiment Tracking**: Use different experiment_name for each run
5. **Version Control**: Commit config changes with code changes

## Requirements

See `requirements.txt` in project root for dependencies.
