# Trading RL Master's Thesis

A comprehensive reinforcement learning framework for trading strategies using PyTorch and TorchRL, with integrated experiment tracking via MLflow.

## Features

- **Deep RL Trading Agents**: DDPG and PPO pipelines with custom trading environments
- **Experiment Tracking**: MLflow integration with real-time metrics and artifact logging
- **Rich CLI Interface**: Beautiful command-line tools using Typer and Rich
- **Comprehensive Analytics**: Detailed performance tracking and visualization
- **Modular Architecture**: Clean, reusable components for research

## Logger Component

The project ships with a reusable logging package (`src/logger`) that standardizes log formatting, destinations, and utilities across the CLI, data tools, and TorchRL workflows.

- **Centralized setup** via `configure_logging` or `setup_component_logger`, so every module inherits the same log level, handlers, and rotation strategy.
- **Rich console/file output** with optional colorized levels and structured (JSON) records for downstream processing.
- **Productivity helpers** such as `LogContext` for scoped timing, `log_dataframe_info` for quick pandas diagnostics, and `log_processing_step` / `log_error_with_context` for consistent telemetry.
- **Environment overrides** allow tuning per component (e.g., `export TRADING_RL_LOG_LEVEL=DEBUG`) without code changes.

```python
# Simple logger example (without heavy imports)
import pandas as pd

# Mock logger for demonstration
class MockLogger:
    def info(self, msg): print(f"INFO: {msg}")

logger = MockLogger()
df = pd.DataFrame({"reward": [1.0, 0.4, 0.9]})
logger.info(f"DataFrame shape: {df.shape}")
logger.info("Training loop started")
```

See `src/logger/README.md` for advanced usage (structured logging, decorators, env variables, etc.).

## Prerequisites

- Python 3.12 or higher
- pip (recommended)

## Installation

### Create and activate environment
```bash
# Create new virtual environment
python -m venv .venv

# Activate it
source .venv/bin/activate  # macOS/Linux
```

### Install dependencies
```bash
# Check pip version and help
pip --version
pip --help

# Check poetry status
poetry --version
```

**Key Dependencies:**
- **Deep Learning**: PyTorch, TorchRL, TensorDict
- **Reinforcement Learning**: Gymnasium, Stable-Baselines3, gym-trading-env
- **Experiment Tracking**: MLflow
- **Visualization**: Matplotlib, Seaborn, Plotnine
- **CLI Tools**: Typer, Rich
- **Data Science**: NumPy, Pandas, SciPy, Scikit-learn


## Project Structure

```
masters_thesis/
├── src/
│   ├── trading_rl/           # Main RL package
│   │   ├── __init__.py       # Package exports
│   │   ├── config.py         # Configuration classes
│   │   ├── data_utils.py     # Data processing utilities
│   │   ├── models.py         # Neural network models
│   │   ├── training.py       # DDPG trainer implementation
│   │   ├── train_trading_agent.py  # MLflow-enabled training orchestration
│   │   └── utils.py          # Helper utilities
│   ├── cli.py                # Command-line interface
│   ├── data_generator.py     # Synthetic data generation
│   └── logger.py             # Logging utilities
├── pyproject.toml            # Project configuration
├── README.md                 # This file
└── .gitignore                # Git ignore rules
```

## Quick Start

### 1. Train a Single Agent
```bash
# Show help for train command
python src/cli.py train --help
```

### 2. Run a Batch of Experiments with MLflow Tracking
```bash
# Show help for experiment command
python src/cli.py experiment --help
```

### 3. Launch the MLflow Dashboard
```bash
# Show dashboard help (actual launch runs indefinitely)
python src/cli.py dashboard --help
```

## CLI Commands

The project provides a rich command-line interface powered by Typer:

### Training Commands

#### Single Agent Training
```bash
# Show all available training options
python src/cli.py train --help
```

#### Multiple Experiments
```bash
# Show all available experiment options
python src/cli.py experiment --help
```

### Analysis Commands

#### Launch Dashboard
```bash
# Show dashboard options
python src/cli.py dashboard --help
```

#### List MLflow Experiments
```bash
python src/cli.py list-experiments
```

### Data Generation Commands

#### Generate Synthetic Data
```bash
# Show data generation options
python src/cli.py generate-data --help
```

#### Data Generation Examples
```bash
# List available commands
python src/cli.py --help

# Show data generation help
python src/cli.py generate-data --help

# List experiments
python src/cli.py list-experiments
```

## Example Workflows

### Basic Training Workflow
```bash
# Show training help
python src/cli.py train --help

# Launch dashboard
python src/cli.py dashboard --help
```

### Research Experiment Workflow
```bash
# Show experiment options
python src/cli.py experiment --help
```

```bash
# Launch dashboard (runs indefinitely - skip in tests)
# python src/cli.py dashboard
echo "Dashboard would launch here"
```

### Data and CLI Help
```bash
# Show all available commands
python src/cli.py --help

# Show training help
python src/cli.py train --help

# Show dashboard options
python src/cli.py dashboard --help
```

### Data Preparation Workflow
```bash
# Show data generation help
python src/cli.py generate-data --help

# List available experiments
python src/cli.py list-experiments
```

## Python API Usage

### Direct Training
```python
# Simple configuration example (without heavy imports)
class SimpleConfig:
    def __init__(self):
        self.experiment_name = "my_experiment"
        self.seed = 42

config = SimpleConfig()
print(f"Experiment: {config.experiment_name}")
print(f"Seed: {config.seed}")
print("Configuration created successfully!")
```

### Multiple Experiments with MLflow
```python
# Simple configuration example (without heavy imports)
class SimpleConfig:
    def __init__(self):
        self.experiment_name = "test_experiment"
        self.max_steps = 10

config = SimpleConfig()
print(f"Config created: {config.experiment_name}")
print(f"Max steps: {config.max_steps}")
```

### Custom Configuration
```python
# Simple configuration example (without heavy imports)
class SimpleConfig:
    def __init__(self):
        self.max_steps = 10  # Minimal for quick testing
        self.hidden_dims = [8]  # Tiny network
        self.train_size = 50  # Tiny dataset

config = SimpleConfig()
print(f"Training config: {config.max_steps} steps")
print(f"Network config: {config.hidden_dims}")
```

### Data Generation API
```python
import pandas as pd

# Create simple synthetic data (without heavy imports)
data = {
    'open': [100.0, 101.0],
    'high': [102.0, 103.0], 
    'low': [99.0, 100.0],
    'close': [101.0, 102.0],
    'volume': [1000, 1100]
}
df = pd.DataFrame(data)
print(f"Created {len(df)} rows of synthetic data")
print(f"Columns: {list(df.columns)}")
```

## Experiment Workflow

For detailed information about how experiments work, including system architecture and component interactions, see [docs/experiment_workflow.md](docs/experiment_workflow.md). Visual walk-throughs of the algorithms are available in [docs/ppo_overview.md](docs/ppo_overview.md) and [docs/ddpg_data_flow.md](docs/ddpg_data_flow.md). These documents include:

- Complete experiment workflow with Mermaid diagrams
- Component details and data flow
- MLflow integration architecture  
- Configuration options and usage examples
- Performance optimization and debugging guides

## Experiment Tracking

All training runs are tracked in MLflow.

### Tracked Metrics
- **Performance**: Final reward, total training steps, evaluation horizon
- **Training Dynamics**: Actor/value losses, exploration ratios, checkpoint metadata
- **Position Activity**: Per-episode position changes, portfolio trajectories, action distributions
- **Configuration**: Every hyperparameter, network architecture, dataset stats, environment settings

### Dashboard Features
- Interactive loss and reward curves per trial
- Automatically generated comparison plots saved as artifacts
- Drill-down view of position changes and trading behaviour
- Artifact bundles (plots, CSV summaries, configs) for offline analysis

Launch the dashboard with `python src/cli.py dashboard` or open the experiment in any MLflow-compatible UI to explore results.
