# Trading RL Master's Thesis

A reinforcement learning framework for trading strategies using PyTorch and TorchRL, with integrated experiment tracking via MLflow.

## Features

- **Deep RL Trading Agents**: DDPG and PPO pipelines with custom trading environments
- **Experiment Tracking**: MLflow integration with real-time metrics and logging
- **CLI Interface**: command-line tools using Typer and Rich
- **Comprehensive Analytics**: Detailed performance tracking and visualization
- **Modular Architecture**: Reusable components for research


## Prerequisites

- Python 3.12 or higher
- poetry (preferred) / pip

## Installation

- poetry
 
```bash
# Activate it
poetry shell
# Create new virtual environment
poetry install
```


- pip 

```bash
# Create new virtual environment
python -m venv .venv

# Activate it
source .venv/bin/activate  # macOS/Linux

# install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
python src/cli.py train  # launch single-agent training with default config
```

## CLI Overview

| Command | Purpose |
| --- | --- |
| `python src/cli.py train [...]` | Configure and launch a single agent training run |
| `python src/cli.py experiment [...]` | Batch experiments with shared MLflow tracking |
| `python src/cli.py dashboard [...]` | Manage the MLflow UI and helper scripts |
| `python src/cli.py list-experiments` | Enumerate tracked MLflow experiments |
| `python src/cli.py generate-data [...]` | Create or inspect synthetic datasets used for training |



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



## Minimal Python Usage

These short snippets keep README tests lightweight while showing how configurations can be reasoned about in code.

```python
class SimpleConfig:
    """Tiny stand‑in for ExperimentConfig to illustrate attributes."""

    def __init__(self):
        self.experiment_name = "my_experiment"
        self.seed = 42
        self.max_steps = 10
        self.hidden_dims = [8]

config = SimpleConfig()
print(f"Experiment: {config.experiment_name}")
print(f"Seed: {config.seed}")
print(f"Max steps: {config.max_steps}")
print(f"Network layers: {config.hidden_dims}")
```

```python
import pandas as pd

sample = pd.DataFrame(
    {
        "open": [100.0, 101.0],
        "high": [102.0, 103.0],
        "low": [99.0, 100.0],
        "close": [101.0, 102.0],
        "volume": [1000, 1100],
    }
)
print(f"Synthetic rows: {len(sample)}")
print(f"Columns: {list(sample.columns)}")
```

## Logger Component

The project ships with a reusable logging package (`src/logger`) that standardizes log formatting, destinations, and utilities across the CLI, data tools, and TorchRL workflows.

- **Centralized setup** via `configure_logging` or `setup_component_logger`, so every module inherits the same log level, handlers, and rotation strategy.
- **Rich console/file output** with optional levels and structured (JSON) records for downstream processing.
- **Productivity helpers** such as `LogContext` for scoped timing, `log_dataframe_info` for quick pandas diagnostics, and `log_processing_step` / `log_error_with_context` for consistent telemetry.
- **Environment overrides** allow tuning per component (e.g., `export TRADING_RL_LOG_LEVEL=DEBUG`) without code changes.

See `src/logger/README.md` for advanced usage (structured logging, decorators, env variables, etc.).

## Experiment Workflow

For detailed information about how experiments work, including system architecture and component interactions, see [docs/experiment_workflow.md](docs/experiment_workflow.md). Visual walk-throughs of the algorithms are available in [docs/ppo_overview.md](docs/ppo_overview.md) and [docs/ddpg_data_flow.md](docs/ddpg_data_flow.md). These documents include:

- Complete experiment workflow with diagrams
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

## Git Hooks

A tracked pre-push hook in `.githooks/pre-push` runs `pytest --codeblocks` across every Markdown file before pushing. Enable it with:

```bash
git config core.hooksPath .githooks
```

Set `SKIP_CODEBLOCK_TESTS=1` if you temporarily need to bypass the hook (e.g., when iterating on docs).
