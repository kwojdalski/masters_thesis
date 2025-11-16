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

### Using Poetry (recommended)

<!--pytest.mark.skip-->
```bash
poetry install          # install dependencies into Poetry-managed venv
poetry shell            # spawn a shell inside that environment
```

### Using pip + venv

<!--pytest.mark.skip-->
```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

## Quick Start


### Common commands

All commands support the `--help` flag, which displays detailed information about their usage and options.

| Command | Purpose |
| --- | --- |
| `python src/cli.py train [...]` | Configure and launch a single agent training run |
| `python src/cli.py experiment [...]` | Batch experiments with shared MLflow tracking |
| `python src/cli.py dashboard [...]` | Manage the MLflow UI and helper scripts |
| `python src/cli.py generate-data [...]` | Create or inspect synthetic datasets used for training |
| `python src/cli.py list-experiments` | Enumerate tracked MLflow experiments |

<!--pytest.mark.skip-->
```bash
python src/cli.py train      # launch single-agent training with default config
python src/cli.py dashboard  # start MLflow UI backed by sqlite:///mlflow.db
```

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

## Logger Component

The project ships with a reusable logging package (`src/logger`) that standardizes log formatting, destinations, and utilities across the CLI, data tools, and TorchRL workflows.

- **Centralized setup** via `configure_logging` or `setup_component_logger`, so every module inherits the same log level, handlers, and rotation strategy.
- **Rich console/file output** with optional levels and structured (JSON) records for downstream processing.
- **Productivity helpers** such as `LogContext` for scoped timing, `log_dataframe_info` for quick pandas diagnostics, and `log_processing_step` / `log_error_with_context` for consistent telemetry.
- **Environment overrides** allow tuning per component (e.g., `export TRADING_RL_LOG_LEVEL=DEBUG`) without code changes.

See `src/logger/README.md` for advanced usage (structured logging, decorators, env variables, etc.).

## Experiment Workflow

For detailed information about how experiments work, including system architecture and component interactions, see [docs/experiment_workflow.md](docs/experiment_workflow.md). Visual walk-throughs of the algorithms are available in [docs/ppo_overview.md](docs/ppo_overview.md) and [docs/ddpg_overview.md](docs/ddpg_overview.md). These documents include:

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
- Drill-down view of position changes and trading behaviour
- Artifact bundles (plots, CSV summaries, configs)
