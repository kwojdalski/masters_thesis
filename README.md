# Trading RL Master's Thesis

A comprehensive reinforcement learning framework for trading strategies using PyTorch and TorchRL, with integrated experiment tracking via MLflow.

## Features

- **Deep RL Trading Agents**: DDPG and PPO pipelines with custom trading environments
- **Experiment Tracking**: MLflow integration with real-time metrics and artifact logging
- **Rich CLI Interface**: Beautiful command-line tools using Typer and Rich
- **Comprehensive Analytics**: Detailed performance tracking and visualization
- **Modular Architecture**: Clean, reusable components for research

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
# Install project dependencies
pip install -e .

# Or using poetry
poetry install
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
# Uses the experiment_name defined in the config unless you override --name
python src/cli.py train --config src/configs/default.yaml --seed 42 --save-plots
```

### 2. Run a Batch of Experiments with MLflow Tracking
```bash
# Runs three trials; MLflow experiment name comes from the YAML unless --name is set
python src/cli.py experiment --config src/configs/upward_drift_ppo.yaml --trials 3 --max-steps 8000
```

### 3. Launch the MLflow Dashboard
```bash
python src/cli.py dashboard --host 0.0.0.0 --port 5000
```

## CLI Commands

The project provides a rich command-line interface powered by Typer:

### Training Commands

#### Single Agent Training
```bash
python src/cli.py train [OPTIONS]

Options:
  --name, -n TEXT         Override experiment name (defaults to config)
  --config, -c PATH       Path to config YAML (uses defaults if omitted)
  --seed, -s INTEGER      Random seed for reproducibility
  --max-steps INTEGER     Maximum training steps
  --actor-lr FLOAT        Actor learning rate override
  --value-lr FLOAT        Value network learning rate override
  --buffer-size INTEGER   Replay buffer size override
  --save-plots            Save training plots to disk
  --log-dir PATH          Logging directory override
```

#### Multiple Experiments
```bash
python src/cli.py experiment [OPTIONS]

Options:
  --name, -n TEXT         Override MLflow experiment name
  --trials, -t INTEGER    Number of trials to run (default: 5)
  --dashboard             Launch MLflow UI after experiments finish
  --config, -c PATH       Path to config YAML (required for custom setups)
  --seed INTEGER          Base seed; each trial uses seed + trial_number
  --max-steps INTEGER     Maximum training steps per trial
  --clear-cache           Clear cached datasets/models before running
  --no-features           Skip feature engineering (raw OHLCV only)
```

### Analysis Commands

#### Launch Dashboard
```bash
python src/cli.py dashboard [OPTIONS]

Options:
  --port, -p INTEGER      MLflow UI port (default: 5000)
  --host TEXT             MLflow UI host (default: localhost)
```

#### List MLflow Experiments
```bash
python src/cli.py list-experiments
```

### Data Generation Commands

#### Generate Synthetic Data
```bash
python src/cli.py generate-data [OPTIONS]

Options:
  --source-dir TEXT      Source directory for parquet files
  --output-dir TEXT      Output directory for synthetic data
  --source-file TEXT     Source parquet file name
  --output-file TEXT     Output file name
  --start-date TEXT      Start date (YYYY-MM-DD)
  --end-date TEXT        End date (YYYY-MM-DD)
  --sample-size INTEGER  Number of rows to sample
  --list                List available source files
  --copy                Copy without modifications

Note: Generated files are saved to `./data/raw/synthetic/` by default.
```

#### Data Generation Examples
```bash
# List available data files
python src/cli.py generate-data --list

# Generate data by date range
python src/cli.py generate-data \
  --source-file binance-BTCUSDT-1h.parquet \
  --output-file BTCUSDT_2023.parquet \
  --start-date 2023-01-01 \
  --end-date 2023-12-31

# Generate random sample
python src/cli.py generate-data \
  --source-file binance-BTCUSDT-1h.parquet \
  --output-file sample_1000.parquet \
  --sample-size 1000

# Copy data without modifications
python src/cli.py generate-data \
  --source-file binance-BTCUSDT-1h.parquet \
  --output-file BTCUSDT_backup.parquet \
  --copy

# Generate training/validation/test sets
python src/cli.py generate-data \
  --source-file binance-BTCUSDT-1h.parquet \
  --output-file train.parquet \
  --start-date 2020-01-01 --end-date 2022-12-31

python src/cli.py generate-data \
  --source-file binance-BTCUSDT-1h.parquet \
  --output-file val.parquet \
  --start-date 2023-01-01 --end-date 2023-12-31
```

## Example Workflows

### Basic Training Workflow
```bash
# Train a single agent with custom parameters
python src/cli.py train \
  --name "ddpg_experiment_1" \
  --seed 42 \
  --max-steps 50000 \
  --actor-lr 0.001 \
  --value-lr 0.002 \
  --save-plots

# View results
python src/cli.py dashboard ddpg_experiment_1.db
```

### Research Experiment Workflow
```bash
# Run multiple experiments for statistical analysis
python src/cli.py experiment \
  --study "parameter_sweep" \
  --trials 20 \
  --dashboard

# Run experiments with position change tracking in intermediate values
python src/cli.py experiment \
  --study "position_analysis" \
  --trials 10 \
  --track-positions \
  --dashboard

# Run dual tracking (both loss and position studies)
python src/cli.py experiment \
  --study "comprehensive_analysis" \
  --trials 15 \
  --dual-tracking

# List all your studies
python src/cli.py list-studies

# Launch dashboard for specific study
python src/cli.py dashboard parameter_sweep.db --port 8080
```

### Synthetic Data Simulations
```bash
# Generate synthetic patterns for controlled testing
python src/cli.py generate-data --source-file binance-BTCUSDT-1h.parquet --output-file sine_wave_validation.parquet --sample-size 500

# Train with pre-configured synthetic patterns
python src/cli.py train --config src/configs/sine_wave.yaml        # Oscillating patterns
python src/cli.py train --config src/configs/upward_drift.yaml     # Trending markets  
python src/cli.py train --config src/configs/mean_reversion.yaml   # Mean-reverting patterns

# Run experiments with synthetic data
python src/cli.py experiment --config src/configs/sine_wave.yaml --name "synthetic_experiments"

# View results
python src/cli.py dashboard
```

### Data Preparation Workflow
```bash
# List available data
python src/cli.py generate-data --list

# Generate training data for specific period
python src/cli.py generate-data \
  --source-file binance-BTCUSDT-1h.parquet \
  --output-file training_2023.parquet \
  --start-date 2023-01-01 \
  --end-date 2023-12-31

# Generate multiple sample sizes for experiments
for n in 100 500 1000 5000; do
  python src/cli.py generate-data \
    --source-file binance-BTCUSDT-1h.parquet \
    --output-file sample_${n}.parquet \
    --sample-size $n
done
```

## Python API Usage

### Direct Training
```python
from trading_rl import run_single_experiment, ExperimentConfig

# Run single experiment
config = ExperimentConfig()
config.experiment_name = "my_experiment"
config.seed = 42

result = run_single_experiment(custom_config=config)
print(f"Final reward: {result['final_metrics']['final_reward']}")
```

### Multiple Experiments with MLflow
```python
from trading_rl import ExperimentConfig, run_multiple_experiments

# Load configuration and run three trials
config = ExperimentConfig.from_yaml("src/configs/upward_drift_ppo.yaml")
experiment_name = run_multiple_experiments(
    n_trials=3,
    base_seed=123,
    custom_config=config,
)

print(f"Results are logged in MLflow under: {experiment_name}")
```

### Custom Configuration
```python
from trading_rl import ExperimentConfig, run_single_experiment

# Create custom configuration
config = ExperimentConfig()
config.training.max_training_steps = 100000
config.training.actor_lr = 0.001
config.training.value_lr = 0.002
config.network.actor_hidden_dims = [256, 256]
config.data.train_size = 10000

# Run with custom config
result = run_single_experiment(custom_config=config)
```

### Data Generation API
```python
from src.data_generator import PriceDataGenerator

# Initialize generator
generator = PriceDataGenerator()

# Generate synthetic data
df = generator.generate_synthetic_sample(
    source_file="binance-BTCUSDT-1h.parquet",
    output_file="custom.parquet",
    start_date="2023-01-01",
    end_date="2023-12-31",
    sample_size=1000
)

# List available files
files = generator.list_source_files()
```

## Experiment Workflow

For detailed information about how experiments work, including system architecture and component interactions, see [docs/experiment_workflow.md](docs/experiment_workflow.md). This documentation includes:

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
