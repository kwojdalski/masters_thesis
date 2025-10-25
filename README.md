# Trading RL Master's Thesis

A comprehensive reinforcement learning framework for trading strategies using PyTorch and TorchRL, with integrated experiment tracking via Optuna.

## Features

- **Deep RL Trading Agents**: DDPG implementation with custom trading environments
- **Experiment Tracking**: Full Optuna integration with real-time training metrics
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
- **Experiment Tracking**: Optuna, Optuna-Dashboard
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
│   │   ├── train_trading_agent.py  # Main training script with Optuna
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
python src/cli.py train --name "my_experiment" --seed 42 --save-plots
```

### 2. Run Multiple Experiments with Optuna
```bash
python src/cli.py experiment --study "trading_study" --trials 10 --dashboard
```

### 3. View Results in Dashboard
```bash
python src/cli.py dashboard trading_study.db
```

## CLI Commands

The project provides a rich command-line interface powered by Typer:

### Training Commands

#### Single Agent Training
```bash
python src/cli.py train [OPTIONS]

Options:
  --name, -n TEXT         Experiment name
  --seed, -s INTEGER      Random seed for reproducibility
  --max-steps INTEGER     Maximum training steps
  --actor-lr FLOAT        Actor learning rate
  --value-lr FLOAT        Value network learning rate
  --buffer-size INTEGER   Replay buffer size
  --save-plots           Save training plots to disk
  --log-dir PATH         Logging directory
```

#### Multiple Experiments
```bash
python src/cli.py experiment [OPTIONS]

Options:
  --study, -s TEXT        Optuna study name
  --trials, -t INTEGER    Number of trials to run
  --storage TEXT          Optuna storage URL
  --dashboard            Launch dashboard after completion
  --config, -c PATH       Custom config file
  --track-positions      Track position changes as intermediate values
  --dual-tracking        Run both loss and position tracking studies
```

### Analysis Commands

#### Launch Dashboard
```bash
python src/cli.py dashboard STUDY_NAME [OPTIONS]

Options:
  --port, -p INTEGER      Dashboard port (default: 8080)
  --host TEXT            Dashboard host (default: localhost)
```

#### List Studies
```bash
python src/cli.py list-studies [OPTIONS]

Options:
  --dir, -d PATH         Directory to search for databases
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

### Multiple Experiments with Optuna
```python
from trading_rl import run_multiple_experiments

# Run multiple experiments
study = run_multiple_experiments("my_study", n_trials=10)

# Access results
print(f"Best trial: {study.best_trial.number}")
print(f"Best reward: {study.best_value}")

# View trial details
for trial in study.trials:
    reward = trial.user_attrs.get('final_reward', 'N/A')
    print(f"Trial {trial.number}: {reward}")
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

The framework provides comprehensive experiment tracking via Optuna:

### Tracked Metrics
- **Performance**: Final reward, training steps, episode rewards
- **Training**: Actor/value losses, training curves, portfolio values
- **Position Activity**: Position changes per episode, trading frequency, position holding duration
- **Configuration**: All hyperparameters, network architecture
- **Data**: Dataset info, start/end dates, data size
- **Environment**: Trading fees, position types, action spaces

### Dashboard Features
- **Interactive Plots**: Real-time training curves and loss visualization
- **Position Tracking**: Monitor how frequently agents change positions
- **Intermediate Values**: Track losses and position changes during training
- **Parameter Analysis**: Hyperparameter importance and relationships
- **Trial Comparison**: Side-by-side comparison of experiments
- **Export Options**: Download results as CSV or visualization images

### Position Change Tracking

The framework provides comprehensive position change tracking with multiple visualization options:

#### **Standard Tracking (User Attributes)**
```python
# Position changes are tracked automatically in all experiments
study = run_multiple_experiments("trading_study", n_trials=5)

# View position activity in dashboard
for trial in study.trials:
    pos_changes = trial.user_attrs.get('total_position_changes', 0)
    avg_changes = trial.user_attrs.get('avg_position_change_per_episode', 0)
    print(f"Trial {trial.number}: {pos_changes} total changes, {avg_changes:.2f} avg per episode")
```

#### **Intermediate Values Tracking (Real-time Plots)**
```python
# Track position changes in intermediate values plot
position_study = run_position_tracking_experiments("position_analysis", n_trials=5)

# Or run both loss and position tracking
loss_study, position_study = run_dual_tracking_experiments("comprehensive", n_trials=5)
```

#### **CLI Position Tracking**
```bash
# Position changes as intermediate values (creates separate plot)
python src/cli.py experiment \
  --study "position_tracking" \
  --trials 10 \
  --track-positions \
  --dashboard

# Dual tracking (creates two studies with different intermediate value plots)
python src/cli.py experiment \
  --study "dual_analysis" \
  --trials 10 \
  --dual-tracking
```

**Available Position Metrics:**
- `total_position_changes` - Total position changes across all episodes
- `avg_position_change_per_episode` - Average changes per episode
- `max_position_changes_per_episode` - Maximum changes in a single episode
- `intermediate_position_changes` - Step-by-step position change progression

**Dashboard Features:**
- **User Attributes Table**: Always shows position metrics for all studies
- **Intermediate Values Plot**: Shows position changes over time when using `--track-positions`
- **Dual Studies**: Compare loss curves and position activity side-by-side
