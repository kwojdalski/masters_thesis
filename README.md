# Trading RL Master's Thesis

Research codebase for trading strategy experiments with TorchRL. It includes
synthetic price generation, Gymnasium-compatible environments, and MLflow-based
experiment tracking.

## Highlights

- PPO, DDPG, and TD3 trainers for discrete and continuous action spaces
- Scenario-driven YAML configs in `src/configs`
- Synthetic data generator (sine wave, upward drift, sampled OHLCV)
- MLflow tracking plus CLI utilities for experiments, checkpoints, and artifacts
- Plotnine-based analytics and reusable logging utilities

## Prerequisites

- Python 3.12
- pip or Poetry

## Installation

### pip + venv (recommended)

<!--pytest.mark.skip-->
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Poetry

<!--pytest.mark.skip-->
```bash
poetry config virtualenvs.in-project true
poetry install
source .venv/bin/activate
```

## Quick Start

Activate the environment before running commands:

<!--pytest.mark.skip-->
```bash
source .venv/bin/activate
```

Common commands (`python src/cli.py --help` for full details):

| Command | Purpose |
| --- | --- |
| `python src/cli.py scenarios` | List available scenario configs |
| `python src/cli.py generate-data --scenario sine_wave` | Generate synthetic data |
| `python src/cli.py train --config src/configs/sine_wave_ppo_no_trend_tradingenv.yaml` | Train a single agent |
| `python src/cli.py experiment --config src/configs/sine_wave_ppo_no_trend.yaml --trials 3` | Run multiple trials |
| `python src/cli.py dashboard` | Launch the MLflow UI |
| `python src/cli.py checkpoints` | Inspect or clean checkpoints |
| `python src/cli.py experiments` | List MLflow experiments |
| `python src/cli.py artifacts --experiment <regex>` | List artifacts per run |

## Configuration

- Scenario YAML files live in `src/configs`.
- Provide a custom config with `--config`.
- Override values with dotlist syntax via `--config-override`, for example:

<!--pytest.mark.skip-->
```bash
python src/cli.py train \
  --config src/configs/sine_wave_ppo_no_trend_tradingenv.yaml \
  --config-override training.max_steps=50000 \
  --config-override training.actor_lr=3e-5
```

## Project Structure

```
masters_thesis/
├── src/
│   ├── cli/                 # CLI command implementations
│   ├── cli.py               # CLI entrypoint
│   ├── configs/             # Scenario and model YAML configs
│   ├── data_generator.py    # Synthetic data generation
│   ├── logger/              # Shared logging utilities
│   └── trading_rl/          # Core RL package
│       ├── envs/            # Environment builders/wrappers
│       ├── rewards/         # Reward functions
│       ├── trainers/        # PPO, DDPG, TD3 trainers
│       └── training.py      # Training loops and helpers
├── data/                    # Raw and synthetic data
├── docs/                    # Experiment and algorithm docs
├── notebooks/               # Research notebooks
├── scripts/                 # Debugging and helper scripts
└── tests/                   # Unit tests
```

## Docs and References

- `docs/experiment_workflow.md`
- `docs/ppo_implementation_overview.md`
- `docs/ddpg_implementation_overview.md`
- `docs/td3_implementation_overview.md`
- `src/trading_rl/README.md`
- `src/logger/README.md`

## Experiment Tracking

Training runs are tracked in MLflow. The default store is `sqlite:///mlflow.db`
with artifacts in `mlruns/`.

<!--pytest.mark.skip-->
```bash
python src/cli.py dashboard
python src/cli.py experiments
python src/cli.py artifacts --experiment sine_wave
```
