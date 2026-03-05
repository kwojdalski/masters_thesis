# Trading RL Master's Thesis

Research codebase for trading strategy experiments with TorchRL. It includes
synthetic price generation, Gymnasium-compatible environments, and MLflow-based
experiment tracking.

## Highlights

- PPO, DDPG, and TD3 trainers for discrete and continuous action spaces
- Scenario-driven YAML configs in `src/configs/scenarios`
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
| `python src/cli.py data generate --scenario sine_wave` | Generate synthetic data |
| `python src/cli.py train --config src/configs/scenarios/sine_wave_ppo_no_trend_tradingenv.yaml` | Train a single agent |
| `python src/cli.py train --config src/configs/scenarios/sine_wave_ppo_no_trend.yaml --trials 3` | Run multiple trials |
| `python src/cli.py dashboard` | Launch the MLflow UI |
| `python src/cli.py checkpoints` | Inspect or clean checkpoints |
| `python src/cli.py experiments` | List MLflow experiments |
| `python src/cli.py artifacts --experiment <regex>` | List artifacts per run |

## Data Download

For downloading real market data, use the dedicated scripts:

```bash
# Download cryptocurrency data (BTC, ETH, etc.)
python scripts/fetch_crypto.py download-crypto --symbols BTC/USDT --timeframe 1h

# Download stock data (requires DATABENTO_API_KEY)
python scripts/fetch_stocks.py download-stocks --symbols AAPL --start-date 2024-01-01
```

### Non-synthetic proprietary stock data (Google Drive)

If you have access to the private shared dataset folder, download it into
`data/raw/stocks` using:

```bash
source .venv/bin/activate
pip install gdown
export GDRIVE_STOCKS_URL="https://drive.google.com/drive/folders/<your-folder-id>"
python scripts/download_stocks_from_gdrive.py
```

Notes:
- The URL must be provided via `GDRIVE_STOCKS_URL` (or passed with `--url`).
- Destination defaults to `data/raw/stocks` and is created automatically.
- You need permission to access the shared Google Drive folder.

See `docs/data_guide.md` for detailed instructions on data download and generation.

## Configuration

- Scenario YAML files live in `src/configs/scenarios`.
- Provide a custom config with `--config`.
- Override values with dotlist syntax via `--config-override`, for example:

<!--pytest.mark.skip-->
```bash
python src/cli.py train \
  --config src/configs/scenarios/sine_wave_ppo_no_trend_tradingenv.yaml \
  --config-override training.max_steps=50000 \
  --config-override training.actor_lr=3e-5
```

## Project Structure

```
masters_thesis/
├── src/
│   ├── cli/                 # CLI command implementations
│   ├── cli.py               # CLI entrypoint
│   ├── configs/
│   │   └── scenarios/       # Scenario YAML configs
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

- `docs/data_guide.md` - Data download and generation guide
- `docs/experiment_workflow.md` - End-to-end training workflow
- `docs/ppo_implementation_overview.md` - PPO algorithm details
- `docs/ddpg_implementation_overview.md` - DDPG algorithm details
- `docs/td3_implementation_overview.md` - TD3 algorithm details
- `src/trading_rl/README.md` - Core RL package overview
- `src/logger/README.md` - Logging utilities

## Experiment Tracking

Training runs are tracked in MLflow. The default store is `sqlite:///mlflow.db`
with artifacts in `mlruns/`.

<!--pytest.mark.skip-->
```bash
python src/cli.py dashboard
python src/cli.py experiments
python src/cli.py artifacts --experiment sine_wave
```
