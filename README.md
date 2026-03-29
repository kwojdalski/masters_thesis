# Trading RL Master's Thesis

Research codebase for trading strategy experiments with TorchRL. It includes
synthetic price generation, Gymnasium-compatible environments, and MLflow-based
experiment tracking.

## Highlights

- PPO, DDPG, and TD3 trainers for discrete and continuous action spaces
- Scenario-driven YAML configs in `src/configs/scenarios`
- Synthetic data generator (sine wave, upward drift, sampled OHLCV)
- MLflow tracking plus CLI utilities for experiments, checkpoints, and artifacts
- Visualization analytics and reusable logging utilities

## Prerequisites

- Python 3.12
- [uv](https://docs.astral.sh/uv/)

## Installation

<!--pytest.mark.skip-->
```bash
uv sync --extra dev
```

## Quick Start

Common commands:

| Command | Purpose |
| --- | --- |
| `uv run python src/cli.py scenarios` | List available scenario configs |
| `uv run python src/cli.py data generate --sine-wave --n-periods 8 --samples-per_period 250 --output-file data/raw/synthetic/sine_wave.parquet` | Generate synthetic sine-wave data |
| `uv run python src/cli.py train --config src/configs/scenarios/sine_wave_ppo_no_trend_tradingenv.yaml` | Train a single agent |
| `uv run python src/cli.py train --config src/configs/scenarios/sine_wave_ppo_no_trend.yaml --trials 3` | Run multiple trials |
| `uv run python src/cli.py validate --scenario aapl_td3_hft_lob` | Validate scenario config and data dependencies |
| `uv run python src/cli.py dashboard` | Launch the MLflow UI |
| `uv run python src/cli.py checkpoints` | Inspect or clean checkpoints |
| `uv run python src/cli.py experiments` | List MLflow experiments |
| `uv run python src/cli.py artifacts --experiment <regex>` | List artifacts per run |

## Data Download

For downloading real market data, use the dedicated scripts:

```bash
# Download cryptocurrency data (BTC, ETH, etc.)
uv run python scripts/fetch_crypto.py download-crypto --symbols BTC/USDT --timeframe 1h

# Download stock data (requires DATABENTO_API_KEY)
uv run python scripts/fetch_stocks.py download-stocks --symbols AAPL --start-date 2024-01-01
```

### Non-synthetic proprietary stock data (Google Drive)

If you have access to the private shared dataset folder, download it into
`data/raw/stocks` using:

```bash
export GDRIVE_STOCKS_URL="https://drive.google.com/drive/folders/<your-folder-id>"

# Download all files (authenticated via Drive API)
export GDRIVE_CLIENT_SECRET_FILE="$HOME/.secrets/gdrive-client-secret.json"
uv run python scripts/download_stocks_from_gdrive.py

# Or pick specific files interactively with fzf
uv run python scripts/download_stocks_from_gdrive.py --interactive
```

Authentication options (set one):
- `GDRIVE_SERVICE_ACCOUNT_FILE` — path to a service-account JSON key (recommended for automation)
- `GDRIVE_CLIENT_SECRET_FILE` — path to OAuth client secrets JSON (opens browser on first run)

Notes:
- The URL must be provided via `GDRIVE_STOCKS_URL` (or passed with `--url`).
- Destination defaults to `data/raw/stocks` and is created automatically.
- The folder does not need to be publicly shared — authenticated downloads use the Drive API directly.

See `docs/data_guide.md` for detailed instructions on data download and generation.

## Configuration

- Scenario YAML files live in `src/configs/scenarios`.
- Provide a custom config with `--config`.
- Override values with dotlist syntax via `--config-override`, for example:

<!--pytest.mark.skip-->
```bash
uv run python src/cli.py train \
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
│   │   ├── scenarios/       # Experiment scenario YAML configs
│   │   ├── data/            # Data-source/data-generation configs
│   │   └── features/        # Feature-set configs
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
