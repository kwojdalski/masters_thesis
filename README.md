# Trading RL Master's Thesis

This codebase supports the research behind the thesis by implementing and
comparing deep reinforcement learning algorithms — [PPO](docs/ppo_implementation_overview.md), [DDPG](docs/ddpg_implementation_overview.md),
[TD3](docs/td3_implementation_overview.md), [SAC](docs/sac_implementation_overview.md), and recurrent PPO —
applied to algorithmic trading across synthetic and real market data (OHLCV and
high-frequency limit order book). The goal is to evaluate whether modern RL
agents can learn profitable, generalizable trading policies under realistic
constraints such as transaction costs, position limits, and non-stationary
market regimes.

The project is built around a scenario-driven experiment framework: each
scenario pairs a dataset, a feature set, a reward function, and an algorithm
config, making runs fully reproducible and comparable. MLflow tracks every
experiment, and all components — environments, trainers, features, rewards —
are modular and independently testable.

## Highlights

- [PPO](docs/ppo_implementation_overview.md), [DDPG](docs/ddpg_implementation_overview.md), [TD3](docs/td3_implementation_overview.md), [SAC](docs/sac_implementation_overview.md), and recurrent PPO trainers for discrete and continuous action spaces
- Scenario-driven YAML configs in `src/configs/scenarios`
- Synthetic data generator (sine wave, upward drift, sampled OHLCV)
- MLflow tracking plus CLI utilities for experiments, checkpoints, and artifacts
- Visualization analytics and reusable logging utilities

## Prerequisites

- Python 3.12 or 3.13
- [uv](https://docs.astral.sh/uv/)

## Installation

<!--pytest.mark.skip-->
```bash
uv sync --extra dev
```

## Rendering the Thesis

`uv sync` only installs Python dependencies. Rendering `thesis/qmd/src/masters-thesis.qmd`
also needs Quarto, a TeX distribution (xelatex), and the Latin Modern Roman
font that the plot theme uses — none of which are Python packages. Tasks for
these are defined with [poethepoet](https://poethepoet.natn.io/) in
`pyproject.toml`:

<!--pytest.mark.skip-->
```bash
uv run poe setup       # one-time: install Quarto, TinyTeX, and thesis fonts
uv run poe thesis-pdf  # render thesis/qmd/src/masters_thesis.pdf
uv run poe thesis-html # render the HTML version
```

`uv run poe setup` runs `brew install --cask quarto`, which needs an
interactive sudo password — run it yourself in a real terminal rather than
through an automated agent. Run `uv run poe --help` to list all tasks.

### Troubleshooting

- **`quarto: command not found` right after `uv run poe setup`.** The
  `sudo` prompt from `brew install --cask quarto` was likely skipped or
  cancelled (e.g. run from a non-interactive shell). Homebrew still records
  the cask as installed even though the installer never ran, so a plain
  retry does nothing (`Warning: Not upgrading quarto, the latest version is
  already installed`). Force it:

  <!--pytest.mark.skip-->
  ```bash
  brew reinstall --cask quarto
  which quarto   # should print a path once it actually succeeded
  uv run poe setup
  ```

- **`findfont: Font family 'Latin Modern Roman' not found`** when
  generating plots. Re-run the font task on its own:

  <!--pytest.mark.skip-->
  ```bash
  uv run poe fonts
  ```

## Reproducing the Thesis Experiments

`uv run thesis-experiments <h1|h2|h3|h4|all>` runs one hypothesis end to end —
guardrails, train, evaluate, the hypothesis-specific report, and export to the
thesis snapshots that `poe thesis-pdf` renders. Each hypothesis trains a fixed
set of scenarios (see `src/masters_thesis/experiments.py`):

| Hypothesis | Tests |
| --- | --- |
| `h1` | Whether TD3 outperforms DDPG, PPO, and a random-policy baseline |
| `h2` | How the observation feature set affects TD3 performance |
| `h3` | Whether the main result is robust to modelling choices (features, reward, transaction costs) |
| `h4` | Whether TD3 learns consistently across independent short trials |

<!--pytest.mark.skip-->
```bash
uv run thesis-experiments h1              # train + evaluate + report + export
uv run thesis-experiments h1 --skip-train # re-evaluate an already-trained agent
uv run thesis-experiments h3 --parallel   # run this hypothesis's scenarios concurrently
uv run thesis-experiments all --dev       # every hypothesis, capped training steps, for a smoke test
```

Run `uv run thesis-experiments --help` (or `<hypothesis> --help`) for the full
option list, including `--config-override` for OmegaConf dotlist overrides.

## Quick Start

Common commands:

| Command | Purpose |
| --- | --- |
| `uv run python src/cli.py scenarios` | List available scenario configs |
| `uv run python src/cli.py data generate --sine-wave --n-periods 8 --samples-per_period 250 --output-file data/raw/synthetic/sine_wave.parquet` | Generate synthetic sine-wave data |
| `uv run python src/cli.py train --scenario sine_wave/ppo_no_trend` | Train a single agent |
| `uv run python src/cli.py train --scenario sine_wave/ppo_no_trend --trials 3` | Run multiple trials |
| `uv run python src/cli.py train --config sine_wave/ppo_no_trend --from-last-checkpoint --additional-steps 5000` | Resume from last checkpoint |
| `uv run python src/cli.py evaluate --scenario <name>` | Evaluate a trained checkpoint: metrics, benchmarks, plots, statistical tests |
| `uv run python src/cli.py validate guardrails --scenario <name>` | Pre-flight sanity checks before a training run (also `--all` for every scenario) |
| `uv run python src/cli.py validate config --scenario <name>` | Validate scenario config and data dependencies |
| `uv run python src/cli.py validate data --scenario <name>` | Validate prepared dataset (NaN, inf, duplicate index, zero-variance features, LOB delta checks) |
| `uv run python src/cli.py feature-research --scenario sine_wave/ppo_no_trend` | Run offline feature scoring |
| `uv run python src/cli.py dashboard` | Launch the MLflow UI |
| `uv run python src/cli.py checkpoints` | List checkpoints; supports `--delete <regex>`, `--delete-all`, `--dry-run` |
| `uv run python src/cli.py experiments` | List MLflow experiments; supports `--delete <regex>`, `--delete-all`, `--dry-run` |
| `uv run python src/cli.py artifacts --experiment <regex>` | List artifacts per run; supports `--delete`, `--run-id`, `--prefix` |

Global options available on all commands:

| Option | Purpose |
| --- | --- |
| `--verbose` / `-v` | Enable debug-level logging |
| `--log-regex <pattern>` | Only show log lines matching the regex |

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

See [docs/data_guide.md](docs/data_guide.md) for detailed instructions on data download and generation.

## Configuration

Each scenario lives in its own directory under `src/configs/scenarios/<group>/<name>/` and contains up to four component files that are merged in order at load time:

| File | Purpose |
| --- | --- |
| `observation.yaml` | Feature pipeline path (`data.feature_config`) and active columns (`env.feature_columns`) |
| `train.yaml` | Data path, environment, network, and training hyperparameters |
| `evaluate.yaml` | Evaluation-only overrides: benchmarks and statistical tests |
| `feature_selection.yaml` | IC-selected feature subset; applied automatically when `data.automated_selection: true` in `train.yaml` |

Reference a scenario by its directory path or `group/name` shorthand:

<!--pytest.mark.skip-->
```bash
# By shorthand (resolves to src/configs/scenarios/sine_wave/ppo_no_trend/)
uv run python src/cli.py train --scenario sine_wave/ppo_no_trend

# Override individual values at run time
uv run python src/cli.py train \
  --scenario sine_wave/ppo_no_trend \
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
│   │   ├── scenarios/       # Experiment configs — one directory per scenario
│   │   │   ├── pooled/
│   │   │   │   └── td3_hft_lob_state_space_pooled_streaming_selected/
│   │   │   │       ├── observation.yaml     # feature pipeline + active columns
│   │   │   │       ├── train.yaml           # training hyperparameters
│   │   │   │       └── evaluate.yaml        # benchmark + stat-test overrides
│   │   │   ├── btc/
│   │   │   ├── pooled/
│   │   │   ├── sine_wave/
│   │   │   └── synthetic/
│   │   └── data/            # Data-source/data-generation configs
│   ├── data_generator/      # Synthetic data generation (sine, trend, drift, mean-reversion)
│   ├── masters_thesis/      # `thesis-experiments` hypothesis-runner CLI
│   ├── logger/              # Shared logging utilities
│   └── trading_rl/          # Core RL package
│       ├── envs/            # Environment builders/wrappers
│       ├── rewards/         # Reward functions
│       ├── trainers/        # PPO, DDPG, TD3, SAC, recurrent PPO trainers
│       └── training.py      # Training loops and helpers
├── data/                    # Raw and synthetic data
├── docs/                    # Experiment and algorithm docs
├── notebooks/               # Research notebooks
├── scripts/                 # Debugging and helper scripts
└── tests/                   # Unit tests
```

## Agent Skills (.claude / .dsh)

Skills for AI coding agents (Claude Code, DeepSeek Harness) live in `.claude/skills/` and `.claude/agents/`, which are the canonical sources. The `.dsh/skills/` directory consumed by DeepSeek Harness is generated from them:

```bash
uv run scripts/sync_dsh_skills.py          # regenerate .dsh/skills
uv run scripts/sync_dsh_skills.py --check  # verify sync (also runs as a pre-commit hook)
```

Edit `.claude/` sources and re-run the sync — never edit `.dsh/skills/` directly. Generic machine-wide skills sync separately from the dotfiles repository into `~/.dsh/skills/`.

## Development

<!--pytest.mark.skip-->
```bash
uv run poe test           # uv run pytest
uv run poe lint           # uv run ruff check .
uv run pre-commit install # one-time: run the same checks locally on every commit
```

`pre-commit` mirrors the checks that gate every commit, including `ruff` and
`ruff-format`. `.pre-commit-config.yaml` pins an exact ruff version, while
`pyproject.toml`'s `ruff>=0.5.0` is only a floor — `uv run ruff` resolves to
whatever is newest — so the two can drift apart. If `uv run ruff check`
passes but `pre-commit` fails on a rule you don't recognize, bump the `rev:`
in `.pre-commit-config.yaml` to match `uv run ruff --version`.

## Docs and References

**Concepts and background**
- [The Big Picture of Reinforcement Learning](docs/big_picture.md)
- [RL Algorithm Comparison: TD3, DDPG, PPO](docs/comparison.md)
- [Feature Selection in DRL Trading (literature review)](docs/feature_selection_in_drl.md)
- [HFT LOB Feature Formulas](docs/hft_features.md)

**Pipeline and architecture**
- [End-to-End Workflow Overview](docs/overview.md)
- [End-to-End Training Workflow](docs/experiment_workflow.md)
- [Training Pipeline Architecture](docs/training_pipeline.md)
- [Feature Pipeline Architecture](docs/feature_pipeline.md)
- [Data Preparation Flow](docs/prepare_data.md)
- [Thesis Artifact Bridge](docs/thesis_artifact_bridge.md) — how experiment metrics/plots reach the rendered thesis
- [The `step_count` Training Metric](docs/max_step_count_metric.md)
- [Data Download and Generation Guide](docs/data_guide.md)
- [MCP Server Workflow](docs/mcp_server.md)

**Algorithm implementations**
- [PPO Implementation Overview](docs/ppo_implementation_overview.md)
- [DDPG Implementation Overview](docs/ddpg_implementation_overview.md)
- [TD3 Implementation Overview](docs/td3_implementation_overview.md)
- [SAC Implementation Overview](docs/sac_implementation_overview.md)

**Package READMEs**
- [Core RL Package Overview](src/trading_rl/README.md)
- [Logging Utilities](src/logger/README.md)
