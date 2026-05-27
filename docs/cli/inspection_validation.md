# Inspection & Validation Commands

These commands help you inspect datasets, list configurations, and validate data quality before training.

## Table of Contents

- [peek dataset](#peek-dataset)
- [peek configs](#peek-configs)
- [validate config](#validate-config)
- [validate data](#validate-data)
- [scenarios](#scenarios)

---

## peek dataset

Show a summary of the prepared dataset for a scenario. Displays split sizes, date ranges, per-feature statistics, correlations, and raw file inventory.

### Usage

```bash
uv run python src/cli.py peek dataset --scenario <name>
uv run python src/cli.py peek dataset --config <path>
```

### Options

| Option | Short | Default | Description |
|---|---|---|---|
| `--scenario` | `-s` | | Scenario name or path under `src/configs/scenarios` |
| `--config` | `-c` | | Path to experiment config YAML |
| `--config-override` | `-o` | | OmegaConf dotlist override (repeatable) |
| `--top` | `-n` | `20` | Max feature rows to show in stats table |
| `--skip` | `0` | | Skip first N rows before computing feature stats (excludes indicator warm-up) |
| `--corr` | | `false` | Show feature-reward correlation table (Pearson + Spearman) |
| `--export` | | `false` | Export tables as CSV files to `reports/peek/<scenario>/` |

### Output Sections

1. **Splits Table**: train/val/test sizes, column counts, timestamps, time deltas
2. **Feature Statistics**: mean, std, min, max, nulls per feature (with selection status)
3. **Price Log-Returns**: distribution stats for the price series (always-long proxy)
4. **Feature Correlations** (optional): Pearson and Spearman correlation vs log-returns
5. **Raw File Inventory**: events and trades per symbol per split, time delta stats

### Examples

```bash
# Basic peek at a scenario
uv run python src/cli.py peek dataset -s pooled/td3_hft_lob_state_space_pooled_streaming_selected

# Show all features
uv run python src/cli.py peek dataset -s td3_hft_lob --top 100

# Show correlations
uv run python src/cli.py peek dataset -s sine_wave/ppo_no_trend --corr

# Export to CSV
uv run python src/cli.py peek dataset -s sine_wave/ppo_no_trend --export

# Skip warmup rows manually
uv run python src/cli.py peek dataset -s td3_hft_lob --skip 100

# Override config parameters
uv run python src/cli.py peek dataset -s sine_wave/ppo_no_trend -o data.train_size=10000
```

### Warmup Detection

The command automatically detects warmup requirements from the feature config (maximum `window`, `period`, `rolling_window` values). Use `--skip` to override.

### Exported Files (with `--export`)

```
reports/peek/<scenario>/
  splits.json
  raw_file_inventory.json
  feature_stats.csv
  log_return_stats.csv
  correlations.csv
```

---

## peek configs

List config YAML files sorted by most recently modified. Shows algorithm, reward type, and other key fields.

### Usage

```bash
uv run python src/cli.py peek configs
```

### Options

| Option | Short | Default | Description |
|---|---|---|---|
| `--top` | `-n` | `20` | Number of configs to show |
| `--dir` | `-d` | `src/configs` | Directory to search |
| `--filter` | `-f` | | Filter by substring in path |

### Output Columns

| Column | Description |
|---|---|
| modified | Last modification time |
| scenario | Scenario label (stripped of leading `scenarios/`) |
| algorithm | Training algorithm (PPO, DDPG, TD3) |
| reward | Reward type from env config |
| backend | Environment backend (torchrl, gymnasium) |

### Examples

```bash
# Show recent configs
uv run python src/cli.py peek configs

# Filter to TD3 scenarios
uv run python src/cli.py peek configs --filter td3

# Show more results
uv run python src/cli.py peek configs --top 50

# Search specific directory
uv run python src/cli.py peek configs --dir src/configs/scenarios/pooled
```

---

## validate config

Validate experiment config, data dependencies, and feature wiring. Checks that:
- Config file loads correctly
- Data paths exist
- Feature config is valid
- Required parameters are present

### Usage

```bash
uv run python src/cli.py validate config --scenario <name>
uv run python src/cli.py validate config --config <path>
```

### Options

| Option | Short | Description |
|---|---|---|
| `--scenario` | `-s` | Scenario name or path to scenario file |
| `--config` | `-c` | Path to config file |
| `--config-override` | `-o` | OmegaConf override in dotlist format |

### Examples

```bash
# Validate a scenario
uv run python src/cli.py validate config -s sine_wave/ppo_no_trend

# Validate a specific config file
uv run python src/cli.py validate config -c src/configs/scenarios/pooled/td3_hft_lob.yaml
```

---

## validate data

Validate the prepared dataset for a scenario using DataValidator. Checks data quality issues including NaNs, infinities, duplicates, zero variance, LOB deltas, temporal ordering, and data leakage.

### Usage

```bash
uv run python src/cli.py validate data --scenario <name>
uv run python src/cli.py validate data --config <path>
```

### Options

| Option | Short | Default | Description |
|---|---|---|---|
| `--scenario` | `-s` | | Scenario name or path |
| `--config` | `-c` | | Path to config file |
| `--config-override` | `-o` | | OmegaConf override |
| `--no-nan` | | `false` | Skip NaN check |
| `--no-inf` | | `false` | Skip infinity check |
| `--no-duplicates` | | `false` | Skip duplicate index check |
| `--no-zero-variance` | | `false` | Skip zero-variance feature check |
| `--no-lob-deltas` | | `false` | Skip LOB delta check |
| `--no-temporal-order` | | `false` | Skip temporal ordering check |
| `--no-overlap` | | `false` | Skip index overlap / data-leakage check |
| `--no-sizes` | | `false` | Skip split-size vs config check |
| `--lob-levels` | | `5` | Number of LOB levels to check for deltas |
| `--verbose` | `-v` | `false` | Show description for each check |
| `--transpose` | `-t` | `false` | Transpose data glimpse table (shows all columns) |

### Checks Performed

| Check | Description |
|---|---|
| NaN check | Ensures no missing values in feature columns |
| Infinity check | Ensures no infinite values |
| Duplicates check | Ensures no duplicate timestamps in index |
| Zero variance | Flags features with zero or near-zero variance |
| LOB deltas | Validates LOB price/size levels change correctly |
| Temporal order | Ensures timestamps are monotonically increasing |
| Overlap | Ensures no data leakage between train/val/test splits |
| Sizes | Validates split sizes match config expectations |

### Examples

```bash
# Full validation
uv run python src/cli.py validate data -s pooled/td3_hft_lob

# Skip specific checks
uv run python src/cli.py validate data -s sine_wave/ppo_no_trend --no-lob-deltas --no-overlap

# Verbose mode with descriptions
uv run python src/cli.py validate data -s td3_hft_lob --verbose

# Show all columns in glimpse
uv run python src/cli.py validate data -s sine_wave/ppo_no_trend --transpose
```

---

## scenarios

List available scenario configurations. Supports deletion with confirmation.

### Usage

```bash
uv run python src/cli.py scenarios
```

### Options

| Option | Short | Default | Description |
|---|---|---|---|
| `--delete` | | | Delete scenarios matching regex |
| `--delete-all` | | `false` | Delete all scenarios |
| `--force` | | `false` | Delete without confirmation |
| `--dry-run` | | `false` | Show what would be deleted |

### Examples

```bash
# List all scenarios
uv run python src/cli.py scenarios

# Dry run deletion
uv run python src/cli.py scenarios --delete "sine_wave" --dry-run

# Delete with confirmation
uv run python src/cli.py scenarios --delete "sine_wave"

# Force delete without confirmation
uv run python src/cli.py scenarios --delete-all --force
```

## Related Commands

- [Workflow Commands](./workflow_commands.md) - Core ML pipeline commands
- [MLflow Management](./mlflow_management.md) - Experiment state management
- [CLI Overview](./overview.md) - CLI reference overview