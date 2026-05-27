# Workflow Commands

These commands form the core ML pipeline: data generation, feature research, training, evaluation, and result aggregation.

## Table of Contents

- [train](#train)
- [evaluate](#evaluate)
- [prepare-data](#prepare-data)
- [data generate](#data-generate)
- [feature-research](#feature-research)
- [dashboard](#dashboard)
- [collect-results](#collect-results)

---

## train

Train trading agents (single run or multiple trials). Supports checkpoint resumption and multi-trial experiments.

### Usage

```bash
# Basic training
uv run python src/cli.py train --scenario <name>

# Multiple trials
uv run python src/cli.py train --scenario <name> --trials 5
```

### Options

| Option | Short | Default | Description |
|---|---|---|---|
| `--trials` | `-t` | `1` | Number of trials to run (1 = single run) |
| `--name` | `-n` | | MLflow experiment name (defaults to config's experiment_name) |
| `--config` | `-c` | | Path to custom config file |
| `--config-override` | `-o` | | OmegaConf override in dotlist format (repeatable) |
| `--clear-cache` | | `false` | Clear cached datasets and models before running |
| `--from-checkpoint` | | | Path to checkpoint file to resume from |
| `--from-last-checkpoint` | | `false` | Resume from the most recent checkpoint for the experiment |
| `--mlflow-run-id` | | | Resume training into an existing MLflow run ID |
| `--additional-steps` | | | Additional steps to train when resuming |
| `--verbose` | `-v` | `false` | Enable verbose logging |
| `--log-regex` | | | Only show log lines matching this regex |
| `--interactive` | `-i` | `false` | Ask setup questions before training starts |

### Checkpoint Resume

Checkpoint options are only supported for single-run training (`--trials 1`).

| Mode | Description |
|---|---|
| `--from-last-checkpoint` | Auto-finds the most recent checkpoint in the experiment directory |
| `--from-checkpoint <path>` | Use a specific checkpoint file |
| `--additional-steps N` | Train for N additional steps from the checkpoint |
| `--mlflow-run-id <id>` | Resume into an existing MLflow run |

### Examples

```bash
# Basic training with scenario
uv run python src/cli.py train --scenario sine_wave/ppo_no_trend

# With custom config (full path or group/name shorthand)
uv run python src/cli.py train --config src/configs/scenarios/sine_wave/ppo_no_trend.yaml
uv run python src/cli.py train --config sine_wave/ppo_no_trend

# Override config parameters
uv run python src/cli.py train -c sine_wave/ppo_no_trend -o training.max_steps=10000

# Custom experiment name
uv run python src/cli.py train -c sine_wave/ppo_no_trend --name my_experiment

# Multiple trials (hyperparameter sweep)
uv run python src/cli.py train -c sine_wave/ppo_no_trend --trials 5

# Resume from last checkpoint
uv run python src/cli.py train -c sine_wave/ppo_no_trend --from-last-checkpoint --additional-steps 5000

# Resume from specific checkpoint
uv run python src/cli.py train --from-checkpoint logs/my_exp/my_exp_checkpoint_step_1000.pt --additional-steps 10000

# Verbose logging
uv run python src/cli.py train --from-last-checkpoint --additional-steps 5000 --verbose

# Interactive setup
uv run python src/cli.py train --interactive
uv run python src/cli.py train -c sine_wave/ppo_no_trend --interactive
```

---

## evaluate

Evaluate a trained policy from a checkpoint without re-running training.

### Usage

```bash
# Evaluate the latest checkpoint on val and test splits (default)
uv run python src/cli.py evaluate --config <scenario>

# Evaluate only one split
uv run python src/cli.py evaluate -c <scenario> --split test
```

### Options

| Option | Short | Default | Description |
|---|---|---|---|
| `--config` | `-c` | | Path or scenario shorthand for the experiment config YAML |
| `--checkpoint` | | | Path to a .pt checkpoint file (auto-discovered if omitted) |
| `--split` | `-s` | `all` | Data split(s) to evaluate: train, val, test, or all |
| `--only` | | | Components to run (repeatable): metrics, benchmarks, plots, stats |
| `--output-dir` | | `./eval_results` | Directory to write results.json and plot PNGs |
| `--config-override` | `-o` | | OmegaConf dotlist override (repeatable) |
| `--tracking-uri` | | `sqlite:///mlflow.db` | MLflow tracking URI |
| `--no-mlflow` | | `false` | Skip MLflow logging (results only to output-dir) |
| `--data-path` | | | Path to an arbitrary raw parquet file to evaluate on |
| `--save-rollout` | | `false` | Save per-step rollout data to <output-dir>/<split>_rollout.csv |

### Splits

- `train` - Evaluate on training data (for in-sample analysis)
- `val` - Evaluate on validation data (for model selection)
- `test` - Evaluate on test data (for final evaluation)
- `all` - Evaluate on all three splits (default)

### Components

| Component | Description |
|---|---|
| `metrics` | Compute Sharpe, total return, max drawdown, win rate, etc. |
| `benchmarks` | Compare vs buy-and-hold, TWAP, VWAP |
| `plots` | Generate wealth curves, action distributions, etc. |
| `stats` | Run statistical significance tests |

### Output Files

```
<output-dir>/
  results.json           # All metrics and statistics
  wealth_curve.png       # Cumulative return over time
  action_dist.png        # Action distribution histogram
  <split>_rollout.csv    # Per-step rollout data (if --save-rollout)
```

### Examples

```bash
# Evaluate the latest checkpoint on val and test splits
uv run python src/cli.py evaluate --config sine_wave/ppo_no_trend

# Evaluate only one split
uv run python src/cli.py evaluate -c sine_wave/ppo_no_trend --split test

# Only compute metrics and plots, skip benchmarks and stats
uv run python src/cli.py evaluate -c sine_wave/ppo_no_trend --only metrics --only plots

# Use a specific checkpoint
uv run python src/cli.py evaluate -c sine_wave/ppo_no_trend \
    --checkpoint logs/my_exp/my_exp_checkpoint_step_5000.pt

# Skip MLflow, write results locally only
uv run python src/cli.py evaluate -c sine_wave/ppo_no_trend --no-mlflow

# Use a remote MLflow server
uv run python src/cli.py evaluate -c sine_wave/ppo_no_trend \
    --tracking-uri http://localhost:5000

# Evaluate on an arbitrary parquet file (feature pipeline applied automatically)
uv run python src/cli.py evaluate -c pooled/td3_hft_lob_state_space_pooled_streaming_selected \
    --checkpoint logs/my_exp/checkpoint.pt \
    --data-path data/raw/stocks/daily/AAPL/AAPL_2026-03-10_raw_mbp-10_us_hours.parquet \
    --no-mlflow

# Save rollout for offline analysis
uv run python src/cli.py evaluate -c sine_wave/ppo_no_trend --save-rollout
```

---

## prepare-data

Materialise features and populate the data cache without training. Run this once after changing feature configs so subsequent feature-research and train runs hit the cache immediately.

### Usage

```bash
uv run python src/cli.py prepare-data --scenario <name>
```

### Options

| Option | Short | Description |
|---|---|---|
| `--scenario` | `-s` | Scenario name or path under `src/configs/scenarios` |
| `--config` | `-c` | Path to experiment config YAML |
| `--config-override` | `-o` | OmegaConf dotlist override (repeatable) |

### Pipeline Steps

1. **Load raw data** - Read parquet files from `data_paths`
2. **Chronological split** - Split per symbol into train/val/test
3. **Feature pipeline fit** - Fit feature transformations on train data only
4. **Feature transform** - Transform all splits
5. **Cache write** - Write to parquet cache and memmap files

### Cache Locations

```
data/prepared/        # Parquet cache
data/memmap/          # Memmap files per symbol
  {symbol}_train_data.npy
  {symbol}_val_data.npy
  {symbol}_test_data.npy
```

### Examples

```bash
# Prepare data for a scenario
uv run python src/cli.py prepare-data -s sine_wave/ppo_no_trend

# Use specific config file
uv run python src/cli.py prepare-data -c src/configs/scenarios/pooled/td3_hft_lob.yaml

# Override config parameters
uv run python src/cli.py prepare-data -s sine_wave/ppo_no_trend -o data.train_size=50000
```

---

## data generate

Generate synthetic price data from existing parquet files. Supports sine wave patterns and upward drift patterns.

### Usage

```bash
uv run python src/cli.py data generate [options]
```

### Main Options

| Option | Short | Description |
|---|---|---|
| `--scenario` | `-s` | Scenario config name for default parameters |
| `--source-dir` | | Source directory containing parquet files |
| `--output-dir` | | Output directory for synthetic data |
| `--list` | | List available source files |
| `--source-file` | | Source parquet file name |
| `--output-file` | | Output file name |
| `--start-date` | | Start date for filtering (YYYY-MM-DD) |
| `--end-date` | | End date for filtering (YYYY-MM-DD) |
| `--sample-size` | | Number of rows to sample randomly |
| `--copy` | | Copy source file without modifications |
| `--sine-wave` | | Generate sine wave pattern with trend |
| `--upward-drift` | | Generate upward drift pattern |

### Sine Wave Options

| Option | Description |
|---|---|
| `--n-periods` | Number of sine wave periods |
| `--samples-per-period` | Samples per sine wave period |
| `--base-price` | Base price level |
| `--amplitude` | Sine wave amplitude |
| `--trend-slope` | Linear trend slope per step |
| `--volatility` | Random noise factor |

### Upward Drift Options

| Option | Description |
|---|---|
| `--drift-samples` | Number of samples for drift pattern |
| `--drift-rate` | Exponential drift rate per step |
| `--drift-volatility` | Volatility factor for drift pattern |
| `--drift-floor` | Pullback floor multiplier for drift pattern |

### Examples

```bash
# Use scenario for default parameters
uv run python src/cli.py data generate --scenario sine_wave

# Generate sine wave pattern
uv run python src/cli.py data generate --sine-wave \
    --n-periods 10 \
    --samples-per-period 100 \
    --base-price 100 \
    --amplitude 10

# Generate upward drift pattern
uv run python src/cli.py data generate --upward-drift \
    --drift-samples 10000 \
    --drift-rate 0.0001

# Sample from existing file
uv run python src/cli.py data generate \
    --source-file AAPL_2024-01-01_raw_mbp-10.parquet \
    --sample-size 1000

# Copy file without modifications
uv run python src/cli.py data generate --copy \
    --source-file original.parquet \
    --output-file copy.parquet

# List available files
uv run python src/cli.py data generate --list
```

---

## feature-research

Run offline feature scoring and shortlist generation. Computes information coefficient (IC) and information coefficient information ratio (ICIR) per feature across multiple horizons.

### Usage

```bash
uv run python src/cli.py feature-research --config <file>
```

### Options

| Option | Description |
|---|---|
| `--config` | Path to feature research config YAML |
| `--experiment-config` | Path to experiment scenario YAML to derive research settings from |
| `--scenario` | Scenario config name or path under `src/configs/scenarios` |
| `--config-override` | OmegaConf override in dotlist format |

### Research Process

1. **Split data** - Chronological split for IC scoring
2. **Fit pipeline** - Feature pipeline fit on train, transform both splits
3. **Compute IC/ICIR** - Rolling Spearman rank correlation vs Sharpe-proxy target
4. **Aggregate** - Mean ICIR ranking across symbols
5. **Select** - Greedy conditional IC selection with linear residualisation for redundancy
6. **Export** - Write `selected_features.yaml` reduced feature config

### Output

```
reports/feature_research/<scenario>/
  feature_scores.json     # IC/ICIR per feature per horizon
  feature_rankings.json   # Aggregate rankings
  selected_features.yaml  # Reduced feature config
```

### Examples

```bash
# Use feature research config
uv run python src/cli.py feature-research \
    --config src/configs/feature_research/pooled_hft_lob.yaml

# Derive from experiment scenario
uv run python src/cli.py feature-research \
    --scenario pooled/td3_hft_lob_state_space_pooled_streaming_selected
```

---

## dashboard

Launch MLflow UI for viewing experiments, runs, metrics, and artifacts.

### Usage

```bash
uv run python src/cli.py dashboard
```

### Options

| Option | Short | Default | Description |
|---|---|---|---|
| `--port` | `-p` | `5000` | Port for MLflow UI |
| `--host` | | `localhost` | Host for MLflow UI |
| `--tracking-uri` | | | MLflow tracking URI (default sqlite:///mlflow.db) |

### Examples

```bash
# Default MLflow UI
uv run python src/cli.py dashboard

# Custom port
uv run python src/cli.py dashboard --port 8080

# Remote MLflow server
uv run python src/cli.py dashboard --tracking-uri http://localhost:5000

# Expose to network
uv run python src/cli.py dashboard --host 0.0.0.0 --port 5000
```

### UI Features

- Experiment comparison
- Run metrics and parameters
- Artifact visualization
- Parallel coordinates plots
- Hyperparameter optimization tracking

---

## collect-results

Merge per-algorithm evaluation results into a single thesis results directory.

### Usage

```bash
uv run python src/cli.py collect-results \
    --algorithm TD3 --dir ./eval_results/td3 \
    --algorithm DDPG --dir ./eval_results/ddpg \
    --algorithm PPO --dir ./eval_results/ppo \
    --output-dir masters_thesis_results/
```

### Options

| Option | Short | Default | Description |
|---|---|---|---|
| `--algorithm` | `-a` | | Algorithm name (repeat for each) |
| `--dir` | `-d` | | Path to eval_results directory for the algorithm |
| `--output-dir` | `-o` | `masters_thesis_results` | Destination directory for aggregated results |
| `--overwrite` | | `false` | Overwrite existing output |

### Output Structure

```
<output-dir>/
  <algorithm_1>/
    results.json
    plots/
      wealth_curve.png
      action_dist.png
  <algorithm_2>/
    ...
  aggregated_results.json
```

### Examples

```bash
# Collect results from multiple algorithms
uv run python src/cli.py collect-results \
    -a TD3 -d ./eval_results/td3 \
    -a DDPG -d ./eval_results/ddpg \
    -a PPO -d ./eval_results/ppo

# Custom output directory
uv run python src/cli.py collect-results \
    -a TD3 -d ./eval_results/td3 \
    -o my_thesis_results/

# Overwrite existing
uv run python src/cli.py collect-results \
    -a TD3 -d ./eval_results/td3 \
    --overwrite
```

## Related Commands

- [Inspection & Validation](./inspection_validation.md) - Data inspection commands
- [MLflow Management](./mlflow_management.md) - Experiment state management
- [CLI Overview](./overview.md) - CLI reference overview