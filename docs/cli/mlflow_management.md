# MLflow Management Commands

These commands manage MLflow experiments, runs, checkpoints, and artifacts. Use these to clean up resources, track progress, and organize your experimentation workflow.

## Table of Contents

- [checkpoints](#checkpoints)
- [experiments](#experiments)
- [artifacts](#artifacts)

---

## checkpoints

List checkpoints grouped by experiment with size, modification time, and step number. Supports deletion.

### Usage

```bash
uv run python src/cli.py checkpoints
uv run python src/cli.py checkpoints --log-dir <path>
```

### Options

| Option | Short | Default | Description |
|---|---|---|---|
| `--log-dir` | | `logs` | Root directory to scan for checkpoints |
| `--delete` | | | Delete checkpoints matching regex |
| `--delete-all` | | `false` | Delete all checkpoints |
| `--force` | | `false` | Delete without confirmation |
| `--dry-run` | | `false` | Show what would be deleted |

### Output Columns

| Column | Description |
|---|---|
| experiment | Experiment name (derived from directory) |
| checkpoint | Checkpoint filename |
| size | File size (human-readable) |
| modified | Last modification time |
| step | Training step number (extracted from filename) |

### Examples

```bash
# List all checkpoints
uv run python src/cli.py checkpoints

# Scan custom directory
uv run python src/cli.py checkpoints --log-dir custom_logs

# Dry run deletion
uv run python src/cli.py checkpoints --delete ".*_step_5000.pt$" --dry-run

# Delete old checkpoints (up to step 5000)
uv run python src/cli.py checkpoints --delete ".*_step_[0-4]...\.pt$"

# Delete with confirmation
uv run python src/cli.py checkpoints --delete "sine_wave.*" --delete-all

# Force delete without confirmation
uv run python src/cli.py checkpoints --delete-all --force
```

---

## experiments

List available MLflow experiments or permanently delete them. Supports soft-delete and purging.

### Usage

```bash
uv run python src/cli.py experiments
uv run python src/cli.py experiments --tracking-uri <uri>
```

### Options

| Option | Short | Default | Description |
|---|---|---|---|
| `--tracking-uri` | | `sqlite:///mlflow.db` | MLflow tracking URI |
| `--delete` | | | Permanently delete experiments matching regex |
| `--delete-all` | | `false` | Permanently delete all experiments |
| `--purge` | | `false` | Remove soft-deleted experiments from SQLite DB |
| `--force` | | `false` | Skip confirmation prompt |
| `--dry-run` | | `false` | Show what would be deleted |

### Output Columns

| Column | Description |
|---|---|
| experiment_id | MLflow experiment ID |
| name | Experiment name |
| lifecycle_stage | active or deleted |
| creation_time | When experiment was created |
| runs | Number of runs in experiment |

### Soft Delete vs Purge

- **Delete**: Experiments are marked as deleted (`lifecycle_stage = deleted`) but remain in the database
- **Purge**: Permanently removes soft-deleted experiments from the SQLite database (requires `--purge` with `sqlite:///` tracking URI)

### Examples

```bash
# List all experiments
uv run python src/cli.py experiments

# List experiments from remote server
uv run python src/cli.py experiments --tracking-uri http://localhost:5000

# Dry run deletion
uv run python src/cli.py experiments --delete "sine_wave" --dry-run

# Delete experiments matching pattern
uv run python src/cli.py experiments --delete "pooled.*"

# Purge soft-deleted experiments
uv run python src/cli.py experiments --delete-all --purge --force

# Delete all experiments (dangerous)
uv run python src/cli.py experiments --delete-all
```

---

## artifacts

List MLflow artifacts grouped by experiment and run. Supports filtering and deletion.

### Usage

```bash
uv run python src/cli.py artifacts
uv run python src/cli.py artifacts --tracking-uri <uri>
```

### Options

| Option | Short | Default | Description |
|---|---|---|---|
| `--tracking-uri` | | `sqlite:///mlflow.db` | MLflow tracking URI |
| `--experiment` | | | Filter experiments by regex |
| `--run-id` | | | List artifacts for a specific run id |
| `--prefix` | | | Only list artifacts under this path prefix |
| `--delete` | | | Delete artifacts matching regex |
| `--delete-all` | | `false` | Delete all artifacts for selected runs |
| `--force` | | `false` | Delete without confirmation |
| `--dry-run` | | `false` | Show what would be deleted |
| `--max-runs` | | `50` | Maximum runs to show per experiment |

### Output Structure

Artifacts are displayed grouped by experiment and run:
```
<experiment_name> (id: <exp_id>)
  <run_name> (id: <run_id>)
    <artifact_path> (<size>)
```

### Common Artifacts

| Artifact Type | Path | Description |
|---|---|---|
| Checkpoints | `checkpoints/*.pt` | Model weights |
| Plots | `plots/*.png` | Evaluation plots |
| Results | `results.json` | Evaluation metrics |
| Configs | `config.yaml` | Training configuration |

### Examples

```bash
# List all artifacts
uv run python src/cli.py artifacts

# Filter by experiment
uv run python src/cli.py artifacts --experiment "sine_wave"

# List specific run
uv run python src/cli.py artifacts --run-id 1234567890abcdef

# Filter by path prefix
uv run python src/cli.py artifacts --prefix "checkpoints"

# Delete plots
uv run python src/cli.py artifacts --delete ".*\.png$" --delete-all

# Dry run deletion
uv run python src/cli.py artifacts --delete ".*checkpoint.*" --dry-run

# Use remote MLflow
uv run python src/cli.py artifacts --tracking-uri http://localhost:5000
```

---

## Common Workflows

### Clean Up Old Experiments

```bash
# 1. List experiments to identify old ones
uv run python src/cli.py experiments

# 2. Dry run deletion to verify
uv run python src/cli.py experiments --delete "old_experiment_.*" --dry-run

# 3. Delete confirmed
uv run python src/cli.py experiments --delete "old_experiment_.*" --force

# 4. Purge soft-deleted from DB
uv run python src/cli.py experiments --delete-all --purge --force
```

### Find and Remove Large Artifacts

```bash
# 1. List all artifacts to find large ones
uv run python src/cli.py artifacts

# 2. Delete specific large artifacts
uv run python src/cli.py artifacts --run-id 1234567890abcdef --delete-all

# 3. Or delete by pattern
uv run python src/cli.py artifacts --delete ".*\.pt$" --delete-all
```

### Checkpoint Management

```bash
# 1. List checkpoints with sizes
uv run python src/cli.py checkpoints

# 2. Delete early checkpoints (keep only recent)
uv run python src/cli.py checkpoints --delete ".*_step_[0-4]...\.pt$" --dry-run
uv run python src/cli.py checkpoints --delete ".*_step_[0-4]...\.pt$" --force
```

## Related Commands

- [Inspection & Validation](./inspection_validation.md) - Data inspection commands
- [Workflow Commands](./workflow_commands.md) - Core ML pipeline commands
- [CLI Overview](./overview.md) - CLI reference overview