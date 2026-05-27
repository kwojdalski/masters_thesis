# CLI Reference

This section documents all command-line interface (CLI) tools for the trading RL project. The CLI is built with Typer and uses Rich for formatted output.

## Quick Start

```bash
# Run any command with uv
uv run python src/cli.py <command>

# Get help
uv run python src/cli.py --help
uv run python src/cli.py <command> --help
uv run python src/cli.py <subcommand> --help

# Enable verbose logging
uv run python src/cli.py <command> --verbose

# Override config parameters (OmegaConf dotlist)
uv run python src/cli.py train -c my_scenario -o training.max_steps=50000 -o env.batch_size=64
```

## Command Categories

| Category | Commands | Purpose |
|---|---|---|
| **Workflow** | `train`, `evaluate`, `prepare-data`, `data generate`, `feature-research`, `dashboard`, `collect-results` | Core ML workflow: data, training, evaluation, experimentation |
| **Inspection & Validation** | `peek dataset`, `peek configs`, `validate config`, `validate data`, `validate guardrails`, `scenarios` | Inspect datasets, configs, run guardrails, and validate data quality |
| **MLflow Management** | `checkpoints`, `experiments`, `artifacts` | Manage MLflow experiments, runs, checkpoints, and artifacts |

## Category Links

- [Workflow Commands](./workflow_commands.md) - Core ML pipeline commands
- [Inspection & Validation](./inspection_validation.md) - Data inspection and validation
- [MLflow Management](./mlflow_management.md) - Experiment state management

## Global Options

| Option | Short | Description |
|---|---|---|
| `--verbose` | `-v` | Enable DEBUG level logging |
| `--log-regex <pattern>` | | Only show log lines matching this regex |

## Scenario Shorthand

Many commands accept scenarios using shorthand syntax:
```bash
# Full path
--config src/configs/scenarios/pooled/td3_hft_lob.yaml

# Scenario shorthand (looks in src/configs/scenarios/)
--scenario pooled/td3_hft_lob
--scenario td3_hft_lob  # also works if unique

# Config file without extension
--config pooled/td3_hft_lob
```

## Config Overrides

Use `--config-override` (or `-o`) with OmegaConf dotlist syntax:
```bash
# Single override
-o training.max_steps=50000

# Multiple overrides (repeatable)
-o training.max_steps=50000 -o env.batch_size=64 -o training.learning_rate=1e-4
```

## Related Documentation

- [End-to-End Workflow Overview](../overview.md) - Full pipeline diagrams
- [Experiment Workflow](../experiment_workflow.md) - Detailed training workflow
- [Data Guide](../data_guide.md) - Data acquisition and generation