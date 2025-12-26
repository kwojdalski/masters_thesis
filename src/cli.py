#!/usr/bin/env python3
"""
Refactored command-line interface using command classes.
"""

import os
import re
from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from cli.commands import (
    DashboardCommand,
    DashboardParams,
    DataGenerationParams,
    DataGeneratorCommand,
    ExperimentCommand,
    ExperimentParams,
    SineWaveParams,
    TrainingCommand,
    TrainingParams,
    UpwardDriftParams,
)
from logger import configure_logging

# Ensure matplotlib can cache fonts to a writable directory
if "MPLCONFIGDIR" in os.environ:
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
else:
    mpl_cache_dir = Path(".cache/matplotlib")
    mpl_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_cache_dir.resolve())

# Configure project logging for the CLI component before creating loggers
app = typer.Typer(
    help="CLI tools for trading data science project",
    add_completion=False,
)


def _configure_logging(verbose: bool, log_regex: str | None) -> None:
    """Configure logging based on CLI context."""
    env_level = os.environ.get("LOG_LEVEL", "").upper()
    if env_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        env_level = ""
    if verbose:
        level = "DEBUG"
    elif env_level:
        level = env_level
    else:
        level = "INFO"
    if log_regex:
        os.environ["LOG_REGEX"] = log_regex
    else:
        os.environ.pop("LOG_REGEX", None)
    configure_logging(
        component="cli",
        level=level,
        simplified=not verbose,
        log_regex=log_regex,
    )
    os.environ["LOG_LEVEL"] = level


# Add global options
@app.callback()
def main(
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
    log_regex: str | None = typer.Option(
        None,
        "--log-regex",
        help="Only show log lines that match this regex",
    ),
):
    """Global CLI options."""
    _configure_logging(verbose, log_regex)


console = Console()

# Command instances
data_gen_cmd = DataGeneratorCommand(console)
training_cmd = TrainingCommand(console)
experiment_cmd = ExperimentCommand(console)
dashboard_cmd = DashboardCommand(console, default_tracking_uri="sqlite:///mlflow.db")


def _parse_checkpoint_step(filename: str) -> int | None:
    marker = "_checkpoint_step_"
    if marker not in filename:
        return None
    suffix = filename.split(marker, 1)[1]
    if suffix.endswith(".pt"):
        suffix = suffix[:-3]
    try:
        return int(suffix)
    except ValueError:
        return None


def _confirm_delete(items: list[str], force: bool) -> bool:
    if force:
        return True
    console.print("[yellow]Delete the following items?[/yellow]")
    for item in items:
        console.print(f"  {item}")
    return typer.confirm("Proceed with deletion?", default=False)


@app.command(name="checkpoints")
def checkpoints(
    log_dir: Path = typer.Option(  # noqa: B008
        Path("logs"), "--log-dir", help="Root directory to scan for checkpoints"
    ),
    delete: str | None = typer.Option(
        None, "--delete", help="Delete checkpoints matching regex"
    ),
    delete_all: bool = typer.Option(
        False, "--delete-all", help="Delete all checkpoints"
    ),
    force: bool = typer.Option(False, "--force", help="Delete without confirmation"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be deleted"),
):
    """List checkpoints grouped by experiment with size, mtime, and step."""
    checkpoints: dict[str, list[dict[str, str]]] = {}
    checkpoint_paths: list[Path] = []

    if not log_dir.exists():
        console.print(f"[red]Log directory not found: {log_dir}[/red]")
        raise typer.Exit(1)

    for root, _dirs, files in os.walk(log_dir):
        for name in files:
            if not name.endswith(".pt") or "_checkpoint" not in name:
                continue
            experiment = name.split("_checkpoint", 1)[0]
            path = Path(root) / name
            checkpoint_paths.append(path)
            stat = path.stat()
            size_kb = f"{stat.st_size / 1024:.1f} KB"
            modified = datetime.fromtimestamp(stat.st_mtime).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            step = _parse_checkpoint_step(name)
            checkpoints.setdefault(experiment, []).append(
                {
                    "path": str(path),
                    "size": size_kb,
                    "modified": modified,
                    "step": str(step) if step is not None else "-",
                }
            )

    if not checkpoint_paths:
        console.print("[yellow]No checkpoints found.[/yellow]")
        raise typer.Exit(0)

    if delete_all or delete:
        pattern = re.compile(delete) if delete else None
        targets = [
            p
            for p in checkpoint_paths
            if delete_all or (pattern and pattern.search(p.name))
        ]
        if not targets:
            console.print("[yellow]No checkpoints matched for deletion.[/yellow]")
            raise typer.Exit(0)
        if dry_run:
            console.print("[yellow]Dry run: checkpoints to delete[/yellow]")
            for path in targets:
                console.print(f"  {path}")
            raise typer.Exit(0)
        if not _confirm_delete([str(p) for p in targets], force):
            console.print("[yellow]Deletion cancelled.[/yellow]")
            raise typer.Exit(0)
        for path in targets:
            path.unlink(missing_ok=True)
        console.print(f"[green]Deleted {len(targets)} checkpoints.[/green]")
        raise typer.Exit(0)

    for experiment, items in sorted(checkpoints.items()):
        table = Table(title=f"Experiment: {experiment}")
        table.add_column("Checkpoint")
        table.add_column("Step", justify="right")
        table.add_column("Size", justify="right")
        table.add_column("Modified", justify="right")
        for item in sorted(items, key=lambda x: x["path"]):
            table.add_row(item["path"], item["step"], item["size"], item["modified"])
        console.print(table)


@app.command(name="generate-data")
def generate_data(
    scenario: str | None = typer.Option(
        None,
        "--scenario",
        "-s",
        help="Scenario config name (e.g., 'sine_wave') or path to YAML file defining defaults",
        show_default=False,
    ),
    source_dir: str | None = typer.Option(
        None,
        "--source-dir",
        help="Source directory containing parquet files",
        show_default=False,
    ),
    output_dir: str | None = typer.Option(
        None,
        "--output-dir",
        help="Output directory for synthetic data",
        show_default=False,
    ),
    list_files: bool = typer.Option(
        False, "--list", help="List available source files"
    ),
    source_file: str | None = typer.Option(
        None, "--source-file", help="Source parquet file name"
    ),
    output_file: str | None = typer.Option(
        None, "--output-file", help="Output file name"
    ),
    start_date: str | None = typer.Option(
        None, "--start-date", help="Start date for filtering (YYYY-MM-DD)"
    ),
    end_date: str | None = typer.Option(
        None, "--end-date", help="End date for filtering (YYYY-MM-DD)"
    ),
    sample_size: int | None = typer.Option(
        None, "--sample-size", help="Number of rows to sample randomly"
    ),
    copy: bool = typer.Option(
        False, "--copy", help="Copy source file without modifications"
    ),
    sine_wave: bool = typer.Option(
        False, "--sine-wave", help="Generate sine wave pattern with trend"
    ),
    n_periods: int | None = typer.Option(
        None, "--n-periods", help="Number of sine wave periods", show_default=False
    ),
    samples_per_period: int | None = typer.Option(
        None,
        "--samples-per_period",
        help="Samples per sine wave period",
        show_default=False,
    ),
    base_price: float | None = typer.Option(
        None, "--base-price", help="Base price level", show_default=False
    ),
    amplitude: float | None = typer.Option(
        None, "--amplitude", help="Sine wave amplitude", show_default=False
    ),
    trend_slope: float | None = typer.Option(
        None, "--trend-slope", help="Linear trend slope per step", show_default=False
    ),
    volatility: float | None = typer.Option(
        None, "--volatility", help="Random noise factor", show_default=False
    ),
    upward_drift: bool = typer.Option(
        False, "--upward-drift", help="Generate upward drift pattern"
    ),
    drift_samples: int | None = typer.Option(
        None,
        "--drift-samples",
        help="Number of samples for drift pattern",
        show_default=False,
    ),
    drift_rate: float | None = typer.Option(
        None, "--drift-rate", help="Exponential drift rate per step", show_default=False
    ),
    drift_volatility: float | None = typer.Option(
        None,
        "--drift-volatility",
        help="Volatility factor for drift pattern",
        show_default=False,
    ),
    drift_floor: float | None = typer.Option(
        None,
        "--drift-floor",
        help="Pullback floor multiplier for drift pattern",
        show_default=False,
    ),
):
    """Generate synthetic price data from existing parquet files."""

    # Create parameter objects
    params = DataGenerationParams(
        scenario=scenario,
        source_dir=source_dir,
        output_dir=output_dir,
        source_file=source_file,
        output_file=output_file,
        start_date=start_date,
        end_date=end_date,
        sample_size=sample_size,
        copy=copy,
        list_files=list_files,
    )

    sine_wave_params = SineWaveParams(
        enabled=sine_wave,
        n_periods=n_periods,
        samples_per_period=samples_per_period,
        base_price=base_price,
        amplitude=amplitude,
        trend_slope=trend_slope,
        volatility=volatility,
    )

    upward_drift_params = UpwardDriftParams(
        enabled=upward_drift,
        drift_samples=drift_samples,
        drift_rate=drift_rate,
        drift_volatility=drift_volatility,
        drift_floor=drift_floor,
    )

    # Execute command
    data_gen_cmd.execute(params, sine_wave_params, upward_drift_params, start_date)


@app.command()
def train(
    config_file: Path | None = typer.Option(  # noqa: B008
        None, "--config", "-c", help="Path to custom config file"
    ),
    config_override: list[str] | None = typer.Option(
        None,
        "--config-override",
        "-o",
        help="OmegaConf override in dotlist format (repeatable)",
    ),
    from_checkpoint: Path | None = typer.Option(  # noqa: B008
        None,
        "--from-checkpoint",
        help="Path to checkpoint file to resume training from",
    ),
    from_last_checkpoint: bool = typer.Option(
        False,
        "--from-last-checkpoint",
        help="Resume from the most recent checkpoint for the experiment",
    ),
    mlflow_run_id: str | None = typer.Option(
        None,
        "--mlflow-run-id",
        help="Resume training into an existing MLflow run ID",
    ),
    additional_steps: int | None = typer.Option(
        None,
        "--additional-steps",
        help="Additional steps to train when resuming (requires --checkpoint)",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
    log_regex: str | None = typer.Option(
        None,
        "--log-regex",
        help="Only show log lines that match this regex",
    ),
):
    """Train a single trading agent.

    You can resume training from a checkpoint by providing --from-checkpoint flag:

    Example:
        python src/cli.py train --from-checkpoint logs/td3_tradingenv_btc/td3_tradingenv_btc_checkpoint.pt --additional-steps 50000
    """

    if verbose or log_regex:
        _configure_logging(verbose, log_regex)

    params = TrainingParams(
        config_file=config_file,
        config_overrides=config_override,
        from_checkpoint=from_checkpoint,
        from_last_checkpoint=from_last_checkpoint,
        mlflow_run_id=mlflow_run_id,
        additional_steps=additional_steps,
    )

    training_cmd.execute(params)


@app.command()
def experiment(
    experiment_name: str | None = typer.Option(
        None,
        "--name",
        "-n",
        help="MLflow experiment name (defaults to config's experiment_name)",
    ),
    n_trials: int = typer.Option(5, "--trials", "-t", help="Number of trials to run"),
    dashboard: bool = typer.Option(
        False, "--dashboard", help="Launch MLflow UI after experiments"
    ),
    config_file: Path | None = typer.Option(  # noqa: B008
        None, "--config", "-c", help="Path to custom config file"
    ),
    scenario: str | None = typer.Option(
        None,
        "--scenario",
        "-s",
        help="Scenario name (use 'list-scenarios' to see options)",
    ),
    seed: int | None = typer.Option(
        None,
        "--seed",
        help="Set base seed for reproducible experiments (default: random)",
    ),
    max_steps: int | None = typer.Option(
        None, "--max-steps", help="Maximum training steps for each trial"
    ),
    clear_cache: bool = typer.Option(
        False,
        "--clear-cache",
        help="Clear cached datasets and models before running experiments",
    ),
    no_features: bool = typer.Option(
        False,
        "--no-features",
        help="Skip feature engineering and use only raw OHLCV data",
    ),
    generate_data: bool = typer.Option(
        False,
        "--generate-data",
        help="Regenerate data from scenario configuration before experiments",
    ),
):
    """Run multiple experiments with MLflow tracking.

    Each experiment tracks multiple metrics simultaneously:
    - Actor and value losses over training steps
    - Position changes and trading activity
    - Portfolio values and episode rewards
    - All parameters and configurations
    """

    params = ExperimentParams(
        experiment_name=experiment_name,
        n_trials=n_trials,
        dashboard=dashboard,
        config_file=config_file,
        scenario=scenario,
        seed=seed,
        max_steps=max_steps,
        clear_cache=clear_cache,
        no_features=no_features,
        generate_data=generate_data,
    )

    experiment_cmd.execute(params)


@app.command()
def dashboard(
    port: int = typer.Option(5000, "--port", "-p", help="Port for MLflow UI"),
    host: str = typer.Option("localhost", "--host", help="Host for MLflow UI"),
    tracking_uri: str | None = typer.Option(
        None,
        "--tracking-uri",
        help="MLflow tracking URI (default sqlite:///mlflow.db)",
        show_default=False,
    ),
):
    """Launch MLflow UI for viewing experiments."""

    params = DashboardParams(port=port, host=host, tracking_uri=tracking_uri)
    dashboard_cmd.execute(params)


@app.command(name="experiments")
def experiments(
    tracking_uri: str | None = typer.Option(
        None,
        "--tracking-uri",
        help="MLflow tracking URI (default sqlite:///mlflow.db)",
        show_default=False,
    ),
    delete: str | None = typer.Option(
        None, "--delete", help="Delete experiments matching regex"
    ),
    delete_all: bool = typer.Option(
        False, "--delete-all", help="Delete all experiments"
    ),
    force: bool = typer.Option(False, "--force", help="Delete without confirmation"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be deleted"),
):
    """List available MLflow experiments."""
    if not delete and not delete_all:
        dashboard_cmd.list_experiments(tracking_uri)
        return

    import mlflow

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    experiments_list = mlflow.search_experiments()
    pattern = re.compile(delete) if delete else None
    targets = [
        exp
        for exp in experiments_list
        if delete_all or (pattern and pattern.search(exp.name))
    ]
    if not targets:
        console.print("[yellow]No experiments matched for deletion.[/yellow]")
        raise typer.Exit(0)
    if dry_run:
        console.print("[yellow]Dry run: experiments to delete[/yellow]")
        for exp in targets:
            console.print(f"  {exp.name}")
        raise typer.Exit(0)
    if not _confirm_delete([exp.name for exp in targets], force):
        console.print("[yellow]Deletion cancelled.[/yellow]")
        raise typer.Exit(0)
    for exp in targets:
        mlflow.delete_experiment(exp.experiment_id)
    console.print(f"[green]Deleted {len(targets)} experiments.[/green]")


@app.command(name="scenarios")
def scenarios(
    delete: str | None = typer.Option(
        None, "--delete", help="Delete scenarios matching regex"
    ),
    delete_all: bool = typer.Option(False, "--delete-all", help="Delete all scenarios"),
    force: bool = typer.Option(False, "--force", help="Delete without confirmation"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be deleted"),
):
    """List available scenario configurations."""
    import yaml

    config_dir = Path("src/configs")
    if not config_dir.exists():
        console.print("[red]Config directory not found: src/configs[/red]")
        return

    # Find all YAML config files
    config_files = sorted(config_dir.glob("*.yaml"))

    if not config_files:
        console.print("[yellow]No configuration files found[/yellow]")
        return

    if delete_all or delete:
        pattern = re.compile(delete) if delete else None
        targets = [
            f
            for f in config_files
            if delete_all or (pattern and pattern.search(f.stem))
        ]
        if not targets:
            console.print("[yellow]No scenarios matched for deletion.[/yellow]")
            raise typer.Exit(0)
        if dry_run:
            console.print("[yellow]Dry run: scenarios to delete[/yellow]")
            for path in targets:
                console.print(f"  {path}")
            raise typer.Exit(0)
        if not _confirm_delete([str(f) for f in targets], force):
            console.print("[yellow]Deletion cancelled.[/yellow]")
            raise typer.Exit(0)
        for path in targets:
            path.unlink(missing_ok=True)
        console.print(f"[green]Deleted {len(targets)} scenarios.[/green]")
        raise typer.Exit(0)

    console.print("[bold blue]Available Scenario Configurations:[/bold blue]\n")
    for config_file in config_files:
        try:
            with config_file.open("r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}

            # Extract key information
            experiment_name = config.get("experiment_name", "N/A")
            algorithm = config.get("training", {}).get("algorithm", "N/A")
            data_path = config.get("data", {}).get("data_path", "N/A")
            pattern_type = config.get("data_generator", {}).get("pattern_type", "N/A")

            console.print(f"[green]{config_file.stem}[/green]:")
            console.print(f"  Experiment: {experiment_name}")
            console.print(f"  Algorithm: {algorithm}")
            console.print(f"  Pattern: {pattern_type}")
            console.print(
                f"  Data: {Path(data_path).name if data_path != 'N/A' else 'N/A'}"
            )
            console.print()

        except Exception as e:
            console.print(f"[red]{config_file.stem}[/red]: Error reading file ({e})")
            console.print()


def _list_run_artifacts(client, run_id: str, prefix: str | None = None) -> list:
    artifacts = []
    stack = [prefix or ""]
    while stack:
        path = stack.pop()
        for entry in client.list_artifacts(run_id, path):
            if entry.is_dir:
                stack.append(entry.path)
            else:
                artifacts.append(entry)
    return artifacts


def _safe_search_experiments(
    tracking_uri: str | None = None,
) -> tuple[list[dict[str, str]], bool, Path | None]:
    from urllib.parse import urlparse

    import mlflow
    import yaml

    try:
        experiments_list = mlflow.search_experiments()
        return (
            [
                {"experiment_id": exp.experiment_id, "name": exp.name}
                for exp in experiments_list
            ],
            False,
            None,
        )
    except Exception as exc:  # pragma: no cover - fallback for malformed stores
        console.print(
            f"[yellow]Warning: failed to list experiments via MLflow ({exc}). "
            "Falling back to scanning the file store.[/yellow]"
        )

    uri = tracking_uri or mlflow.get_tracking_uri()
    parsed = urlparse(uri)
    if parsed.scheme in ("", "file"):
        base_path = Path(parsed.path or uri)
    else:
        console.print(
            "[red]Unable to recover experiments from non-file tracking URI.[/red]"
        )
        raise typer.Exit(1)

    mlruns_dir = base_path if base_path.name == "mlruns" else base_path / "mlruns"
    if not mlruns_dir.exists():
        console.print(f"[red]MLflow directory not found: {mlruns_dir}[/red]")
        raise typer.Exit(1)

    experiments = []
    for exp_dir in sorted(mlruns_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        meta_path = exp_dir / "meta.yaml"
        if not meta_path.exists():
            continue
        try:
            meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
            exp_id = meta.get("experiment_id")
            exp_name = meta.get("name")
            if exp_id and exp_name:
                experiments.append({"experiment_id": exp_id, "name": exp_name})
        except Exception:
            continue
    return experiments, True, mlruns_dir


def _safe_list_runs_file_store(
    mlruns_dir: Path, experiment_id: str
) -> list[dict[str, str]]:
    import yaml

    runs = []
    exp_dir = mlruns_dir / experiment_id
    if not exp_dir.exists():
        return runs
    for run_dir in exp_dir.iterdir():
        if not run_dir.is_dir():
            continue
        meta_path = run_dir / "meta.yaml"
        if not meta_path.exists():
            continue
        try:
            meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
            run_id = meta.get("run_id") or meta.get("run_uuid")
            if not run_id:
                continue
            run_name = ""
            tags = meta.get("tags") or []
            for tag in tags:
                if tag.get("key") == "mlflow.runName":
                    run_name = tag.get("value") or ""
                    break
            runs.append(
                {
                    "run_id": run_id,
                    "run_name": run_name,
                    "artifact_uri": meta.get("artifact_uri"),
                }
            )
        except Exception:
            continue
    return runs


def _list_artifacts_file_store(
    artifact_uri: str | None, prefix: str | None
) -> list[dict[str, str | int]]:
    from urllib.parse import urlparse

    if not artifact_uri:
        return []
    parsed = urlparse(artifact_uri)
    if parsed.scheme not in ("", "file"):
        return []
    artifact_root = Path(parsed.path or artifact_uri)
    scan_root = artifact_root / prefix if prefix else artifact_root
    if not scan_root.exists():
        return []
    artifacts = []
    for path in scan_root.rglob("*"):
        if path.is_dir():
            continue
        rel_path = path.relative_to(artifact_root)
        artifacts.append(
            {"path": str(rel_path), "file_size": path.stat().st_size}
        )
    return artifacts


def _delete_artifacts_file_store(
    artifact_uri: str | None, entries: list[dict[str, str | int]]
) -> None:
    from urllib.parse import urlparse

    if not artifact_uri:
        return
    parsed = urlparse(artifact_uri)
    if parsed.scheme not in ("", "file"):
        return
    base_path = Path(parsed.path or artifact_uri)
    for entry in entries:
        path = base_path / entry["path"]
        path.unlink(missing_ok=True)
    for path in sorted(base_path.rglob("*"), reverse=True):
        if path.is_dir():
            try:
                path.rmdir()
            except OSError:
                continue


def _delete_artifact_path(
    client, run_id: str, artifact_path: str, artifact_uri: str | None = None
) -> None:
    import mlflow

    if hasattr(client, "delete_artifacts"):
        client.delete_artifacts(run_id, artifact_path)
        return
    if hasattr(mlflow, "artifacts") and hasattr(mlflow.artifacts, "delete_artifacts"):
        mlflow.artifacts.delete_artifacts(run_id, artifact_path)
        return
    if artifact_uri:
        _delete_artifacts_file_store(artifact_uri, [{"path": artifact_path}])
        return
    console.print(
        f"[yellow]Skipping delete for {run_id}:{artifact_path} (unsupported backend).[/yellow]"
    )


@app.command(name="artifacts")
def artifacts(
    tracking_uri: str | None = typer.Option(
        None,
        "--tracking-uri",
        help="MLflow tracking URI (default sqlite:///mlflow.db)",
        show_default=False,
    ),
    experiment: str | None = typer.Option(
        None, "--experiment", help="Filter experiments by regex"
    ),
    run_id: str | None = typer.Option(
        None, "--run-id", help="List artifacts for a specific run id"
    ),
    prefix: str | None = typer.Option(
        None, "--prefix", help="Only list artifacts under this path prefix"
    ),
    delete: str | None = typer.Option(
        None, "--delete", help="Delete artifacts matching regex"
    ),
    delete_all: bool = typer.Option(
        False, "--delete-all", help="Delete all artifacts for selected runs"
    ),
    force: bool = typer.Option(False, "--force", help="Delete without confirmation"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be deleted"),
    max_runs: int = typer.Option(
        50, "--max-runs", help="Maximum runs to show per experiment"
    ),
):
    """List MLflow artifacts grouped by experiment and run."""
    import mlflow
    from mlflow.tracking import MlflowClient

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()

    def _print_run_artifacts(exp_name: str, run, artifacts_list: list) -> None:
        run_name = run.data.tags.get("mlflow.runName", "") if run else ""
        title = f"Run: {run.info.run_id}"
        if run_name:
            title += f" ({run_name})"
        table = Table(title=f"Experiment: {exp_name} | {title}")
        table.add_column("Artifact")
        table.add_column("Size", justify="right")
        if not artifacts_list:
            console.print(f"[yellow]No artifacts for run {run.info.run_id}[/yellow]")
            return
        for entry in sorted(artifacts_list, key=lambda e: e.path):
            size = f"{entry.file_size / 1024:.1f} KB" if entry.file_size else "-"
            table.add_row(entry.path, size)
        console.print(table)

    if run_id:
        run = client.get_run(run_id)
        experiment_obj = client.get_experiment(run.info.experiment_id)
        exp_name = experiment_obj.name if experiment_obj else run.info.experiment_id
        artifacts_list = _list_run_artifacts(client, run_id, prefix)
        if delete_all or delete:
            pattern = re.compile(delete) if delete else None
            targets = [
                entry
                for entry in artifacts_list
                if delete_all or (pattern and pattern.search(entry.path))
            ]
            if not targets:
                console.print("[yellow]No artifacts matched for deletion.[/yellow]")
                raise typer.Exit(0)
            if dry_run:
                console.print("[yellow]Dry run: artifacts to delete[/yellow]")
                for entry in targets:
                    console.print(f"  {run_id}:{entry.path}")
                raise typer.Exit(0)
            if not _confirm_delete([f"{run_id}:{e.path}" for e in targets], force):
                console.print("[yellow]Deletion cancelled.[/yellow]")
                raise typer.Exit(0)
            for entry in targets:
                _delete_artifact_path(
                    client,
                    run_id,
                    entry.path,
                    getattr(run.info, "artifact_uri", None),
                )
            console.print(f"[green]Deleted {len(targets)} artifacts.[/green]")
            return
        _print_run_artifacts(exp_name, run, artifacts_list)
        return

    experiments_list, file_store_fallback, mlruns_dir = _safe_search_experiments(
        tracking_uri
    )
    pattern = re.compile(experiment) if experiment else None
    targets = [
        exp for exp in experiments_list if not pattern or pattern.search(exp["name"])
    ]
    if not targets:
        console.print("[yellow]No experiments matched.[/yellow]")
        raise typer.Exit(0)

    delete_any = delete_all or delete
    for exp in sorted(targets, key=lambda e: e["name"]):
        delete_targets = []
        if file_store_fallback:
            runs = _safe_list_runs_file_store(mlruns_dir, exp["experiment_id"])
            if not runs:
                console.print(
                    f"[yellow]No runs for experiment {exp['name']}.[/yellow]"
                )
                continue
            for run in runs[:max_runs]:
                artifacts_list = _list_artifacts_file_store(
                    run.get("artifact_uri"), prefix
                )
                if delete_any:
                    pattern = re.compile(delete) if delete else None
                    delete_targets.extend(
                        (run["run_id"], run.get("artifact_uri"), entry)
                        for entry in artifacts_list
                        if delete_all
                        or (pattern and pattern.search(entry["path"]))
                    )
                else:
                    title = f"Run: {run['run_id']}"
                    if run.get("run_name"):
                        title += f" ({run['run_name']})"
                    table = Table(title=f"Experiment: {exp['name']} | {title}")
                    table.add_column("Artifact")
                    table.add_column("Size", justify="right")
                    if not artifacts_list:
                        console.print(
                            f"[yellow]No artifacts for run {run['run_id']}[/yellow]"
                        )
                    else:
                        for entry in sorted(
                            artifacts_list, key=lambda e: e["path"]
                        ):
                            size = (
                                f"{entry['file_size'] / 1024:.1f} KB"
                                if entry["file_size"]
                                else "-"
                            )
                            table.add_row(entry["path"], size)
                        console.print(table)
        else:
            runs = mlflow.search_runs(
                experiment_ids=[exp["experiment_id"]],
                max_results=max_runs,
                order_by=["start_time DESC"],
            )
            if runs.empty:
                console.print(
                    f"[yellow]No runs for experiment {exp['name']}.[/yellow]"
                )
                continue
            for _, row in runs.iterrows():
                run = client.get_run(row["run_id"])
                artifacts_list = _list_run_artifacts(client, run.info.run_id, prefix)
                if delete_any:
                    pattern = re.compile(delete) if delete else None
                    delete_targets.extend(
                        (run.info.run_id, None, entry)
                        for entry in artifacts_list
                        if delete_all or (pattern and pattern.search(entry.path))
                    )
                else:
                    _print_run_artifacts(exp["name"], run, artifacts_list)
        if delete_any:
            if not delete_targets:
                console.print("[yellow]No artifacts matched for deletion.[/yellow]")
                raise typer.Exit(0)
            if dry_run:
                console.print("[yellow]Dry run: artifacts to delete[/yellow]")
                for run_id_val, _artifact_uri, entry in delete_targets:
                    path = entry["path"] if isinstance(entry, dict) else entry.path
                    console.print(f"  {run_id_val}:{path}")
                raise typer.Exit(0)
            if not _confirm_delete(
                [
                    f"{rid}:{(e['path'] if isinstance(e, dict) else e.path)}"
                    for rid, _artifact_uri, e in delete_targets
                ],
                force,
            ):
                console.print("[yellow]Deletion cancelled.[/yellow]")
                raise typer.Exit(0)
            for run_id_val, artifact_uri, entry in delete_targets:
                if not artifact_uri:
                    run_info = client.get_run(run_id_val)
                    artifact_uri = getattr(run_info.info, "artifact_uri", None)
                if isinstance(entry, dict):
                    _delete_artifact_path(
                        client, run_id_val, entry["path"], artifact_uri
                    )
                else:
                    _delete_artifact_path(
                        client, run_id_val, entry.path, artifact_uri
                    )
            console.print(f"[green]Deleted {len(delete_targets)} artifacts.[/green]")


if __name__ == "__main__":
    app()
