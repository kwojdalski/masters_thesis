#!/usr/bin/env python3
"""
Refactored command-line interface using command classes.
"""

import os
from pathlib import Path

import typer
from rich.console import Console

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


def _configure_logging(verbose: bool) -> None:
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
    configure_logging(component="cli", level=level, simplified=not verbose)
    os.environ["LOG_LEVEL"] = level


# Add global options
@app.callback()
def main(
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
):
    """Global CLI options."""
    _configure_logging(verbose)


console = Console()

# Command instances
data_gen_cmd = DataGeneratorCommand(console)
training_cmd = TrainingCommand(console)
experiment_cmd = ExperimentCommand(console)
dashboard_cmd = DashboardCommand(console, default_tracking_uri="sqlite:///mlflow.db")


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
    experiment_name: str | None = typer.Option(
        None, "--name", "-n", help="Experiment name (default: auto-generated)"
    ),
    config_file: Path | None = typer.Option(  # noqa: B008
        None, "--config", "-c", help="Path to custom config file"
    ),
    seed: int | None = typer.Option(
        None,
        "--seed",
        "-s",
        help="Set specific seed for reproducibility (default: random)",
    ),
    max_steps: int | None = typer.Option(
        None, "--max-steps", help="Maximum training steps"
    ),
    init_rand_steps: int | None = typer.Option(
        None, "--init-rand-steps", help="Initial random exploration steps"
    ),
    actor_lr: float | None = typer.Option(
        None, "--actor-lr", help="Actor learning rate"
    ),
    value_lr: float | None = typer.Option(
        None, "--value-lr", help="Value network learning rate"
    ),
    buffer_size: int | None = typer.Option(
        None, "--buffer-size", help="Replay buffer size"
    ),
    save_plots: bool = typer.Option(
        False, "--save-plots", help="Save training plots to disk"
    ),
    log_dir: Path | None = typer.Option(None, "--log-dir", help="Logging directory"),  # noqa: B008
    checkpoint: Path | None = typer.Option(  # noqa: B008
        None,
        "--checkpoint",
        "--resume",
        help="Path to checkpoint file to resume training from",
    ),
    additional_steps: int | None = typer.Option(
        None,
        "--additional-steps",
        help="Additional steps to train when resuming (requires --checkpoint)",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
):
    """Train a single trading agent.

    You can resume training from a checkpoint by providing --checkpoint flag:

    Example:
        python src/cli.py train --checkpoint logs/td3_tradingenv_btc/td3_tradingenv_btc_checkpoint.pt --additional-steps 50000
    """

    if verbose:
        _configure_logging(True)

    params = TrainingParams(
        experiment_name=experiment_name,
        config_file=config_file,
        seed=seed,
        max_steps=max_steps,
        init_rand_steps=init_rand_steps,
        actor_lr=actor_lr,
        value_lr=value_lr,
        buffer_size=buffer_size,
        save_plots=save_plots,
        log_dir=log_dir,
        checkpoint_path=checkpoint,
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


@app.command()
def list_experiments(
    tracking_uri: str | None = typer.Option(
        None,
        "--tracking-uri",
        help="MLflow tracking URI (default sqlite:///mlflow.db)",
        show_default=False,
    ),
):
    """List available MLflow experiments."""
    dashboard_cmd.list_experiments(tracking_uri)


@app.command()
def list_scenarios():
    """List available scenario configurations."""
    import yaml

    config_dir = Path("src/configs")
    if not config_dir.exists():
        console.print("[red]Config directory not found: src/configs[/red]")
        return

    console.print("[bold blue]Available Scenario Configurations:[/bold blue]\n")

    # Find all YAML config files
    config_files = sorted(config_dir.glob("*.yaml"))

    if not config_files:
        console.print("[yellow]No configuration files found[/yellow]")
        return

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


if __name__ == "__main__":
    app()
