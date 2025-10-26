#!/usr/bin/env python3
"""
Command-line interface for data generation and trading agent training.
"""

import logging
import shutil
import subprocess
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from data_generator import PriceDataGenerator

# Create the main typer app
app = typer.Typer(help="CLI tools for trading data science project")
console = Console()


@app.command(name="generate-data")
def generate_data(
    source_dir: str = typer.Option(
        "data/raw/binance",
        "--source-dir",
        help="Source directory containing parquet files",
    ),
    output_dir: str = typer.Option(
        "data/raw/synthetic", "--output-dir", help="Output directory for synthetic data"
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
    n_periods: int = typer.Option(3, "--n-periods", help="Number of sine wave periods"),
    samples_per_period: int = typer.Option(
        120, "--samples-per-period", help="Samples per sine wave period"
    ),
    base_price: float = typer.Option(50000.0, "--base-price", help="Base price level"),
    amplitude: float = typer.Option(5000.0, "--amplitude", help="Sine wave amplitude"),
    trend_slope: float = typer.Option(
        50.0, "--trend-slope", help="Linear trend slope per step"
    ),
    volatility: float = typer.Option(0.02, "--volatility", help="Random noise factor"),
    upward_drift: bool = typer.Option(
        False, "--upward-drift", help="Generate upward drift pattern"
    ),
    drift_samples: int = typer.Option(
        500, "--drift-samples", help="Number of samples for drift pattern"
    ),
    drift_rate: float = typer.Option(
        0.0015, "--drift-rate", help="Exponential drift rate per step"
    ),
    drift_volatility: float = typer.Option(
        0.0005, "--drift-volatility", help="Volatility factor for drift pattern"
    ),
    drift_floor: float = typer.Option(
        0.995, "--drift-floor", help="Pullback floor multiplier for drift pattern"
    ),
):
    """Generate synthetic price data from existing parquet files."""
    generator = PriceDataGenerator(source_dir=source_dir, output_dir=output_dir)

    # List available files
    if list_files:
        logger = logging.getLogger(__name__)
        logger.info("Available source files:")
        source_files = generator.list_source_files()
        if not source_files:
            logger.warning("  No parquet files found in source directory")
        else:
            for f in source_files:
                logger.info(f"  - {f}")
        return

    # Generate sine wave pattern
    if sine_wave:
        output_file_name = output_file or "sine_wave_pattern.parquet"
        try:
            df = generator.generate_sine_wave_pattern(
                output_file=output_file_name,
                n_periods=n_periods,
                samples_per_period=samples_per_period,
                base_price=base_price,
                amplitude=amplitude,
                trend_slope=trend_slope,
                volatility=volatility,
                start_date=start_date or "2024-01-01",
            )
            logger = logging.getLogger(__name__)
            logger.info(f"Successfully generated sine wave pattern with {len(df)} rows")
            return
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error generating sine wave pattern: {e}")
            raise typer.Exit(1) from e

    if upward_drift:
        output_file_name = output_file or "upward_drift_pattern.parquet"
        try:
            df = generator.generate_upward_drift_pattern(
                output_file=output_file_name,
                n_samples=drift_samples,
                base_price=base_price,
                drift_rate=drift_rate,
                volatility=drift_volatility,
                pullback_floor=drift_floor,
                start_date=start_date or "2024-01-01",
            )
            logger = logging.getLogger(__name__)
            logger.info(
                f"Successfully generated upward drift pattern with {len(df)} rows"
            )
            return
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error generating upward drift pattern: {e}")
            raise typer.Exit(1) from e

    # Require source file for other operations
    if not source_file:
        logger = logging.getLogger(__name__)
        logger.error(
            "Error: --source-file is required (or use --list to see available files, or --sine-wave for pattern generation)"
        )
        raise typer.Exit(1)

    # Copy operation
    if copy:
        generator.copy_data(source_file, output_file)
        return

    # Generate synthetic data
    try:
        df = generator.generate_synthetic_sample(
            source_file=source_file,
            output_file=output_file,
            start_date=start_date,
            end_date=end_date,
            sample_size=sample_size,
        )
        logger = logging.getLogger(__name__)
        logger.info(f"Successfully generated synthetic data with {len(df)} rows")
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error generating data: {e}")
        raise typer.Exit(1) from e


# Add the other trading-related commands to the main app


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
):
    """Train a single trading agent."""
    from trading_rl import ExperimentConfig, run_single_experiment

    console.print("[bold blue]Starting Trading Agent Training[/bold blue]")

    # Load base configuration
    if config_file:
        config = ExperimentConfig.from_yaml(config_file)
        console.print(f"[blue]Loaded config from: {config_file}[/blue]")
    else:
        config = ExperimentConfig()

    # Handle seed: random by default, specific if provided
    if seed is not None:
        config.seed = seed
        console.print(f"[blue]Using specified seed: {seed}[/blue]")
    else:
        import random

        generated_seed = random.randint(1, 100000)  # noqa: S311
        config.seed = generated_seed
        console.print(f"[yellow]Using random seed: {generated_seed}[/yellow]")

    # Override with CLI parameters
    if experiment_name:
        config.experiment_name = experiment_name
    if max_steps is not None:
        config.training.max_training_steps = max_steps
    if actor_lr is not None:
        config.training.actor_lr = actor_lr
    if value_lr is not None:
        config.training.value_lr = value_lr
    if buffer_size is not None:
        config.training.buffer_size = buffer_size
    if log_dir is not None:
        config.logging.log_dir = str(log_dir)

    console.print(f"Experiment: [green]{config.experiment_name}[/green]")
    console.print(f"Seed: [green]{config.seed}[/green]")
    console.print(f"Max steps: [green]{config.training.max_training_steps}[/green]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Training agent...", total=None)

        try:
            result = run_single_experiment(custom_config=config)
            progress.update(task, description="Training complete!")

            console.print("\n[bold green]Training completed successfully![/bold green]")
            console.print(
                f"Final reward: [green]{result['final_metrics']['final_reward']:.4f}[/green]"
            )
            console.print(
                f"Training steps: [green]{result['final_metrics']['training_steps']}[/green]"
            )

            if save_plots:
                plots_dir = Path(config.logging.log_dir) / "plots"
                plots_dir.mkdir(exist_ok=True, parents=True)

                console.print(f"\nSaving plots to: [blue]{plots_dir}[/blue]")
                result["plots"]["loss"].save(
                    plots_dir / f"{config.experiment_name}_losses.png"
                )
                result["plots"]["reward"].save(
                    plots_dir / f"{config.experiment_name}_rewards.png"
                )
                result["plots"]["action"].save(
                    plots_dir / f"{config.experiment_name}_actions.png"
                )
                console.print("[green]Plots saved successfully![/green]")

        except Exception as e:
            progress.update(task, description="Training failed!")
            console.print(f"\n[bold red]Training failed: {e}[/bold red]")
            raise typer.Exit(1) from e


@app.command()
def experiment(
    experiment_name: str = typer.Option(
        "trading_rl_experiment", "--name", "-n", help="MLflow experiment name"
    ),
    n_trials: int = typer.Option(5, "--trials", "-t", help="Number of trials to run"),
    dashboard: bool = typer.Option(
        False, "--dashboard", help="Launch MLflow UI after experiments"
    ),
    config_file: Path | None = typer.Option(  # noqa: B008
        None, "--config", "-c", help="Path to custom config file"
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
):
    """Run multiple experiments with MLflow tracking.

    Each experiment tracks multiple metrics simultaneously:
    - Actor and value losses over training steps
    - Position changes and trading activity
    - Portfolio values and episode rewards
    - All parameters and configurations
    """
    from trading_rl import ExperimentConfig, run_multiple_experiments

    if clear_cache:
        console.print("[blue]Clearing caches before experiments...[/blue]")
        try:
            from trading_rl.cache_utils import (
                clear_data_cache,
                clear_model_cache,
                clear_training_cache,
            )
        except ImportError:
            console.print(
                "[yellow]Cache utilities unavailable; skipping cache clear.[/yellow]"
            )
        else:
            clear_data_cache()
            clear_model_cache()
            clear_training_cache()
            console.print("[green]Caches cleared.[/green]")

    # Create config and override with CLI parameters
    if config_file:
        config = ExperimentConfig.from_yaml(config_file)
        console.print(f"[blue]Loaded config from: {config_file}[/blue]")
    else:
        config = ExperimentConfig()

    if max_steps is not None:
        config.training.max_training_steps = max_steps

    if no_features:
        config.data.no_features = True

    # Handle seed for experiments
    if seed is not None:
        console.print(
            f"[blue]Using base seed: {seed} (each trial will use seed+trial_number)[/blue]"
        )
        base_seed = seed
    else:
        import random

        base_seed = random.randint(1, 100000)  # noqa: S311
        console.print(
            f"[yellow]Using random base seed: {base_seed} (each trial will use seed+trial_number)[/yellow]"
        )

    console.print(f"[bold blue]Running {n_trials} experiments[/bold blue]")
    console.print(f"Experiment: [green]{experiment_name}[/green]")
    if max_steps is not None:
        episodes = max_steps // 200  # frames_per_batch = 200
        console.print(f"Max steps: [green]{max_steps}[/green] (~{episodes} episodes)")
    console.print(
        "[dim]Tracking losses, position changes, rewards, and all parameters in MLflow[/dim]"
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Running {n_trials} trials...", total=None)

        try:
            # Run experiments with unified tracking (always plot positions)
            experiment_result = run_multiple_experiments(
                n_trials=n_trials,
                base_seed=base_seed,
                custom_config=config,
                experiment_name=experiment_name
            )
            progress.update(task, description="Experiments complete!")

            console.print("\n[bold green]All experiments completed![/bold green]")
            console.print(f"MLflow experiment: [green]{experiment_result}[/green]")
            console.print(f"Total trials: [green]{n_trials}[/green]")
            console.print(
                "\n[dim]Check MLflow UI for detailed metrics and comparisons[/dim]"
            )

            # Launch MLflow UI if requested
            if dashboard:
                console.print(
                    f"\n[blue]Launching MLflow UI for experiment: {experiment_name}[/blue]"
                )
                console.print(
                    "[dim]MLflow UI will be available at http://localhost:5000[/dim]"
                )
                console.print(
                    "[dim]View metrics: losses, position changes, rewards, and parameters[/dim]"
                )
                try:
                    mlflow_cmd = shutil.which("mlflow")
                    if not mlflow_cmd:
                        console.print(
                            "[red]MLflow not found. Install with: pip install mlflow[/red]"
                        )
                        return
                    subprocess.run([mlflow_cmd, "ui"], check=False)  # noqa: S603
                except KeyboardInterrupt:
                    console.print("\n[yellow]MLflow UI stopped[/yellow]")
                except FileNotFoundError:
                    console.print(
                        "[red]MLflow not found. Install with: pip install mlflow[/red]"
                    )

        except Exception as e:
            progress.update(task, description="Experiments failed!")
            console.print(f"\n[bold red]Experiments failed: {e}[/bold red]")
            raise typer.Exit(1) from e


@app.command()
def dashboard(
    port: int = typer.Option(5000, "--port", "-p", help="Port for MLflow UI"),
    host: str = typer.Option("localhost", "--host", help="Host for MLflow UI"),
):
    """Launch MLflow UI for viewing experiments."""

    console.print("[blue]Starting MLflow UI[/blue]")
    console.print(f"URL: [blue]http://{host}:{port}[/blue]")
    console.print("[dim]Press Ctrl+C to stop[/dim]")

    try:
        mlflow_cmd = shutil.which("mlflow")
        if not mlflow_cmd:
            console.print(
                "[red]MLflow not found. Install with: pip install mlflow[/red]"
            )
            raise typer.Exit(1)
        subprocess.run(  # noqa: S603
            [
                mlflow_cmd,
                "ui",
                "--host",
                host,
                "--port",
                str(port),
            ],
            check=False,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]MLflow UI stopped[/yellow]")
    except FileNotFoundError as e:
        console.print("[red]MLflow not found. Install with: pip install mlflow[/red]")
        raise typer.Exit(1) from e


@app.command()
def list_experiments():
    """List available MLflow experiments."""
    import mlflow

    console.print("[bold blue]Available MLflow Experiments:[/bold blue]")

    try:
        experiments = mlflow.search_experiments()

        if not experiments:
            console.print("[yellow]No experiments found[/yellow]")
            return

        for exp in experiments:
            console.print(f"\n[green]{exp.name}[/green]:")
            console.print(f"  ID: {exp.experiment_id}")
            console.print(f"  Lifecycle: {exp.lifecycle_stage}")

            # Get runs for this experiment
            runs = mlflow.search_runs(
                experiment_ids=[exp.experiment_id], max_results=10
            )
            console.print(f"  Runs: {len(runs)}")

            if len(runs) > 0:
                best_run = (
                    runs.loc[runs["metrics.final_reward"].idxmax()]
                    if "metrics.final_reward" in runs.columns
                    else None
                )
                if (
                    best_run is not None
                    and best_run["metrics.final_reward"] is not None
                ):
                    console.print(
                        f"  Best reward: {best_run['metrics.final_reward']:.4f}"
                    )

    except Exception as e:
        console.print(f"[red]Error reading MLflow experiments: {e}[/red]")


if __name__ == "__main__":
    app()
