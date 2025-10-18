"""
Command-line interface for data generation and trading agent training.
"""

import argparse
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from data_generator import PriceDataGenerator


def setup_data_generator_parser(subparsers):
    """Setup argument parser for data generator commands."""
    parser = subparsers.add_parser(
        "generate-data",
        help="Generate synthetic price data from existing parquet files",
        description="Generate synthetic price data by sampling or filtering existing data",
    )

    parser.add_argument(
        "--source-dir",
        type=str,
        default="data/raw/binance",
        help="Source directory containing parquet files (default: data/raw/binance)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw/synthetic",
        help="Output directory for synthetic data (default: data/raw/synthetic)",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List available source files",
    )

    parser.add_argument(
        "--source-file",
        type=str,
        help="Source parquet file name (e.g., binance-BTCUSDT-1h.parquet)",
    )

    parser.add_argument(
        "--output-file",
        type=str,
        help="Output file name (default: source file with _synthetic suffix)",
    )

    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date for filtering (format: YYYY-MM-DD)",
    )

    parser.add_argument(
        "--end-date",
        type=str,
        help="End date for filtering (format: YYYY-MM-DD)",
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        help="Number of rows to sample randomly",
    )

    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy source file without modifications",
    )

    # Synthetic pattern generation arguments
    parser.add_argument(
        "--sine-wave",
        action="store_true",
        help="Generate sine wave pattern with trend (ignores source-file)",
    )

    parser.add_argument(
        "--n-periods",
        type=int,
        default=3,
        help="Number of sine wave periods (default: 3)",
    )

    parser.add_argument(
        "--samples-per-period",
        type=int,
        default=120,
        help="Samples per sine wave period (default: 120)",
    )

    parser.add_argument(
        "--base-price",
        type=float,
        default=50000.0,
        help="Base price level (default: 50000.0)",
    )

    parser.add_argument(
        "--amplitude",
        type=float,
        default=5000.0,
        help="Sine wave amplitude (default: 5000.0)",
    )

    parser.add_argument(
        "--trend-slope",
        type=float,
        default=50.0,
        help="Linear trend slope per step (default: 50.0)",
    )

    parser.add_argument(
        "--volatility",
        type=float,
        default=0.02,
        help="Random noise factor (default: 0.02)",
    )

    parser.add_argument(
        "--upward-drift",
        action="store_true",
        help="Generate a strong upward drift pattern (ignores source-file)",
    )

    parser.add_argument(
        "--drift-samples",
        type=int,
        default=500,
        help="Number of samples for the drift pattern (default: 500)",
    )

    parser.add_argument(
        "--drift-rate",
        type=float,
        default=0.0015,
        help="Exponential drift rate per step for the drift pattern (default: 0.0015)",
    )

    parser.add_argument(
        "--drift-volatility",
        type=float,
        default=0.0005,
        help="Volatility factor for the drift pattern (default: 0.0005)",
    )

    parser.add_argument(
        "--drift-floor",
        type=float,
        default=0.995,
        help="Pullback floor multiplier for the drift pattern (default: 0.995)",
    )

    parser.set_defaults(func=handle_data_generator)


def handle_data_generator(args):
    """Handle data generator commands."""
    generator = PriceDataGenerator(
        source_dir=args.source_dir, output_dir=args.output_dir
    )

    # List available files
    if args.list:
        print("Available source files:")
        source_files = generator.list_source_files()
        if not source_files:
            print("  No parquet files found in source directory")
        else:
            for f in source_files:
                print(f"  - {f}")
        return

    # Generate sine wave pattern
    if args.sine_wave:
        output_file = args.output_file or "sine_wave_pattern.parquet"
        try:
            df = generator.generate_sine_wave_pattern(
                output_file=output_file,
                n_periods=args.n_periods,
                samples_per_period=args.samples_per_period,
                base_price=args.base_price,
                amplitude=args.amplitude,
                trend_slope=args.trend_slope,
                volatility=args.volatility,
                start_date=args.start_date or "2024-01-01",
            )
            print(f"\nSuccessfully generated sine wave pattern with {len(df)} rows")
            return
        except Exception as e:
            print(f"Error generating sine wave pattern: {e}")
            sys.exit(1)

    if args.upward_drift:
        output_file = args.output_file or "upward_drift_pattern.parquet"
        try:
            df = generator.generate_upward_drift_pattern(
                output_file=output_file,
                n_samples=args.drift_samples,
                base_price=args.base_price,
                drift_rate=args.drift_rate,
                volatility=args.drift_volatility,
                pullback_floor=args.drift_floor,
                start_date=args.start_date or "2024-01-01",
            )
            print(
                f"\nSuccessfully generated upward drift pattern with {len(df)} rows"
            )
            return
        except Exception as e:
            print(f"Error generating upward drift pattern: {e}")
            sys.exit(1)

    # Require source file for other operations
    if not args.source_file:
        print("Error: --source-file is required (or use --list to see available files, or --sine-wave for pattern generation)")
        sys.exit(1)

    # Copy operation
    if args.copy:
        generator.copy_data(args.source_file, args.output_file)
        return

    # Generate synthetic data
    try:
        df = generator.generate_synthetic_sample(
            source_file=args.source_file,
            output_file=args.output_file,
            start_date=args.start_date,
            end_date=args.end_date,
            sample_size=args.sample_size,
        )
        print(f"\nSuccessfully generated synthetic data with {len(df)} rows")
    except Exception as e:
        print(f"Error generating data: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="CLI tools for trading data science project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(
        title="commands", description="Available commands", dest="command"
    )

    # Setup command parsers
    setup_data_generator_parser(subparsers)

    # Parse arguments
    args = parser.parse_args()

    # Show help if no command specified
    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute command
    args.func(args)


# Typer CLI for Trading Agent
app = typer.Typer(
    name="trading-rl",
    help="Trading RL CLI for training and experimentation",
    add_completion=False,
)
console = Console()


@app.command()
def train(
    experiment_name: str | None = typer.Option(
        None, "--name", "-n", help="Experiment name (default: auto-generated)"
    ),
    config_file: Path | None = typer.Option(
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
    log_dir: Path | None = typer.Option(None, "--log-dir", help="Logging directory"),
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

        generated_seed = random.randint(1, 100000)
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
            raise typer.Exit(1)


@app.command()
def experiment(
    experiment_name: str = typer.Option(
        "trading_rl_experiment", "--name", "-n", help="MLflow experiment name"
    ),
    n_trials: int = typer.Option(5, "--trials", "-t", help="Number of trials to run"),
    dashboard: bool = typer.Option(
        False, "--dashboard", help="Launch MLflow UI after experiments"
    ),
    config_file: Path | None = typer.Option(
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
):
    """Run multiple experiments with MLflow tracking.

    Each experiment tracks multiple metrics simultaneously:
    - Actor and value losses over training steps
    - Position changes and trading activity
    - Portfolio values and episode rewards
    - All parameters and configurations
    """
    import subprocess

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

    # Handle seed for experiments
    if seed is not None:
        console.print(
            f"[blue]Using base seed: {seed} (each trial will use seed+trial_number)[/blue]"
        )
        base_seed = seed
    else:
        import random

        base_seed = random.randint(1, 100000)
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
                experiment_name, n_trials, base_seed, config
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
                    subprocess.run(["mlflow", "ui"])
                except KeyboardInterrupt:
                    console.print("\n[yellow]MLflow UI stopped[/yellow]")
                except FileNotFoundError:
                    console.print(
                        "[red]MLflow not found. Install with: pip install mlflow[/red]"
                    )

        except Exception as e:
            progress.update(task, description="Experiments failed!")
            console.print(f"\n[bold red]Experiments failed: {e}[/bold red]")
            raise typer.Exit(1)


@app.command()
def dashboard(
    port: int = typer.Option(5000, "--port", "-p", help="Port for MLflow UI"),
    host: str = typer.Option("localhost", "--host", help="Host for MLflow UI"),
):
    """Launch MLflow UI for viewing experiments."""
    import subprocess

    console.print("[blue]Starting MLflow UI[/blue]")
    console.print(f"URL: [blue]http://{host}:{port}[/blue]")
    console.print("[dim]Press Ctrl+C to stop[/dim]")

    try:
        subprocess.run(
            [
                "mlflow",
                "ui",
                "--host",
                host,
                "--port",
                str(port),
            ]
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]MLflow UI stopped[/yellow]")
    except FileNotFoundError:
        console.print("[red]MLflow not found. Install with: pip install mlflow[/red]")
        raise typer.Exit(1)


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
    # Check if we should use the Typer CLI
    if len(sys.argv) > 1 and sys.argv[1] in [
        "train",
        "experiment",
        "dashboard",
        "list-experiments",
    ]:
        app()
    else:
        # Fall back to original argparse CLI
        main()
