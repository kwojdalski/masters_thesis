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

    # Require source file for other operations
    if not args.source_file:
        print("Error: --source-file is required (or use --list to see available files)")
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
        None, "--seed", "-s", help="Random seed for reproducibility"
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
    config = ExperimentConfig()

    # Override with CLI parameters
    if experiment_name:
        config.experiment_name = experiment_name
    if seed is not None:
        config.seed = seed
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
    study_name: str = typer.Option(
        "trading_rl_study", "--study", "-s", help="Optuna study name"
    ),
    n_trials: int = typer.Option(5, "--trials", "-t", help="Number of trials to run"),
    storage_url: str | None = typer.Option(
        None, "--storage", help="Optuna storage URL (default: SQLite)"
    ),
    dashboard: bool = typer.Option(
        False, "--dashboard", help="Launch Optuna dashboard after experiments"
    ),
    config_file: Path | None = typer.Option(
        None, "--config", "-c", help="Path to custom config file"
    ),
):
    """Run multiple experiments with Optuna tracking."""
    import subprocess

    from trading_rl import run_multiple_experiments

    console.print(f"[bold blue]Running {n_trials} experiments[/bold blue]")
    console.print(f"Study: [green]{study_name}[/green]")

    if storage_url:
        # Create study with custom storage
        from trading_rl.train_trading_agent import create_optuna_study

        study = create_optuna_study(study_name, storage_url)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Running {n_trials} trials...", total=None)

        try:
            study = run_multiple_experiments(study_name, n_trials)
            progress.update(task, description="Experiments complete!")

            console.print("\n[bold green]All experiments completed![/bold green]")
            console.print(f"Best trial: [green]{study.best_trial.number}[/green]")
            console.print(f"Best reward: [green]{study.best_value:.4f}[/green]")

            # Show trial summary
            console.print("\n[bold]Trial Summary:[/bold]")
            for trial in study.trials:
                status = "✓" if trial.state.name == "COMPLETE" else "✗"
                reward = trial.user_attrs.get("final_reward", "N/A")
                console.print(f"  {status} Trial {trial.number}: reward={reward}")

            # Launch dashboard if requested
            if dashboard:
                db_path = f"{study_name}.db"
                console.print(
                    f"\n[blue]Launching Optuna dashboard for {db_path}[/blue]"
                )
                console.print(
                    "[dim]Dashboard will be available at http://localhost:8080[/dim]"
                )
                try:
                    subprocess.run(["optuna-dashboard", f"sqlite:///{db_path}"])
                except KeyboardInterrupt:
                    console.print("\n[yellow]Dashboard stopped[/yellow]")
                except FileNotFoundError:
                    console.print(
                        "[red]optuna-dashboard not found. Install with: pip install optuna-dashboard[/red]"
                    )

        except Exception as e:
            progress.update(task, description="Experiments failed!")
            console.print(f"\n[bold red]Experiments failed: {e}[/bold red]")
            raise typer.Exit(1)


@app.command()
def dashboard(
    study_name: str = typer.Argument(help="Study name or database file"),
    port: int = typer.Option(8080, "--port", "-p", help="Port for dashboard"),
    host: str = typer.Option("localhost", "--host", help="Host for dashboard"),
):
    """Launch Optuna dashboard for viewing results."""
    import subprocess

    # Handle both study name and file path
    if study_name.endswith(".db"):
        db_path = study_name
    else:
        db_path = f"{study_name}.db"

    if not Path(db_path).exists():
        console.print(f"[red]Database file not found: {db_path}[/red]")
        raise typer.Exit(1)

    console.print("[blue]Starting Optuna dashboard[/blue]")
    console.print(f"Database: [green]{db_path}[/green]")
    console.print(f"URL: [blue]http://{host}:{port}[/blue]")
    console.print("[dim]Press Ctrl+C to stop[/dim]")

    try:
        subprocess.run(
            [
                "optuna-dashboard",
                f"sqlite:///{db_path}",
                "--host",
                host,
                "--port",
                str(port),
            ]
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Dashboard stopped[/yellow]")
    except FileNotFoundError:
        console.print(
            "[red]optuna-dashboard not found. Install with: pip install optuna-dashboard[/red]"
        )
        raise typer.Exit(1)


@app.command()
def list_studies(
    directory: Path = typer.Option(
        ".", "--dir", "-d", help="Directory to search for study databases"
    ),
):
    """List available Optuna studies."""
    import optuna

    console.print("[bold blue]Available Studies:[/bold blue]")

    db_files = list(directory.glob("*.db"))
    if not db_files:
        console.print("[yellow]No study databases found[/yellow]")
        return

    for db_file in db_files:
        try:
            study_summaries = optuna.get_all_study_summaries(f"sqlite:///{db_file}")
            console.print(f"\n[green]{db_file.name}[/green]:")

            if not study_summaries:
                console.print("  [dim]No studies found[/dim]")
                continue

            for summary in study_summaries:
                console.print(f"  • [blue]{summary.study_name}[/blue]")
                console.print(f"    Trials: {summary.n_trials}")
                if summary.best_trial:
                    console.print(f"    Best value: {summary.best_trial.value:.4f}")

        except Exception as e:
            console.print(f"  [red]Error reading {db_file.name}: {e}[/red]")


if __name__ == "__main__":
    # Check if we should use the Typer CLI
    if len(sys.argv) > 1 and sys.argv[1] in [
        "train",
        "experiment",
        "dashboard",
        "list-studies",
    ]:
        app()
    else:
        # Fall back to original argparse CLI
        main()
