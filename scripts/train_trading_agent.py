#!/usr/bin/env python3
"""CLI for training trading RL agents.

This script provides a command-line interface for training trading agents
using various RL algorithms (PPO, TD3, DDPG) with different environment backends.
"""

import sys
from pathlib import Path

import typer
from rich.console import Console

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trading_rl.train_trading_agent import run_experiment_from_config

app = typer.Typer(
    help="Train trading RL agents with configurable algorithms and environments",
    add_completion=False,
)
console = Console()


@app.command()
def main(
    config: Path = typer.Argument(
        ...,
        help="Path to YAML configuration file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    trials: int = typer.Option(
        1,
        "--trials",
        "-n",
        help="Number of training trials to run",
        min=1,
    ),
):
    """Train a trading RL agent using the specified configuration.

    Example:
        train-agent src/configs/tradingenv_ppo_example.yaml

        train-agent src/configs/my_config.yaml --trials 5
    """
    console.print(f"\n[bold blue]Training RL Agent[/bold blue]")
    console.print(f"Config: [cyan]{config}[/cyan]")
    console.print(f"Trials: [cyan]{trials}[/cyan]\n")

    try:
        experiment_name = run_experiment_from_config(
            config_path=str(config),
            n_trials=trials,
        )

        console.print(
            f"\n[bold green]✓ Training complete![/bold green]\n"
            f"Experiment: [cyan]{experiment_name}[/cyan]\n"
            f"Check MLflow UI for results: [yellow]mlflow ui[/yellow]"
        )

    except Exception as e:
        console.print(f"\n[bold red]✗ Training failed:[/bold red] {e}\n")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
