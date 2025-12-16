"""Training command implementation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.progress import Progress, SpinnerColumn, TextColumn

from .base_command import BaseCommand


@dataclass
class TrainingParams:
    """Parameters for single agent training."""

    experiment_name: str | None = None
    config_file: Path | None = None
    seed: int | None = None
    max_steps: int | None = None
    init_rand_steps: int | None = None
    actor_lr: float | None = None
    value_lr: float | None = None
    buffer_size: int | None = None
    save_plots: bool = False
    log_dir: Path | None = None


class TrainingCommand(BaseCommand):
    """Command for training a single trading agent."""

    def execute(self, params: TrainingParams) -> None:
        """Execute single training run."""
        try:
            self.console.print("[bold blue]Starting Trading Agent Training[/bold blue]")

            # Load and configure experiment
            config = self._load_training_config(params)

            # Display configuration
            self._display_config(config)

            # Run training with progress tracking
            result = self._run_training_with_progress(config)

            # Save plots if requested
            if params.save_plots:
                self._save_training_plots(result, config, params)

        except Exception as e:
            self.handle_error(e, "Training")

    def _load_training_config(self, params: TrainingParams):
        """Load and configure training parameters."""
        from trading_rl import ExperimentConfig

        # Load base configuration
        if params.config_file:
            config = ExperimentConfig.from_yaml(params.config_file)
            self.console.print(f"[blue]Loaded config from: {params.config_file}[/blue]")
        else:
            config = ExperimentConfig()

        # Handle seed generation
        config.seed = self.resolve_seed(params.seed)

        # Apply CLI overrides
        self._apply_training_overrides(config, params)

        return config

    def _apply_training_overrides(self, config, params: TrainingParams) -> None:
        """Apply CLI parameter overrides to config."""
        if params.experiment_name:
            config.experiment_name = params.experiment_name
        if params.max_steps is not None:
            config.training.max_steps = params.max_steps
        if params.init_rand_steps is not None:
            config.training.init_rand_steps = params.init_rand_steps
        if params.actor_lr is not None:
            config.training.actor_lr = params.actor_lr
        if params.value_lr is not None:
            config.training.value_lr = params.value_lr
        if params.buffer_size is not None:
            config.training.buffer_size = params.buffer_size
        if params.log_dir is not None:
            config.logging.log_dir = str(params.log_dir)

    def _display_config(self, config) -> None:
        """Display training configuration."""
        self.console.print(f"Experiment: [green]{config.experiment_name}[/green]")
        self.console.print(f"Seed: [green]{config.seed}[/green]")
        self.console.print(f"Max steps: [green]{config.training.max_steps}[/green]")

    def _run_training_with_progress(self, config) -> dict[str, Any]:
        """Run training with progress display."""
        from trading_rl import run_single_experiment

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Training agent...", total=None)

            try:
                result = run_single_experiment(custom_config=config)
                progress.update(task, description="Training complete!")

                self._display_training_results(result)
                return result

            except Exception as e:
                progress.update(task, description="Training failed!")
                raise e

    def _display_training_results(self, result: dict[str, Any]) -> None:
        """Display training results."""
        self.console.print(
            "\n[bold green]Training completed successfully![/bold green]"
        )

        if "final_metrics" in result:
            metrics = result["final_metrics"]
            if "final_reward" in metrics:
                self.console.print(
                    f"Final reward: [green]{metrics['final_reward']:.4f}[/green]"
                )
            if "training_steps" in metrics:
                self.console.print(
                    f"Training steps: [green]{metrics['training_steps']}[/green]"
                )

    def _save_training_plots(
        self, result: dict[str, Any], config, params: TrainingParams
    ) -> None:
        """Save training plots to disk."""

        plots_dir = Path(config.logging.log_dir) / "plots"
        plots_dir.mkdir(exist_ok=True, parents=True)
        # (Skipping rewrite as per instructions)
        return result
