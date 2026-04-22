"""Training command implementation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from cli.services import validate_experiment_config
from trading_rl import ExperimentConfig, run_single_experiment

from .base_command import BaseCommand


@dataclass
class TrainingParams:
    """Parameters for single agent training."""

    experiment_name: str | None = None
    config_file: Path | None = None
    scenario: str | None = None
    config_overrides: list[str] | None = None
    seed: int | None = None
    max_steps: int | None = None
    checkpoint_path: Path | None = None  # Path to checkpoint to resume from
    additional_steps: int | None = None  # Additional steps when resuming
    from_checkpoint: Path | None = None  # Path to checkpoint alias
    from_last_checkpoint: bool = False  # Resume from most recent checkpoint
    mlflow_run_id: str | None = None  # Resume MLflow run by ID


class TrainingCommand(BaseCommand):
    """Command for training a single trading agent."""

    def execute(self, params: TrainingParams) -> None:
        """Execute single training run."""
        try:
            if params.from_checkpoint and params.from_last_checkpoint:
                raise ValueError(
                    "Use only one of --from-checkpoint or --from-last-checkpoint."
                )

            # Load and configure experiment
            config = self._load_training_config(params)
            self._resolve_checkpoint_path(config, params)

            if params.checkpoint_path:
                self.console.print(
                    "[bold blue]Resuming Training from Checkpoint[/bold blue]"
                )
                self.console.print(f"Checkpoint: [cyan]{params.checkpoint_path}[/cyan]")
            else:
                self.console.print(
                    "[bold blue]Starting Trading Agent Training[/bold blue]"
                )

            # Display configuration
            self._display_config(config, params)

            # Run training with progress tracking
            result = self._run_training_with_progress(config, params)

            # Save plots if requested
            if config.logging.save_plots:
                self._save_training_plots(result, config, params)

        except Exception as e:
            self.handle_error(e, "Training")

    def _load_training_config(self, params: TrainingParams) -> Any:
        """Load and configure training parameters."""
        # Load base configuration
        if params.config_file and params.scenario:
            raise ValueError("Cannot specify both --config and --scenario.")

        if params.config_file:
            config_path = (
                params.config_file
                if params.config_file.exists()
                else self._resolve_scenario_config_path(str(params.config_file))
            )
            config = ExperimentConfig.from_yaml(
                config_path, overrides=params.config_overrides
            )
            self.console.print(f"[blue]Loaded config from: {config_path}[/blue]")
        elif params.scenario:
            config_file = self._resolve_scenario_config_path(params.scenario)
            config = ExperimentConfig.from_yaml(
                config_file, overrides=params.config_overrides
            )
            self.console.print(
                f"[blue]Loaded config from scenario: {params.scenario} -> {config_file}[/blue]"
            )
        else:
            if params.config_overrides:
                raise ValueError("--config-override requires --config or --scenario.")
            config = ExperimentConfig()

        # Apply CLI overrides
        self._apply_training_overrides(config, params)

        # Handle seed generation
        config.seed = self.resolve_seed(config.seed)
        self._prevalidate_or_fail(config)

        return config

    def _prevalidate_or_fail(self, config: Any) -> None:
        """Run validation before training starts and fail fast on errors."""
        report = validate_experiment_config(config)
        if report.has_warnings:
            self.console.print(
                f"[yellow]Validation warnings: {report.warning_count}[/yellow]"
            )
            for issue in report.issues:
                if issue.severity == "warning":
                    self.console.print(
                        f"[yellow]- {issue.check} ({issue.code}): {issue.message}[/yellow]"
                    )
        if report.has_errors:
            error_lines = [
                f"- {issue.check} ({issue.code}): {issue.message}"
                for issue in report.issues
                if issue.severity == "error"
            ]
            raise ValueError(
                "Validation failed before training:\n" + "\n".join(error_lines)
            )

    def _apply_training_overrides(self, config: Any, params: TrainingParams) -> None:
        """Apply CLI parameter overrides to config."""
        if params.experiment_name:
            config.experiment_name = params.experiment_name
        if params.seed is not None:
            config.seed = params.seed
        if params.max_steps is not None:
            config.training.max_steps = params.max_steps

    def _resolve_scenario_config_path(self, scenario: str) -> Path:
        """Resolve scenario name to config file path."""
        candidate_path = Path(scenario)

        if candidate_path.is_dir():
            candidate_path = candidate_path / "config.yaml"

        search_paths = [
            candidate_path,
            Path("src/configs/scenarios") / scenario,
            Path("src/configs/scenarios") / f"{scenario}.yaml",
        ]

        for path in search_paths:
            if path.exists():
                return path.resolve()

        raise ValueError(
            f"Scenario '{scenario}' not found. Provide a valid path or name in src/configs/scenarios."
        )

    def _display_config(self, config: Any, params: TrainingParams) -> None:
        """Display training configuration."""
        self.console.print(f"Experiment: [green]{config.experiment_name}[/green]")
        self.console.print(f"Seed: [green]{config.seed}[/green]")
        if params.checkpoint_path and params.additional_steps:
            self.console.print(
                f"Additional steps: [green]{params.additional_steps}[/green]"
            )
        else:
            self.console.print(f"Max steps: [green]{config.training.max_steps}[/green]")

    def _resolve_checkpoint_path(self, config: Any, params: TrainingParams) -> None:
        """Resolve checkpoint path from aliases or latest checkpoint option."""
        if params.checkpoint_path:
            return
        if params.from_checkpoint:
            params.checkpoint_path = params.from_checkpoint
            return
        if not params.from_last_checkpoint:
            return

        log_dir = Path(config.logging.log_dir)
        pattern = f"{config.experiment_name}_checkpoint*.pt"
        matches = list(log_dir.rglob(pattern))
        if not matches:
            raise FileNotFoundError(
                f"No checkpoints found for {config.experiment_name} in {log_dir}"
            )
        latest = max(matches, key=lambda p: p.stat().st_mtime)
        params.checkpoint_path = latest

    def _run_training_with_progress(
        self, config: Any, params: TrainingParams
    ) -> dict[str, Any]:
        """Run training with progress display."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Training agent...", total=None)

            try:
                # Handle checkpoint resumption
                if params.checkpoint_path:
                    result = self._resume_from_checkpoint(config, params, progress)
                else:
                    result = run_single_experiment(custom_config=config)

                if result.get("interrupted"):
                    progress.update(
                        task, description="Training interrupted; final evaluation complete!"
                    )
                else:
                    progress.update(task, description="Training complete!")

                self._display_training_results(result)
                return result

            except Exception as e:
                progress.update(task, description="Training failed!")
                raise e

    def _resume_from_checkpoint(
        self, config: Any, params: TrainingParams, progress: Any
    ) -> dict[str, Any]:
        """Resume training from a checkpoint file.

        This is now a thin wrapper around run_single_experiment.
        """
        # Validate checkpoint exists
        if not Path(params.checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {params.checkpoint_path}")

        self.console.print(
            f"[cyan]Resuming from checkpoint: {params.checkpoint_path}...[/cyan]"
        )

        # Call run_single_experiment with checkpoint parameters
        result = run_single_experiment(
            custom_config=config,
            checkpoint_path=str(params.checkpoint_path),
            additional_steps=params.additional_steps,
            progress_bar=progress,
        )

        if result.get("interrupted"):
            self.console.print(
                "[yellow]Training resumed, interrupted, and final evaluation completed.[/yellow]"
            )
        else:
            self.console.print(
                "[green]Training resumed and completed successfully![/green]"
            )
        return result

    def _display_training_results(self, result: dict[str, Any]) -> None:
        """Display training results in a formatted table."""
        table = Table(title="Training Results", show_header=True, header_style="bold")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        # Extract metrics from nested structure
        final_metrics = result.get("final_metrics", {})
        eval_report = final_metrics.get("evaluation_report", {})

        # Define metrics to display: (source_dict, key, display_name, format_spec)
        metrics_spec = [
            # Core training metrics
            (final_metrics, "final_reward", "Final Reward", ".4f"),
            (final_metrics, "training_steps", "Training Steps", ","),
            # Trading performance metrics
            (eval_report, "total_return", "Total Return", ".2%"),
            (eval_report, "annualized_return_cagr", "CAGR", ".2%"),
            (eval_report, "sharpe_ratio", "Sharpe Ratio", ".3f"),
            (eval_report, "sortino_ratio", "Sortino Ratio", ".3f"),
            (eval_report, "max_drawdown", "Max Drawdown", ".2%"),
            (eval_report, "win_rate", "Win Rate", ".2%"),
            (eval_report, "profit_factor", "Profit Factor", ".3f"),
        ]

        # Add rows for available metrics
        for source, key, display_name, fmt in metrics_spec:
            if key in source:
                value = source[key]
                table.add_row(display_name, f"{value:{fmt}}")

        self.console.print(table)

    def _save_training_plots(
        self, result: dict[str, Any], config, params: TrainingParams
    ) -> None:
        """Save training plots to disk."""

        plots_dir = Path(config.logging.log_dir) / "plots"
        plots_dir.mkdir(exist_ok=True, parents=True)
        # (Skipping rewrite as per instructions)
        return result
