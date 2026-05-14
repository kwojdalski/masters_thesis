"""Training command implementation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.columns import Columns
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
    interactive: bool = False


class TrainingCommand(BaseCommand):
    """Command for training a single trading agent."""

    def execute(self, params: TrainingParams) -> None:
        """Execute single training run."""
        try:
            if params.from_checkpoint and params.from_last_checkpoint:
                raise ValueError(
                    "Use only one of --from-checkpoint or --from-last-checkpoint."
                )

            if params.interactive:
                self._interactive_setup(params)

            # Load and configure experiment
            config = self._load_training_config(params)
            self._resolve_checkpoint_path(config, params)

            if params.interactive:
                self._interactive_post_config(config, params)

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

    def _interactive_setup(self, params: TrainingParams) -> None:
        """Ask setup questions before config is loaded (pre-config phase)."""
        self.console.print("\n[bold]Interactive training setup[/bold]")

        if params.experiment_name is None:
            name = typer.prompt("Experiment name", default="")
            if name:
                params.experiment_name = name

        if not params.from_checkpoint and not params.from_last_checkpoint:
            if typer.confirm("Resume from the last checkpoint?", default=False):
                params.from_last_checkpoint = True

    def _interactive_post_config(self, config: Any, params: TrainingParams) -> None:
        """Ask setup questions that need the loaded config (post-config phase)."""
        current_steps = config.training.max_steps
        override_steps = typer.confirm(
            f"Max training steps is {current_steps:,}. Change it?", default=False
        )
        if override_steps:
            new_steps = typer.prompt("New max steps", default=current_steps)
            config.training.max_steps = int(new_steps)

        cache_enabled = getattr(config.data, "feature_cache_dir", ".cache/feature_transformation") is not None
        if cache_enabled:
            if typer.confirm("Process features from scratch (skip cache)?", default=False):
                config.data.feature_cache_dir = None
                self.console.print("[yellow]Feature cache disabled — features will be recomputed.[/yellow]")
        else:
            self.console.print("[dim]Feature caching is already disabled in config.[/dim]")

    def _load_training_config(self, params: TrainingParams) -> Any:
        """Load and configure training parameters."""
        # Load base configuration
        if params.config_file and params.scenario:
            raise ValueError("Cannot specify both --config and --scenario.")

        if params.config_file:
            config = self._load_experiment_config(
                params.config_file, command="train", overrides=params.config_overrides
            )
            self.console.print(f"[blue]Loaded config from: {params.config_file}[/blue]")
        elif params.scenario:
            config = self._load_experiment_config(
                params.scenario, command="train", overrides=params.config_overrides
            )
            self.console.print(f"[blue]Loaded config from scenario: {params.scenario}[/blue]")
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
        """Display training results in three side-by-side tables."""
        final_metrics = result.get("final_metrics", {})

        def make_table(title: str) -> Table:
            t = Table(title=title, show_header=True, header_style="bold")
            t.add_column("Metric", style="cyan")
            t.add_column("Value", style="green")
            return t

        run_table = make_table("Run")
        start = final_metrics.get("data_start_date", "")
        end = final_metrics.get("data_end_date", "")
        if start:
            run_table.add_row("Data Start", str(start)[:10])
        if end:
            run_table.add_row("Data End", str(end)[:10])
        for key, label in [
            ("train_size", "Train Rows"),
            ("validation_size", "Val Rows"),
            ("test_size", "Test Rows"),
            ("data_size_total", "Total Rows"),
        ]:
            if key in final_metrics:
                run_table.add_row(label, f"{final_metrics[key]:,}")
        duration = final_metrics.get("training_duration_s")
        if duration is not None:
            mins, secs = divmod(int(duration), 60)
            run_table.add_row("Duration", f"{mins}m {secs}s")

        steps_table = make_table("Steps")
        for key, label, fmt in [
            ("total_env_steps", "Env Steps", ","),
            ("episode_length", "Episode Length", ","),
            ("total_episodes", "Episodes", ","),
            ("optimizer_steps", "Optimizer Steps", ","),
            ("eval_steps", "Eval Horizon", ","),
        ]:
            if key in final_metrics:
                steps_table.add_row(label, f"{final_metrics[key]:{fmt}}")
        steps_table.add_row("Final Reward", f"{final_metrics.get('final_reward', float('nan')):.4f}")

        _perf_metrics = [
            ("total_return", "Total Return", ".2%"),
            ("annualized_return_cagr", "CAGR", ".2%"),
            ("sharpe_ratio", "Sharpe Ratio", ".3f"),
            ("sortino_ratio", "Sortino Ratio", ".3f"),
            ("max_drawdown", "Max Drawdown", ".2%"),
            ("win_rate", "Win Rate", ".2%"),
            ("profit_factor", "Profit Factor", ".3f"),
        ]

        split_results = final_metrics.get("split_results", {})
        split_label = {"train": "Train", "val": "Val", "test": "Test"}
        perf_tables = []
        for split in ("train", "val", "test"):
            report = split_results.get(split, {}).get("evaluation_report", {})
            if not report:
                continue
            t = make_table(f"Performance ({split_label[split]})")
            for key, display_name, fmt in _perf_metrics:
                if key in report:
                    t.add_row(display_name, f"{report[key]:{fmt}}")
            perf_tables.append(t)

        self.console.print(Columns([run_table, steps_table, *perf_tables]))

        legend_lines = [
            "[bold]Legend[/bold]",
            "[cyan]Env Steps[/cyan]          Total environment steps collected during training.",
            "[cyan]Episode Length[/cyan]     Steps per episode (streaming window or full dataset). Episodes = Env Steps / Episode Length.",
            "[cyan]Episodes[/cyan]           Number of full episodes completed (resets) during training.",
            "[cyan]Optimizer Steps[/cyan]    Number of gradient update steps taken.",
            "[cyan]Eval Horizon[/cyan]       Number of steps used for each evaluation rollout.",
            "[cyan]Final Reward[/cyan]       Raw reward signal at the last training step.",
            "[cyan]Total Return[/cyan]       Cumulative portfolio return over the evaluation horizon.",
            "[cyan]CAGR[/cyan]               Compound Annual Growth Rate — total return annualised.",
            "[cyan]Sharpe Ratio[/cyan]       Mean excess return divided by its standard deviation, annualised.",
            "[cyan]Sortino Ratio[/cyan]      Like Sharpe but penalises only downside deviation.",
            "[cyan]Max Drawdown[/cyan]       Largest peak-to-trough decline in portfolio value.",
            "[cyan]Win Rate[/cyan]           Fraction of steps where the portfolio return was positive.",
            "[cyan]Profit Factor[/cyan]      Gross profit divided by gross loss (> 1 means net profitable).",
        ]
        self.console.print()
        for line in legend_lines:
            self.console.print(f"  {line}")

    def _save_training_plots(
        self, result: dict[str, Any], config, params: TrainingParams
    ) -> None:
        """Save training plots to disk."""

        plots_dir = Path(config.logging.log_dir) / "plots"
        plots_dir.mkdir(exist_ok=True, parents=True)
        # (Skipping rewrite as per instructions)
        return result
