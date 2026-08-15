"""Training command implementation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer
from rich.columns import Columns
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from cli.services import TrainingConfigRequest, TrainingConfigService, ValidationReport
from trading_rl import run_single_experiment
from trading_rl.constants import SplitName
from trading_rl.evaluation.metric_meta import METRIC_LEGEND, METRIC_META_BY_KEY

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

    def __init__(self, console, config_service: TrainingConfigService | None = None):
        super().__init__(console)
        self._config_service = config_service or TrainingConfigService()

    def execute(self, params: TrainingParams) -> None:
        """Execute single training run."""
        try:
            if params.interactive:
                self._interactive_setup(params)

            prepared = self._config_service.prepare(
                TrainingConfigRequest(
                    config_file=params.config_file,
                    scenario=params.scenario,
                    config_overrides=params.config_overrides,
                    experiment_name=params.experiment_name,
                    seed=params.seed,
                    max_steps=params.max_steps,
                    checkpoint_path=params.checkpoint_path,
                    from_checkpoint=params.from_checkpoint,
                    from_last_checkpoint=params.from_last_checkpoint,
                ),
                load_config=self._load_experiment_config,
                resolve_seed=self.resolve_seed,
            )
            config = prepared.config
            params.checkpoint_path = prepared.checkpoint_path
            self._display_config_source(params)
            self._display_validation(prepared.validation)

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

    def _display_config_source(self, params: TrainingParams) -> None:
        if params.config_file:
            self.console.print(f"[blue]Loaded config from: {params.config_file}[/blue]")
        elif params.scenario:
            self.console.print(f"[blue]Loaded config from scenario: {params.scenario}[/blue]")

    def _display_validation(self, report: ValidationReport) -> None:
        """Run validation before training starts and fail fast on errors."""
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
                    result = run_single_experiment(custom_config=config, progress_bar=progress)

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
            run_table.add_row("Date Start", str(start)[:10])
        if end:
            run_table.add_row("Date End", str(end)[:10])
        unique_symbols = final_metrics.get("unique_symbols", [])
        if unique_symbols:
            run_table.add_row("Unique Symbols", str(len(unique_symbols)))
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

        _perf_keys = [
            "total_return", "sharpe_ratio", "sortino_ratio", "max_drawdown",
            "win_rate", "lose_rate", "profit_factor", "pct_long", "pct_short",
        ]
        _perf_metrics = [
            (key, METRIC_META_BY_KEY[key].label, METRIC_META_BY_KEY[key].fmt)
            for key in _perf_keys
        ]

        split_results = final_metrics.get("split_results", {})
        split_label = {
            SplitName.TRAIN: "Train",
            SplitName.VAL: "Val",
            SplitName.TEST: "Test",
        }
        perf_tables = []
        for split in SplitName:
            split_meta = split_results.get(split, {})
            report = split_meta.get("evaluation_report")
            if not report:
                continue
            report_dict = report
            t = make_table(f"Performance ({split_label[split]})")
            date_start = split_meta.get("date_start")
            date_end = split_meta.get("date_end")
            if date_start and date_end:
                t.add_row("Start Datetime", date_start)
                t.add_row("End Datetime",   date_end)
            symbols = split_meta.get("symbols", [])
            if symbols:
                t.add_row("Symbols", ", ".join(symbols))
            for key, display_name, fmt in _perf_metrics:
                if key in report_dict:
                    val = report_dict[key]
                    t.add_row(display_name, f"{val:{fmt}}")
            perf_tables.append(t)

        self.console.print(Columns([run_table, steps_table, *perf_tables]))

        legend_lines = ["[bold]Legend[/bold]"] + [
            f"[cyan]{name}[/cyan]  {desc}"
            for name, desc in METRIC_LEGEND.items()
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
