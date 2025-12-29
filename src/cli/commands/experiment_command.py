"""Experiment command implementation."""

from dataclasses import dataclass
from pathlib import Path

import typer
from rich.progress import Progress, SpinnerColumn, TextColumn

from .base_command import BaseCommand


@dataclass
class ExperimentParams:
    """Parameters for running multiple experiments."""
    experiment_name: str | None = None
    n_trials: int = 5
    config_file: Path | None = None
    scenario: str | None = None
    config_overrides: list[str] | None = None
    clear_cache: bool = False


class ExperimentCommand(BaseCommand):
    """Command for running multiple experiments with MLflow tracking."""
    
    def execute(self, params: ExperimentParams) -> None:
        """Execute multiple experiments."""
        try:
            # Clear cache if requested
            if params.clear_cache:
                self._clear_caches()
            
            # Load and configure experiments
            config = self._load_experiment_config(params)
            base_seed = self.resolve_seed(config.seed)
            
            # Display experiment info
            self._display_experiment_info(config, params, base_seed)
            
            # Run experiments
            experiment_result = self._run_experiments_with_progress(
                config, params, base_seed
            )
            
        except Exception as e:
            self.handle_error(e, "Experiments")
    
    def _clear_caches(self) -> None:
        """Clear all caches before experiments."""
        self.console.print("[blue]Clearing caches before experiments...[/blue]")
        try:
            from trading_rl.cache_utils import (
                clear_data_cache,
                clear_model_cache,
                clear_training_cache,
            )
            clear_data_cache()
            clear_model_cache()
            clear_training_cache()
            self.console.print("[green]Caches cleared.[/green]")
        except ImportError:
            self.console.print(
                "[yellow]Cache utilities unavailable; skipping cache clear.[/yellow]"
            )
    
    def _load_experiment_config(self, params: ExperimentParams):
        """Load and configure experiment parameters."""
        from trading_rl import ExperimentConfig
        
        # Validate that only one of config_file or scenario is provided
        if params.config_file and params.scenario:
            raise typer.BadParameter("Cannot specify both --config and --scenario. Use one or the other.")
        
        # Create config and override with CLI parameters
        if params.config_file:
            config = ExperimentConfig.from_yaml(
                params.config_file, overrides=params.config_overrides
            )
            self.console.print(f"[blue]Loaded config from file: {params.config_file}[/blue]")
        elif params.scenario:
            # Resolve scenario to config file
            config_file = self._resolve_scenario_config_path(params.scenario)
            config = ExperimentConfig.from_yaml(
                config_file, overrides=params.config_overrides
            )
            self.console.print(f"[blue]Loaded config from scenario: {params.scenario} -> {config_file}[/blue]")
        else:
            if params.config_overrides:
                raise typer.BadParameter(
                    "--config-override requires --config or --scenario."
                )
            config = ExperimentConfig()

        # Determine effective experiment name (CLI override wins, otherwise config file)
        effective_experiment_name = params.experiment_name or config.experiment_name
        config.experiment_name = effective_experiment_name
        
        return config
    
    def _display_experiment_info(self, config, params: ExperimentParams, base_seed: int) -> None:
        """Display experiment configuration info."""
        self.console.print(f"[bold blue]Running {params.n_trials} experiments[/bold blue]")
        self.console.print(f"Experiment: [green]{config.experiment_name}[/green]")
        
        self.console.print(
            f"[blue]Using base seed: {base_seed} (each trial will use seed+trial_number)[/blue]"
        )
        self.console.print(
            "[dim]Tracking losses, position changes, rewards, and all parameters in MLflow[/dim]"
        )
    
    def _run_experiments_with_progress(
        self, config, params: ExperimentParams, base_seed: int
    ) -> str:
        """Run experiments with progress tracking."""
        from trading_rl import run_multiple_experiments
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task(f"Running {params.n_trials} trials...", total=None)
            
            try:
                # Run experiments with unified tracking (always plot positions)
                experiment_result = run_multiple_experiments(
                    n_trials=params.n_trials,
                    base_seed=base_seed,
                    custom_config=config,
                    experiment_name=config.experiment_name,
                )
                progress.update(task, description="Experiments complete!")
                
                self._display_experiment_results(experiment_result, params.n_trials)
                return experiment_result
                
            except Exception as e:
                progress.update(task, description="Experiments failed!")
                raise e
    
    def _display_experiment_results(self, experiment_result: str, n_trials: int) -> None:
        """Display experiment results."""
        self.console.print("\n[bold green]All experiments completed![/bold green]")
        self.console.print(f"MLflow experiment: [green]{experiment_result}[/green]")
        self.console.print(f"Total trials: [green]{n_trials}[/green]")
        self.console.print(
            "\n[dim]Check MLflow UI for detailed metrics and comparisons[/dim]"
        )
    
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

        raise typer.BadParameter(
            f"Scenario '{scenario}' not found. Provide a valid path or name in src/configs/scenarios. "
                f"Use 'scenarios' to see available options."
        )
