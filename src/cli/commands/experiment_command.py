"""Experiment command implementation."""

import shutil
import subprocess
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
    dashboard: bool = False
    config_file: Path | None = None
    scenario: str | None = None
    seed: int | None = None
    max_steps: int | None = None
    clear_cache: bool = False
    no_features: bool = False
    generate_data: bool = False


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
            base_seed = self.resolve_seed(params.seed)
            
            # Generate data if requested
            if params.generate_data:
                self._generate_data_from_config(config, params.scenario)
            
            # Display experiment info
            self._display_experiment_info(config, params, base_seed)
            
            # Run experiments
            experiment_result = self._run_experiments_with_progress(
                config, params, base_seed
            )
            
            # Launch dashboard if requested
            if params.dashboard:
                self._launch_dashboard(config.experiment_name)
                
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
            config = ExperimentConfig.from_yaml(params.config_file)
            self.console.print(f"[blue]Loaded config from file: {params.config_file}[/blue]")
        elif params.scenario:
            # Resolve scenario to config file
            config_file = self._resolve_scenario_config_path(params.scenario)
            config = ExperimentConfig.from_yaml(config_file)
            self.console.print(f"[blue]Loaded config from scenario: {params.scenario} -> {config_file}[/blue]")
        else:
            config = ExperimentConfig()

        if params.max_steps is not None:
            config.training.max_steps = params.max_steps

        if params.no_features:
            config.data.no_features = True

        # Determine effective experiment name (CLI override wins, otherwise config file)
        effective_experiment_name = params.experiment_name or config.experiment_name
        config.experiment_name = effective_experiment_name
        
        return config
    
    def _display_experiment_info(self, config, params: ExperimentParams, base_seed: int) -> None:
        """Display experiment configuration info."""
        self.console.print(f"[bold blue]Running {params.n_trials} experiments[/bold blue]")
        self.console.print(f"Experiment: [green]{config.experiment_name}[/green]")
        
        if params.max_steps is not None:
            episodes = params.max_steps // 200  # frames_per_batch = 200
            self.console.print(f"Max steps: [green]{params.max_steps}[/green] (~{episodes} episodes)")
        
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
    
    def _launch_dashboard(self, experiment_name: str) -> None:
        """Launch MLflow dashboard for the experiment."""
        self.console.print(
            f"\n[blue]Launching MLflow UI for experiment: {experiment_name}[/blue]"
        )
        self.console.print(
            "[dim]MLflow UI will be available at http://localhost:5000[/dim]"
        )
        self.console.print(
            "[dim]View metrics: losses, position changes, rewards, and parameters[/dim]"
        )
        
        try:
            mlflow_cmd = shutil.which("mlflow")
            if not mlflow_cmd:
                self.console.print(
                    "[red]MLflow not found. Install with: pip install mlflow[/red]"
                )
                return
            subprocess.run([mlflow_cmd, "ui"], check=False)  # noqa: S603
        except KeyboardInterrupt:
            self.console.print("\n[yellow]MLflow UI stopped[/yellow]")
        except FileNotFoundError:
            self.console.print(
                "[red]MLflow not found. Install with: pip install mlflow[/red]"
            )
    
    def _generate_data_from_config(self, config, provided_scenario: str | None = None) -> None:
        """Generate data based on the scenario configuration."""
        self.console.print("[blue]Generating data from scenario configuration...[/blue]")
        
        try:
            # Use provided scenario or extract from config
            scenario_name = provided_scenario or self._extract_scenario_name(config)
            
            if not scenario_name:
                self.console.print("[yellow]Could not determine scenario for data generation[/yellow]")
                return
            
            # Import and use DataGeneratorCommand
            from .data_generator_command import (
                DataGeneratorCommand,
                DataGenerationParams,
                SineWaveParams,
                UpwardDriftParams,
            )
            
            data_gen_cmd = DataGeneratorCommand(self.console)
            
            # Extract output file name from config
            output_file = None
            if hasattr(config, 'data') and hasattr(config.data, 'data_path'):
                from pathlib import Path
                data_path = Path(config.data.data_path)
                output_file = data_path.name
                self.console.print(f"[dim]Generating data for: {output_file}[/dim]")
            
            # Create parameters for data generation
            params = DataGenerationParams(scenario=scenario_name, output_file=output_file)
            sine_wave_params = SineWaveParams()
            upward_drift_params = UpwardDriftParams()
            
            # Generate data
            data_gen_cmd.execute(params, sine_wave_params, upward_drift_params)
            
            self.console.print("[green]Data generation completed![/green]")
            
        except Exception as e:
            self.console.print(f"[yellow]Data generation failed: {e}[/yellow]")
            self.logger.warning(f"Data generation failed: {e}")
    
    def _extract_scenario_name(self, config) -> str | None:
        """Extract scenario name from config for data generation."""
        # Try to get from config file path
        if hasattr(config, '_config_file_path'):
            config_path = Path(config._config_file_path)
            scenario_name = config_path.stem
            return scenario_name
        
        # Try to infer from experiment name
        if hasattr(config, 'experiment_name') and config.experiment_name:
            experiment_name = config.experiment_name
            
            # Map common experiment names to scenario names
            scenario_mappings = {
                'sine_wave_ppo_no_trend': 'sine_wave_ppo_no_trend',
                'sine_wave_ppo_trend': 'sine_wave_ppo_trend',
                'upward_trend_ppo': 'upward_trend_ppo',
                'white_noise_no_trend_ppo': 'white_noise_no_trend_ppo',
                'mean_reversion': 'mean_reversion',
            }
            
            if experiment_name in scenario_mappings:
                return scenario_mappings[experiment_name]
        
        return None
    
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
