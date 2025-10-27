"""Dashboard command implementation."""

import shutil
import subprocess
from dataclasses import dataclass

import typer

from .base_command import BaseCommand


@dataclass
class DashboardParams:
    """Parameters for MLflow dashboard."""
    port: int = 5000
    host: str = "localhost"


class DashboardCommand(BaseCommand):
    """Command for managing MLflow dashboard."""
    
    def execute(self, params: DashboardParams) -> None:
        """Launch MLflow UI."""
        self.console.print("[blue]Starting MLflow UI[/blue]")
        self.console.print(f"URL: [blue]http://{params.host}:{params.port}[/blue]")
        self.console.print("[dim]Press Ctrl+C to stop[/dim]")
        
        try:
            self._launch_mlflow_ui(params)
        except KeyboardInterrupt:
            self.console.print("\n[yellow]MLflow UI stopped[/yellow]")
        except FileNotFoundError as e:
            self.handle_error(e, "MLflow UI launch")
    
    def list_experiments(self) -> None:
        """List available MLflow experiments."""
        import mlflow
        
        self.console.print("[bold blue]Available MLflow Experiments:[/bold blue]")
        
        try:
            experiments = mlflow.search_experiments()
            
            if not experiments:
                self.console.print("[yellow]No experiments found[/yellow]")
                return
            
            for exp in experiments:
                self._display_experiment_summary(exp)
                
        except Exception as e:
            self.handle_error(e, "Reading MLflow experiments")
    
    def _launch_mlflow_ui(self, params: DashboardParams) -> None:
        """Launch MLflow UI subprocess."""
        mlflow_cmd = shutil.which("mlflow")
        if not mlflow_cmd:
            raise FileNotFoundError("MLflow not found. Install with: pip install mlflow")
        
        subprocess.run([
            mlflow_cmd, "ui",
            "--host", params.host,
            "--port", str(params.port),
        ], check=False)  # noqa: S603
    
    def _display_experiment_summary(self, exp) -> None:
        """Display summary information for an experiment."""
        import mlflow
        
        self.console.print(f"\n[green]{exp.name}[/green]:")
        self.console.print(f"  ID: {exp.experiment_id}")
        self.console.print(f"  Lifecycle: {exp.lifecycle_stage}")

        # Get runs for this experiment
        try:
            runs = mlflow.search_runs(
                experiment_ids=[exp.experiment_id], max_results=10
            )
            self.console.print(f"  Runs: {len(runs)}")

            if len(runs) > 0:
                self._display_best_run(runs)
        except Exception as e:
            self.logger.warning(f"Could not retrieve runs for experiment {exp.name}: {e}")
    
    def _display_best_run(self, runs) -> None:
        """Display best run information if available."""
        if "metrics.final_reward" in runs.columns:
            best_run = runs.loc[runs["metrics.final_reward"].idxmax()]
            if best_run["metrics.final_reward"] is not None:
                self.console.print(
                    f"  Best reward: {best_run['metrics.final_reward']:.4f}"
                )