"""Base command class with common functionality."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import typer
import yaml
from rich.console import Console

from logger import get_logger


class BaseCommand(ABC):
    """Base class for all CLI commands."""
    
    def __init__(self, console: Console):
        self.console = console
        self.logger = get_logger(self.__class__.__name__)
    
    def load_config(self, config_path: Path | None) -> dict[str, Any]:
        """Load configuration from YAML file."""
        if not config_path:
            return {}
        
        self.logger.info("Loading config from %s", config_path)
        with config_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    
    def handle_error(self, error: Exception, context: str) -> None:
        """Standardized error handling."""
        self.console.print(f"[bold red]{context} failed: {error}[/bold red]")
        self.logger.error(f"{context} failed", exc_info=error)
        raise typer.Exit(1) from error
    
    def resolve_seed(self, provided_seed: int | None) -> int:
        """Resolve seed value - use provided or generate random."""
        if provided_seed is not None:
            self.console.print(f"[blue]Using specified seed: {provided_seed}[/blue]")
            return provided_seed
        
        import random
        generated_seed = random.randint(1, 100000)  # noqa: S311
        self.console.print(f"[yellow]Using random seed: {generated_seed}[/yellow]")
        return generated_seed
    
    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """Execute the command."""
        pass