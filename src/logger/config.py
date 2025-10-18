"""
Configuration management for the logging package.

This module provides configuration options and defaults for logging
across different components of the project.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class LoggingConfig:
    """Configuration class for logging settings."""

    level: str = "INFO"
    log_dir: str = "logs"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    console_output: bool = True
    colored_output: bool = True
    structured_logging: bool = False
    include_function_info: bool = True

    def __post_init__(self):
        """Validate configuration after initialization."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.level.upper() not in valid_levels:
            raise ValueError(
                f"Invalid log level: {self.level}. Must be one of {valid_levels}"
            )


# Component-specific configurations
COMPONENT_CONFIGS: dict[str, LoggingConfig] = {
    "tardis_downloader": LoggingConfig(
        level=os.getenv("TARDIS_LOG_LEVEL", "INFO"),
        log_dir="logs/tardis_downloader",
        structured_logging=os.getenv("TARDIS_STRUCTURED_LOGS", "false").lower()
        == "true",
    ),
    "market_data_fetcher": LoggingConfig(
        level=os.getenv("MARKET_DATA_LOG_LEVEL", "INFO"),
        log_dir="logs/market_data_fetcher",
        structured_logging=os.getenv("MARKET_DATA_STRUCTURED_LOGS", "false").lower()
        == "true",
    ),
    "pyth_downloader": LoggingConfig(
        level=os.getenv("PYTH_LOG_LEVEL", "INFO"),
        log_dir="logs/pyth_downloader",
        structured_logging=os.getenv("PYTH_STRUCTURED_LOGS", "false").lower() == "true",
    ),
    "dex_arbs": LoggingConfig(
        level=os.getenv("DEX_ARBS_LOG_LEVEL", "INFO"),
        log_dir="logs/dex_arbs",
        structured_logging=os.getenv("DEX_ARBS_STRUCTURED_LOGS", "false").lower()
        == "true",
    ),
    "price_shift_cost": LoggingConfig(
        level=os.getenv("PRICE_SHIFT_COST_LOG_LEVEL", "INFO"),
        log_dir="logs/price_shift_cost",
    ),
    "simulation": LoggingConfig(
        level=os.getenv("SIMULATION_LOG_LEVEL", "INFO"), log_dir="logs/simulation"
    ),
    "compare_oracles": LoggingConfig(
        level=os.getenv("ORACLE_LOG_LEVEL", "INFO"), log_dir="logs/compare_oracles"
    ),
}


def get_component_config(component: str) -> LoggingConfig:
    """
    Get logging configuration for a specific component.

    Args:
        component: Component name

    Returns:
        LoggingConfig instance for the component
    """
    return COMPONENT_CONFIGS.get(component, LoggingConfig())


def get_global_config() -> LoggingConfig:
    """
    Get global logging configuration from environment variables.

    Returns:
        LoggingConfig instance with global settings
    """
    return LoggingConfig(
        level=os.getenv("RL_LOG_LEVEL", "INFO"),
        log_dir=os.getenv("RL_LOG_DIR", "logs"),
        max_file_size=int(os.getenv("RL_LOG_MAX_SIZE", str(10 * 1024 * 1024))),
        backup_count=int(os.getenv("RL_LOG_BACKUP_COUNT", "5")),
        console_output=os.getenv("RL_CONSOLE_LOGS", "true").lower() == "true",
        colored_output=os.getenv("RL_COLORED_LOGS", "true").lower() == "true",
        structured_logging=os.getenv("RL_STRUCTURED_LOGS", "false").lower() == "true",
    )


def create_component_logger_config(
    component: str,
    submodule: str | None = None,
    override_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Create a logger configuration dictionary for a specific component.

    Args:
        component: Component name
        submodule: Optional submodule name
        override_config: Optional configuration overrides

    Returns:
        Configuration dictionary ready for logger setup
    """
    base_config = get_component_config(component)

    if override_config:
        for key, value in override_config.items():
            if hasattr(base_config, key):
                setattr(base_config, key, value)

    logger_name = f"{component}.{submodule}" if submodule else component

    log_file = None
    if base_config.log_dir:
        log_dir = Path(base_config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{logger_name.replace('.', '_')}.log"

    return {
        "name": logger_name,
        "level": base_config.level,
        "log_file": str(log_file) if log_file else None,
        "console_output": base_config.console_output,
        "colored_output": base_config.colored_output,
        "structured_logging": base_config.structured_logging,
        "max_file_size": base_config.max_file_size,
        "backup_count": base_config.backup_count,
    }
