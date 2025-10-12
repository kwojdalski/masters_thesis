"""
Centralized logging package for the Weles project.

This package provides consistent logging configuration and utilities
across all Weles components including tardis_downloader, market_data_fetcher,
pyth_downloader, dex_arbs, and other modules.
"""

from logger.core import (
    ColoredFormatter,
    StructuredFormatter,
    configure_weles_logging,
    get_logger,
    setup_component_logger,
    setup_logging,
)
from logger.utils import (
    LogContext,
    log_dataframe_info,
    log_error_with_context,
    log_function_call,
    log_performance_metrics,
    log_processing_step,
)

__all__ = [
    "ColoredFormatter",
    "LogContext",
    "StructuredFormatter",
    "configure_weles_logging",
    "get_logger",
    "log_dataframe_info",
    "log_error_with_context",
    "log_function_call",
    "log_performance_metrics",
    "log_processing_step",
    "setup_component_logger",
    "setup_logging",
]
