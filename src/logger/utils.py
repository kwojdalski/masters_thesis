"""
Utility functions for common logging tasks across the project.

This module provides helper functions that make logging more convenient
and consistent across different components.
"""

import functools
import time
from collections.abc import Callable
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from loguru import logger as _loguru_logger
from rich.console import Console
from rich.table import Table

from logger.core import is_level_enabled

_GREEN = "\033[32m"
_RESET = "\033[0m"
_BANNER_WIDTH = 100


def log_banner(logger: Any, message: str) -> None:
    """Log a fully green separator banner at INFO level."""
    inner = f"  {message}  "
    dashes = max(0, _BANNER_WIDTH - len(inner))
    left = dashes // 2
    right = dashes - left
    line = "=" * left + inner + "=" * right
    logger.info("{}{}{}", _GREEN, line, _RESET)


def log_dataframe_info(
    logger: Any, df: Any, name: str = "DataFrame", level: str = "INFO"
) -> None:
    """Log comprehensive information about a pandas DataFrame."""
    if not is_level_enabled(level.upper()):
        return

    logger.log(level.upper(), "dataframe shape name={} n_rows={} n_cols={}", name, *df.shape)
    logger.log(level.upper(), "dataframe columns name={} columns={}", name, df.columns.tolist())

    if hasattr(df, "dtypes"):
        logger.trace("dataframe dtypes name={} dtypes={}", name, df.dtypes.to_dict())

    if hasattr(df, "memory_usage"):
        try:
            total_memory = df.memory_usage(deep=True).sum()
            logger.trace("dataframe memory name={} mb={:.2f}", name, total_memory / 1024 / 1024)
        except Exception as e:
            logger.trace("dataframe memory error name={} err={}", name, e)

    if hasattr(df, "isnull"):
        null_counts = df.isnull().sum()
        if null_counts.any():
            logger.trace("dataframe nulls name={} nulls={}", name, null_counts[null_counts > 0].to_dict())

    if is_level_enabled("TRACE") and hasattr(df, "head"):
        logger.trace("dataframe sample name={}\n{}", name, df.head())


def log_processing_step(
    logger: Any,
    step: str,
    details: str | None = None,
    extra_data: dict[str, Any] | None = None,
) -> None:
    """Log a processing step with consistent formatting."""
    message = f"Processing step: {step}"
    if details:
        message += f" - {details}"
    if extra_data:
        message += f" | {extra_data}"
    logger.info("{}", message)


def log_error_with_context(
    logger: Any,
    error: Exception,
    context: str,
    extra_data: dict[str, Any] | None = None,
) -> None:
    """Log an error with additional context information."""
    error_msg = f"Error in {context}: {type(error).__name__}: {error!s}"
    if extra_data:
        error_msg += f" | {extra_data}"
    logger.error("{}", error_msg)
    logger.opt(exception=True).debug("error details context={}", context)


def log_function_call(
    logger: Any,
    func_name: str,
    args: tuple | None = None,
    kwargs: dict[str, Any] | None = None,
    level: str = "DEBUG",
) -> None:
    """Log function call details."""
    if not is_level_enabled(level.upper()):
        return

    call_info = f"Calling function: {func_name}"

    if args:
        args_str = str(args)
        if len(args_str) > 200:
            args_str = args_str[:200] + "..."
        call_info += f" with args: {args_str}"

    if kwargs:
        kwargs_str = str(kwargs)
        if len(kwargs_str) > 200:
            kwargs_str = kwargs_str[:200] + "..."
        call_info += f" with kwargs: {kwargs_str}"

    logger.log(level.upper(), "{}", call_info)


def log_performance_metrics(
    logger: Any,
    operation: str,
    duration: float,
    extra_metrics: dict[str, int | float | str] | None = None,
) -> None:
    """Log performance metrics for operations."""
    perf_msg = f"Performance - {operation}: {duration:.3f}s"
    if extra_metrics:
        perf_msg += f" | Metrics: {extra_metrics}"
    logger.info("{}", perf_msg)


@contextmanager
def LogContext(
    logger: Any,
    operation: str,
    log_start: bool = True,
    log_end: bool = True,
    log_performance: bool = True,
    level: str = "INFO",
):
    """Context manager for logging operation start/end and performance."""
    start_time = time.time()

    if log_start and is_level_enabled(level.upper()):
        logger.log(level.upper(), "start operation={}", operation)

    try:
        yield
    except Exception as e:
        duration = time.time() - start_time
        log_error_with_context(
            logger, e, operation, {"duration_seconds": duration, "operation": operation}
        )
        raise
    finally:
        duration = time.time() - start_time

        if log_end and is_level_enabled(level.upper()):
            logger.log(level.upper(), "complete operation={}", operation)

        if log_performance:
            log_performance_metrics(logger, operation, duration)


def logged_function(
    logger: Any | None = None,
    level: str = "DEBUG",
    log_args: bool = False,
    log_result: bool = False,
    log_performance: bool = True,
) -> Callable:
    """Decorator to automatically log function calls and performance."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            func_logger = logger or _loguru_logger

            if log_args:
                log_function_call(func_logger, func.__name__, args, kwargs, level)
            else:
                log_function_call(func_logger, func.__name__, level=level)

            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time

                if log_performance:
                    log_performance_metrics(func_logger, f"Function {func.__name__}", duration)

                if log_result:
                    result_str = str(result)
                    if len(result_str) > 100:
                        result_str = result_str[:100] + "..."
                    func_logger.debug("Function {} returned: {}", func.__name__, result_str)

                return result

            except Exception as e:
                duration = time.time() - start_time
                log_error_with_context(
                    func_logger,
                    e,
                    f"Function {func.__name__}",
                    {"duration_seconds": duration, "function": func.__name__},
                )
                raise

        return wrapper

    return decorator


def setup_component_specific_logger(
    component: str, submodule: str | None = None, level: str = "INFO"
) -> Any:
    """Return the loguru logger (name parameters kept for API compat)."""
    return _loguru_logger


def _fmt_cell(value: Any) -> str:
    if pd.isna(value):
        return "NaN"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def print_df_head(
    df: pd.DataFrame,
    n_rows: int = 5,
    title: str = "DataFrame Head",
    max_columns: int = 12,
    paginate: bool = False,
    columns_per_page: int = 10,
    transpose: bool = False,
) -> None:
    """Print the first n_rows of a DataFrame as a Rich table."""
    if df.empty:
        return

    console = Console()
    head_df = df.head(n_rows)
    all_columns = list(head_df.columns)
    index_labels = [str(i) for i in head_df.index]

    if transpose:
        def _build_transposed_table(col_chunk: list[str], page_info: str) -> Table:
            t = Table(title=f"{title}{page_info}")
            t.add_column("Column", style="cyan")
            for lbl in index_labels:
                t.add_column(lbl, justify="right")
            for col in col_chunk:
                t.add_row(str(col), *[_fmt_cell(head_df.at[idx, col]) for idx in head_df.index])
            return t

        if paginate:
            n_pages = max(1, -(-len(all_columns) // columns_per_page))
            for page, start in enumerate(range(0, len(all_columns), columns_per_page), 1):
                chunk = all_columns[start : start + columns_per_page]
                info = f" — rows {start + 1}–{start + len(chunk)} of {len(all_columns)} (page {page}/{n_pages})"
                console.print(_build_transposed_table(chunk, info))
        else:
            visible = all_columns[:max_columns]
            hidden = len(all_columns) - len(visible)
            console.print(_build_transposed_table(visible, ""))
            if hidden > 0:
                console.print(
                    f"[dim]Showing {len(visible)} of {len(all_columns)} columns "
                    f"({hidden} hidden). Pass paginate=True to see all.[/dim]"
                )

    else:
        def _build_table(col_chunk: list[str], page_info: str) -> Table:
            t = Table(title=f"{title}{page_info}")
            t.add_column("index", style="cyan")
            for col in col_chunk:
                t.add_column(str(col), justify="right")
            for idx, row in head_df.iterrows():
                t.add_row(str(idx), *[_fmt_cell(row[col]) for col in col_chunk])
            return t

        if paginate:
            n_pages = max(1, -(-len(all_columns) // columns_per_page))
            for page, start in enumerate(range(0, len(all_columns), columns_per_page), 1):
                chunk = all_columns[start : start + columns_per_page]
                info = f" — columns {start + 1}–{start + len(chunk)} of {len(all_columns)} (page {page}/{n_pages})"
                console.print(_build_table(chunk, info))
        else:
            visible = all_columns[:max_columns]
            hidden = len(all_columns) - len(visible)
            console.print(_build_table(visible, ""))
            if hidden > 0:
                console.print(
                    f"[dim]Showing {len(visible)} of {len(all_columns)} columns "
                    f"({hidden} hidden). Pass paginate=True to see all.[/dim]"
                )
