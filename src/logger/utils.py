"""
Utility functions for common logging tasks across the project.

This module provides helper functions that make logging more convenient
and consistent across different components.
"""

import functools
import logging
import time
from collections.abc import Callable
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any


def log_dataframe_info(
    logger: logging.Logger, df, name: str = "DataFrame", level: str = "INFO"
):
    """
    Log comprehensive information about a pandas DataFrame.

    Args:
        logger: Logger instance
        df: Pandas DataFrame
        name: Name to use in log messages
        level: Logging level to use
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    if not logger.isEnabledFor(log_level):
        return

    logger.log(log_level, f"{name} shape: {df.shape}")
    logger.log(log_level, f"{name} columns: {df.columns.tolist()}")

    if hasattr(df, "dtypes"):
        logger.debug(f"{name} dtypes: {df.dtypes.to_dict()}")

    if hasattr(df, "memory_usage"):
        try:
            total_memory = df.memory_usage(deep=True).sum()
            logger.debug(f"{name} memory usage: {total_memory / 1024 / 1024:.2f} MB")
        except Exception as e:
            logger.debug(f"Could not calculate memory usage for {name}: {e}")

    if hasattr(df, "isnull"):
        null_counts = df.isnull().sum()
        if null_counts.any():
            logger.debug(
                f"{name} null values: {null_counts[null_counts > 0].to_dict()}"
            )

    # Log sample data if DEBUG level
    if logger.isEnabledFor(logging.DEBUG) and hasattr(df, "head"):
        logger.debug(f"{name} sample data:\n{df.head()}")


def log_processing_step(
    logger: logging.Logger,
    step: str,
    details: str | None = None,
    extra_data: dict[str, Any] | None = None,
):
    """
    Log a processing step with consistent formatting and optional structured data.

    Args:
        logger: Logger instance
        step: Description of the processing step
        details: Optional additional details
        extra_data: Optional structured data to include
    """
    message = f"Processing step: {step}"
    if details:
        message += f" - {details}"

    if extra_data:
        # Create a custom LogRecord with extra data for structured logging
        extra_record = logging.LogRecord(
            name=logger.name,
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg=message,
            args=(),
            exc_info=None,
        )
        extra_record.extra_data = extra_data
        logger.handle(extra_record)
    else:
        logger.info(message)


def log_error_with_context(
    logger: logging.Logger,
    error: Exception,
    context: str,
    extra_data: dict[str, Any] | None = None,
):
    """
    Log an error with additional context information and structured data.

    Args:
        logger: Logger instance
        error: Exception that occurred
        context: Context description where the error occurred
        extra_data: Optional structured data for debugging
    """
    error_msg = f"Error in {context}: {type(error).__name__}: {error!s}"

    if extra_data:
        extra_record = logging.LogRecord(
            name=logger.name,
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg=error_msg,
            args=(),
            exc_info=None,
        )
        extra_record.extra_data = extra_data
        logger.handle(extra_record)
    else:
        logger.error(error_msg)

    logger.debug(f"Full error details for {context}:", exc_info=True)


def log_function_call(
    logger: logging.Logger,
    func_name: str,
    args: tuple | None = None,
    kwargs: dict[str, Any] | None = None,
    level: str = "DEBUG",
):
    """
    Log function call details.

    Args:
        logger: Logger instance
        func_name: Name of the function being called
        args: Function arguments (will be truncated if too long)
        kwargs: Function keyword arguments (will be truncated if too long)
        level: Logging level to use
    """
    log_level = getattr(logging, level.upper(), logging.DEBUG)

    if not logger.isEnabledFor(log_level):
        return

    call_info = f"Calling function: {func_name}"

    if args:
        # Truncate long arguments for readability
        args_str = str(args)
        if len(args_str) > 200:
            args_str = args_str[:200] + "..."
        call_info += f" with args: {args_str}"

    if kwargs:
        kwargs_str = str(kwargs)
        if len(kwargs_str) > 200:
            kwargs_str = kwargs_str[:200] + "..."
        call_info += f" with kwargs: {kwargs_str}"

    logger.log(log_level, call_info)


def log_performance_metrics(
    logger: logging.Logger,
    operation: str,
    duration: float,
    extra_metrics: dict[str, int | float | str] | None = None,
):
    """
    Log performance metrics for operations.

    Args:
        logger: Logger instance
        operation: Description of the operation
        duration: Duration in seconds
        extra_metrics: Optional additional metrics (e.g., rows_processed, memory_used)
    """
    perf_msg = f"Performance - {operation}: {duration:.3f}s"

    extra_data = {
        "operation": operation,
        "duration_seconds": duration,
        "timestamp": datetime.now(tz=UTC).isoformat(),
    }

    if extra_metrics:
        extra_data.update(extra_metrics)
        perf_msg += f" | Metrics: {extra_metrics}"

    # Create structured log entry
    extra_record = logging.LogRecord(
        name=logger.name,
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg=perf_msg,
        args=(),
        exc_info=None,
    )
    extra_record.extra_data = extra_data
    logger.handle(extra_record)


@contextmanager
def LogContext(
    logger: logging.Logger,
    operation: str,
    log_start: bool = True,
    log_end: bool = True,
    log_performance: bool = True,
    level: str = "INFO",
):
    """
    Context manager for logging operation start/end and performance.

    Args:
        logger: Logger instance
        operation: Description of the operation
        log_start: Whether to log operation start
        log_end: Whether to log operation end
        log_performance: Whether to log performance metrics
        level: Logging level to use

    Usage:
        with LogContext(logger, "Data processing"):
            # Your code here
            pass
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    start_time = time.time()

    if log_start and logger.isEnabledFor(log_level):
        logger.log(log_level, f"Starting: {operation}")

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

        if log_end and logger.isEnabledFor(log_level):
            logger.log(log_level, f"Completed: {operation}")

        if log_performance:
            log_performance_metrics(logger, operation, duration)


def logged_function(
    logger: logging.Logger | None = None,
    level: str = "DEBUG",
    log_args: bool = False,
    log_result: bool = False,
    log_performance: bool = True,
):
    """
    Decorator to automatically log function calls and performance.

    Args:
        logger: Logger instance (if None, will get logger from function's module)
        level: Logging level to use
        log_args: Whether to log function arguments
        log_result: Whether to log function result
        log_performance: Whether to log performance metrics

    Usage:
        @logged_function(logger=my_logger, log_performance=True)
        def my_function(arg1, arg2):
            return "result"
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get logger if not provided
            func_logger = logger or logging.getLogger(func.__module__)

            # Log function call
            if log_args:
                log_function_call(func_logger, func.__name__, args, kwargs, level)
            else:
                log_function_call(func_logger, func.__name__, level=level)

            # Execute function with performance tracking
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time

                if log_performance:
                    log_performance_metrics(
                        func_logger, f"Function {func.__name__}", duration
                    )

                if log_result:
                    result_str = str(result)
                    if len(result_str) > 100:
                        result_str = result_str[:100] + "..."
                    func_logger.debug(
                        f"Function {func.__name__} returned: {result_str}"
                    )

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
) -> logging.Logger:
    """
    Set up a logger specifically for a component/submodule combination.

    This is useful for getting consistent logger names across the project.

    Args:
        component: Main component name (e.g., 'tardis_downloader')
        submodule: Optional submodule name (e.g., 'utils', 'analysis')
        level: Logging level

    Returns:
        Configured logger
    """
    if submodule:
        logger_name = f"{component}.{submodule}"
    else:
        logger_name = component

    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    return logger
