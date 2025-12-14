"""Debug decorators for function call tracing.

Provides decorators for tracing function execution during debugging.
"""

import functools
import logging
import threading
import time
from typing import Any, Callable

# Thread-local storage for call depth tracking
_call_depth = threading.local()


def _get_call_depth() -> int:
    """Get current call depth for indentation."""
    if not hasattr(_call_depth, "depth"):
        _call_depth.depth = 0
    return _call_depth.depth


def _set_call_depth(depth: int) -> None:
    """Set call depth for indentation."""
    _call_depth.depth = depth


def trace_calls(show_return: bool = False) -> Callable:
    """Decorator to trace function calls when LOG_LEVEL=DEBUG.

    Logs function entry with arguments, execution time, and optionally return values.
    Uses indentation to show call hierarchy.

    Args:
        show_return: If True, log return values (default: False)

    Usage:
        @trace_calls()
        def my_function(arg1, arg2):
            ...

        @trace_calls(show_return=True)
        def my_function_with_return(arg1):
            return result

    Example output (DEBUG mode):
        → [TRACE] run_single_experiment(custom_config=ExperimentConfig(...))
          ↳ [TRACE] setup_mlflow_experiment(config=..., experiment_name='test')
          ← [TRACE] setup_mlflow_experiment returned 'experiment_123' (0.023s)
          ↳ [TRACE] prepare_data(data_path='data.parquet', ...)
          ← [TRACE] prepare_data returned DataFrame(500 rows) (0.145s)
        ← [TRACE] run_single_experiment completed (45.2s)
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            import os

            # Get logger for the function's module
            logger = logging.getLogger(func.__module__)

            # Also check root logger level if module logger is not set
            root_logger = logging.getLogger()
            # Check environment variable directly in case logging isn't configured yet
            env_log_level = os.getenv("LOG_LEVEL", "").upper()
            is_debug = (
                logger.isEnabledFor(logging.DEBUG)
                or root_logger.isEnabledFor(logging.DEBUG)
                or env_log_level == "DEBUG"
            )

            # Only trace if DEBUG level is enabled
            if not is_debug:
                return func(*args, **kwargs)

            # Ensure logging is configured and at DEBUG level
            if not logging.getLogger().hasHandlers():
                # No handlers configured yet, use basicConfig
                logging.basicConfig(
                    level=logging.DEBUG,
                    format="%(levelname)s - %(name)s - %(message)s",
                )
            elif env_log_level == "DEBUG":
                # Env var says DEBUG - ensure logger and handlers are configured
                if not logger.isEnabledFor(logging.DEBUG):
                    logger.setLevel(logging.DEBUG)
                # Also ensure root logger handlers are at DEBUG level
                for handler in logging.getLogger().handlers:
                    if handler.level > logging.DEBUG:
                        handler.setLevel(logging.DEBUG)

            # Get current depth for indentation
            depth = _get_call_depth()
            indent = "  " * depth
            arrow_in = "→" if depth == 0 else "↳"
            arrow_out = "←"

            # Format function call
            func_name = func.__name__
            module_name = func.__module__.split(".")[-1]

            # Format arguments (truncate long values)
            args_repr = []

            # Add positional args
            for i, arg in enumerate(args):
                arg_str = _format_arg_value(arg)
                # Try to get parameter name from function signature
                try:
                    import inspect

                    sig = inspect.signature(func)
                    param_names = list(sig.parameters.keys())
                    if i < len(param_names):
                        args_repr.append(f"{param_names[i]}={arg_str}")
                    else:
                        args_repr.append(arg_str)
                except Exception:
                    args_repr.append(arg_str)

            # Add keyword args
            for key, value in kwargs.items():
                value_str = _format_arg_value(value)
                args_repr.append(f"{key}={value_str}")

            args_str = ", ".join(args_repr)

            # Log entry
            logger.debug(
                f"{indent}{arrow_in} [TRACE] {module_name}.{func_name}({args_str})"
            )

            # Increase depth for nested calls
            _set_call_depth(depth + 1)

            # Execute function and measure time
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time

                # Log exit with optional return value
                if show_return:
                    return_str = _format_arg_value(result)
                    logger.debug(
                        f"{indent}{arrow_out} [TRACE] {module_name}.{func_name} "
                        f"returned {return_str} ({execution_time:.3f}s)"
                    )
                else:
                    logger.debug(
                        f"{indent}{arrow_out} [TRACE] {module_name}.{func_name} "
                        f"completed ({execution_time:.3f}s)"
                    )

                return result

            except Exception as e:
                execution_time = time.time() - start_time
                logger.debug(
                    f"{indent}{arrow_out} [TRACE] {module_name}.{func_name} "
                    f"raised {type(e).__name__}: {e} ({execution_time:.3f}s)"
                )
                raise

            finally:
                # Restore depth
                _set_call_depth(depth)

        return wrapper

    return decorator


def _format_arg_value(value: Any, max_length: int = 60) -> str:
    """Format argument value for logging (with truncation).

    Args:
        value: Value to format
        max_length: Maximum string length before truncation

    Returns:
        Formatted string representation
    """
    try:
        # Handle common types with custom formatting
        if isinstance(value, str):
            if len(value) > max_length:
                return f"'{value[:max_length]}...'"
            return f"'{value}'"

        elif isinstance(value, (int, float, bool, type(None))):
            return str(value)

        elif hasattr(value, "__class__"):
            class_name = value.__class__.__name__

            # Handle DataFrames
            if class_name == "DataFrame":
                shape = getattr(value, "shape", None)
                if shape:
                    return f"DataFrame({shape[0]} rows × {shape[1]} cols)"
                return "DataFrame(...)"

            # Handle Path objects
            elif class_name in ("Path", "PosixPath", "WindowsPath"):
                path_str = str(value)
                if len(path_str) > max_length:
                    return f"Path('...{path_str[-max_length:]}')"
                return f"Path('{path_str}')"

            # Handle config objects
            elif "Config" in class_name:
                return f"{class_name}(...)"

            # Handle tensors
            elif class_name in ("Tensor", "ndarray"):
                shape = getattr(value, "shape", None)
                if shape:
                    return f"{class_name}(shape={shape})"
                return f"{class_name}(...)"

            # Default: just show class name
            else:
                return f"{class_name}(...)"

        else:
            # Fallback: convert to string and truncate
            value_str = str(value)
            if len(value_str) > max_length:
                return f"{value_str[:max_length]}..."
            return value_str

    except Exception:
        # If anything fails, just return a safe placeholder
        return "<value>"
