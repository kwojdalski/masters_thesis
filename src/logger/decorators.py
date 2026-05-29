"""Debug decorators for function call tracing.

Provides decorators for tracing function execution during debugging.
"""

import functools
import threading
import time
from typing import Any, Callable

from loguru import logger

from logger.core import is_level_enabled

# Thread-local storage for call depth tracking
_call_depth = threading.local()


def _get_call_depth() -> int:
    if not hasattr(_call_depth, "depth"):
        _call_depth.depth = 0
    return _call_depth.depth


def _set_call_depth(depth: int) -> None:
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

            env_log_level = os.getenv("LOG_LEVEL", "").upper()
            is_debug = is_level_enabled("DEBUG") or env_log_level == "DEBUG"

            if not is_debug:
                return func(*args, **kwargs)

            depth = _get_call_depth()
            indent = "  " * depth
            arrow_in = "→" if depth == 0 else "↳"
            arrow_out = "←"

            func_name = func.__name__
            module_name = func.__module__.split(".")[-1]

            args_repr = []
            for i, arg in enumerate(args):
                arg_str = _format_arg_value(arg)
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

            for key, value in kwargs.items():
                args_repr.append(f"{key}={_format_arg_value(value)}")

            args_str = ", ".join(args_repr)

            logger.debug("{}{} [TRACE] {}.{}({})", indent, arrow_in, module_name, func_name, args_str)

            _set_call_depth(depth + 1)
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time

                if show_return:
                    logger.debug(
                        "{}{} [TRACE] {}.{} returned {} ({:.3f}s)",
                        indent, arrow_out, module_name, func_name,
                        _format_arg_value(result), execution_time,
                    )
                else:
                    logger.debug(
                        "{}{} [TRACE] {}.{} completed ({:.3f}s)",
                        indent, arrow_out, module_name, func_name, execution_time,
                    )

                return result

            except Exception as e:
                execution_time = time.time() - start_time
                logger.debug(
                    "{}{} [TRACE] {}.{} raised {}: {} ({:.3f}s)",
                    indent, arrow_out, module_name, func_name,
                    type(e).__name__, e, execution_time,
                )
                raise

            finally:
                _set_call_depth(depth)

        return wrapper

    return decorator


def _format_arg_value(value: Any, max_length: int = 60) -> str:
    """Format argument value for logging (with truncation)."""
    try:
        if isinstance(value, str):
            if len(value) > max_length:
                return f"'{value[:max_length]}...'"
            return f"'{value}'"

        elif isinstance(value, (int, float, bool, type(None))):
            return str(value)

        elif hasattr(value, "__class__"):
            class_name = value.__class__.__name__

            if class_name == "DataFrame":
                shape = getattr(value, "shape", None)
                if shape:
                    return f"DataFrame({shape[0]} rows × {shape[1]} cols)"
                return "DataFrame(...)"

            elif class_name in ("Path", "PosixPath", "WindowsPath"):
                path_str = str(value)
                if len(path_str) > max_length:
                    return f"Path('...{path_str[-max_length:]}')"
                return f"Path('{path_str}')"

            elif "Config" in class_name:
                return f"{class_name}(...)"

            elif class_name in ("Tensor", "ndarray"):
                shape = getattr(value, "shape", None)
                if shape:
                    return f"{class_name}(shape={shape})"
                return f"{class_name}(...)"

            else:
                return f"{class_name}(...)"

        else:
            value_str = str(value)
            if len(value_str) > max_length:
                return f"{value_str[:max_length]}..."
            return value_str

    except Exception:
        return "<value>"
