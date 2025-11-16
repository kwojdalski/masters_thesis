# Weles Centralized Logging Package

This package provides consistent, comprehensive logging functionality across all components of the Weles project, including `tardis_downloader`, `market_data_fetcher`, `pyth_downloader`, `dex_arbs`, and other modules.

## Features

- **Centralized Configuration**: Consistent logging setup across all components
- **Colored Console Output**: Enhanced readability with color-coded log levels
- **Structured Logging**: JSON format support for machine-readable logs
- **File Rotation**: Automatic log file rotation to manage disk space
- **Performance Tracking**: Built-in performance monitoring utilities
- **Context Management**: Easy start/end logging with automatic performance tracking
- **Decorator Support**: Automatic function call logging and performance tracking
- **DataFrame Utilities**: Specialized logging for pandas DataFrame operations
- **Environment Configuration**: Component-specific settings via environment variables

## Quick Start

### Basic Usage

```python
from logger import get_logger, configure_logging

# Set up logging for your component
configure_logging(
    component="tardis_downloader",
    debug=False,
    log_dir="logs",
    structured_logging=False
)

# Get a logger instance
logger = get_logger(__name__)

# Use it
logger.info("Processing started")
logger.warning("This is a warning")
logger.error("An error occurred")
```

### Component-Specific Setup

```python
from logger import configure_logging, get_logger

# Each component can have its own configuration
configure_logging("tardis_downloader", debug=True, log_dir="logs/tardis")
configure_logging("market_data_fetcher", structured_logging=True)
configure_logging("pyth_downloader", debug=False)

# Get component-specific loggers
tardis_logger = get_logger("tardis_downloader.utils")
market_logger = get_logger("market_data_fetcher.core")
pyth_logger = get_logger("pyth_downloader.analyzer")
```

## Advanced Features

### DataFrame Logging

```python
from logger import get_logger, log_dataframe_info
import pandas as pd

logger = get_logger(__name__)
df = pd.DataFrame({"price": [1.0, 1.1, 1.2], "volume": [100, 200, 150]})

# Automatically log DataFrame information
log_dataframe_info(logger, df, "Market Data")
```

### Performance Tracking with Context Manager

```python
from logger import get_logger, LogContext

logger = get_logger(__name__)

# Define example functions for demonstration
def process_data():
    print("Processing data...")

def validate_results():
    print("Validating results...")

with LogContext(logger, "Data processing pipeline", log_performance=True):
    # Your code here
    process_data()
    validate_results()
    # Automatically logs start, end, and performance metrics
```

### Function Decorators

```python
from logger import get_logger, log_function_call

logger = get_logger(__name__)

def process_market_data(symbol: str, exchange: str):
    # Log function call with performance tracking
    log_function_call(logger, "process_market_data", {"symbol": symbol, "exchange": exchange})

    # Function logic here
    result = {"processed": True, "symbol": symbol, "exchange": exchange}

    logger.info(f"Processed market data: {result}")
    return result

# Example usage
result = process_market_data("BTCUSDT", "binance")
```

### Structured Logging

```python
from logger import configure_logging, get_logger, log_processing_step

# Enable structured (JSON) logging
configure_logging("market_analysis", structured_logging=True)
logger = get_logger(__name__)

# Logs will be formatted as JSON
log_processing_step(
    logger,
    "Price aggregation",
    extra_data={
        "sources": ["binance", "coinbase"],
        "method": "median",
        "symbol": "USDC/USD"
    }
)
```

### Error Handling with Context

```python
from logger import get_logger, log_error_with_context

logger = get_logger(__name__)

try:
    risky_operation()
except Exception as e:
    log_error_with_context(
        logger,
        e,
        "Price calculation",
        extra_data={
            "symbol": "USDC/USD",
            "exchange": "binance",
            "step": "aggregation"
        }
    )
```

## Environment Configuration

You can configure logging behavior using environment variables:

```bash
# Global settings
export WELES_LOG_LEVEL=DEBUG
export WELES_LOG_DIR=./logs
export WELES_STRUCTURED_LOGS=true
export WELES_COLORED_LOGS=false

# Component-specific settings
export TARDIS_LOG_LEVEL=INFO
export TARDIS_STRUCTURED_LOGS=true
export MARKET_DATA_LOG_LEVEL=DEBUG
export PYTH_LOG_LEVEL=WARNING
export DEX_ARBS_LOG_LEVEL=INFO
```

## Log Levels

- **DEBUG**: Detailed information for debugging
- **INFO**: General information about program execution
- **WARNING**: Warning messages for potentially harmful situations
- **ERROR**: Error messages for serious problems
- **CRITICAL**: Critical errors that may cause program termination

## File Organization

Logs are organized by component in separate directories:

```
logs/
├── tardis_downloader/
│   ├── tardis_downloader_20240101.log
│   └── tardis_downloader_utils_20240101.log
├── market_data_fetcher/
│   └── market_data_fetcher_20240101.log
├── pyth_downloader/
│   └── pyth_downloader_20240101.log
└── dex_arbs/
    └── dex_arbs_20240101.log
```

## Integration with Existing Components

### Updating tardis_downloader

Replace existing logging setup:

```python
# Old way
import logging
logger = logging.getLogger(__name__)

# New way
from logger import get_logger, configure_logging

configure_logging("tardis_downloader")
logger = get_logger(__name__)
```

### Updating market_data_fetcher

```python
import pandas as pd
from logger import get_logger, configure_logging, log_dataframe_info

configure_logging("market_data_fetcher", debug=True)
logger = get_logger(__name__)

# Create sample market data for demonstration
market_df = pd.DataFrame({
    'timestamp': ['2024-01-01', '2024-01-02'],
    'price': [50000, 51000],
    'volume': [100, 150]
})

# Enhanced DataFrame logging
log_dataframe_info(logger, market_df, "Market Data")
```

## Performance Considerations

- Log files are automatically rotated when they exceed 10MB (configurable)
- Up to 5 backup files are kept by default (configurable)
- Structured logging has minimal performance overhead
- DEBUG level logging can be disabled in production for better performance

## Migration Guide

1. **Install the package**: The logger package is now available at `src/logger`

2. **Update imports**: Replace standard logging imports:
   ```python
   # Before
   import logging
   logger = logging.getLogger(__name__)

   # After
   from logger import get_logger, configure_logging
   configure_logging("your_component")
   logger = get_logger(__name__)
   ```

3. **Configure per component**: Add component-specific configuration at the start of your main modules

4. **Enhance logging**: Use the utility functions for better structured logging:
   ```python
   from logger import log_dataframe_info, log_processing_step, LogContext
   ```

## Comprehensive Examples

### Basic Usage Example

```python
from logger import get_logger, configure_logging

# Basic logger setup
logger = get_logger(__name__)

# Or set up component-specific logging
configure_logging(
    component="tardis_downloader",
    debug=True,
    log_dir="logs",
    structured_logging=False,
)

logger.info("This is a basic info message")
logger.debug("This is a debug message")
logger.warning("This is a warning message")
```

### DataFrame Logging Example

```python
import pandas as pd
from logger import get_logger, log_dataframe_info, log_processing_step

logger = get_logger(__name__)

# Create sample DataFrame
df = pd.DataFrame(
    {
        "timestamp": pd.date_range("2024-01-01", periods=1000, freq="1min"),
        "price": [1.0 + i * 0.001 for i in range(1000)],
        "volume": [100 + i for i in range(1000)],
    }
)

# Log DataFrame info
log_dataframe_info(logger, df, "Market Data")

# Process the data
log_processing_step(logger, "Data filtering", "Filtering by volume > 500")

filtered_df = df[df["volume"] > 500]
log_dataframe_info(logger, filtered_df, "Filtered Data")
```

### Error Handling Example

```python
from logger import get_logger, log_error_with_context

logger = get_logger(__name__)

try:
    # Simulate some processing that might fail
    data = {"symbol": "USDC/USD", "exchange": "binance"}
    result = 1 / 0  # This will raise ZeroDivisionError

except Exception as e:
    log_error_with_context(
        logger,
        e,
        "Price calculation",
        extra_data={
            "symbol": data.get("symbol"),
            "exchange": data.get("exchange"),
            "step": "division_operation",
        },
    )
```

### Performance Logging Example

```python
import time
from logger import get_logger, log_performance_metrics

logger = get_logger(__name__)

# Manual performance logging
start_time = time.time()
time.sleep(0.1)  # Simulate work
duration = time.time() - start_time

log_performance_metrics(
    logger,
    "Data processing",
    duration,
    extra_metrics={"rows_processed": 1000, "memory_mb": 50.5, "cpu_percent": 25.3},
)
```

### Context Manager Example

```python
import time
from logger import get_logger, LogContext, log_processing_step

logger = get_logger(__name__)

# Using context manager for automatic start/end logging
with LogContext(logger, "Complete data pipeline", log_performance=True):
    log_processing_step(logger, "Data loading")
    time.sleep(0.05)  # Simulate loading

    log_processing_step(logger, "Data transformation")
    time.sleep(0.03)  # Simulate transformation

    log_processing_step(logger, "Data validation")
    time.sleep(0.02)  # Simulate validation
```

### Decorated Function Example

```python
import time
from logger import logged_function

@logged_function(log_performance=True, log_args=True)
def process_market_data(symbol: str, exchange: str, timeframe: str = "1h"):
    """Function with automatic logging of calls, performance, and errors."""
    
    time.sleep(0.01)  # Simulate work
    
    return {
        "symbol": symbol,
        "exchange": exchange,
        "timeframe": timeframe,
        "data_points": 100,
    }

# Usage
result = process_market_data("USDC/USD", "binance", "5m")
```

### Structured Logging Example

```python
from logger import configure_logging, get_logger, log_processing_step

# Configure with structured logging
configure_logging(
    component="market_analysis", structured_logging=True, include_console=True
)

logger = get_logger(__name__)

# Regular log message - will be formatted as JSON
logger.info("Processing market data")

# Log with structured data
log_processing_step(
    logger,
    "Price aggregation",
    extra_data={
        "sources": ["binance", "coinbase", "kraken"],
        "method": "median",
        "timestamp": "2024-01-01T12:00:00Z",
        "symbol": "USDC/USD",
    },
)
```

### Component-Specific Configuration Example

```python
from logger import get_logger, configure_logging

# Tardis downloader logging
tardis_logger = get_logger("tardis_downloader.utils")
configure_logging("tardis_downloader", debug=False, log_dir="logs/tardis")

# Market data fetcher logging
market_logger = get_logger("market_data_fetcher.core")
configure_logging(
    "market_data_fetcher", debug=True, log_dir="logs/market_data"
)

# Pyth downloader logging
pyth_logger = get_logger("pyth_downloader.analyzer")
configure_logging("pyth_downloader", debug=False, structured_logging=True)

# Each logger will have component-specific configuration
tardis_logger.info("Downloading Tardis data")
market_logger.debug("Fetching market data")
pyth_logger.info("Analyzing Pyth network data")
```

## Best Practices

1. **Configure early**: Set up logging configuration at the start of your application
2. **Use meaningful names**: Use descriptive logger names that reflect the component/module
3. **Log at appropriate levels**: Use DEBUG for development, INFO for general flow, WARNING/ERROR for issues
4. **Include context**: Use the structured logging features to include relevant context
5. **Monitor performance**: Use the performance tracking features for critical operations
6. **Handle errors gracefully**: Use `log_error_with_context` for better error reporting
