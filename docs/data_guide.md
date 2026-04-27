# Data Download and Generation Guide

This guide covers how to obtain data for training reinforcement learning agents in this project.

## Overview

The project supports three methods for obtaining training data:

1. **Download cryptocurrency data** - Real OHLCV data from exchanges (Bitfinex, Binance, etc.)
2. **Download stock data** - Real tick/OHLCV data from Databento or Polygon
3. **Generate synthetic data** - Programmatically created price patterns

## 1. Downloading Cryptocurrency Data

Download historical cryptocurrency data from multiple exchanges using `scripts/fetch_crypto.py`.

### Supported Exchanges

- `bitfinex2` - Bitfinex (high liquidity, good historical data)
- `binance` - Binance (largest exchange by volume)
- `kraken` - Kraken (US-based exchange)
- `coinbasepro` - Coinbase Pro (US-regulated)

### CLI Usage

```bash
# Download BTC/USDT from Bitfinex (default)
uv run python scripts/fetch_crypto.py download-crypto

# Download from Binance with custom timeframe
uv run python scripts/fetch_crypto.py download-crypto \
  --exchange binance \
  --symbols BTC/USDT \
  --timeframe 1h \
  --start-date 2024-01-01

# Download multiple pairs
uv run python scripts/fetch_crypto.py download-crypto \
  --symbols "BTC/USDT,ETH/USDT,SOL/USDT" \
  --timeframe 1d

# List available exchanges
uv run python scripts/fetch_crypto.py list-exchanges
```

### Interactive Mode

```bash
uv run python scripts/fetch_crypto.py download-crypto --interactive
```

### Output

Data is saved to `data/raw/crypto/` with naming pattern:
```
{exchange}-{symbol}-{timeframe}.parquet
```

Example: `bitfinex2-BTCUSDT-1h.parquet`

## 2. Downloading Stock Data

Download historical stock market data from Databento using `scripts/fetch_stocks.py`.

### Prerequisites

Set your Databento API key:
```bash
export DATABENTO_API_KEY='your-api-key-here'
```

Get your API key from: https://databento.com

### Supported Datasets

- `XNAS.ITCH` - NASDAQ
- `XNYS.TRADES` - NYSE

### Supported Schemas

**Aggregated (OHLCV):**
- `trades` - Individual trades aggregated to bars (1h, 1d, 5m)

**Raw (Tick/Order Book):**
- `trades` - Individual trades (tick data)
- `tbbo` - Top of book (best bid/offer)
- `mbp-1` - Market by price level 1
- `mbp-10` - Market by price level 10 (full order book depth)

### CLI Usage

```bash
# Download AAPL hourly OHLCV (default)
uv run python scripts/fetch_stocks.py download-stocks

# Download multiple stocks with custom dates
uv run python scripts/fetch_stocks.py download-stocks \
  --symbols "AAPL,MSFT,GOOGL" \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --timeframe 1h

# Download raw order book data
uv run python scripts/fetch_stocks.py download-stocks \
  --symbols AAPL \
  --raw \
  --schema mbp-10 \
  --start-date 2024-01-01 \
  --end-date 2024-01-31

# Download from NYSE
uv run python scripts/fetch_stocks.py download-stocks \
  --symbols IBM \
  --dataset XNYS.TRADES

# List available datasets
uv run python scripts/fetch_stocks.py list-datasets
```

### Interactive Mode

```bash
uv run python scripts/fetch_stocks.py download-stocks --interactive
```

### Output

Data is saved to `data/raw/stocks/` with naming pattern:

**Aggregated:**
```
{symbol}_{start_date}_{end_date}_{timeframe}.parquet
```

**Raw:**
```
{symbol}_{start_date}_{end_date}_raw_{schema}.parquet
```

Example: `AAPL_2024-01-01_2024-03-31_1h.parquet`

## 3. Generating Synthetic Data

Generate synthetic price data for testing and experimentation using the `PriceDataGenerator` class.

### Python API

```python
from data_generator import PriceDataGenerator

# Initialize generator
generator = PriceDataGenerator(
    source_dir="data/raw/binance",
    output_dir="data/raw/synthetic"
)

# Generate sine wave pattern (predictable)
df = generator.generate_sine_wave_pattern(
    output_file="BTCUSDT_2023.parquet",
    n_periods=5,
    samples_per_period=100,
    base_price=50000.0,
    amplitude=30.0,
    trend_slope=0.0,
    start_date="2023-01-01"
)

# Sample from existing data
df = generator.generate_synthetic_sample(
    source_file="binance-BTCUSDT-1h.parquet",
    output_file="BTCUSDT_sample.parquet",
    start_date="2024-01-01",
    end_date="2024-12-31",
    sample_size=1000
)

# Copy data without modifications
generator.copy_data(
    source_file="binance-BTCUSDT-1h.parquet",
    output_file="BTCUSDT_copy.parquet"
)
```

### Use Cases

**Sine Wave Pattern:**
- Predictable price movements for testing buy/sell strategies
- Useful for validating that agents can learn simple patterns
- Default configuration creates ~500 hourly candles with upward trend

**Sampling:**
- Create smaller datasets for faster iteration
- Extract specific date ranges
- Reproducible sampling (random_state=42)

**Copying:**
- Organize data into synthetic folder
- Rename files for clarity
- Keep original data intact

## Data Pipeline Integration

Once data is downloaded/generated, it flows through the training pipeline:

### 1. Data Loading

```yaml
# In scenario YAML config
data:
  data_path: "./data/raw/crypto/bitfinex2-BTCUSDT-1h.parquet"
  download_data: false  # Set to true to auto-download on each run
```

### 2. Feature Engineering

```yaml
data:
  feature_config: "src/configs/features/btc_with_volume.yaml"
```

Features are automatically computed and normalized. See [docs/prepare_data.md](./prepare_data.md) for details.

### 3. Train/Val/Test Split

Data is split BEFORE feature engineering to prevent data leakage:
- Training set: Fit feature transformations
- Validation set: Monitor overfitting
- Test set: Final evaluation

## Quick Start Examples

### Download BTC data and train

```bash
# 1. Download BTC data
uv run python scripts/fetch_crypto.py download-crypto \
  --exchange binance \
  --symbols BTC/USDT \
  --timeframe 1h \
  --start-date 2024-01-01

# 2. Update config to use downloaded data
# Edit src/configs/scenarios/your_scenario.yaml:
#   data_path: "./data/raw/crypto/binance-BTCUSDT-1h.parquet"

# 3. Train agent
uv run python src/cli.py train --config your_scenario
```

### Download stock data and train

```bash
# 1. Set API key
export DATABENTO_API_KEY='your-key'

# 2. Download AAPL data
uv run python scripts/fetch_stocks.py download-stocks \
  --symbols AAPL \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --timeframe 1h

# 3. Update config
# Edit scenario YAML:
#   data_path: "./data/raw/stocks/AAPL_2024-01-01_2024-12-31_1h.parquet"

# 4. Train agent
uv run python src/cli.py train --config your_scenario
```

### Generate synthetic data for testing

```python
from data_generator import PriceDataGenerator

gen = PriceDataGenerator(output_dir="data/raw/synthetic")

# Create predictable sine wave for testing
gen.generate_sine_wave_pattern(
    output_file="test_pattern.parquet",
    n_periods=3,
    samples_per_period=200,
    base_price=100.0,
    amplitude=10.0
)

# Use in config:
# data_path: "./data/raw/synthetic/test_pattern.parquet"
```

## See Also

- [Experiment Workflow](./experiment_workflow.md) - End-to-end training workflow
- [Data Preparation Details](./prepare_data.md) - Feature engineering and normalization
- [PPO Implementation](./ppo_implementation_overview.md) - PPO algorithm details
- [DDPG Implementation](./ddpg_implementation_overview.md) - DDPG algorithm details
- [TD3 Implementation](./td3_implementation_overview.md) - TD3 algorithm details
- [Trading RL Package](../src/trading_rl/README.md) - Core RL package overview
