# Stock Data Fetcher CLI - Usage Guide

Download stock market data from Databento with automatic rate limiting, download tracking, and parallel downloads.

## Features

✅ **Rate Limiting** - Skip re-downloads within 24 hours
✅ **Download Tracking** - Remembers what was downloaded
✅ **Parallel Downloads** - Download multiple symbols simultaneously
✅ **Batch Processing** - Configure multiple downloads in YAML
✅ **Progress Monitoring** - View download history and cache stats

## Quick Start

### 1. Set up API key

```bash
export DATABENTO_API_KEY='your-api-key-here'
```

### 2. Download a single symbol

```bash
python scripts/fetch_stocks.py download-stocks --symbols AAPL --start-date 2024-01-01 --end-date 2024-03-31
```

### 3. Download multiple symbols (parallel)

```bash
python scripts/fetch_stocks.py download-stocks \
  --symbols AAPL,MSFT,GOOGL,TSLA \
  --start-date 2024-01-01 \
  --end-date 2024-03-31 \
  --parallel \
  --max-workers 4
```

### 4. Batch download from config

```bash
python scripts/fetch_stocks.py batch --config src/configs/data/batch_download_example.yaml
```

## Commands

### `download-stocks` - Single/Multiple Symbol Download

Download one or more stock symbols with automatic rate limiting.

```bash
python scripts/fetch_stocks.py download-stocks [OPTIONS]
```

**Options:**
- `--symbols, -s` - Comma-separated symbols (default: AAPL)
- `--start-date, -d` - Start date YYYY-MM-DD (default: 2024-01-01)
- `--end-date, -e` - End date YYYY-MM-DD (default: 2024-03-31)
- `--dataset` - Dataset (XNAS.ITCH for NASDAQ, XNYS.TRADES for NYSE)
- `--schema` - Data schema: trades, tbbo, mbp-1, mbp-10 (default: trades)
- `--timeframe, -t` - Aggregation: 1h, 1d, 5m (default: 1h)
- `--aggregate/--raw` - Aggregate to OHLCV or keep raw (default: aggregate)
- `--parallel/--sequential` - Parallel or sequential downloads (default: parallel)
- `--max-workers` - Number of parallel workers (default: 4)
- `--force, -f` - Force download even if within rate limit
- `--interactive, -i` - Interactive mode with prompts

**Examples:**

```bash
# Download AAPL hourly data
python scripts/fetch_stocks.py download-stocks --symbols AAPL

# Download multiple symbols in parallel
python scripts/fetch_stocks.py download-stocks \
  --symbols AAPL,MSFT,GOOGL \
  --parallel \
  --max-workers 4

# Download raw order book data
python scripts/fetch_stocks.py download-stocks \
  --symbols AAPL \
  --schema mbp-10 \
  --raw

# Force re-download (ignore rate limit)
python scripts/fetch_stocks.py download-stocks \
  --symbols AAPL \
  --force
```

### `batch` - Batch Download from YAML Config

Download multiple jobs defined in a YAML configuration file.

```bash
python scripts/fetch_stocks.py batch --config CONFIG_FILE [--force]
```

**Options:**
- `--config, -c` - Path to batch config YAML (required)
- `--force, -f` - Force all downloads

**Example Config:** `src/configs/data/batch_download_example.yaml`

```yaml
settings:
  output_dir: "data/raw/stocks"
  source: "databento"
  rate_limit_hours: 24
  parallel_downloads: true
  max_workers: 4

downloads:
  - name: "nasdaq_tech_hourly"
    symbols: [AAPL, MSFT, GOOGL, AMZN]
    start_date: "2024-01-01"
    end_date: "2024-03-31"
    dataset: "XNAS.ITCH"
    timeframe: "1h"

  - name: "nyse_financial_daily"
    symbols: [JPM, BAC, GS]
    start_date: "2024-01-01"
    end_date: "2024-03-31"
    dataset: "XNYS.TRADES"
    timeframe: "1d"
```

**Usage:**

```bash
python scripts/fetch_stocks.py batch --config src/configs/data/batch_download_example.yaml
```

### `download-history` - View Download History

Show recent downloads and cache statistics.

```bash
python scripts/fetch_stocks.py download-history [--hours 24]
```

**Options:**
- `--hours, -h` - Show downloads from last N hours (default: 24)

**Example Output:**

```
DOWNLOAD CACHE STATISTICS
Total downloads: 12
Total symbols: 45
Total files: 45
Total size: 1250.35 MB
Oldest download: 2024-03-01 10:30:00
Newest download: 2024-03-02 14:15:00

Recent downloads (last 24h):
AAPL, MSFT, GOOGL
  Date range: 2024-01-01 to 2024-03-31
  Downloaded: 2.5h ago
  Rows: 15420
```

### `clear-cache` - Clear Download Cache

Clear all download tracking cache (allows re-downloading).

```bash
python scripts/fetch_stocks.py clear-cache
```

### `list-datasets` - List Available Datasets

Show all available Databento datasets.

```bash
python scripts/fetch_stocks.py list-datasets
```

## Rate Limiting

The CLI automatically tracks downloads and skips re-downloads within the rate limit window (default: 24 hours).

**How it works:**
1. When you download data, a hash of parameters (symbols, dates, schema) is saved
2. Next time you try the same download, it checks if it was done within 24h
3. If yes → skip download, if no → proceed

**Override rate limit:**
```bash
# Force re-download even if within 24h window
python scripts/fetch_stocks.py download-stocks --symbols AAPL --force
```

**Clear all tracking:**
```bash
python scripts/fetch_stocks.py clear-cache
```

## Parallel Downloads

Download multiple symbols simultaneously for faster processing.

**Parallel mode (default for multiple symbols):**
```bash
python scripts/fetch_stocks.py download-stocks \
  --symbols AAPL,MSFT,GOOGL,AMZN,TSLA \
  --parallel \
  --max-workers 4  # Download 4 symbols at once
```

**Sequential mode:**
```bash
python scripts/fetch_stocks.py download-stocks \
  --symbols AAPL,MSFT,GOOGL \
  --sequential  # Download one by one
```

**Performance:**
- Parallel: ~4x faster for 4 workers
- Best for many symbols with similar date ranges
- Rate limits are per-API key, so use appropriate max_workers

## Output Files

Downloaded files are saved as Parquet files:

**Aggregated data:**
```
data/raw/stocks/AAPL_2024-01-01_2024-03-31_1h.parquet
```

**Raw data:**
```
data/raw/stocks/AAPL_2024-01-01_2024-03-31_raw_mbp-10.parquet
```

**Inspect downloaded data:**
```bash
python -c "import pandas as pd; df = pd.read_parquet('data/raw/stocks/AAPL_2024-01-01_2024-03-31_1h.parquet'); print(df.info()); print(df.head())"
```

## Troubleshooting

### API key not found
```
Error: DATABENTO_API_KEY environment variable not set

Fix: export DATABENTO_API_KEY='your-key'
```

### Download failed
```
Error: Failed to download data

Check:
1. API key is correct
2. Symbol exists (AAPL, not aapl)
3. Date range is valid
4. Weles project at ../weles
```

### Too many re-downloads
```
# Clear cache to allow fresh downloads
python scripts/fetch_stocks.py clear-cache
```

## Best Practices

1. **Use batch config for regular downloads** - Easier to track and repeat
2. **Check download history first** - Avoid redundant downloads
3. **Use parallel for many symbols** - 4 workers is a good default
4. **Don't force unless necessary** - Rate limiting saves API quota
5. **Aggregate data for training** - Raw data is huge, only use for HFT research

## Example Workflow

```bash
# 1. Check what you've downloaded
python scripts/fetch_stocks.py download-history

# 2. Download new data (auto-skips if recent)
python scripts/fetch_stocks.py download-stocks \
  --symbols AAPL,MSFT,GOOGL \
  --start-date 2024-01-01 \
  --end-date 2024-03-31

# 3. Or use batch config for complex setups
python scripts/fetch_stocks.py batch --config my_downloads.yaml

# 4. Verify files
ls -lh data/raw/stocks/

# 5. Use in training
# Update your scenario YAML:
#   data:
#     data_path: 'data/raw/stocks/AAPL_2024-01-01_2024-03-31_1h.parquet'
```
