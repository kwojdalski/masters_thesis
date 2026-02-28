"""
Stock data fetcher using market_data_fetcher from weles project.

This module provides a unified interface for downloading stock market data
from various sources (Databento, Polygon) using the weles market_data_fetcher.
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from trading_rl.data_fetchers.base import BaseMarketDataFetcher

# Add weles to path if needed
WELES_PATH = Path(__file__).parent.parent.parent.parent.parent / "weles" / "src"
if WELES_PATH.exists() and str(WELES_PATH) not in sys.path:
    sys.path.insert(0, str(WELES_PATH))

try:
    from market_data_fetcher import DataSource, get_market_data
except ImportError as e:
    raise ImportError(
        f"Could not import market_data_fetcher from weles project. "
        f"Expected path: {WELES_PATH}. "
        f"Error: {e}"
    ) from e


class StockDataFetcher(BaseMarketDataFetcher):
    """
    Fetcher for stock market data using weles market_data_fetcher.

    This class provides a modular interface for downloading stock data from
    NYSE/NASDAQ and other exchanges via Databento or Polygon data sources.

    Example:
        fetcher = StockDataFetcher(output_dir="data/raw/stocks")
        df = fetcher.fetch_stock_data(
            symbols=["AAPL", "MSFT"],
            start_date="2023-01-01",
            end_date="2023-12-31",
            source="databento",
            dataset="XNAS.ITCH"
        )
    """

    def __init__(
        self,
        output_dir: str = "data/raw/stocks",
        log_level: int | str | None = None,
    ):
        """
        Initialize the stock data fetcher.

        Parameters
        ----------
        output_dir : str
            Directory to save downloaded data
        log_level : int | str | None
            Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        super().__init__(output_dir=output_dir, log_level=log_level)

    def fetch_market_data(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str | None = None,
        source: str = "databento",
        dataset: str | None = None,
        schema: str = "trades",
        timeframe: str | None = "1h",
        aggregate: bool = True,
        save_to_file: bool = True,
        output_filename: str | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Fetch stock market data from specified data source.

        Parameters
        ----------
        symbols : list[str]
            List of stock symbols (e.g., ["AAPL", "MSFT", "TSLA"])
        start_date : str
            Start date in format 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'
        end_date : str | None
            End date (if None, uses current date)
        source : str
            Data source: "databento" or "polygon"
        dataset : str | None
            Dataset identifier (e.g., "XNAS.ITCH" for NASDAQ, "XNYS.TRADES" for NYSE)
            If None, uses default for the source
        schema : str
            Data schema:
            - "trades" - Individual trades (tick data)
            - "tbbo" - Top of book (best bid/offer)
            - "mbp-1" - Market by price level 1 (order book top)
            - "mbp-10" - Market by price level 10 (order book depth)
            - "ohlcv-1h" - Pre-aggregated OHLCV bars
        timeframe : str | None
            Timeframe for aggregation (e.g., "1h", "1d").
            Only used if aggregate=True. If None and aggregate=True, defaults to "1h"
        aggregate : bool
            Whether to aggregate data to OHLCV format.
            - True: Convert to OHLCV bars at specified timeframe
            - False: Keep raw tick/order book data (no aggregation)
        save_to_file : bool
            Whether to save data to parquet file
        output_filename : str | None
            Custom output filename (auto-generated if None)
        **kwargs
            Additional parameters passed to market_data_fetcher

        Returns
        -------
        pd.DataFrame
            Stock market data - OHLCV format if aggregate=True, raw format otherwise

        Examples:
            # Aggregated OHLCV data
            df = fetcher.fetch_market_data(
                symbols=["AAPL"],
                start_date="2024-01-01",
                end_date="2024-01-31",
                source="databento",
                dataset="XNAS.ITCH",
                schema="trades",
                timeframe="1h",
                aggregate=True
            )

            # Raw tick-level trades
            df = fetcher.fetch_market_data(
                symbols=["AAPL"],
                start_date="2024-01-01",
                end_date="2024-01-01",
                source="databento",
                dataset="XNAS.ITCH",
                schema="trades",
                aggregate=False  # Keep raw trades
            )

            # Raw order book (top of book)
            df = fetcher.fetch_market_data(
                symbols=["AAPL"],
                start_date="2024-01-01",
                end_date="2024-01-01",
                source="databento",
                dataset="XNAS.ITCH",
                schema="tbbo",  # Best bid/offer
                aggregate=False  # Keep raw quotes
            )

            # Order book depth (10 levels)
            df = fetcher.fetch_market_data(
                symbols=["AAPL"],
                start_date="2024-01-01",
                end_date="2024-01-01",
                source="databento",
                dataset="XNAS.ITCH",
                schema="mbp-10",  # 10 price levels
                aggregate=False  # Keep raw order book
            )
        """
        self.logger.info(
            f"Fetching stock data: symbols={symbols}, dates={start_date} to {end_date}, "
            f"source={source}, dataset={dataset}"
        )

        # Prepare date range
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        # Convert dates to market_data_fetcher format
        dates = self._generate_date_range(start_date, end_date)

        # Map source to DataSource enum
        if source.lower() == "databento":
            data_source = DataSource.DATABENTO
            dataset = dataset or "XNAS.ITCH"  # Default to NASDAQ
        elif source.lower() == "polygon":
            data_source = DataSource.POLYGON
        else:
            raise ValueError(f"Unsupported data source: {source}")

        # Fetch data using market_data_fetcher
        self.logger.debug(
            f"Calling market_data_fetcher with: source={data_source}, "
            f"symbols={symbols}, dates={len(dates)} date points, dataset={dataset}"
        )

        try:
            df = get_market_data(
                sources=data_source,
                dates=dates,
                symbols=symbols,
                venues=dataset,  # Use venues parameter for dataset
                schema=schema,
                normalize=True,
                merge_all=True,
                cache_enabled=True,
                **kwargs,
            )

            if df is None or df.empty:
                self.logger.warning("No data returned from market_data_fetcher")
                return pd.DataFrame()

            self.logger.info(f"Fetched {len(df)} rows from {source}")

            # Convert to OHLCV format if aggregation is requested
            if aggregate:
                if timeframe is None:
                    timeframe = "1h"
                    self.logger.info("No timeframe specified, defaulting to 1h")
                df = self._convert_to_ohlcv(df, schema, timeframe)
            else:
                self.logger.info(
                    f"Keeping raw {schema} data without aggregation ({len(df)} rows)"
                )

            # Save to file if requested - one file per instrument
            if save_to_file:
                # Determine file suffix based on aggregation
                if aggregate and timeframe:
                    file_suffix = timeframe
                else:
                    file_suffix = f"raw_{schema}"

                # Check if we have multiple symbols
                if "symbol" in df.columns and len(df["symbol"].unique()) > 1:
                    # Save each symbol separately
                    self.logger.info(
                        f"Splitting data into separate files for {len(df['symbol'].unique())} symbols"
                    )
                    saved_files = []
                    for symbol in df["symbol"].unique():
                        symbol_df = df[df["symbol"] == symbol].copy()

                        # Generate filename for this symbol
                        if output_filename is None:
                            symbol_filename = (
                                f"{symbol}_{start_date}_{end_date}_{file_suffix}.parquet"
                            )
                        else:
                            # Use provided filename as template, insert symbol
                            base_name = output_filename.replace(".parquet", "")
                            symbol_filename = f"{base_name}_{symbol}.parquet"

                        output_path = self.output_dir / symbol_filename
                        symbol_df.to_parquet(output_path)
                        saved_files.append(output_path)
                        self.logger.info(
                            f"Saved {len(symbol_df)} rows for {symbol} to {output_path}"
                        )

                    self.logger.info(
                        f"Saved {len(saved_files)} files: {[f.name for f in saved_files]}"
                    )
                else:
                    # Single symbol or no symbol column - save as one file
                    if output_filename is None:
                        # Auto-generate filename
                        if len(symbols) == 1:
                            symbol_str = symbols[0]
                        else:
                            symbol_str = "_".join(symbols[:3])
                            if len(symbols) > 3:
                                symbol_str += f"_and_{len(symbols)-3}_more"
                        output_filename = (
                            f"{symbol_str}_{start_date}_{end_date}_{file_suffix}.parquet"
                        )

                    output_path = self.output_dir / output_filename
                    df.to_parquet(output_path)
                    self.logger.info(f"Saved data to {output_path}")

            return df

        except Exception as e:
            self.logger.error(f"Error fetching stock data: {e}")
            raise

    def fetch_stock_data(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str | None = None,
        source: str = "databento",
        dataset: str | None = None,
        schema: str = "trades",
        timeframe: str | None = "1h",
        aggregate: bool = True,
        save_to_file: bool = True,
        output_filename: str | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Convenience method that wraps fetch_market_data with stock-specific defaults.

        This maintains backward compatibility with existing code.
        """
        return self.fetch_market_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            source=source,
            dataset=dataset,
            schema=schema,
            timeframe=timeframe,
            aggregate=aggregate,
            save_to_file=save_to_file,
            output_filename=output_filename,
            **kwargs,
        )

    def fetch_from_config(
        self,
        config_path: str,
        output_filename: str | None = None,
    ) -> pd.DataFrame:
        """
        Fetch stock data using parameters from YAML config file.

        Parameters
        ----------
        config_path : str
            Path to YAML configuration file
        output_filename : str | None
            Custom output filename (uses config value if None)

        Returns
        -------
        pd.DataFrame
            Stock market data

        Example config:
            stock_data:
              symbols: ["AAPL", "MSFT"]
              start_date: "2023-01-01"
              end_date: "2023-12-31"
              source: "databento"
              dataset: "XNAS.ITCH"
              schema: "trades"
              timeframe: "1h"
        """
        config = self.load_config(config_path)
        stock_config = config.get("stock_data", {})

        self.logger.info(f"Fetching stock data from config: {config_path}")

        return self.fetch_market_data(
            symbols=stock_config.get("symbols", []),
            start_date=stock_config.get("start_date"),
            end_date=stock_config.get("end_date"),
            source=stock_config.get("source", "databento"),
            dataset=stock_config.get("dataset"),
            schema=stock_config.get("schema", "trades"),
            timeframe=stock_config.get("timeframe", "1h"),
            save_to_file=stock_config.get("save_to_file", True),
            output_filename=output_filename or stock_config.get("output_filename"),
            **stock_config.get("additional_params", {}),
        )

    def _generate_date_range(
        self, start_date: str, end_date: str, freq: str = "D"
    ) -> list[str]:
        """
        Generate list of date strings for market_data_fetcher.

        Parameters
        ----------
        start_date : str
            Start date
        end_date : str
            End date
        freq : str
            Frequency for date range ('D' for daily, 'H' for hourly)

        Returns
        -------
        list[str]
            List of date strings in 'YYYY-MM-DD HH:MM:SS' format
        """
        # For market data, daily granularity is usually sufficient
        # The API will return intraday data within each date
        date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
        # Convert to format expected by market_data_fetcher
        return [date.strftime("%Y-%m-%d %H:%M:%S") for date in date_range]

    def _convert_to_ohlcv(
        self, df: pd.DataFrame, schema: str, timeframe: str
    ) -> pd.DataFrame:
        """
        Convert raw data to OHLCV format expected by trading environment.

        Parameters
        ----------
        df : pd.DataFrame
            Raw data from market_data_fetcher
        schema : str
            Data schema type
        timeframe : str
            Target timeframe for resampling

        Returns
        -------
        pd.DataFrame
            Data in OHLCV format with columns: open, high, low, close, volume
        """
        self.logger.debug(f"Converting to OHLCV format: schema={schema}, timeframe={timeframe}")

        # If already in OHLCV format, return as is
        required_cols = {"open", "high", "low", "close", "volume"}
        if required_cols.issubset(df.columns):
            self.logger.debug("Data already in OHLCV format")
            return df[["open", "high", "low", "close", "volume"]]

        # For trades data, aggregate to OHLCV
        if schema == "trades":
            if "price" not in df.columns:
                raise ValueError("Trades data must have 'price' column")

            # Ensure timestamp is in index
            if "timestamp" in df.columns:
                df = df.set_index("timestamp")

            # Resample to desired timeframe
            ohlcv = df.resample(timeframe).agg(
                {
                    "price": ["first", "max", "min", "last"],
                    "volume": "sum" if "volume" in df.columns else "count",
                }
            )

            # Flatten column names
            ohlcv.columns = ["open", "high", "low", "close", "volume"]

            # Fill missing values
            ohlcv = ohlcv.fillna(method="ffill").dropna()

            self.logger.info(
                f"Converted {len(df)} trades to {len(ohlcv)} OHLCV bars ({timeframe})"
            )

            return ohlcv

        # For quotes/tbbo data
        elif schema in ["tbbo", "quotes"]:
            if "price" in df.columns:
                # Use mid price from normalized data
                if "timestamp" in df.columns:
                    df = df.set_index("timestamp")

                ohlcv = df.resample(timeframe).agg(
                    {
                        "price": ["first", "max", "min", "last"],
                        "volume": "sum" if "volume" in df.columns else "count",
                    }
                )

                ohlcv.columns = ["open", "high", "low", "close", "volume"]
                ohlcv = ohlcv.fillna(method="ffill").dropna()

                self.logger.info(
                    f"Converted {len(df)} quotes to {len(ohlcv)} OHLCV bars ({timeframe})"
                )

                return ohlcv

        self.logger.warning(
            f"Could not convert schema '{schema}' to OHLCV format. "
            f"Returning raw data with available columns: {list(df.columns)}"
        )
        return df


    def list_available_datasets(self, source: str = "databento") -> dict[str, str]:
        """
        List commonly used datasets for each source.

        Parameters
        ----------
        source : str
            Data source ("databento" or "polygon")

        Returns
        -------
        dict[str, str]
            Dictionary mapping dataset names to descriptions
        """
        if source.lower() == "databento":
            return {
                "XNAS.ITCH": "NASDAQ (US tech stocks)",
                "XNYS.TRADES": "NYSE (US stocks)",
                "XASE.TRADES": "NYSE American (small caps)",
                "ARCX.TRADES": "NYSE Arca (ETFs)",
                "IEXG.TOPS": "IEX (US stocks)",
            }
        elif source.lower() == "polygon":
            return {
                "stocks": "US stocks (all exchanges)",
                "options": "US options",
                "indices": "Market indices",
                "crypto": "Cryptocurrencies",
            }
        else:
            return {}
