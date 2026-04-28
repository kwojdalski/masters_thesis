"""Stock market data fetcher backed by the Databento SDK."""

from __future__ import annotations

import importlib
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from trading_rl.data_fetchers.base import BaseMarketDataFetcher


class StockDataFetcher(BaseMarketDataFetcher):
    """Fetch stock market data from Databento.

    The class keeps the public interface used by ``scripts/fetch_stocks.py``:
    data can be saved as raw records or aggregated to OHLCV bars.
    """

    def __init__(
        self,
        output_dir: str = "data/raw/stocks",
        log_level: int | str | None = None,
    ) -> None:
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
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Fetch stock market data.

        Parameters
        ----------
        symbols
            Stock symbols such as ``["AAPL", "MSFT"]``.
        start_date
            Start timestamp accepted by Databento, commonly ``YYYY-MM-DD``.
        end_date
            End timestamp. Defaults to the current UTC date.
        source
            Data source. Currently only ``"databento"`` is implemented.
        dataset
            Databento dataset such as ``"XNAS.ITCH"``.
        schema
            Databento schema such as ``"trades"``, ``"tbbo"``, ``"mbp-1"``,
            or ``"mbp-10"``.
        timeframe
            Pandas resampling frequency used when ``aggregate=True``.
        aggregate
            If true, convert records to OHLCV bars.
        save_to_file
            Save the fetched frame to parquet files.
        output_filename
            Optional explicit parquet filename for single-file output.
        **kwargs
            Extra Databento request parameters. Commonly ``stype_in``.
        """
        if not symbols:
            raise ValueError("At least one symbol is required")

        source_normalized = source.lower()
        if source_normalized != "databento":
            raise NotImplementedError(
                f"Unsupported stock data source: {source}. "
                "Only the Databento-backed StockDataFetcher is implemented."
            )

        dataset = dataset or "XNAS.ITCH"
        end_date = end_date or datetime.now(tz=UTC).strftime("%Y-%m-%d")
        self.logger.info(
            "Fetching stock data: symbols=%s, dates=%s to %s, source=%s, "
            "dataset=%s, schema=%s",
            symbols,
            start_date,
            end_date,
            source_normalized,
            dataset,
            schema,
        )

        df = self._fetch_databento_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            dataset=dataset,
            schema=schema,
            **kwargs,
        )

        if df.empty:
            self.logger.warning("no data returned from databento")
            return df

        if aggregate:
            if timeframe is None:
                timeframe = "1h"
                self.logger.info("no timeframe specified, defaulting to 1h")
            df = self._convert_to_ohlcv(df, schema=schema, timeframe=timeframe)
        else:
            self.logger.info("keep raw data schema=%s", schema)

        if save_to_file:
            suffix = timeframe if aggregate and timeframe else f"raw_{schema}"
            self._save_stock_data(
                df=df,
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                file_suffix=suffix,
                output_filename=output_filename,
            )

        return df

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
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Compatibility wrapper around :meth:`fetch_market_data`."""
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
        """Fetch stock data using a YAML config with a ``stock_data`` section."""
        config = self.load_config(config_path)
        stock_config = config.get("stock_data", {})

        self.logger.info("fetch stock data config=%s", config_path)
        return self.fetch_market_data(
            symbols=stock_config.get("symbols", []),
            start_date=stock_config.get("start_date"),
            end_date=stock_config.get("end_date"),
            source=stock_config.get("source", "databento"),
            dataset=stock_config.get("dataset"),
            schema=stock_config.get("schema", "trades"),
            timeframe=stock_config.get("timeframe", "1h"),
            aggregate=stock_config.get("aggregate", True),
            save_to_file=stock_config.get("save_to_file", True),
            output_filename=output_filename or stock_config.get("output_filename"),
            **stock_config.get("additional_params", {}),
        )

    def _fetch_databento_data(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
        dataset: str,
        schema: str,
        **kwargs: Any,
    ) -> pd.DataFrame:
        db = self._import_databento()
        api_key = kwargs.pop("api_key", None) or os.getenv("DATABENTO_API_KEY")
        if not api_key:
            raise RuntimeError(
                "DATABENTO_API_KEY environment variable is not set. "
                "Export it before downloading Databento data."
            )

        request = {
            "dataset": dataset,
            "symbols": symbols,
            "schema": schema,
            "start": start_date,
            "end": end_date,
            "stype_in": kwargs.pop("stype_in", "raw_symbol"),
        }
        request.update(self._supported_databento_kwargs(kwargs))

        client = db.Historical(api_key)
        store = client.timeseries.get_range(**request)
        if hasattr(store, "to_df"):
            df = store.to_df()
        elif isinstance(store, pd.DataFrame):
            df = store
        else:
            raise TypeError(
                "Databento get_range returned an unsupported object without to_df()."
            )

        return self._normalize_databento_frame(df, symbols=symbols)

    @staticmethod
    def _supported_databento_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
        supported = {
            "stype_out",
            "limit",
            "path",
        }
        ignored = {"cache_enabled", "cache_ttl_hours", "normalize", "merge_all"}
        unknown = set(kwargs) - supported - ignored
        if unknown:
            raise TypeError(
                "Unsupported Databento request parameter(s): "
                f"{', '.join(sorted(unknown))}"
            )
        return {key: value for key, value in kwargs.items() if key in supported}

    @staticmethod
    def _import_databento() -> Any:
        try:
            return importlib.import_module("databento")
        except ImportError as exc:
            raise ImportError(
                "The 'databento' package is required for stock downloads. "
                "Install project dependencies with: uv sync"
            ) from exc

    @staticmethod
    def _normalize_databento_frame(
        df: pd.DataFrame,
        symbols: list[str],
    ) -> pd.DataFrame:
        normalized = df.copy()
        if isinstance(normalized.index, pd.DatetimeIndex):
            normalized.index.name = normalized.index.name or "timestamp"
        elif "timestamp" in normalized.columns:
            normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], utc=True)
            normalized = normalized.set_index("timestamp")
        elif "ts_event" in normalized.columns:
            normalized["ts_event"] = pd.to_datetime(normalized["ts_event"], utc=True)
            normalized = normalized.set_index("ts_event")

        if "symbol" not in normalized.columns and len(symbols) == 1:
            normalized["symbol"] = symbols[0]

        if "volume" not in normalized.columns and "size" in normalized.columns:
            normalized["volume"] = normalized["size"]

        return normalized.sort_index()

    def _convert_to_ohlcv(
        self,
        df: pd.DataFrame,
        schema: str,
        timeframe: str,
    ) -> pd.DataFrame:
        required_cols = {"open", "high", "low", "close", "volume"}
        if required_cols.issubset(df.columns):
            return self._ensure_datetime_index(df)

        working = self._ensure_datetime_index(df)
        if not isinstance(working.index, pd.DatetimeIndex):
            raise ValueError("Data must have a datetime index or timestamp column")

        if schema not in {"trades", "tbbo", "quotes", "mbp-1", "mbp-10"}:
            raise ValueError(f"Cannot aggregate unsupported Databento schema: {schema}")

        working = self._add_price_column(working)
        frames = []
        if "symbol" in working.columns:
            for symbol, symbol_df in working.groupby("symbol", sort=False):
                ohlcv = self._resample_ohlcv(symbol_df, timeframe)
                ohlcv["symbol"] = symbol
                frames.append(ohlcv)
        else:
            frames.append(self._resample_ohlcv(working, timeframe))

        result = pd.concat(frames).sort_index()
        self.logger.info(
            "Converted %d %s records to %d OHLCV bars (%s)",
            len(df),
            schema,
            len(result),
            timeframe,
        )
        return result

    @staticmethod
    def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
        working = df.copy()
        if isinstance(working.index, pd.DatetimeIndex):
            return working.sort_index()
        for column in ("timestamp", "ts_event", "time"):
            if column in working.columns:
                working[column] = pd.to_datetime(working[column], utc=True)
                return working.set_index(column).sort_index()
        return working

    @staticmethod
    def _add_price_column(df: pd.DataFrame) -> pd.DataFrame:
        working = df.copy()
        if "price" in working.columns:
            return working

        bid_columns = ("bid_px", "bid_price", "bid")
        ask_columns = ("ask_px", "ask_price", "ask")
        bid_col = next((column for column in bid_columns if column in working.columns), None)
        ask_col = next((column for column in ask_columns if column in working.columns), None)
        if bid_col and ask_col:
            working["price"] = (working[bid_col] + working[ask_col]) / 2
            return working

        raise ValueError(
            "Cannot aggregate records without a 'price' column or bid/ask columns"
        )

    @staticmethod
    def _resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        price_bars = df["price"].resample(timeframe).ohlc()
        if "volume" in df.columns:
            volume = df["volume"].resample(timeframe).sum()
        else:
            volume = df["price"].resample(timeframe).count()
        ohlcv = price_bars.assign(volume=volume)
        return ohlcv.ffill().dropna()

    def _save_stock_data(
        self,
        df: pd.DataFrame,
        symbols: list[str],
        start_date: str,
        end_date: str,
        file_suffix: str,
        output_filename: str | None,
    ) -> None:
        if "symbol" in df.columns and len(df["symbol"].dropna().unique()) > 1:
            for symbol in df["symbol"].dropna().unique():
                symbol_df = df[df["symbol"] == symbol].copy()
                filename = self._symbol_filename(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    file_suffix=file_suffix,
                    output_filename=output_filename,
                )
                self._write_parquet(symbol_df, filename)
            return

        if output_filename is None:
            symbol_label = symbols[0] if len(symbols) == 1 else "_".join(symbols)
            output_filename = (
                f"{symbol_label}_{start_date}_{end_date}_{file_suffix}.parquet"
            )
        self._write_parquet(df, output_filename)

    @staticmethod
    def _symbol_filename(
        symbol: str,
        start_date: str,
        end_date: str,
        file_suffix: str,
        output_filename: str | None,
    ) -> str:
        if output_filename is None:
            return f"{symbol}_{start_date}_{end_date}_{file_suffix}.parquet"
        base_name = output_filename.removesuffix(".parquet")
        return f"{base_name}_{symbol}.parquet"

    def _write_parquet(self, df: pd.DataFrame, filename: str) -> Path:
        output_path = self.output_dir / filename
        df.to_parquet(output_path)
        self.logger.info("save rows=%d path=%s", len(df), output_path)
        return output_path

    def list_available_datasets(self, source: str = "databento") -> dict[str, str]:
        """List common dataset identifiers."""
        if source.lower() == "databento":
            return {
                "XNAS.ITCH": "NASDAQ TotalView-ITCH",
                "XNYS.TRADES": "NYSE trades",
                "XASE.TRADES": "NYSE American trades",
                "ARCX.TRADES": "NYSE Arca trades and ETFs",
                "IEXG.TOPS": "IEX TOPS",
            }
        if source.lower() == "polygon":
            return {
                "stocks": "US stocks",
                "options": "US options",
                "indices": "Market indices",
                "crypto": "Cryptocurrencies",
            }
        return {}
