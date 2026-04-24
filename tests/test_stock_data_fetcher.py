from types import SimpleNamespace

import pandas as pd
import pytest

from trading_rl.data_fetchers import StockDataFetcher


class _FakeStore:
    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def to_df(self) -> pd.DataFrame:
        return self._df.copy()


def test_databento_fetcher_aggregates_trades_and_saves_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    calls = {}
    raw_df = pd.DataFrame(
        {
            "price": [10.0, 12.0, 11.0],
            "size": [100, 50, 25],
            "symbol": ["AAPL", "AAPL", "AAPL"],
        },
        index=pd.to_datetime(
            [
                "2024-01-02T09:00:00Z",
                "2024-01-02T09:30:00Z",
                "2024-01-02T10:00:00Z",
            ]
        ),
    )

    class FakeTimeseries:
        def get_range(self, **kwargs):
            calls.update(kwargs)
            return _FakeStore(raw_df)

    class FakeHistorical:
        def __init__(self, api_key: str) -> None:
            calls["api_key"] = api_key
            self.timeseries = FakeTimeseries()

    monkeypatch.setenv("DATABENTO_API_KEY", "test-key")
    monkeypatch.setattr(
        StockDataFetcher,
        "_import_databento",
        staticmethod(lambda: SimpleNamespace(Historical=FakeHistorical)),
    )

    fetcher = StockDataFetcher(output_dir=str(tmp_path))
    df = fetcher.fetch_stock_data(
        symbols=["AAPL"],
        start_date="2024-01-02",
        end_date="2024-01-03",
        dataset="XNAS.ITCH",
        schema="trades",
        timeframe="1h",
        save_to_file=True,
    )

    assert calls["api_key"] == "test-key"
    assert calls["dataset"] == "XNAS.ITCH"
    assert calls["symbols"] == ["AAPL"]
    assert calls["schema"] == "trades"
    assert calls["stype_in"] == "raw_symbol"
    assert list(df.columns) == ["open", "high", "low", "close", "volume", "symbol"]
    assert df.iloc[0].to_dict() == {
        "open": 10.0,
        "high": 12.0,
        "low": 10.0,
        "close": 12.0,
        "volume": 150,
        "symbol": "AAPL",
    }
    assert (tmp_path / "AAPL_2024-01-02_2024-01-03_1h.parquet").exists()


def test_databento_fetcher_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
    monkeypatch.setattr(
        StockDataFetcher,
        "_import_databento",
        staticmethod(lambda: SimpleNamespace(Historical=object)),
    )

    fetcher = StockDataFetcher(output_dir="data/raw/stocks")
    with pytest.raises(RuntimeError, match="DATABENTO_API_KEY"):
        fetcher.fetch_stock_data(
            symbols=["AAPL"],
            start_date="2024-01-02",
            end_date="2024-01-03",
            save_to_file=False,
        )


def test_fetcher_rejects_unsupported_databento_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABENTO_API_KEY", "test-key")
    monkeypatch.setattr(
        StockDataFetcher,
        "_import_databento",
        staticmethod(lambda: SimpleNamespace(Historical=object)),
    )

    fetcher = StockDataFetcher(output_dir="data/raw/stocks")
    with pytest.raises(TypeError, match="Unsupported Databento request parameter"):
        fetcher.fetch_stock_data(
            symbols=["AAPL"],
            start_date="2024-01-02",
            end_date="2024-01-03",
            save_to_file=False,
            unexpected=True,
        )
