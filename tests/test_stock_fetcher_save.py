"""Regression test for _save_stock_data naming files after requested symbols
instead of symbols actually present in the fetched data (issue #464)."""

from __future__ import annotations

import pandas as pd

from trading_rl.data_fetchers.stock_fetcher import StockDataFetcher


class TestSaveStockDataFilename:
    def test_filename_reflects_symbols_actually_present_not_requested(
        self, tmp_path, caplog
    ):
        fetcher = StockDataFetcher(output_dir=str(tmp_path))
        df = pd.DataFrame(
            {
                "symbol": ["AAPL", "AAPL"],
                "price": [100.0, 101.0],
            }
        )

        fetcher._save_stock_data(
            df,
            symbols=["AAPL", "MSFT"],
            start_date="2024-01-02",
            end_date="2024-01-03",
            file_suffix="1h",
            output_filename=None,
        )

        written = list(tmp_path.glob("*.parquet"))
        assert len(written) == 1
        assert written[0].name == "AAPL_2024-01-02_2024-01-03_1h.parquet"
        assert "MSFT" not in written[0].name

    def test_filename_unchanged_when_requested_symbols_match_data(self, tmp_path):
        fetcher = StockDataFetcher(output_dir=str(tmp_path))
        df = pd.DataFrame({"symbol": ["AAPL", "AAPL"], "price": [100.0, 101.0]})

        fetcher._save_stock_data(
            df,
            symbols=["AAPL"],
            start_date="2024-01-02",
            end_date="2024-01-03",
            file_suffix="1h",
            output_filename=None,
        )

        written = list(tmp_path.glob("*.parquet"))
        assert len(written) == 1
        assert written[0].name == "AAPL_2024-01-02_2024-01-03_1h.parquet"
