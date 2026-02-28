import logging

import numpy as np
import pandas as pd
import pytest

from trading_rl import ExperimentConfig
from trading_rl.train_trading_agent import _ensure_close_column_for_hft


def _lob_df() -> pd.DataFrame:
    idx = pd.date_range("2026-02-25", periods=3, freq="s", tz="UTC")
    return pd.DataFrame(
        {
            "ask_px_00": [101.0, np.nan, 103.0],
            "bid_px_00": [99.0, 100.0, np.nan],
            "price": [100.0, 100.2, 103.1],
            "feature_hft_bid_px_00": [0.1, 0.2, 0.3],
        },
        index=idx,
    ).astype(
        {
            "ask_px_00": "float64",
            "bid_px_00": "float64",
            "price": "float64",
            "feature_hft_bid_px_00": "float64",
        }
    )


def test_hft_derives_close_from_top_of_book_with_price_fallback():
    config = ExperimentConfig()
    config.env.mode = "hft"
    logger = logging.getLogger("test_hft")

    train_df = _lob_df()
    val_df = _lob_df()
    test_df = _lob_df()

    train_out, val_out, test_out = _ensure_close_column_for_hft(
        train_df, val_df, test_df, config, logger
    )

    expected_close = pd.Series([100.0, 100.2, 103.1], index=train_df.index)
    pd.testing.assert_series_equal(train_out["close"], expected_close, check_names=False)
    pd.testing.assert_series_equal(val_out["close"], expected_close, check_names=False)
    pd.testing.assert_series_equal(test_out["close"], expected_close, check_names=False)


def test_hft_close_derivation_requires_l1_bid_ask():
    config = ExperimentConfig()
    config.env.mode = "hft"
    logger = logging.getLogger("test_hft")

    df = pd.DataFrame({"price": [1.0, 2.0, 3.0]})

    with pytest.raises(ValueError, match="ask_px_00/bid_px_00"):
        _ensure_close_column_for_hft(df, df, df, config, logger)


def test_non_hft_mode_leaves_data_untouched():
    config = ExperimentConfig()
    config.env.mode = "mft"
    logger = logging.getLogger("test_hft")

    df = _lob_df()
    train_out, val_out, test_out = _ensure_close_column_for_hft(df, df, df, config, logger)

    assert "close" not in train_out.columns
    assert "close" not in val_out.columns
    assert "close" not in test_out.columns
