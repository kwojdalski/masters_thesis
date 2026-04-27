import pandas as pd

from trading_rl.features import FeaturePipeline


def _sample_df() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=5, freq="h")
    return pd.DataFrame(
        {
            "open": [10.0, 11.0, 12.0, 13.0, 14.0],
            "high": [11.0, 12.0, 13.0, 14.0, 15.0],
            "low": [9.0, 10.0, 11.0, 12.0, 13.0],
            "close": [10.5, 11.5, 12.5, 13.5, 14.5],
            "volume": [100, 110, 120, 130, 140],
        },
        index=idx,
    )


def test_column_value_feature_pass_through_without_normalization():
    df = _sample_df()
    pipeline = FeaturePipeline.from_config_dict(
        [
            {
                "name": "close",
                "feature_type": "column_value",
                "params": {"column": "close"},
                "normalize": False,
            }
        ]
    )
    pipeline.fit(df)
    out = pipeline.transform(df)

    assert list(out.columns) == ["feature_close"]
    pd.testing.assert_series_equal(out["feature_close"], df["close"], check_names=False)


def test_column_value_feature_normalizes_with_train_fit():
    train_df = _sample_df()
    test_df = train_df.copy()
    test_df["close"] = test_df["close"] + 10.0

    # Use global (StandardScaler) normalization so fit+transform on the same
    # data produces a zero-mean output. The default "running" mode is an online
    # normalizer that intentionally does not produce zero-mean output on the
    # training split (it normalizes each step by past-only statistics).
    pipeline = FeaturePipeline.from_config_dict(
        [
            {
                "name": "close",
                "feature_type": "column_value",
                "params": {"column": "close"},
                "normalize": True,
                "normalization_method": "global",
            }
        ]
    )
    pipeline.fit(train_df)
    train_out = pipeline.transform(train_df)
    test_out = pipeline.transform(test_df)

    # Z-score normalization on train split should center train feature near zero.
    assert abs(float(train_out["feature_close"].mean())) < 1e-7
    # Test split uses train-fitted scaler, so a shifted series should not be centered.
    assert abs(float(test_out["feature_close"].mean())) > 1.0

