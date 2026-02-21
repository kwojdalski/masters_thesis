# Data Preparation Workflow (`prepare_data`)

This document explains how data preparation is handled in the current training workflow.

## Entry Point

Data preparation is orchestrated by:

- `prepare_data(...)` in `src/trading_rl/data_utils.py`

It is called from:

- `build_training_context(...)` in `src/trading_rl/train_trading_agent.py`

## Inputs

`prepare_data` accepts:

- `data_path`: path to source dataset (`.pkl`, `.pickle`, `.parquet`)
- `train_size`: number of rows used for training split
- `no_features`: if `True`, skip feature engineering
- `feature_config_path`: YAML path for feature pipeline config
- optional download arguments (`download_if_missing`, exchange/symbol/timeframe/since)

## Governing Principles

1. Split before feature engineering (anti-leakage).
2. Fit feature normalization on train split only.
3. Transform train/test using the same fitted feature parameters.
4. Keep raw OHLCV columns and append engineered feature columns.

## Step-by-Step Flow

1. Validate file presence (or download if configured).
2. Load raw data with cache support (`joblib.Memory`).
3. Drop raw NaNs (`df = df.dropna()`).
4. Split into:
   - `train_df_raw = df[:train_size]`
   - `test_df_raw = df[train_size:]`
5. If `no_features=True`, return raw train/test splits immediately.
6. Build feature pipeline:
   - from YAML (`FeaturePipeline.from_yaml(feature_config_path)`), or
   - default pipeline (`create_default_pipeline()`).
7. Fit on train raw only:
   - `pipeline.fit(train_df_raw)`
8. Transform both splits:
   - `train_features = pipeline.transform(train_df_raw)`
   - `test_features = pipeline.transform(test_df_raw)`
9. Concatenate:
   - `train_df = pd.concat([train_df_raw, train_features], axis=1)`
   - `test_df = pd.concat([test_df_raw, test_features], axis=1)`
10. Return `(train_df, test_df)`.

## Feature Pipeline Behavior

Feature pipeline is implemented in `src/trading_rl/features/pipeline.py`.

- Features are configured in YAML under `features:`.
- Each item maps to a registered feature implementation (`feature_type`).
- Required input columns are validated before fit/transform.
- Unknown `feature_type` raises a validation error.

### Example feature config

```yaml
features:
  - name: "log_return"
    feature_type: "log_return"
    normalize: true
```

## Where Features Enter the State Space

Features are appended to the DataFrame in `prepare_data` and then passed to environment creation in `build_training_context`.

For `tradingenv` backend, `env.feature_columns` controls which columns are used as observations.

## Config Paths

Use repo-root-relative paths, e.g.:

- `src/configs/features/...`
- `src/configs/data/...`
- `src/configs/scenarios/...`

## Notes and Caveats

1. `FeaturePipeline.transform` drops NaNs in the feature-only output; concatenation uses index alignment.
2. If transformed features drop early rows (warmup windows), feature columns can be NaN on unmatched raw rows after concat.
3. Keep feature names stable (`feature_*`) for predictable environment wiring and logging.

## Minimal Operational Checklist

1. Confirm dataset file exists and format is supported.
2. Confirm `train_size` is sensible for your horizon.
3. Confirm `feature_config_path` points to `src/configs/features/...`.
4. Confirm `env.feature_columns` matches generated feature names.
5. Validate final train/test shapes and sample head before training.
