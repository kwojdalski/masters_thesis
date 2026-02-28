from pathlib import Path

import pandas as pd

from cli.services.config_validation_service import validate_experiment_config
from trading_rl import ExperimentConfig


def _write_dataset(path: Path, rows: int = 20, include_lob: bool = False) -> Path:
    idx = pd.date_range("2024-01-01", periods=rows, freq="h")
    data = {
        "open": range(rows),
        "high": [x + 1 for x in range(rows)],
        "low": [x - 1 for x in range(rows)],
        "close": [100 + x for x in range(rows)],
        "volume": [1000 + x for x in range(rows)],
    }

    if include_lob:
        for level in range(1, 4):
            data[f"bid_px_{level}"] = [99.5 + x + (level - 1) * 0.1 for x in range(rows)]
            data[f"ask_px_{level}"] = [100.5 + x + (level - 1) * 0.1 for x in range(rows)]
            data[f"bid_sz_{level}"] = [200 + x + level for x in range(rows)]
            data[f"ask_sz_{level}"] = [210 + x + level for x in range(rows)]

    df = pd.DataFrame(data, index=idx)
    df.to_parquet(path)
    return path


def _write_feature_config(
    path: Path,
    feature_type: str = "log_return",
    *,
    name: str = "f1",
    domain: str | None = None,
    params: dict[str, str | int | float] | None = None,
) -> Path:
    lines = [
        "features:",
        f'  - name: "{name}"',
        f'    feature_type: "{feature_type}"',
        "    normalize: true",
    ]
    if domain is not None:
        lines.append(f'    domain: "{domain}"')
    if params:
        lines.append("    params:")
        for key, value in params.items():
            if isinstance(value, str):
                lines.append(f'      {key}: "{value}"')
            else:
                lines.append(f"      {key}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_valid_config_no_errors(tmp_path: Path):
    data_path = _write_dataset(tmp_path / "data.parquet")
    feat_cfg = _write_feature_config(tmp_path / "features.yaml")

    config = ExperimentConfig()
    config.data.data_path = str(data_path)
    config.data.train_size = 10
    config.data.validation_size = 5
    config.data.feature_config = str(feat_cfg)
    config.env.price_columns = ["close"]
    config.env.feature_columns = ["feature_f1"]

    report = validate_experiment_config(config)
    assert report.error_count == 0


def test_missing_feature_config_is_error(tmp_path: Path):
    data_path = _write_dataset(tmp_path / "data.parquet")
    config = ExperimentConfig()
    config.data.data_path = str(data_path)
    config.data.feature_config = str(tmp_path / "missing_features.yaml")

    report = validate_experiment_config(config)
    assert report.has_errors
    assert any(i.code == "FEATURE_CONFIG_MISSING" for i in report.issues)


def test_unknown_feature_type_is_error(tmp_path: Path):
    data_path = _write_dataset(tmp_path / "data.parquet")
    feat_cfg = _write_feature_config(tmp_path / "bad_features.yaml", feature_type="foo")

    config = ExperimentConfig()
    config.data.data_path = str(data_path)
    config.data.feature_config = str(feat_cfg)

    report = validate_experiment_config(config)
    assert report.has_errors
    assert any(i.code == "FEATURE_PIPELINE_INVALID" for i in report.issues)


def test_infeasible_split_is_error(tmp_path: Path):
    data_path = _write_dataset(tmp_path / "data.parquet", rows=10)
    feat_cfg = _write_feature_config(tmp_path / "features.yaml")
    config = ExperimentConfig()
    config.data.data_path = str(data_path)
    config.data.feature_config = str(feat_cfg)
    config.data.train_size = 10

    report = validate_experiment_config(config)
    assert report.has_errors
    assert any(i.code == "TRAIN_SPLIT_INVALID" for i in report.issues)


def test_missing_env_feature_column_is_error(tmp_path: Path):
    data_path = _write_dataset(tmp_path / "data.parquet")
    feat_cfg = _write_feature_config(tmp_path / "features.yaml", feature_type="log_return")

    config = ExperimentConfig()
    config.data.data_path = str(data_path)
    config.data.feature_config = str(feat_cfg)
    config.env.feature_columns = ["feature_not_existing"]

    report = validate_experiment_config(config)
    assert report.has_errors
    assert any(i.code == "FEATURE_COLUMNS_MISSING" for i in report.issues)


def test_missing_data_path_is_warning_only(tmp_path: Path):
    feat_cfg = _write_feature_config(tmp_path / "features.yaml")
    config = ExperimentConfig()
    config.data.data_path = str(tmp_path / "missing.parquet")
    config.data.feature_config = str(feat_cfg)

    report = validate_experiment_config(config)
    assert report.warning_count >= 1
    assert any(i.code == "DATA_PATH_MISSING" for i in report.issues)
    assert report.error_count == 0


def test_mft_mode_rejects_hft_features(tmp_path: Path):
    data_path = _write_dataset(tmp_path / "data.parquet")
    feat_cfg = _write_feature_config(
        tmp_path / "features_hft_domain.yaml",
        feature_type="log_return",
        domain="hft",
    )

    config = ExperimentConfig()
    config.data.data_path = str(data_path)
    config.data.feature_config = str(feat_cfg)
    config.env.mode = "mft"
    config.env.feature_columns = ["feature_f1"]

    report = validate_experiment_config(config)
    assert report.has_errors
    assert any(i.code == "MFT_HAS_HFT_FEATURES" for i in report.issues)


def test_hft_mode_requires_hft_features(tmp_path: Path):
    data_path = _write_dataset(tmp_path / "data.parquet")
    feat_cfg = _write_feature_config(tmp_path / "features_shared.yaml")

    config = ExperimentConfig()
    config.data.data_path = str(data_path)
    config.data.feature_config = str(feat_cfg)
    config.env.mode = "hft"
    config.env.feature_columns = ["feature_f1"]

    report = validate_experiment_config(config)
    assert report.has_errors
    assert any(i.code == "HFT_FEATURES_MISSING" for i in report.issues)


def test_hft_mode_rejects_mft_features(tmp_path: Path):
    data_path = _write_dataset(tmp_path / "data.parquet")
    feat_cfg = _write_feature_config(
        tmp_path / "features_mft_domain.yaml",
        feature_type="log_return",
        domain="mft",
    )

    config = ExperimentConfig()
    config.data.data_path = str(data_path)
    config.data.feature_config = str(feat_cfg)
    config.env.mode = "hft"
    config.env.feature_columns = ["feature_f1"]

    report = validate_experiment_config(config)
    assert report.has_errors
    assert any(i.code == "HFT_HAS_MFT_FEATURES" for i in report.issues)


def test_hft_mode_requires_lob_input_columns(tmp_path: Path):
    data_path = _write_dataset(tmp_path / "data.parquet", include_lob=False)
    feat_cfg = _write_feature_config(
        tmp_path / "features_hft_column_value.yaml",
        feature_type="column_value",
        name="hft_bid_px_1",
        domain="hft",
        params={"column": "bid_px_1"},
    )

    config = ExperimentConfig()
    config.data.data_path = str(data_path)
    config.data.feature_config = str(feat_cfg)
    config.env.mode = "hft"
    config.env.feature_columns = ["feature_hft_bid_px_1"]

    report = validate_experiment_config(config)
    assert report.has_errors
    assert any(i.code == "FEATURE_INPUT_COLUMNS_MISSING" for i in report.issues)


def test_hft_mode_with_hft_lob_features_is_valid(tmp_path: Path):
    data_path = _write_dataset(tmp_path / "data_lob.parquet", include_lob=True)
    feat_cfg = _write_feature_config(
        tmp_path / "features_hft_column_value.yaml",
        feature_type="column_value",
        name="hft_bid_px_1",
        domain="hft",
        params={"column": "bid_px_1"},
    )

    config = ExperimentConfig()
    config.data.data_path = str(data_path)
    config.data.feature_config = str(feat_cfg)
    config.data.train_size = 10
    config.data.validation_size = 5
    config.env.mode = "hft"
    config.env.price_columns = ["close"]
    config.env.feature_columns = ["feature_hft_bid_px_1"]

    report = validate_experiment_config(config)
    assert report.error_count == 0
