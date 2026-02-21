from pathlib import Path

import pandas as pd

from cli.services.config_validation_service import validate_experiment_config
from trading_rl import ExperimentConfig


def _write_dataset(path: Path, rows: int = 20) -> Path:
    idx = pd.date_range("2024-01-01", periods=rows, freq="h")
    df = pd.DataFrame(
        {
            "open": range(rows),
            "high": [x + 1 for x in range(rows)],
            "low": [x - 1 for x in range(rows)],
            "close": [100 + x for x in range(rows)],
            "volume": [1000 + x for x in range(rows)],
        },
        index=idx,
    )
    df.to_parquet(path)
    return path


def _write_feature_config(path: Path, feature_type: str = "log_return") -> Path:
    path.write_text(
        "features:\n"
        f'  - name: "f1"\n'
        f'    feature_type: "{feature_type}"\n'
        "    normalize: true\n",
        encoding="utf-8",
    )
    return path


def test_valid_config_no_errors(tmp_path: Path):
    data_path = _write_dataset(tmp_path / "data.parquet")
    feat_cfg = _write_feature_config(tmp_path / "features.yaml")

    config = ExperimentConfig()
    config.data.data_path = str(data_path)
    config.data.train_size = 10
    config.data.validation_size = 5
    config.data.no_features = False
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
    config = ExperimentConfig()
    config.data.data_path = str(data_path)
    config.data.no_features = True
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
    config = ExperimentConfig()
    config.data.data_path = str(tmp_path / "missing.parquet")
    config.data.no_features = True
    config.data.feature_config = None

    report = validate_experiment_config(config)
    assert report.warning_count >= 1
    assert any(i.code == "DATA_PATH_MISSING" for i in report.issues)
    assert report.error_count == 0
