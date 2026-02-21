from pathlib import Path

import pytest
import typer
from rich.console import Console

from cli.commands import TrainingCommand, TrainingParams
from cli.commands.validation_command import ValidationCommand, ValidationParams


def _write_config(path: Path, data_path: str, feature_config: str | None = None) -> Path:
    feature_cfg_line = (
        f'  feature_config: "{feature_config}"\n' if feature_config is not None else ""
    )
    text = (
        'experiment_name: "validate_test"\n'
        "data:\n"
        f'  data_path: "{data_path}"\n'
        "  train_size: 10\n"
        "  validation_size: 5\n"
        "  no_features: false\n"
        f"{feature_cfg_line}"
        "env:\n"
        '  backend: "tradingenv"\n'
        '  price_columns: ["close"]\n'
        '  feature_columns: ["feature_f1"]\n'
        "training:\n"
        '  algorithm: "PPO"\n'
        "  max_steps: 50\n"
        "  frames_per_batch: 10\n"
        "  init_rand_steps: 0\n"
        "  eval_steps: 10\n"
    )
    path.write_text(text, encoding="utf-8")
    return path


def _write_feature_config(path: Path, feature_type: str = "log_return") -> Path:
    path.write_text(
        "features:\n"
        '  - name: "f1"\n'
        f'    feature_type: "{feature_type}"\n'
        "    normalize: true\n",
        encoding="utf-8",
    )
    return path


def _write_dataset(path: Path) -> Path:
    import pandas as pd

    idx = pd.date_range("2024-01-01", periods=30, freq="h")
    df = pd.DataFrame(
        {
            "open": range(30),
            "high": [x + 1 for x in range(30)],
            "low": [x - 1 for x in range(30)],
            "close": [100 + x for x in range(30)],
            "volume": [1000 + x for x in range(30)],
        },
        index=idx,
    )
    df.to_parquet(path)
    return path


def test_validate_command_success_exit_code(tmp_path: Path):
    data_path = _write_dataset(tmp_path / "data.parquet")
    feat_cfg = _write_feature_config(tmp_path / "features.yaml")
    cfg = _write_config(tmp_path / "cfg.yaml", str(data_path), str(feat_cfg))

    console = Console(record=True, force_terminal=False)
    cmd = ValidationCommand(console)
    cmd.execute(ValidationParams(config_file=cfg))


def test_validate_command_fails_with_errors(tmp_path: Path):
    data_path = _write_dataset(tmp_path / "data.parquet")
    cfg = _write_config(tmp_path / "cfg.yaml", str(data_path), str(tmp_path / "missing.yaml"))

    console = Console(record=True, force_terminal=False)
    cmd = ValidationCommand(console)
    with pytest.raises(typer.Exit):
        cmd.execute(ValidationParams(config_file=cfg))


def test_train_fails_fast_when_validation_errors_exist(tmp_path: Path):
    data_path = _write_dataset(tmp_path / "data.parquet")
    cfg = _write_config(tmp_path / "cfg.yaml", str(data_path), str(tmp_path / "missing.yaml"))

    console = Console(record=True, force_terminal=False)
    cmd = TrainingCommand(console)
    with pytest.raises(typer.Exit):
        cmd.execute(TrainingParams(config_file=cfg))


def test_train_continues_when_only_validation_warnings(tmp_path: Path, monkeypatch):
    cfg = _write_config(tmp_path / "cfg.yaml", str(tmp_path / "missing.parquet"), None)

    console = Console(record=True, force_terminal=False)
    cmd = TrainingCommand(console)

    monkeypatch.setattr(
        cmd, "_run_training_with_progress", lambda config, params: {"final_metrics": {}}
    )
    cmd.execute(TrainingParams(config_file=cfg))
