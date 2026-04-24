from pathlib import Path

import pandas as pd
from rich.console import Console

from cli.commands import FeatureResearchCommand, FeatureResearchParams


def _write_dataset(path: Path) -> Path:
    periods = 80
    idx = pd.date_range("2024-01-01", periods=periods, freq="h")
    close = pd.Series(range(periods), index=idx, dtype=float) + 100.0
    df = pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "volume": 1000.0 + close * 5.0,
        },
        index=idx,
    )
    df.to_parquet(path)
    return path


def _write_feature_config(path: Path) -> Path:
    path.write_text(
        "features:\n"
        '  - name: "lag1"\n'
        '    feature_type: "return_lag"\n'
        "    normalize: true\n"
        "    params:\n"
        '      column: "close"\n'
        "      lag: 1\n"
        '  - name: "lag2"\n'
        '    feature_type: "return_lag"\n'
        "    normalize: true\n"
        "    params:\n"
        '      column: "close"\n'
        "      lag: 2\n"
        '  - name: "trend"\n'
        '    feature_type: "trend"\n'
        "    normalize: false\n",
        encoding="utf-8",
    )
    return path


def _write_config(
    path: Path,
    data_path: Path,
    feature_config: Path,
    output_dir: Path,
) -> Path:
    path.write_text(
        "experiment_name: offline_feature_research_test\n"
        "data:\n"
        f'  data_path: "{data_path}"\n'
        "  train_size: 40\n"
        "  validation_size: 20\n"
        f'  feature_config: "{feature_config}"\n'
        "research:\n"
        "  horizon: 1\n"
        "  top_k: 2\n"
        "  icir_threshold: 0.0\n"
        f'  output_dir: "{output_dir}"\n',
        encoding="utf-8",
    )
    return path


def test_feature_research_command_creates_outputs(tmp_path: Path):
    data_path = _write_dataset(tmp_path / "data.parquet")
    feature_config = _write_feature_config(tmp_path / "features.yaml")
    output_dir = tmp_path / "research_outputs"
    config_path = _write_config(
        tmp_path / "research_config.yaml",
        data_path,
        feature_config,
        output_dir,
    )

    cmd = FeatureResearchCommand(Console(record=True, force_terminal=False))
    cmd.execute(
        FeatureResearchParams(
            config_file=config_path,
        )
    )

    assert (output_dir / "feature_scores.csv").exists()
    assert (output_dir / "feature_correlations.csv").exists()
    assert (output_dir / "selected_features.yaml").exists()
    assert (output_dir / "summary.md").exists()

    scores = pd.read_csv(output_dir / "feature_scores.csv")
    assert not scores.empty
    assert "feature" in scores.columns

    selected_yaml = (output_dir / "selected_features.yaml").read_text(encoding="utf-8")
    assert "features:" in selected_yaml
