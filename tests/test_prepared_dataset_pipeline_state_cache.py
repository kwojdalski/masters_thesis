"""Regression tests for feature-pipeline scaler state across the prepared cache.

A prepared-cache hit must restore the same normalization scaler statistics a
cache-miss run computed, not silently drop them (issue #365).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from trading_rl.config import ExperimentConfig
from trading_rl.data.preparation import build_prepared_dataset

LOGGER = logging.getLogger(__name__)


def _write_dataset(path: Path, periods: int = 72) -> Path:
    idx = pd.date_range("2024-01-01", periods=periods, freq="h")
    close = pd.Series(range(periods), index=idx, dtype=float) + 100.0
    pd.DataFrame({"close": close}, index=idx).to_parquet(path)
    return path


def _write_normalized_feature_groups(path: Path) -> Path:
    # _build_single_symbol_splits only wires a caller-visible FeaturePipeline
    # (and therefore captures its scaler state) via _resolve_feature_pipeline,
    # which reads data.feature_groups -- not data.feature_config. Use the
    # groups form here so cache-miss already captures state upstream and this
    # test isolates the cache-hit restore behaviour issue #365 is about.
    path.write_text(
        "groups:\n"
        "  lag_group:\n"
        "    description: test group\n"
        "    features:\n"
        '      - name: "lag1"\n'
        '        feature_type: "return_lag"\n'
        "        normalize: true\n"
        "        normalization_method: running\n"
        "        params:\n"
        '          column: "close"\n'
        "          lag: 1\n",
        encoding="utf-8",
    )
    return path


def _config(tmp_path: Path) -> ExperimentConfig:
    data_path = _write_dataset(tmp_path / "data.parquet")
    feature_groups = _write_normalized_feature_groups(tmp_path / "feature_groups.yaml")
    return ExperimentConfig.from_dict(
        {
            "data": {
                "data_path": str(data_path),
                "train_size": 36,
                "validation_size": 18,
                "download_data": False,
                "feature_groups": str(feature_groups),
                "warmup_rows": 0,
                "lazy_load": True,
                "prepared_data_dir": str(tmp_path / "prepared"),
            },
            "env": {
                "backend": "tradingenv",
                "price_column": "close",
                "feature_columns": ["feature_lag1"],
            },
            "training": {
                "algorithm": "TD3",
                "max_steps": 10,
                "frames_per_batch": 5,
            },
            "evaluation": {"eval_steps": 3},
            "logging": {"log_dir": str(tmp_path / "logs")},
        }
    )


def test_cache_hit_preserves_feature_pipeline_state(tmp_path: Path) -> None:
    """A cache-hit PreparedDataset must carry the scaler state a cache-miss
    run computed, not None."""
    config = _config(tmp_path)

    miss_dataset = build_prepared_dataset(config, LOGGER)
    assert miss_dataset.feature_pipeline_state is not None
    assert (Path(config.data.prepared_data_dir) / "pipeline_state.pkl").exists()

    hit_dataset = build_prepared_dataset(config, LOGGER)

    assert hit_dataset.feature_pipeline_state is not None
    assert hit_dataset.feature_pipeline_state == miss_dataset.feature_pipeline_state


def test_legacy_cache_without_pipeline_state_rebuilds_instead_of_dropping_state(
    tmp_path: Path,
) -> None:
    """A prepared cache written before pipeline-state persistence existed
    (or with its state file removed) must be rebuilt, not silently accepted
    with feature_pipeline_state=None."""
    config = _config(tmp_path)

    build_prepared_dataset(config, LOGGER)
    state_path = Path(config.data.prepared_data_dir) / "pipeline_state.pkl"
    checksum_path = Path(config.data.prepared_data_dir) / "pipeline_state.pkl.sha256"
    state_path.unlink()
    checksum_path.unlink()

    rebuilt_dataset = build_prepared_dataset(config, LOGGER)

    assert rebuilt_dataset.feature_pipeline_state is not None
    assert state_path.exists()
