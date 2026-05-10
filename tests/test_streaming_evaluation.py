import logging
from pathlib import Path

import pandas as pd

import trading_rl.pipeline.evaluation as evaluation_module
from trading_rl.config import ExperimentConfig
from trading_rl.data_loading import save_prepared_splits, save_symbol_memmap
from trading_rl.data_utils import _prepared_cache_compatible


def _priced_frame(n_rows: int) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n_rows, freq="s")
    close = pd.Series([100.0 + i for i in range(n_rows)], index=idx, dtype=float)
    return pd.DataFrame(
        {
            "close": close,
            "feature_signal": close.pct_change().fillna(0.0),
        },
        index=idx,
    )


def _cache_config(tmp_path: Path, data_path: Path) -> ExperimentConfig:
    return ExperimentConfig.from_dict(
        {
            "data": {
                "data_path": str(data_path),
                "train_size": 5,
                "validation_size": 2,
                "test_size": 2,
                "lazy_load": True,
                "prepared_data_dir": str(tmp_path / "prepared"),
                "memmap_dir": str(tmp_path / "memmap"),
                "download_data": False,
            },
            "env": {
                "backend": "tradingenv",
                "price_column": "close",
                "feature_columns": ["feature_signal"],
            },
            "training": {
                "algorithm": "TD3",
                "max_steps": 10,
                "frames_per_batch": 5,
                "eval_steps": 3,
            },
            "logging": {"log_dir": str(tmp_path / "logs")},
        }
    )


def test_prepared_cache_rejects_stale_memmap_rows(tmp_path: Path) -> None:
    data_path = tmp_path / "raw.parquet"
    _priced_frame(12).to_parquet(data_path)
    config = _cache_config(tmp_path, data_path)
    prepared_dir = Path(config.data.prepared_data_dir)
    memmap_dir = Path(config.data.memmap_dir)

    save_prepared_splits(
        _priced_frame(5),
        _priced_frame(2),
        _priced_frame(2),
        prepared_dir,
    )
    save_symbol_memmap(_priced_frame(10), memmap_dir, prefix="0")

    assert not _prepared_cache_compatible(
        config,
        prepared_dir,
        memmap_dir,
        logger=logging.getLogger(__name__),
    )


def test_evaluation_context_disables_training_memmaps(monkeypatch) -> None:
    calls = []

    class BuilderProbe:
        def create(self, df, config, *, use_memmap=True):
            calls.append(use_memmap)
            return object()

    monkeypatch.setattr(evaluation_module, "AlgorithmicEnvironmentBuilder", BuilderProbe)
    config = ExperimentConfig()
    config.training.eval_steps = 3

    ctx = evaluation_module.build_evaluation_context_for_split(
        split="val",
        df=_priced_frame(5),
        config=config,
    )

    assert calls == [False]
    assert ctx.max_steps == 3
