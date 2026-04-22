from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_rl.config import ExperimentConfig
from trading_rl.data_utils import PreparedDataset
from trading_rl.train_trading_agent import (
    build_experiment_runtime,
    build_training_context,
    run_single_experiment,
)


def _write_dataset(path: Path, periods: int = 72) -> Path:
    idx = pd.date_range("2024-01-01", periods=periods, freq="h")
    close = pd.Series(range(periods), index=idx, dtype=float) + 100.0
    df = pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "volume": 1000.0 + close * 2.0,
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
        '  - name: "trend"\n'
        '    feature_type: "trend"\n'
        "    normalize: false\n",
        encoding="utf-8",
    )
    return path


def _make_config(tmp_path: Path) -> ExperimentConfig:
    data_path = _write_dataset(tmp_path / "data.parquet")
    feature_config = _write_feature_config(tmp_path / "features.yaml")
    return ExperimentConfig.from_dict(
        {
            "experiment_name": "pipeline_smoke",
            "data": {
                "data_path": str(data_path),
                "train_size": 36,
                "validation_size": 18,
                "download_data": False,
                "feature_config": str(feature_config),
            },
            "env": {
                "backend": "tradingenv",
                "price_column": "close",
                "feature_columns": ["feature_lag1", "feature_trend"],
            },
            "training": {
                "algorithm": "PPO",
                "max_steps": 8,
                "frames_per_batch": 4,
                "init_rand_steps": 0,
                "eval_steps": 4,
                "log_interval": 4,
            },
            "logging": {
                "log_dir": str(tmp_path / "logs"),
                "log_level": "WARNING",
                "save_plots": False,
            },
            "tracking": {
                "tracking_uri": f"file://{tmp_path / 'mlruns'}",
            },
        }
    )


class TestPipelineSmoke:
    """Smoke tests for the training pipeline.

    Verify that data loading, feature engineering, and experiment execution
    complete without error and produce structurally correct outputs.
    These tests use minimal training steps and stub out MLflow I/O.
    """

    pytestmark = pytest.mark.smoke

    def test_build_training_context(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)

        context = build_training_context(
            config=config,
            create_mlflow_callback=False,
        )

        assert not context["train_df"].empty
        assert not context["val_df"].empty
        assert not context["test_df"].empty
        assert isinstance(context["prepared_dataset"], PreparedDataset)
        assert context["training_bundle"].trainer is context["trainer"]
        assert context["training_bundle"].train_env is context["env"]
        assert context["prepared_dataset"].price_column == "close"
        assert "feature_lag1" in context["train_df"].columns
        assert "feature_trend" in context["train_df"].columns
        assert context["n_obs"] == 2
        assert context["n_act"] >= 1
        assert context["trainer"] is not None

        train_df = context["train_df"]
        val_df = context["val_df"]
        test_df = context["test_df"]

        # Splits must not overlap in time
        assert train_df.index.max() < val_df.index.min(), "train and val splits overlap"
        assert val_df.index.max() < test_df.index.min(), "val and test splits overlap"

        # Feature columns must contain only finite values — NaN/inf here would silently
        # corrupt observations fed to the neural network
        feature_cols = ["feature_lag1", "feature_trend"]
        for col in feature_cols:
            bad_train = (~np.isfinite(train_df[col].to_numpy())).sum()
            assert bad_train == 0, f"train_df['{col}'] has {bad_train} non-finite values"

    def test_run_single_experiment(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        import trading_rl.train_trading_agent as training_module

        config = _make_config(tmp_path)
        real_build_experiment_runtime = training_module.build_experiment_runtime

        def patched_build_experiment_runtime(*args, **kwargs):
            runtime = real_build_experiment_runtime(*args, **kwargs)
            trainer = runtime.training_bundle.trainer
            trainer.logs = {"loss_value": [1.0], "loss_actor": [0.5]}
            trainer.total_count = 1
            trainer.setup_periodic_evaluation = lambda **_kwargs: None
            trainer.setup_periodic_explainability = lambda **_kwargs: None
            trainer.train = lambda callback=None: dict(trainer.logs)
            trainer.evaluate = lambda df, max_steps, config, algorithm, eval_env: (
                None,
                None,
                None,
                0.1,
                [0.0],
                None,
                None,
            )
            trainer.save_checkpoint = (
                lambda path: Path(path).write_text("stub checkpoint", encoding="utf-8")
            )
            return runtime

        monkeypatch.setattr(
            training_module,
            "build_experiment_runtime",
            patched_build_experiment_runtime,
        )
        monkeypatch.setattr(
            training_module,
            "build_evaluation_report_for_trainer",
            lambda *args, **kwargs: {"total_return": 0.0, "sharpe_ratio": 0.0},
        )
        monkeypatch.setattr(
            training_module,
            "_run_explainability_analysis",
            lambda **kwargs: None,
        )
        monkeypatch.setattr(
            training_module.MLflowTrainingCallback,
            "log_evaluation_plots",
            staticmethod(lambda **kwargs: None),
        )
        monkeypatch.setattr(
            training_module.MLflowTrainingCallback,
            "log_evaluation_report",
            staticmethod(lambda *args, **kwargs: None),
        )
        monkeypatch.setattr(
            training_module.MLflowTrainingCallback,
            "log_final_metrics",
            staticmethod(lambda *args, **kwargs: None),
        )

        result = run_single_experiment(custom_config=config)

        assert result["interrupted"] is False
        assert result["final_metrics"]["experiment_name"] == "pipeline_smoke"
        assert result["final_metrics"]["evaluation_split"] == "test"
        assert set(result["final_metrics"]["split_results"]) == {"train", "val", "test"}
        checkpoint_paths = list(Path(config.logging.log_dir).glob("*_checkpoint*.pt"))
        assert checkpoint_paths
