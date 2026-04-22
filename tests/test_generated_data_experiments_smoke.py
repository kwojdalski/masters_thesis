from __future__ import annotations

import math
from pathlib import Path

import mlflow
import numpy as np
import pytest
import trading_rl.train_trading_agent as training_module
import yaml

from data_generator import PriceDataGenerator
from trading_rl.config import ExperimentConfig
from trading_rl.train_trading_agent import run_single_experiment


SMOKE_SCENARIOS = (
    Path("src/configs/scenarios/sine_wave/ppo_no_trend.yaml"),
    Path("src/configs/scenarios/synthetic/upward_trend_td3_tradingenv.yaml"),
    Path("src/configs/scenarios/synthetic/upward_trend_ddpg_tradingenv.yaml"),
)

_CORE_REPORT_KEYS = ("total_return", "sharpe_ratio", "max_drawdown")


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _generate_dataset_for_scenario(tmp_path: Path, scenario_path: Path) -> Path:
    scenario_config = _load_yaml(scenario_path)
    data_config_path = Path(str(scenario_config["data"]["data_config"]))
    data_generator_config = _load_yaml(data_config_path)["data_generator"]

    output_dir = tmp_path / "generated_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = Path(str(scenario_config["data"]["data_path"])).name
    generator = PriceDataGenerator(output_dir=str(output_dir))
    pattern_type = str(data_generator_config["pattern_type"]).lower()

    if pattern_type == "sine_wave":
        generator.generate_sine_wave_pattern(
            output_file=output_file,
            n_periods=4,
            samples_per_period=40,
            base_price=float(data_generator_config["base_price"]),
            amplitude=float(data_generator_config["amplitude"]),
            trend_slope=float(data_generator_config["trend_slope"]),
            volatility=float(data_generator_config["volatility"]),
            start_date=str(data_generator_config["start_date"]),
        )
    elif pattern_type == "upward_drift":
        generator.generate_upward_drift_pattern(
            output_file=output_file,
            n_samples=160,
            base_price=float(data_generator_config["base_price"]),
            drift_rate=float(data_generator_config["drift_rate"]),
            volatility=float(data_generator_config["volatility"]),
            pullback_floor=float(data_generator_config.get("pullback_floor", 0.995)),
            start_date=str(data_generator_config["start_date"]),
        )
    else:
        raise ValueError(f"Unsupported synthetic pattern for smoke test: {pattern_type}")

    return output_dir / output_file


def _make_smoke_config(tmp_path: Path, scenario_path: Path, data_path: Path) -> ExperimentConfig:
    tracking_db = tmp_path / "mlflow.db"
    log_dir = tmp_path / "logs"

    return ExperimentConfig.from_yaml(
        scenario_path,
        overrides=[
            "seed=7",
            f"data.data_path={data_path}",
            "data.train_size=96",
            "data.validation_size=32",
            "training.max_steps=8",
            "training.init_rand_steps=4",
            "training.frames_per_batch=4",
            "training.optim_steps_per_batch=1",
            "training.sample_size=4",
            "training.buffer_size=32",
            "training.eval_steps=4",
            "training.eval_interval=1000",
            "training.log_interval=1000",
            "training.checkpoint_interval=0",
            "training.ppo_epochs=1",
            "network.actor_hidden_dims=[16,8]",
            "network.value_hidden_dims=[16,8]",
            f"logging.log_dir={log_dir}",
            "logging.log_level=WARNING",
            "logging.save_plots=false",
            f"tracking.tracking_uri=sqlite:///{tracking_db}",
            "explainability.enabled=false",
            "statistical_testing.enabled=false",
        ],
    )


def _assert_split_result_sane(split: str, split_result: dict) -> None:
    """Assert that a single split result is structurally valid and numerically sane."""
    for key in ("final_reward", "last_positions", "evaluation_report"):
        assert key in split_result, f"split '{split}' missing key '{key}'"

    # final_reward must be a finite number
    final_reward = split_result["final_reward"]
    assert math.isfinite(float(final_reward)), (
        f"split '{split}' final_reward is not finite: {final_reward}"
    )

    # Core evaluation metrics must be present and finite (some optional metrics like
    # alpha/recovery_time may legitimately be NaN when data is insufficient)
    report = split_result["evaluation_report"]
    for key in _CORE_REPORT_KEYS:
        assert key in report, f"split '{split}' evaluation_report missing '{key}'"
        assert math.isfinite(float(report[key])), (
            f"split '{split}' {key} is not finite: {report[key]}"
        )

    # max_drawdown is bounded below by -100% — a worse value indicates a
    # portfolio accounting bug (e.g. short positions losing more than 1x capital)
    assert report["max_drawdown"] >= -1.0, (
        f"split '{split}' max_drawdown {report['max_drawdown']:.4f} < -1.0"
    )

    # last_positions must be non-empty and contain only finite values.
    # NaN or inf here means the policy network produced garbage outputs.
    positions = split_result["last_positions"]
    assert len(positions) > 0, f"split '{split}' last_positions is empty"
    bad = [p for p in positions if not math.isfinite(float(p))]
    assert not bad, f"split '{split}' last_positions contains non-finite values: {bad[:5]}"


class TestGeneratedDataScenarioSmoke:
    """Smoke tests for full experiment runs on synthetic generated data.

    Each scenario generates a small in-memory dataset from a known pattern
    (sine wave, upward drift) and runs the full training + evaluation pipeline
    with minimal steps. Tests verify that the pipeline completes and that
    evaluation outputs are numerically sane for the given market pattern.
    MLflow artifact I/O is stubbed out.
    """

    pytestmark = pytest.mark.smoke

    @pytest.mark.parametrize("scenario_path", SMOKE_SCENARIOS, ids=lambda path: path.stem)
    def test_scenario(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        scenario_path: Path,
    ) -> None:
        mlflow.end_run()
        monkeypatch.setattr(
            training_module.MLflowTrainingCallback,
            "log_evaluation_plots",
            staticmethod(lambda **_kwargs: None),
        )
        monkeypatch.setattr(
            training_module.MLflowTrainingCallback,
            "log_evaluation_report",
            staticmethod(lambda *_args, **_kwargs: None),
        )
        monkeypatch.setattr(
            training_module.MLflowTrainingCallback,
            "log_final_metrics",
            staticmethod(lambda *_args, **_kwargs: None),
        )

        data_path = _generate_dataset_for_scenario(tmp_path, scenario_path)
        config = _make_smoke_config(tmp_path, scenario_path, data_path)

        result = run_single_experiment(custom_config=config)

        assert result["interrupted"] is False
        assert result["final_metrics"]["experiment_name"] == config.experiment_name
        assert result["final_metrics"]["evaluation_split"] == "test"
        assert result["final_metrics"]["data_size_total"] == 160
        assert result["final_metrics"]["train_size"] == 96
        assert result["final_metrics"]["validation_size"] == 32
        assert result["final_metrics"]["test_size"] == 32
        assert result["final_metrics"]["n_observations"] >= 1
        assert result["final_metrics"]["n_actions"] >= 1
        assert set(result["final_metrics"]["split_results"]) == {"train", "val", "test"}
        assert list((tmp_path / "logs").glob("*_checkpoint*.pt"))

        split_results = result["final_metrics"]["split_results"]
        for split in ("train", "val", "test"):
            _assert_split_result_sane(split, split_results[split])

        # Upward-drift data has a guaranteed rising price series. Even a random agent
        # should not lose more than 50% on it — a deeper loss signals inverted action
        # semantics, sign-flipped rewards, or broken portfolio accounting.
        if "upward_trend" in config.experiment_name:
            test_report = split_results["test"]["evaluation_report"]
            assert test_report["total_return"] > -0.5, (
                f"Total return {test_report['total_return']:.3f} is suspiciously low "
                "for a monotonically rising market"
            )
