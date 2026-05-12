from __future__ import annotations

from dataclasses import dataclass
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


SCENARIO_PATHS = (
    Path("src/configs/scenarios/sine_wave/ppo_no_trend.yaml"),
    Path("src/configs/scenarios/sine_wave/td3_no_trend_tradingenv.yaml"),
    Path("src/configs/scenarios/synthetic/upward_trend_td3_tradingenv.yaml"),
    Path("src/configs/scenarios/synthetic/upward_trend_ddpg_tradingenv.yaml"),
)

_CORE_REPORT_KEYS = ("total_return", "sharpe_ratio", "max_drawdown")
_EXTENDED_REPORT_KEYS = (
    "annualized_return_cagr",
    "annualized_volatility",
    "sortino_ratio",
    "calmar_ratio",
    "average_drawdown",
    "max_drawdown_duration",
    "var_95",
    "cvar_95",
    "downside_deviation",
    "win_rate",
    "hit_rate",
    "turnover",
    "average_holding_period",
    "expectancy_per_period",
)
_EXTENDED_FINITE_REPORT_KEYS = (
    "annualized_volatility",
    "average_drawdown",
    "max_drawdown_duration",
    "var_95",
    "cvar_95",
    "downside_deviation",
    "win_rate",
    "hit_rate",
    "turnover",
    "average_holding_period",
    "expectancy_per_period",
)


@dataclass(frozen=True)
class GeneratedDataScenarioCase:
    scenario_path: Path
    data_size: int = 160
    train_size: int = 96
    validation_size: int = 32
    max_steps: int = 8
    init_rand_steps: int = 4
    frames_per_batch: int = 4
    optim_steps_per_batch: int = 1
    sample_size: int = 4
    buffer_size: int = 32
    eval_steps: int = 4
    eval_interval: int = 1000
    actor_hidden_dims: tuple[int, ...] = (16, 8)
    value_hidden_dims: tuple[int, ...] = (16, 8)
    extended_checks: bool = False
    reward_trend_checks: bool = False

    @property
    def test_size(self) -> int:
        return self.data_size - self.train_size - self.validation_size

    @property
    def id(self) -> str:
        suffix = "extended" if self.extended_checks else "fast"
        return f"{self.scenario_path.stem}_{suffix}"


SMOKE_CASES = tuple(GeneratedDataScenarioCase(path) for path in SCENARIO_PATHS)
PUSH_SMOKE_CASES = tuple(
    GeneratedDataScenarioCase(
        path,
        data_size=320,
        train_size=192,
        validation_size=64,
        max_steps=10_000,
        init_rand_steps=100,
        frames_per_batch=2_000,
        optim_steps_per_batch=1,
        sample_size=64,
        buffer_size=10_000,
        eval_steps=64,
        eval_interval=1,
        actor_hidden_dims=(32, 16),
        value_hidden_dims=(32, 16),
        extended_checks=True,
        reward_trend_checks=True,
    )
    for path in SCENARIO_PATHS
)


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _format_int_list(values: tuple[int, ...]) -> str:
    return "[" + ",".join(str(value) for value in values) + "]"


def _generate_dataset_for_scenario(
    tmp_path: Path,
    case: GeneratedDataScenarioCase,
) -> Path:
    scenario_config = _load_yaml(case.scenario_path)
    data_config_path = Path(str(scenario_config["data"]["data_config"]))
    data_generator_config = _load_yaml(data_config_path)["data_generator"]

    output_dir = tmp_path / "generated_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = Path(str(scenario_config["data"]["data_path"])).name
    generator = PriceDataGenerator(output_dir=str(output_dir))
    pattern_type = str(data_generator_config["pattern_type"]).lower()

    if pattern_type == "sine_wave":
        n_periods = 4
        if case.data_size % n_periods != 0:
            raise ValueError(
                f"Sine-wave smoke data_size must be divisible by {n_periods}: "
                f"{case.data_size}"
            )
        generator.generate_sine_wave_pattern(
            output_file=output_file,
            n_periods=n_periods,
            samples_per_period=case.data_size // n_periods,
            base_price=float(data_generator_config["base_price"]),
            amplitude=float(data_generator_config["amplitude"]),
            trend_slope=float(data_generator_config["trend_slope"]),
            volatility=float(data_generator_config["volatility"]),
            start_date=str(data_generator_config["start_date"]),
        )
    elif pattern_type == "upward_drift":
        generator.generate_upward_drift_pattern(
            output_file=output_file,
            n_samples=case.data_size,
            base_price=float(data_generator_config["base_price"]),
            drift_rate=float(data_generator_config["drift_rate"]),
            volatility=float(data_generator_config["volatility"]),
            pullback_floor=float(data_generator_config.get("pullback_floor", 0.995)),
            start_date=str(data_generator_config["start_date"]),
        )
    else:
        raise ValueError(f"Unsupported synthetic pattern for smoke test: {pattern_type}")

    return output_dir / output_file


def _make_smoke_config(
    tmp_path: Path,
    case: GeneratedDataScenarioCase,
    data_path: Path,
) -> ExperimentConfig:
    tracking_db = tmp_path / "mlflow.db"
    log_dir = tmp_path / "logs"

    return ExperimentConfig.from_yaml(
        case.scenario_path,
        overrides=[
            "seed=7",
            f"data.data_path={data_path}",
            f"data.train_size={case.train_size}",
            f"data.validation_size={case.validation_size}",
            f"training.max_steps={case.max_steps}",
            f"training.init_rand_steps={case.init_rand_steps}",
            f"training.frames_per_batch={case.frames_per_batch}",
            f"training.optim_steps_per_batch={case.optim_steps_per_batch}",
            f"training.sample_size={case.sample_size}",
            f"training.buffer_size={case.buffer_size}",
            f"training.eval_steps={case.eval_steps}",
            f"training.eval_interval={case.eval_interval}",
            "training.log_interval=1000",
            "training.checkpoint_interval=0",
            "training.ppo_epochs=1",
            f"network.actor_hidden_dims={_format_int_list(case.actor_hidden_dims)}",
            f"network.value_hidden_dims={_format_int_list(case.value_hidden_dims)}",
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


def _assert_split_result_extended(split: str, split_result: dict) -> None:
    """Assert richer metric coverage for the larger pre-push smoke runs."""
    report = split_result["evaluation_report"]

    for key in _EXTENDED_REPORT_KEYS:
        assert key in report, f"split '{split}' evaluation_report missing '{key}'"

    for key in _EXTENDED_FINITE_REPORT_KEYS:
        assert math.isfinite(float(report[key])), (
            f"split '{split}' {key} is not finite: {report[key]}"
        )

    for key in ("win_rate", "hit_rate"):
        assert 0.0 <= report[key] <= 1.0, (
            f"split '{split}' {key} must be in [0, 1]: {report[key]}"
        )

    assert report["annualized_volatility"] >= 0.0, (
        f"split '{split}' annualized_volatility is negative: "
        f"{report['annualized_volatility']}"
    )
    assert report["downside_deviation"] >= 0.0, (
        f"split '{split}' downside_deviation is negative: "
        f"{report['downside_deviation']}"
    )
    assert report["turnover"] >= 0.0, (
        f"split '{split}' turnover is negative: {report['turnover']}"
    )
    assert report["average_holding_period"] > 0.0, (
        f"split '{split}' average_holding_period must be positive: "
        f"{report['average_holding_period']}"
    )


def _assert_reward_trend(logs: dict, case: GeneratedDataScenarioCase) -> None:
    """Assert pre-push reward evaluations are finite and do not collapse.

    Strict monotonic improvement is too brittle for stochastic RL. The useful
    push-time signal is that deterministic eval rewards are sampled at the
    requested cadence and do not materially collapse.
    """
    rewards = np.asarray(logs.get("eval_reward_mean", []), dtype=float)
    expected_points = case.max_steps // case.frames_per_batch

    assert rewards.size >= expected_points, (
        f"Expected at least {expected_points} eval_reward_mean values "
        f"for {case.max_steps} steps with {case.frames_per_batch}-step batches; "
        f"got {rewards.size}: {rewards.tolist()}"
    )

    non_finite = rewards[~np.isfinite(rewards)]
    assert non_finite.size == 0, (
        f"eval_reward_mean contains non-finite values: {non_finite.tolist()}"
    )

    collapse_tolerance = max(1e-6, abs(float(rewards[0])) * 0.10)
    lowest_reward = float(np.min(rewards))
    assert lowest_reward + collapse_tolerance >= rewards[0], (
        "eval_reward_mean materially collapsed across 2k-step checks: "
        f"first={rewards[0]:.6g}, min={lowest_reward:.6g}, "
        f"tolerance={collapse_tolerance:.6g}, rewards={rewards.tolist()}"
    )

    assert rewards[-1] + collapse_tolerance >= rewards[0], (
        "final eval_reward_mean is materially below the first 2k-step check: "
        f"first={rewards[0]:.6g}, final={rewards[-1]:.6g}, "
        f"tolerance={collapse_tolerance:.6g}, rewards={rewards.tolist()}"
    )


def _stub_mlflow_artifact_logging(monkeypatch: pytest.MonkeyPatch) -> None:
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


def _run_generated_data_scenario(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    case: GeneratedDataScenarioCase,
) -> None:
    mlflow.end_run()
    _stub_mlflow_artifact_logging(monkeypatch)

    data_path = _generate_dataset_for_scenario(tmp_path, case)
    config = _make_smoke_config(tmp_path, case, data_path)

    result = run_single_experiment(custom_config=config)
    logs = result["logs"]

    assert result["interrupted"] is False
    assert result["final_metrics"]["experiment_name"] == config.experiment_name
    assert result["final_metrics"]["evaluation_split"] == "test"
    assert result["final_metrics"]["data_size_total"] == case.data_size
    assert result["final_metrics"]["train_size"] == case.train_size
    assert result["final_metrics"]["validation_size"] == case.validation_size
    assert result["final_metrics"]["test_size"] == case.test_size
    assert result["final_metrics"]["n_observations"] >= 1
    assert result["final_metrics"]["n_actions"] >= 1
    assert set(result["final_metrics"]["split_results"]) == {"train", "val", "test"}
    assert list((tmp_path / "logs").glob("*_checkpoint*.pt"))

    split_results = result["final_metrics"]["split_results"]
    for split in ("train", "val", "test"):
        _assert_split_result_sane(split, split_results[split])
        if case.extended_checks:
            _assert_split_result_extended(split, split_results[split])

    # Upward-drift data has a guaranteed rising price series. Even a random agent
    # should not lose more than 50% on it — a deeper loss signals inverted action
    # semantics, sign-flipped rewards, or broken portfolio accounting.
    if "upward_trend" in config.experiment_name:
        test_report = split_results["test"]["evaluation_report"]
        assert test_report["total_return"] > -0.5, (
            f"Total return {test_report['total_return']:.3f} is suspiciously low "
            "for a monotonically rising market"
        )

    if case.reward_trend_checks:
        _assert_reward_trend(logs, case)


class TestGeneratedDataScenarioSmoke:
    """Smoke tests for full experiment runs on synthetic generated data.

    Each scenario generates a small in-memory dataset from a known pattern
    (sine wave, upward drift) and runs the full training + evaluation pipeline
    with minimal steps. Tests verify that the pipeline completes and that
    evaluation outputs are numerically sane for the given market pattern.
    MLflow artifact I/O is stubbed out.
    """

    pytestmark = pytest.mark.smoke

    @pytest.mark.parametrize("case", SMOKE_CASES, ids=lambda case: case.id)
    def test_scenario(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        case: GeneratedDataScenarioCase,
    ) -> None:
        _run_generated_data_scenario(monkeypatch, tmp_path, case)


class TestGeneratedDataScenarioPushSmoke:
    """Larger generated-data smoke tests intended for the pre-push hook."""

    pytestmark = pytest.mark.push_smoke

    @pytest.mark.parametrize("case", PUSH_SMOKE_CASES, ids=lambda case: case.id)
    def test_scenario(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        case: GeneratedDataScenarioCase,
    ) -> None:
        _run_generated_data_scenario(monkeypatch, tmp_path, case)
