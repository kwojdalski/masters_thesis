from __future__ import annotations

from types import ModuleType

import pytest
from typer.testing import CliRunner

from masters_thesis import experiments


@pytest.mark.parametrize(
    ("command", "description"),
    [
        ("h1", "continuous-control agents beat a random-policy baseline"),
        ("h2", "transaction-cost assumption affects TD3 performance"),
        ("h3", "observation feature set affects TD3 performance"),
        ("h4", "reward function changes the learned policy"),
        ("h5", "execution latency affects TD3 performance"),
    ],
)
def test_hypothesis_help_describes_the_research_question(
    command: str, description: str
) -> None:
    result = CliRunner().invoke(experiments.app, [command, "--help"])

    assert result.exit_code == 0
    assert description in result.stdout


@pytest.fixture
def runner() -> ModuleType:
    return experiments


@pytest.mark.parametrize(
    ("hypothesis", "expected_scenarios"),
    [
        ("h1", experiments._H1_SCENARIOS),
        ("h2", experiments._H2_SCENARIOS),
        ("h3", experiments._H3_SCENARIOS),
        ("h4", experiments._H4_SCENARIOS),
        ("h5", experiments._H5_SCENARIOS),
    ],
)
def test_each_list_hypothesis_trains_its_own_scenarios(
    runner: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    hypothesis: str,
    expected_scenarios: list[str],
) -> None:
    trained: list[list[str]] = []
    monkeypatch.setattr(runner, "_check_guardrails", lambda _scenarios, _args: None)
    monkeypatch.setattr(
        runner,
        "_train_all",
        lambda scenarios, _args, extra_overrides=None: trained.append(list(scenarios)),
    )
    monkeypatch.setattr(runner, "_evaluate_all", lambda *_a, **_kw: None)

    runner.run_hypothesis(
        hypothesis,
        runner._apply_experiment_set(
            runner.RunArgs(overrides=[], skip_eval=True), "full", False
        ),
    )

    assert trained == [list(expected_scenarios)]


def test_h4_reward_scenarios_are_the_logreturn_and_dsr_baselines() -> None:
    assert experiments._H4_SCENARIOS == [
        "pooled/td3_h4_reward_logreturn",
        "pooled/td3_h4_reward_dsr",
    ]


def test_no_scenario_is_shared_between_hypotheses() -> None:
    """Two hypotheses sharing a scenario share its output directory.

    The runner derives log_dir from the scenario name, so when h3 and h4 both
    listed `td3_hft_lob_state_space_pooled_streaming_selected`, the second to
    run overwrote the first's results.json, checkpoints and rollouts. h4 also
    listed the h1 scenario and destroyed h1's 3M-step evaluation mid-run.
    """
    seen: dict[str, str] = {}
    for hypothesis in ("h1", "h2", "h3", "h4", "h5"):
        for scenario in experiments._SCENARIOS[hypothesis]:
            assert scenario not in seen, (
                f"{scenario} is in both {seen[scenario]} and {hypothesis}; "
                "they would write the same logs/<scenario> directory"
            )
            seen[scenario] = hypothesis


def test_run_all_covers_every_list_hypothesis(
    runner: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str] = []
    monkeypatch.setattr(runner, "run_hypothesis", lambda hyp, _args: seen.append(hyp))

    result = CliRunner().invoke(runner.app, ["all", "--skip-train", "--skip-eval"])

    assert result.exit_code == 0, result.stdout
    assert seen == ["h1", "h2", "h3", "h4", "h5"]


@pytest.mark.parametrize(
    ("hypothesis", "config_name"),
    [
        ("h2", "src/configs/h2_transaction_cost.yaml"),
        ("h4", "src/configs/h4_reward_design.yaml"),
        ("h5", "src/configs/h5_execution_latency.yaml"),
    ],
)
def test_sensitivity_report_config_flag_is_spliced_into_the_command(
    runner: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    hypothesis: str,
    config_name: str,
) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(runner, "_run_simple", commands.append)

    runner._run_report(
        hypothesis,
        runner._apply_experiment_set(runner.RunArgs(overrides=[]), "full", False),
    )

    cmd = commands[0]
    assert cmd[:4] == ["uv", "run", "python", cmd[3]]
    assert cmd[3].endswith("scripts/sensitivity_report.py")
    assert cmd[cmd.index("--config") + 1] == config_name
    assert "--results-root" in cmd


def test_h1_report_takes_no_config_flag(
    runner: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(runner, "_run_simple", commands.append)

    runner._run_report(
        "h1", runner._apply_experiment_set(runner.RunArgs(overrides=[]), "full", False)
    )

    cmd = commands[0]
    assert cmd[3].endswith("scripts/h1_performance_report.py")
    assert "--config" not in cmd


@pytest.mark.parametrize("command", ["h1", "h2", "h3", "h4", "h5", "all"])
def test_max_train_seconds_is_offered_by_every_hypothesis_command(
    command: str,
) -> None:
    """The wall-clock cap must not drift back to being h1-only.

    It was originally declared inline on h1 alone, so `h2 --max-train-seconds`
    failed with an unknown-option error while the identical h1 invocation
    worked.
    """
    result = CliRunner().invoke(experiments.app, [command, "--help"])

    assert result.exit_code == 0
    assert "--max-train-seconds" in result.stdout


@pytest.mark.parametrize("hypothesis", ["h1", "h2", "h3", "h4", "h5"])
def test_max_train_seconds_reaches_the_train_command(
    runner: ModuleType, monkeypatch: pytest.MonkeyPatch, hypothesis: str
) -> None:
    """Being accepted by the parser is not enough — it must reach training."""
    extras: list[list[str]] = []
    monkeypatch.setattr(runner, "_check_guardrails", lambda _scenarios, _args: None)
    monkeypatch.setattr(runner, "_evaluate_all", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        runner,
        "_train_all",
        lambda _scenarios, _args, extra_overrides=None: extras.append(
            extra_overrides or []
        ),
    )

    runner.run_hypothesis(
        hypothesis,
        runner._apply_experiment_set(
            runner.RunArgs(overrides=[], skip_eval=True, max_train_seconds=240),
            "full",
            False,
        ),
    )

    assert "training.max_train_seconds=240" in extras[0]


def test_h5_latency_scenarios_sweep_exec_latency_only() -> None:
    """Every H5 arm must be the H1 configuration with one field changed.

    The ladder only measures latency if `env.exec_latency_us` is the single
    varying factor; a drifting feature set, reward or fee would confound it the
    way the pre-isolation H2 ladder was confounded by train_size and reward.
    """
    import yaml

    scenarios = experiments._H5_SCENARIOS
    assert scenarios == [
        "pooled/td3_h5_latency_0_dsr",
        "pooled/td3_h5_latency_10us_dsr",
        "pooled/td3_h5_latency_100us_dsr",
        "pooled/td3_h5_latency_1ms_dsr",
        "pooled/td3_h5_latency_5ms_dsr",
    ]

    root = experiments._REPO_ROOT / "src" / "configs" / "scenarios"
    baseline_dir = root / "pooled/td3_hft_lob_state_space_pooled_streaming_selected_dsr"

    expected_us = [0.0, 10.0, 100.0, 1000.0, 5000.0]
    for scenario, want_us in zip(scenarios, expected_us, strict=True):
        arm_dir = root / scenario
        arm = yaml.safe_load((arm_dir / "train.yaml").read_text())
        base = yaml.safe_load((baseline_dir / "train.yaml").read_text())

        assert arm["env"]["exec_latency_us"] == want_us, scenario
        # Latency modelling is only honoured by the tradingenv streaming
        # backend; the gym_trading backends raise on non-zero latency.
        assert arm["env"]["backend"] == "tradingenv", scenario

        # Everything else must match the H1 baseline byte for byte.
        arm["env"].pop("exec_latency_us")
        base["env"].pop("exec_latency_us")
        assert arm == base, f"{scenario} differs from the H1 baseline beyond latency"

        for side in ("observation.yaml", "evaluate.yaml"):
            assert (arm_dir / side).read_text() == (baseline_dir / side).read_text(), (
                f"{scenario}/{side} differs from the H1 baseline"
            )
