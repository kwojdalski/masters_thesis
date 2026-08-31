from __future__ import annotations

from types import ModuleType

import pytest
from typer.testing import CliRunner

from masters_thesis import experiments


@pytest.mark.parametrize(
    ("command", "description"),
    [
        ("h1", "TD3 outperforms DDPG, PPO, and a random-policy baseline"),
        ("h2", "observation feature set affects TD3 performance"),
        ("h3", "main result is robust to modelling choices"),
        ("h4", "TD3 learns consistently across independent short trials"),
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


def test_h4_builds_train_eval_report_and_export_commands(
    runner: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    tee_commands: list[list[str]] = []
    simple_commands: list[list[str]] = []
    checked: list[list[str]] = []
    exported: list[list[str]] = []
    monkeypatch.delenv("EXTRA_TRAIN_ARGS", raising=False)
    monkeypatch.delenv("EXTRA_EVAL_ARGS", raising=False)
    monkeypatch.setattr(
        runner, "_check_guardrails", lambda scenarios, _args: checked.append(scenarios)
    )
    monkeypatch.setattr(
        runner, "_run_tee", lambda command, _log: tee_commands.append(command)
    )
    monkeypatch.setattr(runner, "_run_simple", simple_commands.append)
    monkeypatch.setattr(runner, "_export_all", exported.append)

    runner.run_h4(
        "scenario/name",
        3,
        1_000,
        runner.RunArgs(overrides=["training.seed=7"], verbose=True),
    )

    assert checked == [["scenario/name"]]
    assert tee_commands[0][4:9] == ["train", "-c", "scenario/name", "--trials", "3"]
    assert "training.max_steps=1000" in tee_commands[0]
    assert "evaluation.eval_fraction=0.05" in tee_commands[0]
    assert "training.seed=7" in tee_commands[0]
    assert tee_commands[0][-1] == "--verbose"
    assert tee_commands[1][4:7] == ["evaluate", "-c", "scenario/name"]
    assert "training.max_steps=1000" in tee_commands[1]
    assert tee_commands[1][-1] == "--verbose"
    report_command = simple_commands[0]
    assert report_command[report_command.index("--scenario") :][:6] == [
        "--scenario",
        "scenario/name",
        "--n-trials",
        "3",
        "--max-steps",
        "1000",
    ]
    assert report_command[report_command.index("--output-dir") + 1].endswith(
        "logs/name"
    )
    assert exported == [["scenario/name"]]


def test_h4_skip_eval_stops_after_training(
    runner: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    tee_commands: list[list[str]] = []
    monkeypatch.delenv("EXTRA_TRAIN_ARGS", raising=False)
    monkeypatch.setattr(
        runner, "_run_tee", lambda command, _log: tee_commands.append(command)
    )
    monkeypatch.setattr(
        runner, "_run_simple", lambda _command: pytest.fail("report should not run")
    )
    monkeypatch.setattr(
        runner, "_export_all", lambda _scenarios: pytest.fail("export should not run")
    )

    runner.run_h4(
        "scenario/name",
        2,
        500,
        runner.RunArgs(skip_eval=True, skip_guardrails=True),
    )

    assert len(tee_commands) == 1
    assert tee_commands[0][4] == "train"


def test_h4_parallel_runs_one_job_per_trial_with_distinct_seeds_and_dirs(
    runner: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs_seen: list[list[tuple[list[str], object]]] = []
    tee_commands: list[list[str]] = []
    monkeypatch.delenv("EXTRA_TRAIN_ARGS", raising=False)
    monkeypatch.delenv("EXTRA_EVAL_ARGS", raising=False)
    monkeypatch.setattr(runner, "_check_guardrails", lambda scenarios, _args: None)
    monkeypatch.setattr(
        runner,
        "_run_parallel_jobs",
        lambda label, jobs, max_workers: jobs_seen.append(jobs),
    )
    monkeypatch.setattr(
        runner, "_run_tee", lambda command, _log: tee_commands.append(command)
    )
    monkeypatch.setattr(runner, "_run_simple", lambda _command: None)
    monkeypatch.setattr(runner, "_export_all", lambda _scenarios: None)
    monkeypatch.setattr(runner.random, "randint", lambda _lo, _hi: 42)

    runner.run_h4(
        "scenario/name",
        3,
        1_000,
        runner.RunArgs(parallel=True, max_parallel=2),
    )

    jobs = jobs_seen[0]
    assert len(jobs) == 3
    scenario_dir = runner._EXPERIMENT_OUTPUT_DIR / "name"
    seeds = []
    for i, (cmd, log) in enumerate(jobs):
        assert cmd[4:7] == ["train", "-c", "scenario/name"]
        assert "--trials" not in cmd
        trial_dir = scenario_dir / f"trial_{i}"
        assert f"logging.log_dir={trial_dir}" in cmd
        seed_flag = next(o for o in cmd if o.startswith("seed="))
        seeds.append(int(seed_flag.split("=", 1)[1]))
        assert log == trial_dir / f"trial_{i}_train.log"
    assert seeds == [42, 43, 44]

    # Post-training evaluate reads the last trial's own directory, since
    # each parallel trial has its own log_dir instead of one shared dir.
    last_trial_dir = scenario_dir / "trial_2"
    assert f"logging.log_dir={last_trial_dir}" in tee_commands[0]


def test_debug_set_supplies_small_h4_defaults(
    runner: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[tuple[str, int, int, runner.RunArgs]] = []
    monkeypatch.setattr(
        runner,
        "run_h4",
        lambda scenario, trials, steps, args: captured.append(
            (scenario, trials, steps, args)
        ),
    )

    result = CliRunner().invoke(
        runner.app, ["h4", "--debug", "--skip-train", "--skip-eval"]
    )

    assert result.exit_code == 0, result.stdout
    _, trials, steps, args = captured[0]
    assert (trials, steps) == (2, 1000)
    assert args.experiment_set.name == "debug"
    assert "data.max_rows_per_file=20000" in args.overrides
    assert not args.experiment_set.export_to_thesis
