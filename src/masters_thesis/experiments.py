#!/usr/bin/env python3
"""Run H1, H2, H3, H4, or all hypothesis experiments.

Steps for each hypothesis
-------------------------
0. Guardrails   -- validate config for all scenarios (auto-skipped with --dev)
1. Train        -- fit the agent(s)
2. Evaluate     -- compute metrics, benchmarks, plots
3. Report       -- run the hypothesis-specific summary script
4. Export       -- write thesis snapshots for Quarto rendering

Examples
--------
    uv run thesis-experiments h1
    uv run thesis-experiments h2 --skip-train
    uv run thesis-experiments h3 --parallel
    uv run thesis-experiments h4 --trials 5 --steps 200000
    uv run thesis-experiments all
    uv run thesis-experiments h1 --max-train-seconds 300
    uv run thesis-experiments h1 -o training.max_steps=50000
    uv run thesis-experiments h2 -o evaluation.eval_steps=500
    uv run thesis-experiments h1 --dev
    uv run thesis-experiments all --dev --dev-steps 500
"""

from __future__ import annotations

import concurrent.futures
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape

from logger.config import get_global_config as _get_global_logging_config
from trading_rl.config import EXPERIMENT_OUTPUT_DIR

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CLI = ["uv", "run", "python", str(_REPO_ROOT / "src" / "cli.py")]
# Plain-text subprocess captures (guardrails/train/eval .log files) vs.
# structured evaluate output (results.json, benchmark_tables/, plots) --
# these are separately configurable (RL_LOG_DIR vs EXPERIMENT_OUTPUT_DIR)
# even though both default to "logs".
_TEXT_LOG_DIR = _REPO_ROOT / Path(_get_global_logging_config().log_dir)
_EXPERIMENT_OUTPUT_DIR = _REPO_ROOT / EXPERIMENT_OUTPUT_DIR

_con = Console()
_err = Console(stderr=True)

# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

_H1_SCENARIOS = [
    "pooled/td3_hft_lob_state_space_pooled_streaming_selected_dsr",
    "pooled/ddpg_hft_lob_state_space_pooled_streaming_selected_dsr",
    "pooled/ppo_hft_lob_state_space_pooled_streaming_selected_dsr",
    "pooled/random_hft_lob_state_space_pooled_streaming_selected_dsr",
]

_H2_SCENARIOS = [
    "pooled/td3_h3_features_minimal",
    "pooled/td3_hft_lob_state_space_pooled_streaming_selected",  # shared baseline
    "pooled/td3_h3_features_full",
]

# Deduplicate while preserving first-occurrence order — baseline appears in
# multiple axes (feature, reward, transaction-cost) and is trained only once.
_H3_SCENARIOS = list(
    dict.fromkeys(
        [
            "pooled/td3_h3_features_minimal",
            "pooled/td3_hft_lob_state_space_pooled_streaming_selected",  # baseline
            "pooled/td3_h3_features_full",
            "pooled/td3_hft_lob_state_space_pooled_streaming_selected_dsr",
            "pooled/td3_h3_fees_1e6",
            "pooled/td3_h3_fees_1e5",
            "pooled/td3_h3_fees_1e4",
        ]
    )
)

_H4_SCENARIO = "pooled/td3_hft_lob_state_space_pooled_streaming_selected_dsr"

_SCENARIOS: dict[str, list[str]] = {
    "h1": _H1_SCENARIOS,
    "h2": _H2_SCENARIOS,
    "h3": _H3_SCENARIOS,
}

_EVAL_ONLY: dict[str, list[str]] = {
    "h1": ["metrics", "benchmarks", "plots"],
    "h2": ["metrics", "plots"],
    "h3": ["metrics", "plots"],
}

_REPORT_SCRIPTS: dict[str, str] = {
    "h1": "h1_performance_report.py",
    "h2": "h2_feature_sensitivity_report.py",
    "h3": "h3_sensitivity_report.py",
}

# ---------------------------------------------------------------------------
# Shared run args
# ---------------------------------------------------------------------------


@dataclass
class RunArgs:
    skip_train: bool = False
    skip_eval: bool = False
    parallel: bool = False
    verbose: bool = False
    skip_guardrails: bool = False
    overrides: list[str] = field(default_factory=list)
    dev: bool = False
    dev_steps: int = 2000
    max_train_seconds: int | None = None
    # Concurrent scenario subprocesses under --parallel. Each one independently
    # loads its own copy of the val split for the periodic-eval env (~666 MB as
    # a DataFrame for the pooled data) on top of its replay buffer and torch
    # runtime, so running all of them at once (7 for H3) exhausts memory (#517).
    max_parallel: int = 2


# ---------------------------------------------------------------------------
# Low-level subprocess helpers
# ---------------------------------------------------------------------------


def _scenario_name(scenario: str) -> str:
    return scenario.split("/")[-1]


def _log_file(scenario: str, suffix: str) -> Path:
    return _TEXT_LOG_DIR / f"{_scenario_name(scenario)}_{suffix}.log"


def _override_flags(overrides: list[str]) -> list[str]:
    flags: list[str] = []
    for kv in overrides:
        flags += ["--config-override", kv]
    return flags


def _watch_hint(label: str, log_files: list[Path]) -> None:
    paths = " ".join(str(f) for f in log_files)
    _con.print()
    if shutil.which("multitail"):
        _con.print(f"[dim]Monitor {label} logs:[/dim]")
        _con.print(f"  [cyan]multitail -s {len(log_files)} {escape(paths)}[/cyan]")
    else:
        _con.print(
            f"[dim]Monitor {label} logs (install multitail for split-pane view):[/dim]"
        )
        _con.print(f"  [cyan]tail -f {escape(paths)}[/cyan]")
    _con.print()


def _run_tee(cmd: list[str], log_file: Path) -> None:
    """Run command streaming output to both terminal and log file."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    _con.print(f"  [dim]-> {escape(str(log_file))}[/dim]")
    env = os.environ.copy()
    env["FORCE_COLOR"] = "1"  # Rich detects pipe and disables colors without this
    with log_file.open("w") as fh:
        proc = subprocess.Popen(  # noqa: S603 — cmd built internally from _CLI + scenario names, not external input
            cmd,
            cwd=_REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
        )
        assert proc.stdout is not None
        for raw in proc.stdout:
            text = raw.decode(errors="replace")
            sys.stdout.write(text)
            sys.stdout.flush()
            fh.write(text)
        proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def _run_capture(cmd: list[str], log_file: Path) -> None:
    """Run command capturing all output to log file (for parallel threads)."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["NO_COLOR"] = "1"
    with log_file.open("w") as fh:
        subprocess.run(cmd, cwd=_REPO_ROOT, stdout=fh, stderr=fh, env=env, check=True)  # noqa: S603 — cmd built internally from _CLI + scenario names, not external input


def _run_simple(cmd: list[str]) -> None:
    _con.print(f"[dim]$ {escape(' '.join(cmd))}[/dim]")
    subprocess.run(cmd, cwd=_REPO_ROOT, check=True)  # noqa: S603 — cmd built internally from _CLI + scenario names, not external input


def _run_parallel_jobs(
    label: str, jobs: list[tuple[list[str], Path]], max_workers: int = 2
) -> None:
    for _, log in jobs:
        _con.print(
            f"  [cyan]{escape(_scenario_name(str(log.stem)))}[/cyan]  [dim]->[/dim]  [dim]{escape(str(log))}[/dim]  [dim](background)[/dim]"
        )

    n_workers = max(1, min(max_workers, len(jobs)))
    if n_workers < len(jobs):
        _con.print(
            f"[dim]running {len(jobs)} {label} job(s) {n_workers} at a time "
            f"(--max-parallel={max_workers})[/dim]"
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_run_capture, cmd, log): log for cmd, log in jobs}
        _watch_hint(label, [log for _, log in jobs])
        _con.print(f"Waiting for [bold]{len(futures)}[/bold] {label} job(s)...")
        failed: list[Path] = []
        for future in concurrent.futures.as_completed(futures):
            log = futures[future]
            try:
                future.result()
                _con.print(f"  [green]done:[/green] {log.name}")
            except subprocess.CalledProcessError as exc:
                _con.print(
                    f"  [red]FAILED[/red] (rc={exc.returncode}): {escape(str(log))}"
                )
                failed.append(log)

    if failed:
        _err.print(
            f"\n[red]{len(failed)} {label} job(s) failed:[/red] "
            f"{escape(', '.join(f.name for f in failed))}"
        )
        raise typer.Exit(
            code=1,
        )


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------


def _check_guardrails(scenarios: list[str], args: RunArgs) -> None:
    _con.print(
        f"\n[bold]Pre-flight:[/bold] checking guardrails for [cyan]{len(scenarios)}[/cyan] scenario(s)"
    )
    passed: list[str] = []
    failed: list[str] = []

    for scenario in scenarios:
        log = _log_file(scenario, "guardrails")
        log.parent.mkdir(parents=True, exist_ok=True)
        _con.print(f"  Checking [cyan]{escape(scenario)}[/cyan]...")
        cmd = [*_CLI, "validate", "guardrails", "-c", scenario]
        if args.verbose:
            cmd.append("--verbose")
        with log.open("w") as fh:
            result = subprocess.run(  # noqa: S603 — cmd built internally from _CLI + scenario names, not external input
                cmd, cwd=_REPO_ROOT, stdout=fh, stderr=fh, check=False
            )
        if result.returncode == 0:
            _con.print(f"    [green][PASS][/green] {escape(scenario)}")
            passed.append(scenario)
        else:
            _con.print(
                f"    [red][FAIL][/red] {escape(scenario)}  [dim](see {escape(str(log))})[/dim]"
            )
            failed.append(scenario)

    _con.print(
        f"\nGuardrails summary:  "
        f"passed=[green]{len(passed)}[/green]  "
        f"failed=[{'red' if failed else 'green'}]{len(failed)}[/{'red' if failed else 'green'}]"
    )

    if failed:
        _con.print("\n[red]Failed scenarios:[/red]")
        for s in failed:
            _con.print(
                f"  [dim]-[/dim] {escape(s)}  [dim](logs: {escape(str(_log_file(s, 'guardrails')))})[/dim]"
            )
        _err.print(
            "\n[red]Fix the guardrail issues or run with --skip-guardrails to proceed anyway.[/red]"
        )
        raise typer.Exit(code=1)

    _con.print("[green]All scenarios passed guardrails.[/green]\n")


def _train_all(
    scenarios: list[str], args: RunArgs, extra_overrides: list[str] | None = None
) -> None:
    overrides = list(args.overrides) + (extra_overrides or [])

    def _cmd(scenario: str) -> list[str]:
        cmd = [*_CLI, "train", "-c", scenario, *_override_flags(overrides)]
        if args.verbose:
            cmd.append("--verbose")
        return cmd

    if args.parallel:
        _run_parallel_jobs(
            "training",
            [(_cmd(s), _log_file(s, "train")) for s in scenarios],
            max_workers=args.max_parallel,
        )
    else:
        for scenario in scenarios:
            log = _log_file(scenario, "train")
            _con.print(f"Training [cyan]{escape(scenario)}[/cyan]")
            _run_tee(_cmd(scenario), log)
            _con.print("  [green]done.[/green]")


def _evaluate_all(scenarios: list[str], eval_only: list[str], args: RunArgs) -> None:
    overrides = list(args.overrides)

    def _cmd(scenario: str) -> list[str]:
        output_dir = str(_EXPERIMENT_OUTPUT_DIR / _scenario_name(scenario))
        cmd = [
            *_CLI,
            "evaluate",
            "-c",
            scenario,
            "--output-dir",
            output_dir,
            *[flag for only in eval_only for flag in ("--only", only)],
            *_override_flags(overrides),
        ]
        if args.verbose:
            cmd.append("--verbose")
        return cmd

    if args.parallel:
        _run_parallel_jobs(
            "evaluation",
            [(_cmd(s), _log_file(s, "eval")) for s in scenarios],
            max_workers=args.max_parallel,
        )
    else:
        for scenario in scenarios:
            log = _log_file(scenario, "eval")
            output_dir = _EXPERIMENT_OUTPUT_DIR / _scenario_name(scenario)
            _con.print(
                f"Evaluating [cyan]{escape(scenario)}[/cyan]  [dim]->[/dim]  [dim]{escape(str(output_dir))}[/dim]"
            )
            _run_tee(_cmd(scenario), log)
            _con.print("  [green]done.[/green]")


def _run_report(hypothesis: str) -> None:
    script = _REPO_ROOT / "scripts" / _REPORT_SCRIPTS[hypothesis]
    _con.print(f"\n[bold cyan]=== {hypothesis.upper()}: Report ===[/bold cyan]")
    _run_simple(["uv", "run", "python", str(script)])


def _export_all(scenarios: list[str]) -> None:
    export_script = str(_REPO_ROOT / "scripts" / "export_eval_to_thesis.py")
    for scenario in scenarios:
        _con.print(f"  Exporting [cyan]{escape(scenario)}[/cyan] ...")
        _run_simple(["uv", "run", "python", export_script, "--scenario", scenario])
    _con.print("[green]Thesis snapshots updated.[/green]")


# ---------------------------------------------------------------------------
# Hypothesis runner
# ---------------------------------------------------------------------------


def run_hypothesis(hypothesis: str, args: RunArgs) -> None:
    scenarios = _SCENARIOS[hypothesis]
    eval_only = _EVAL_ONLY[hypothesis]
    skip_guardrails = args.skip_guardrails or args.dev

    _TEXT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    _EXPERIMENT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.dev:
        _con.print(
            f"[yellow][dev][/yellow] guardrails skipped, "
            f"training capped at [bold]{args.dev_steps}[/bold] steps per scenario"
        )

    # Step 0 — Guardrails
    if not skip_guardrails:
        _check_guardrails(scenarios, args)

    # Step 1 — Train
    if not args.skip_train:
        _con.print(f"\n[bold cyan]=== {hypothesis.upper()}: Training ===[/bold cyan]")
        extra: list[str] = []
        if args.max_train_seconds:
            extra.append(f"training.max_train_seconds={args.max_train_seconds}")
        if args.dev:
            extra.append(f"training.max_steps={args.dev_steps}")
            extra.append("training.skip_guardrails=true")
        _train_all(scenarios, args, extra_overrides=extra)
        _con.print()

    # Steps 2–4 — Evaluate, Report, Export
    if not args.skip_eval:
        _con.print(f"\n[bold cyan]=== {hypothesis.upper()}: Evaluating ===[/bold cyan]")
        _evaluate_all(scenarios, eval_only, args)

        _run_report(hypothesis)

        _con.print(
            f"\n[bold cyan]=== {hypothesis.upper()}: Export to thesis ===[/bold cyan]"
        )
        _export_all(scenarios)
    else:
        _con.print(
            f"[dim]=== {hypothesis.upper()}: skipping evaluate, report, and export (--skip-eval) ===[/dim]"
        )


def run_h4(
    scenario: str,
    trials: int,
    steps: int,
    args: RunArgs,
) -> None:
    """Run the H4 multi-trial learning-progression workflow."""
    _TEXT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    _EXPERIMENT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_guardrails:
        _check_guardrails([scenario], args)

    if not args.skip_train:
        _con.print(
            f"\n[bold cyan]=== H4: Training {trials} trials "
            f"(max_steps={steps}) ===[/bold cyan]"
        )
        train_overrides = [
            f"training.max_steps={steps}",
            "evaluation.eval_fraction=0.05",
            "training.temp_eval.max_steps=5000",
            "evaluation.skip_final_eval=true",
            *args.overrides,
            *shlex.split(os.environ.get("EXTRA_TRAIN_ARGS", "")),
        ]
        cmd = [
            *_CLI,
            "train",
            "-c",
            scenario,
            "--trials",
            str(trials),
            *_override_flags(train_overrides),
        ]
        if args.verbose:
            cmd.append("--verbose")
        _run_tee(cmd, _log_file(scenario, "train"))

    if args.skip_eval:
        _con.print(
            "[dim]=== H4: skipping evaluate, report, and export (--skip-eval) ===[/dim]"
        )
        return

    _con.print(f"\n[bold cyan]=== H4: Evaluating {trials} trials ===[/bold cyan]")
    eval_overrides = [
        f"training.max_steps={steps}",
        *args.overrides,
        *shlex.split(os.environ.get("EXTRA_EVAL_ARGS", "")),
    ]
    eval_cmd = [
        *_CLI,
        "evaluate",
        "-c",
        scenario,
        *_override_flags(eval_overrides),
    ]
    if args.verbose:
        eval_cmd.append("--verbose")
    _run_tee(eval_cmd, _log_file(scenario, "eval"))

    _con.print("\n[bold cyan]=== H4: Learning progression report ===[/bold cyan]")
    _run_simple(
        [
            "uv",
            "run",
            "python",
            str(_REPO_ROOT / "scripts" / "h4_learning_progression_report.py"),
            "--scenario",
            scenario,
            "--n-trials",
            str(trials),
            "--max-steps",
            str(steps),
        ]
    )

    _con.print("\n[bold cyan]=== H4: Export to thesis ===[/bold cyan]")
    _export_all([scenario])


# ---------------------------------------------------------------------------
# Typer app
# ---------------------------------------------------------------------------

app = typer.Typer(
    help="Run H1, H2, H3, H4, or all hypothesis experiments.",
    rich_markup_mode="rich",
    no_args_is_help=True,
)

# Reusable option type aliases
_SkipTrain = Annotated[bool, typer.Option("--skip-train", help="Skip training step.")]
_SkipEval = Annotated[
    bool, typer.Option("--skip-eval", help="Skip evaluate / report / export steps.")
]
_Parallel = Annotated[
    bool, typer.Option("--parallel", help="Run scenarios concurrently.")
]
_MaxParallel = Annotated[
    int,
    typer.Option(
        "--max-parallel",
        min=1,
        metavar="N",
        help="Max concurrent scenarios under [bold]--parallel[/bold] (default: 2). "
        "Each holds its own copy of the val split, so raise with care.",
    ),
]
_Verbose = Annotated[
    bool, typer.Option("--verbose", "-v", help="Enable debug logging in subcommands.")
]
_SkipGuardrails = Annotated[
    bool, typer.Option("--skip-guardrails", help="Skip pre-flight guardrails check.")
]
_Overrides = Annotated[
    list[str] | None,
    typer.Option(
        "-o",
        "--config-override",
        help="OmegaConf dotlist override forwarded to both train and evaluate. Repeatable.",
    ),
]
_Dev = Annotated[
    bool,
    typer.Option(
        "--dev",
        help="Dev mode: skip guardrails and cap training at [bold]--dev-steps[/bold] steps.",
    ),
]
_DevSteps = Annotated[
    int,
    typer.Option(
        "--dev-steps",
        help="Training steps per scenario in [bold]--dev[/bold] mode (default: 2000).",
    ),
]


@app.command()
def h1(
    skip_train: _SkipTrain = False,
    skip_eval: _SkipEval = False,
    parallel: _Parallel = False,
    max_parallel: _MaxParallel = 2,
    verbose: _Verbose = False,
    skip_guardrails: _SkipGuardrails = False,
    overrides: _Overrides = None,
    dev: _Dev = False,
    dev_steps: _DevSteps = 2000,
    max_train_seconds: Annotated[
        int | None,
        typer.Option(
            "--max-train-seconds",
            metavar="N",
            help="Cap training wall-clock time per scenario (forwarded as training.max_train_seconds=N).",
        ),
    ] = None,
) -> None:
    """Test whether TD3 outperforms DDPG, PPO, and a random-policy baseline.

    Trains and evaluates the four agents with the selected LOB state space and
    differential-Sharpe reward, then generates the H1 performance comparison.
    """
    run_hypothesis(
        "h1",
        RunArgs(
            skip_train=skip_train,
            skip_eval=skip_eval,
            parallel=parallel,
            max_parallel=max_parallel,
            verbose=verbose,
            skip_guardrails=skip_guardrails,
            overrides=overrides or [],
            dev=dev,
            dev_steps=dev_steps,
            max_train_seconds=max_train_seconds,
        ),
    )


@app.command()
def h2(
    skip_train: _SkipTrain = False,
    skip_eval: _SkipEval = False,
    parallel: _Parallel = False,
    max_parallel: _MaxParallel = 2,
    verbose: _Verbose = False,
    skip_guardrails: _SkipGuardrails = False,
    overrides: _Overrides = None,
    dev: _Dev = False,
    dev_steps: _DevSteps = 2000,
) -> None:
    """Test how the observation feature set affects TD3 performance.

    Compares minimal, selected, and full feature specifications while holding
    the learning algorithm and the remaining experiment design fixed.
    """
    run_hypothesis(
        "h2",
        RunArgs(
            skip_train=skip_train,
            skip_eval=skip_eval,
            parallel=parallel,
            max_parallel=max_parallel,
            verbose=verbose,
            skip_guardrails=skip_guardrails,
            overrides=overrides or [],
            dev=dev,
            dev_steps=dev_steps,
        ),
    )


@app.command()
def h3(
    skip_train: _SkipTrain = False,
    skip_eval: _SkipEval = False,
    parallel: _Parallel = False,
    max_parallel: _MaxParallel = 2,
    verbose: _Verbose = False,
    skip_guardrails: _SkipGuardrails = False,
    overrides: _Overrides = None,
    dev: _Dev = False,
    dev_steps: _DevSteps = 2000,
) -> None:
    """Test whether the main result is robust to modelling choices.

    Runs sensitivity analyses across feature specifications, reward variants,
    and transaction-cost assumptions, then produces the H3 robustness report.
    """
    run_hypothesis(
        "h3",
        RunArgs(
            skip_train=skip_train,
            skip_eval=skip_eval,
            parallel=parallel,
            max_parallel=max_parallel,
            verbose=verbose,
            skip_guardrails=skip_guardrails,
            overrides=overrides or [],
            dev=dev,
            dev_steps=dev_steps,
        ),
    )


@app.command()
def h4(
    scenario: Annotated[
        str,
        typer.Option(
            "--scenario", envvar="SCENARIO", help="Scenario configuration name."
        ),
    ] = _H4_SCENARIO,
    trials: Annotated[
        int,
        typer.Option(
            "--trials", envvar="N_TRIALS", min=1, help="Number of independent trials."
        ),
    ] = 5,
    steps: Annotated[
        int,
        typer.Option(
            "--steps",
            envvar="STEPS",
            min=1,
            help="Maximum training steps per trial.",
        ),
    ] = 200_000,
    skip_train: _SkipTrain = False,
    skip_eval: _SkipEval = False,
    verbose: _Verbose = False,
    skip_guardrails: _SkipGuardrails = False,
    overrides: _Overrides = None,
) -> None:
    """Test whether TD3 learns consistently across independent short trials.

    Runs repeated seeds with a bounded training budget, evaluates the resulting
    checkpoints, and compares their learning progression with the baseline.
    """
    run_h4(
        scenario,
        trials,
        steps,
        RunArgs(
            skip_train=skip_train,
            skip_eval=skip_eval,
            verbose=verbose,
            skip_guardrails=skip_guardrails,
            overrides=overrides or [],
        ),
    )


@app.command(name="all")
def run_all(
    skip_train: _SkipTrain = False,
    skip_eval: _SkipEval = False,
    parallel: _Parallel = False,
    max_parallel: _MaxParallel = 2,
    verbose: _Verbose = False,
    skip_guardrails: _SkipGuardrails = False,
    overrides: _Overrides = None,
    dev: _Dev = False,
    dev_steps: _DevSteps = 2000,
) -> None:
    """Run [bold]H1[/bold], [bold]H2[/bold], and [bold]H3[/bold] in sequence."""
    args = RunArgs(
        skip_train=skip_train,
        skip_eval=skip_eval,
        parallel=parallel,
        max_parallel=max_parallel,
        verbose=verbose,
        skip_guardrails=skip_guardrails,
        overrides=overrides or [],
        dev=dev,
        dev_steps=dev_steps,
    )
    for hyp in ("h1", "h2", "h3"):
        run_hypothesis(hyp, args)
    _con.print("\n[bold green]All done.[/bold green]")


if __name__ == "__main__":
    app()
