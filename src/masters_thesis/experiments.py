#!/usr/bin/env python3
"""Run H1, H2, H3, H4, H5, or all hypothesis experiments.

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
    uv run thesis-experiments h4 --skip-train
    uv run thesis-experiments h5
    uv run thesis-experiments all
    uv run thesis-experiments h2 --max-train-seconds 300
    uv run thesis-experiments h1 -o training.max_steps=50000
    uv run thesis-experiments h2 -o evaluation.eval_steps=500
    uv run thesis-experiments h1 --dev
    uv run thesis-experiments all --dev --dev-steps 500
"""

from __future__ import annotations

import concurrent.futures
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import typer
import yaml
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
_EXPERIMENT_SET_DIR = _REPO_ROOT / "src" / "configs" / "experiment_sets"

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

# Transaction-cost sensitivity. The 0 bp arm is the H1 agent itself, and each
# fee arm differs from it by `env.trading_fees` alone.
#
# The previous ladder was built on `td3_hft_lob_state_space_pooled_streaming_selected`
# (log-return reward, `train_size: 50000`). That agent holds a near-constant
# position -- turnover 5.5e-04 against the H1 agent's 0.43 -- so a per-trade fee
# had nothing to act on and the ladder was inert at every fee level and every
# training budget. It also meant the sweep did not measure the cost sensitivity
# of the agent H1 actually reports. Both are fixed by sweeping the H1 agent.
_H2_SCENARIOS = [
    "pooled/td3_h2_fees_0_dsr",  # 0 bp; byte-identical to the H1 scenario
    "pooled/td3_h2_fees_1e6_dsr",
    "pooled/td3_h2_fees_1e5_dsr",
    "pooled/td3_h2_fees_1e4_dsr",
]

# Feature specification: minimal / selected / full state representations. All
# three arms now share the H1 train.yaml and differ only in observation.yaml, so
# the feature set is the single varying factor. Previously the minimal arm also
# carried a smaller network ([64, 32] vs [128, 64]), confounding feature count
# with model capacity.
_H3_SCENARIOS = [
    "pooled/td3_h3_features_minimal",
    "pooled/td3_h3_features_selected",
    "pooled/td3_h3_features_full",
]

# Reward-function design. Both arms are the H1 configuration; only reward_type
# and its scaling differ. The previous pair also differed in train_size (50000
# vs null), actor_weight_decay and reward_scale, so no difference could be
# attributed to the reward.
_H4_SCENARIOS = [
    "pooled/td3_h4_reward_logreturn",
    "pooled/td3_h4_reward_dsr",
]

# Execution latency. Each arm is the H1 configuration differing only by
# `env.exec_latency_us`, which delays the fill by the number of rows spanning
# that many microseconds: the agent observes rows [0 .. N-k-1] and is filled at
# rows [k .. N-1].
#
# The ladder is specified in microseconds, not ticks, because inter-event
# spacing varies roughly eight-fold across the six instruments (median gap
# 21 us on TSLA against 173 us on META); a fixed tick count would mean a
# different physical delay per symbol. FixedTimedLatency resolves the value
# against each episode's own timestamps at reset.
#
# Latency requires the `tradingenv` streaming backend -- the gym_trading
# backends raise on non-zero latency params rather than silently ignoring them
# (envs/builder.py). Every H5 arm inherits `backend: tradingenv` from H1.
_H5_SCENARIOS = [
    "pooled/td3_h5_latency_0_dsr",  # 0 us; byte-identical to the H1 scenario
    "pooled/td3_h5_latency_100us_dsr",
    "pooled/td3_h5_latency_1ms_dsr",
    "pooled/td3_h5_latency_5ms_dsr",
]

_SCENARIOS: dict[str, list[str]] = {
    "h1": _H1_SCENARIOS,
    "h2": _H2_SCENARIOS,
    "h3": _H3_SCENARIOS,
    "h4": _H4_SCENARIOS,
    "h5": _H5_SCENARIOS,
}

_EVAL_ONLY: dict[str, list[str]] = {
    # "stats" runs the t_test / sharpe_bootstrap / sortino_bootstrap /
    # mann_whitney declared in each h1 scenario's evaluate.yaml; without it the
    # exported statistical_tests.json is only the benchmark comparison table
    # (experiment-audit 2026-08-31 finding #9).
    "h1": ["metrics", "benchmarks", "plots", "stats"],
    # h2-h4 render no figures: every table in 06-02 reads metrics via
    # load_scenario_metrics, and the only show_plot calls in the thesis are the
    # three h1 figures in 06-03. Generating plots here wrote ~1.4 GB of per-step
    # plot CSV per scenario that nothing read. The per-step rollout is still
    # retained (--save-rollout in _evaluate_all), so any figure can be rebuilt
    # offline without re-running the policy.
    "h2": ["metrics"],
    "h3": ["metrics"],
    "h4": ["metrics"],
    "h5": ["metrics"],
}

# Values are argv tails for scripts/<name>; _run_report splices in the
# --results-root flag. The two sensitivity axes share one generic script with a
# per-axis --config.
_REPORT_SCRIPTS: dict[str, list[str]] = {
    "h1": ["h1_performance_report.py"],
    "h2": ["sensitivity_report.py", "--config", "src/configs/h2_transaction_cost.yaml"],
    "h3": ["h3_feature_sensitivity_report.py"],
    "h4": ["sensitivity_report.py", "--config", "src/configs/h4_reward_design.yaml"],
    "h5": [
        "sensitivity_report.py",
        "--config",
        "src/configs/h5_execution_latency.yaml",
    ],
}


@dataclass(frozen=True)
class ExperimentSet:
    """A named collection of scenarios and shared pipeline settings."""

    name: str
    output_root: Path
    export_to_thesis: bool
    hypotheses: dict[str, object]
    overrides: list[str]


def _load_experiment_set(name: str) -> ExperimentSet:
    if not name or Path(name).name != name:
        raise typer.BadParameter(f"Invalid experiment set name: {name!r}")
    path = _EXPERIMENT_SET_DIR / f"{name}.yaml"
    if not path.is_file():
        available = ", ".join(
            sorted(p.stem for p in _EXPERIMENT_SET_DIR.glob("*.yaml"))
        )
        raise typer.BadParameter(
            f"Unknown experiment set {name!r}. Available sets: {available or 'none'}"
        )
    with path.open() as fh:
        raw = yaml.safe_load(fh) or {}
    hypotheses = raw.get("hypotheses")
    if not isinstance(hypotheses, dict):
        raise typer.BadParameter(f"Experiment set {name!r} has no hypotheses mapping")
    overrides = raw.get("overrides", [])
    if not isinstance(overrides, list) or not all(
        isinstance(v, str) for v in overrides
    ):
        raise typer.BadParameter(f"Experiment set {name!r} overrides must be strings")
    output_root = Path(raw.get("output_root", EXPERIMENT_OUTPUT_DIR))
    if not output_root.is_absolute():
        output_root = _REPO_ROOT / output_root
    return ExperimentSet(
        name=name,
        output_root=output_root,
        export_to_thesis=bool(raw.get("export_to_thesis", True)),
        hypotheses=hypotheses,
        overrides=overrides,
    )


def _resolve_set_name(set_name: str, debug: bool) -> str:
    if debug and set_name != "full":
        raise typer.BadParameter("Use either --debug or --set, not both")
    return "debug" if debug else set_name


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
    # runtime, so running all of them at once (4 for H2) exhausts memory (#517).
    max_parallel: int = 2
    experiment_set: ExperimentSet = field(
        default_factory=lambda: _load_experiment_set("full")
    )


def _apply_experiment_set(args: RunArgs, set_name: str, debug: bool) -> RunArgs:
    experiment_set = _load_experiment_set(_resolve_set_name(set_name, debug))
    args.experiment_set = experiment_set
    # Set defaults come first so explicit CLI overrides remain authoritative.
    args.overrides = [*experiment_set.overrides, *args.overrides]
    _con.print(
        f"[dim]experiment set:[/dim] [cyan]{experiment_set.name}[/cyan]  "
        f"[dim]output:[/dim] {escape(str(experiment_set.output_root))}"
    )
    return args


# ---------------------------------------------------------------------------
# Low-level subprocess helpers
# ---------------------------------------------------------------------------


def _scenario_name(scenario: str) -> str:
    return scenario.split("/")[-1]


def _log_file(scenario: str, suffix: str, args: RunArgs | None = None) -> Path:
    root = args.experiment_set.output_root if args else _TEXT_LOG_DIR
    return root / f"{_scenario_name(scenario)}_{suffix}.log"


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
    """Run command streaming output to both terminal and log file.

    On Ctrl-C the SIGINT also reaches the child (shared process group), but we
    still terminate and reap it here: without this, KeyboardInterrupt aborts the
    read loop before ``proc.wait()``, leaving an unreaped subprocess and an
    unclosed pipe that surface as ResourceWarnings during interpreter shutdown.
    """
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
        try:
            for raw in proc.stdout:
                text = raw.decode(errors="replace")
                sys.stdout.write(text)
                sys.stdout.flush()
                fh.write(text)
            proc.wait()
        except BaseException:
            # Ctrl-C or a mid-stream error: stop the child and reap it so no
            # orphan process or unclosed pipe outlives this call.
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            raise
        finally:
            proc.stdout.close()
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
        log = _log_file(scenario, "guardrails", args)
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
                f"  [dim]-[/dim] {escape(s)}  [dim](logs: {escape(str(_log_file(s, 'guardrails', args)))})[/dim]"
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
        scenario_overrides = [
            f"logging.log_dir={args.experiment_set.output_root / _scenario_name(scenario)}",
            *overrides,
        ]
        cmd = [*_CLI, "train", "-c", scenario, *_override_flags(scenario_overrides)]
        if args.verbose:
            cmd.append("--verbose")
        return cmd

    if args.parallel:
        _run_parallel_jobs(
            "training",
            [(_cmd(s), _log_file(s, "train", args)) for s in scenarios],
            max_workers=args.max_parallel,
        )
    else:
        for scenario in scenarios:
            log = _log_file(scenario, "train", args)
            _con.print(f"Training [cyan]{escape(scenario)}[/cyan]")
            _run_tee(_cmd(scenario), log)
            _con.print("  [green]done.[/green]")


def _evaluate_all(scenarios: list[str], eval_only: list[str], args: RunArgs) -> None:
    overrides = list(args.overrides)

    def _cmd(scenario: str) -> list[str]:
        output_dir = str(args.experiment_set.output_root / _scenario_name(scenario))
        scenario_overrides = [f"logging.log_dir={output_dir}", *overrides]
        cmd = [
            *_CLI,
            "evaluate",
            "-c",
            scenario,
            "--output-dir",
            output_dir,
            # Every pooled scenario sets data.val_data_paths, and without this
            # flag `evaluate` skips val/test entirely and reports the TRAIN
            # split instead -- silently turning in-sample numbers into the
            # thesis's out-of-sample results.
            "--per-symbol",
            # The complete per-step trace (action, simple_return,
            # cumulative_log_return) per split, ~10 MB. Every metric in the
            # results tables can be recomputed from it offline via
            # build_metric_report, so a metrics bug no longer costs a re-run of
            # the policy -- which is hours per scenario at pooled sizes.
            "--save-rollout",
            *[flag for only in eval_only for flag in ("--only", only)],
            *_override_flags(scenario_overrides),
        ]
        if args.verbose:
            cmd.append("--verbose")
        return cmd

    if args.parallel:
        _run_parallel_jobs(
            "evaluation",
            [(_cmd(s), _log_file(s, "eval", args)) for s in scenarios],
            max_workers=args.max_parallel,
        )
    else:
        for scenario in scenarios:
            log = _log_file(scenario, "eval", args)
            output_dir = args.experiment_set.output_root / _scenario_name(scenario)
            _con.print(
                f"Evaluating [cyan]{escape(scenario)}[/cyan]  [dim]->[/dim]  [dim]{escape(str(output_dir))}[/dim]"
            )
            _run_tee(_cmd(scenario), log)
            _con.print("  [green]done.[/green]")


def _run_report(hypothesis: str, args: RunArgs) -> None:
    script_args = _REPORT_SCRIPTS[hypothesis]
    script = _REPO_ROOT / "scripts" / script_args[0]
    _con.print(f"\n[bold cyan]=== {hypothesis.upper()}: Report ===[/bold cyan]")
    _run_simple(
        [
            "uv",
            "run",
            "python",
            str(script),
            *script_args[1:],
            "--results-root",
            str(args.experiment_set.output_root),
        ]
    )


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
    configured = args.experiment_set.hypotheses.get(hypothesis, _SCENARIOS[hypothesis])
    if not isinstance(configured, list) or not all(
        isinstance(value, str) for value in configured
    ):
        raise typer.BadParameter(
            f"Experiment set {args.experiment_set.name!r} must define "
            f"{hypothesis} as a list of scenarios"
        )
    scenarios = configured
    eval_only = _EVAL_ONLY[hypothesis]
    skip_guardrails = args.skip_guardrails or args.dev

    args.experiment_set.output_root.mkdir(parents=True, exist_ok=True)

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
        if skip_guardrails:
            extra.append("training.skip_guardrails=true")
        _train_all(scenarios, args, extra_overrides=extra)
        _con.print()

    # Steps 2–4 — Evaluate, Report, Export
    if not args.skip_eval:
        _con.print(f"\n[bold cyan]=== {hypothesis.upper()}: Evaluating ===[/bold cyan]")
        _evaluate_all(scenarios, eval_only, args)

        _run_report(hypothesis, args)

        if args.experiment_set.export_to_thesis:
            _con.print(
                f"\n[bold cyan]=== {hypothesis.upper()}: Export to thesis ===[/bold cyan]"
            )
            _export_all(scenarios)
    else:
        _con.print(
            f"[dim]=== {hypothesis.upper()}: skipping evaluate, report, and export (--skip-eval) ===[/dim]"
        )


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
_MaxTrainSeconds = Annotated[
    int | None,
    typer.Option(
        "--max-train-seconds",
        metavar="N",
        help="Cap training wall-clock time per scenario (forwarded as training.max_train_seconds=N).",
    ),
]
_SetName = Annotated[
    str,
    typer.Option(
        "--set",
        metavar="NAME",
        help="Named experiment set from src/configs/experiment_sets (default: full).",
    ),
]
_DebugSet = Annotated[
    bool,
    typer.Option(
        "--debug",
        help="Shorthand for --set debug (small data, budgets, and isolated outputs).",
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
    set_name: _SetName = "full",
    debug: _DebugSet = False,
    max_train_seconds: _MaxTrainSeconds = None,
) -> None:
    """Test whether continuous-control agents beat a random-policy baseline.

    Trains and evaluates TD3, DDPG, PPO, and Random with the selected LOB state
    space and differential-Sharpe reward, then generates the H1 performance
    comparison. The learners are not expected to separate from each other; the
    comparison tests the learned-versus-unlearned boundary.
    """
    run_hypothesis(
        "h1",
        _apply_experiment_set(
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
            set_name,
            debug,
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
    set_name: _SetName = "full",
    debug: _DebugSet = False,
    max_train_seconds: _MaxTrainSeconds = None,
) -> None:
    """Test how the transaction-cost assumption affects TD3 performance.

    Sweeps proportional fee levels from 0 bp to 1 bp on the shared baseline,
    holding the algorithm, feature set, and reward fixed, then reports where
    the learned edge changes sign.
    """
    run_hypothesis(
        "h2",
        _apply_experiment_set(
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
            set_name,
            debug,
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
    set_name: _SetName = "full",
    debug: _DebugSet = False,
    max_train_seconds: _MaxTrainSeconds = None,
) -> None:
    """Test how the observation feature set affects TD3 performance.

    Compares minimal, selected, and full feature specifications while holding
    the learning algorithm, reward, and transaction cost fixed.
    """
    run_hypothesis(
        "h3",
        _apply_experiment_set(
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
            set_name,
            debug,
        ),
    )


@app.command()
def h4(
    skip_train: _SkipTrain = False,
    skip_eval: _SkipEval = False,
    parallel: _Parallel = False,
    max_parallel: _MaxParallel = 2,
    verbose: _Verbose = False,
    skip_guardrails: _SkipGuardrails = False,
    overrides: _Overrides = None,
    dev: _Dev = False,
    dev_steps: _DevSteps = 2000,
    set_name: _SetName = "full",
    debug: _DebugSet = False,
    max_train_seconds: _MaxTrainSeconds = None,
) -> None:
    """Test how the reward function changes the learned policy.

    Compares the log-return baseline against the Differential Sharpe Ratio
    reward on an otherwise identical configuration, then reports the effect on
    performance and policy behaviour.
    """
    run_hypothesis(
        "h4",
        _apply_experiment_set(
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
            set_name,
            debug,
        ),
    )


@app.command()
def h5(
    skip_train: _SkipTrain = False,
    skip_eval: _SkipEval = False,
    parallel: _Parallel = False,
    max_parallel: _MaxParallel = 2,
    verbose: _Verbose = False,
    skip_guardrails: _SkipGuardrails = False,
    overrides: _Overrides = None,
    dev: _Dev = False,
    dev_steps: _DevSteps = 2000,
    set_name: _SetName = "full",
    debug: _DebugSet = False,
    max_train_seconds: _MaxTrainSeconds = None,
) -> None:
    """Test how execution latency affects TD3 performance.

    Sweeps the delay between observing the order book and reaching the market
    from 0 to 5 ms on an otherwise identical configuration, then reports the
    effect on performance and policy behaviour.
    """
    run_hypothesis(
        "h5",
        _apply_experiment_set(
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
            set_name,
            debug,
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
    set_name: _SetName = "full",
    debug: _DebugSet = False,
    max_train_seconds: _MaxTrainSeconds = None,
) -> None:
    """Run [bold]H1[/bold] through [bold]H5[/bold] in sequence."""
    args = _apply_experiment_set(
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
        set_name,
        debug,
    )
    for hyp in ("h1", "h2", "h3", "h4", "h5"):
        run_hypothesis(hyp, args)
    _con.print("\n[bold green]All done.[/bold green]")


if __name__ == "__main__":
    app()
