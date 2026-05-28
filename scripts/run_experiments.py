#!/usr/bin/env python3
"""Run H1, H2, H3, or all hypothesis experiments.

Steps for each hypothesis
-------------------------
0. Guardrails   -- validate config for all scenarios (skip with --skip-guardrails)
1. Train        -- fit the agent(s)                  (skip with --skip-train)
2. Evaluate     -- compute metrics, benchmarks, plots (skip with --skip-eval)
3. Report       -- run the hypothesis-specific summary script
4. Export       -- write thesis snapshots for Quarto rendering

Usage
-----
    uv run python scripts/run_experiments.py h1
    uv run python scripts/run_experiments.py h2 --skip-train
    uv run python scripts/run_experiments.py h3 --parallel
    uv run python scripts/run_experiments.py all
    uv run python scripts/run_experiments.py h1 --max-train-seconds 300
    uv run python scripts/run_experiments.py h1 -o training.max_steps=50000
    uv run python scripts/run_experiments.py h2 -o evaluation.eval_steps=500
    uv run python scripts/run_experiments.py h1 --skip-train --parallel
    uv run python scripts/run_experiments.py all --skip-guardrails

Config overrides (-o / --config-override) are forwarded to both train and
evaluate.  Use --skip-eval or --skip-train to isolate overrides to one step.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import shutil
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CLI = ["uv", "run", "python", str(_REPO_ROOT / "src" / "cli.py")]
_LOG_DIR = _REPO_ROOT / "logs"

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
_H3_SCENARIOS = list(dict.fromkeys([
    "pooled/td3_h3_features_minimal",
    "pooled/td3_hft_lob_state_space_pooled_streaming_selected",   # baseline
    "pooled/td3_h3_features_full",
    "pooled/td3_hft_lob_state_space_pooled_streaming_selected_dsr",
    "pooled/td3_h3_fees_1e6",
    "pooled/td3_h3_fees_1e5",
    "pooled/td3_h3_fees_1e4",
]))

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
# Low-level subprocess helpers
# ---------------------------------------------------------------------------

def _scenario_name(scenario: str) -> str:
    return scenario.split("/")[-1]


def _log_file(scenario: str, suffix: str) -> Path:
    return _LOG_DIR / f"{_scenario_name(scenario)}_{suffix}.log"


def _override_flags(overrides: list[str]) -> list[str]:
    flags: list[str] = []
    for kv in overrides:
        flags += ["--config-override", kv]
    return flags


def _watch_hint(label: str, log_files: list[Path]) -> None:
    print()
    paths = " ".join(str(f) for f in log_files)
    if shutil.which("multitail"):
        print(f"Monitor {label} logs:")
        print(f"  multitail -s {len(log_files)} {paths}")
    else:
        print(f"Monitor {label} logs (install multitail for split-pane view):")
        print(f"  tail -f {paths}")
    print()


def _run_tee(cmd: list[str], log_file: Path) -> None:
    """Run command, streaming output to both terminal and log file."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"  -> {log_file}")
    with log_file.open("w") as fh:
        proc = subprocess.Popen(
            cmd,
            cwd=_REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
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
    """Run command, capturing all output to log file (used in parallel mode)."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["NO_COLOR"] = "1"
    with log_file.open("w") as fh:
        subprocess.run(cmd, cwd=_REPO_ROOT, stdout=fh, stderr=fh, env=env, check=True)


def _run_simple(cmd: list[str]) -> None:
    """Run command with inherited stdio."""
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=_REPO_ROOT, check=True)


def _run_parallel_jobs(
    label: str,
    jobs: list[tuple[list[str], Path]],
) -> None:
    log_files = [log for _, log in jobs]
    for cmd, log in jobs:
        print(f"  {_scenario_name(str(log).split('_')[0])}  ->  {log}  (background)")

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        futures = {
            executor.submit(_run_capture, cmd, log): log
            for cmd, log in jobs
        }
        _watch_hint(label, log_files)
        print(f"Waiting for {len(futures)} {label} job(s)...")
        failed: list[Path] = []
        for future in concurrent.futures.as_completed(futures):
            log = futures[future]
            try:
                future.result()
                print(f"  done: {log.name}")
            except subprocess.CalledProcessError as exc:
                print(f"  FAILED (rc={exc.returncode}): {log}")
                failed.append(log)

    if failed:
        raise SystemExit(
            f"{len(failed)} {label} job(s) failed. "
            f"Check logs: {', '.join(str(f) for f in failed)}"
        )

# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def _check_guardrails(scenarios: list[str], args: argparse.Namespace) -> None:
    print(f"=== Pre-flight: Checking guardrails for {len(scenarios)} scenario(s) ===")
    passed: list[str] = []
    failed: list[str] = []

    for scenario in scenarios:
        log = _log_file(scenario, "guardrails")
        log.parent.mkdir(parents=True, exist_ok=True)
        print(f"  Checking {scenario}...")
        cmd = [*_CLI, "validate", "guardrails", "-c", scenario]
        if args.verbose:
            cmd.append("--verbose")
        with log.open("w") as fh:
            result = subprocess.run(
                cmd, cwd=_REPO_ROOT, stdout=fh, stderr=fh, check=False
            )
        if result.returncode == 0:
            print(f"    [PASS] {scenario}")
            passed.append(scenario)
        else:
            print(f"    [FAIL] {scenario}  (see {log})")
            failed.append(scenario)

    print(f"\nGuardrails summary:  passed={len(passed)}  failed={len(failed)}")

    if failed:
        print("\nFailed scenarios:")
        for s in failed:
            print(f"  - {s}  (logs: {_log_file(s, 'guardrails')})")
        print("\nFix the guardrail issues or run with --skip-guardrails to proceed anyway.")
        raise SystemExit("Guardrails check failed.")

    print("All scenarios passed guardrails.\n")


def _train_all(
    scenarios: list[str],
    args: argparse.Namespace,
    extra_overrides: list[str] | None = None,
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
        )
    else:
        for scenario in scenarios:
            log = _log_file(scenario, "train")
            print(f"Training {scenario}  ->  {log}")
            _run_tee(_cmd(scenario), log)
            print("  done.")


def _evaluate_all(
    scenarios: list[str],
    eval_only: list[str],
    args: argparse.Namespace,
) -> None:
    overrides = list(args.overrides)

    def _cmd(scenario: str) -> list[str]:
        output_dir = str(_LOG_DIR / _scenario_name(scenario))
        cmd = [
            *_CLI, "evaluate",
            "-c", scenario,
            "--output-dir", output_dir,
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
        )
    else:
        for scenario in scenarios:
            log = _log_file(scenario, "eval")
            output_dir = _LOG_DIR / _scenario_name(scenario)
            print(f"Evaluating {scenario}  ->  {output_dir}")
            _run_tee(_cmd(scenario), log)
            print("  done.")


def _run_report(hypothesis: str) -> None:
    script = _REPO_ROOT / "scripts" / _REPORT_SCRIPTS[hypothesis]
    print(f"=== {hypothesis.upper()}: Report ===")
    _run_simple(["uv", "run", "python", str(script)])


def _export_all(scenarios: list[str]) -> None:
    export_script = str(_REPO_ROOT / "scripts" / "export_eval_to_thesis.py")
    for scenario in scenarios:
        print(f"  Exporting {scenario} ...")
        _run_simple(["uv", "run", "python", export_script, "--scenario", scenario])
    print("Thesis snapshots updated.")


# ---------------------------------------------------------------------------
# Hypothesis runners
# ---------------------------------------------------------------------------

def run_hypothesis(hypothesis: str, args: argparse.Namespace) -> None:
    scenarios = _SCENARIOS[hypothesis]
    eval_only = _EVAL_ONLY[hypothesis]

    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Step 0 — Guardrails
    if not args.skip_guardrails:
        _check_guardrails(scenarios, args)

    # Step 1 — Train
    if not args.skip_train:
        print(f"=== {hypothesis.upper()}: Training ===")
        extra: list[str] = []
        max_secs = getattr(args, "max_train_seconds", None)
        if max_secs:
            extra.append(f"training.max_train_seconds={max_secs}")
        _train_all(scenarios, args, extra_overrides=extra)
        print()

    # Steps 2–4 — Evaluate, Report, Export
    if not args.skip_eval:
        print(f"=== {hypothesis.upper()}: Evaluating ===")
        _evaluate_all(scenarios, eval_only, args)
        print()

        _run_report(hypothesis)

        print(f"=== {hypothesis.upper()}: Export to thesis ===")
        _export_all(scenarios)
    else:
        print(f"=== {hypothesis.upper()}: Skipping evaluate, report, and export (--skip-eval) ===")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--skip-train", action="store_true", help="Skip training step.")
    p.add_argument("--skip-eval", action="store_true", help="Skip evaluate/report/export steps.")
    p.add_argument("--parallel", action="store_true", help="Run scenarios concurrently.")
    p.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging in subcommands.")
    p.add_argument("--skip-guardrails", action="store_true", help="Skip pre-flight guardrails check.")
    p.add_argument(
        "-o", "--config-override",
        dest="overrides",
        action="append",
        default=[],
        metavar="K=V",
        help="OmegaConf dotlist override forwarded to both train and evaluate. Repeatable.",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run H1, H2, H3, or all hypothesis experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="hypothesis", required=True)

    # h1
    h1 = sub.add_parser("h1", help="H1: Compare TD3, DDPG, PPO, Random agents with DSR reward.")
    _add_common_args(h1)
    h1.add_argument(
        "--max-train-seconds",
        type=int,
        metavar="N",
        help="Cap training wall-clock time per scenario (forwarded as training.max_train_seconds=N).",
    )

    # h2
    h2 = sub.add_parser("h2", help="H2: Compare minimal / selected / full feature specifications.")
    _add_common_args(h2)

    # h3
    h3 = sub.add_parser("h3", help="H3: Sensitivity — features, reward, transaction cost.")
    _add_common_args(h3)

    # all
    all_p = sub.add_parser("all", help="Run H1, H2, and H3 in sequence.")
    _add_common_args(all_p)

    args = parser.parse_args()

    if args.hypothesis == "all":
        for hyp in ("h1", "h2", "h3"):
            run_hypothesis(hyp, args)
    else:
        run_hypothesis(args.hypothesis, args)

    print("\nAll done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
