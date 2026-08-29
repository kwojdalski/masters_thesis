#!/usr/bin/env python3
"""
H4: Learning progression report — merge multiple short trials to analyze
whether the algorithm is learning and can outperform a dummy strategy.

Analyzes:
1. Performance progression across trials and steps
2. Comparison against dummy/baseline strategies
3. Statistical significance of improvements
4. Learning curves and convergence indicators
"""

import json
import math
from dataclasses import dataclass
from datetime import UTC
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from rich.console import Console
from rich.table import Table
from scipy import stats

console = Console()


def load_results(eval_dir: Path) -> dict[str, Any]:
    """Load results.json from evaluation directory."""
    results_file = eval_dir / "results.json"
    if not results_file.exists():
        console.print(f"[yellow]No results.json found in {eval_dir}[/yellow]")
        return {}
    with results_file.open() as f:
        return json.load(f)


def find_trial_results(
    scenario: str, n_trials: int, max_steps: int
) -> list[dict[str, Any]]:
    """Find results from all trials for the given scenario."""
    # MLflow experiment name is typically derived from scenario
    experiment_name = scenario.replace("/", "_")
    console.print(f"Searching for experiment: {experiment_name}")

    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        order_by=["start_time ASC"],
        max_results=n_trials * 10,  # Buffer
    )

    trial_results = []
    for _, row in runs.iterrows():
        run_id = row["run_id"]
        run_name = row.get("tags.mlflow.runName", "")

        # Check if this is a training run (not evaluation)
        if row.get("status") != "FINISHED":
            continue

        # Get metrics
        metrics = {}
        for col in runs.columns:
            if col.startswith("metrics."):
                metric_name = col.replace("metrics.", "")
                value = row[col]
                if pd.notna(value):
                    metrics[metric_name] = float(value)

        if metrics:
            trial_results.append(
                {
                    "run_id": run_id,
                    "run_name": run_name,
                    "metrics": metrics,
                    "params": dict(row.get("params", {})),
                }
            )

        if len(trial_results) >= n_trials:
            break

    return trial_results


def _json_safe(obj: object) -> object:
    """Replace non-finite floats with None so json.dump emits valid JSON.

    The vs-baseline columns are NaN when no usable random baseline exists, and
    Python's json module would otherwise write a bare ``NaN`` token, which is
    not valid JSON for strict readers.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


@dataclass(frozen=True)
class RandomBaseline:
    """Metrics of the cost-aware random-policy baseline for a scenario."""

    total_return: float
    win_rate: float
    scenario: str
    source: str
    source_split: str | None

    @property
    def usable(self) -> bool:
        """True when the baseline was evaluated on the same split as the trials.

        Trial metrics come from the periodic in-training evaluation, which runs
        on the validation split. A baseline exported from the train split is
        not a like-for-like comparison, so it is reported as unavailable rather
        than silently differenced against.
        """
        return self.source_split in {"val", "test"}


def _derive_random_scenario(scenario: str) -> str:
    """Map an agent scenario to its matching random-policy scenario.

    ``pooled/td3_hft_lob_..._dsr`` -> ``pooled/random_hft_lob_..._dsr``.  The
    random scenario shares the data, features and transaction costs, so its
    metrics are the cost-aware baseline this report should difference against.
    """
    prefix, _, name = scenario.rpartition("/")
    _algo, sep, rest = name.partition("_")
    if not sep:
        return scenario
    random_name = f"random{sep}{rest}"
    return f"{prefix}/{random_name}" if prefix else random_name


def load_random_baseline(scenario: str) -> RandomBaseline | None:
    """Load the random-policy baseline metrics for a scenario, or None.

    Reads the thesis snapshot written by export_eval_to_thesis.py for the
    matching random scenario.  Returns None when it is absent so the caller can
    show the comparison as unavailable instead of inventing values.
    """
    experiment = scenario.replace("/", "_")
    snapshot = (
        Path("thesis/qmd/results")
        / experiment
        / "latest_finished"
        / "evaluation_report.json"
    )
    if not snapshot.exists():
        return None
    try:
        with snapshot.open() as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None

    total_return = data.get("total_return")
    win_rate = data.get("win_rate")
    if not isinstance(total_return, int | float) or not isinstance(
        win_rate, int | float
    ):
        return None

    return RandomBaseline(
        total_return=float(total_return),
        win_rate=float(win_rate),
        scenario=scenario,
        source=str(snapshot),
        source_split=data.get("__source_split__"),
    )


def compare_to_baseline(
    trial_results: list[dict[str, Any]],
    baseline: RandomBaseline | None = None,
) -> pd.DataFrame:
    """Compare trial performance against the random-policy baseline.

    ``baseline`` is the cost-aware random baseline for the same scenario.  When
    it is None or was evaluated on a different split, the vs-baseline columns
    are NaN and the caller renders them as "n/a" -- a missing baseline is made
    visible rather than replaced with an assumed value.
    """
    if not trial_results:
        return pd.DataFrame()

    # Extract key metrics across trials
    metrics_data = []
    for i, trial in enumerate(trial_results):
        m = trial["metrics"]
        metrics_data.append(
            {
                "trial": i + 1,
                "total_return": m.get("eval_total_return", 0.0),
                "sharpe_ratio": m.get("eval_sharpe_ratio", 0.0),
                "win_rate": m.get("eval_win_rate", 0.0),
                "max_drawdown": m.get("eval_max_drawdown", 0.0),
                "profit_factor": m.get("eval_profit_factor", 0.0),
            }
        )

    df = pd.DataFrame(metrics_data)

    # Difference against the measured random baseline, never against assumed
    # values: a random policy paying transaction costs does not have exactly
    # zero expected return, and its realised win rate over a finite sample is
    # not exactly 0.5 (the measured baseline for the H4 scenario is +6.9e-4
    # and 52%). NaN when no usable baseline exists, so the margin is shown as
    # "n/a" rather than silently computed against a placeholder.
    if baseline is not None and baseline.usable:
        df["dummy_total_return"] = baseline.total_return
        df["dummy_win_rate"] = baseline.win_rate
    else:
        df["dummy_total_return"] = float("nan")
        df["dummy_win_rate"] = float("nan")
    df["vs_dummy_return"] = df["total_return"] - df["dummy_total_return"]
    df["vs_dummy_win_rate"] = df["win_rate"] - df["dummy_win_rate"]

    return df


def compute_learning_significance(df: pd.DataFrame) -> dict[str, Any]:
    """Statistical tests to determine if learning occurred."""
    if df.empty:
        return {}

    results = {
        "n_trials": len(df),
        "mean_return": float(df["total_return"].mean()),
        "std_return": float(df["total_return"].std()),
        "mean_sharpe": float(df["sharpe_ratio"].mean()),
        "mean_win_rate": float(df["win_rate"].mean()),
    }

    # One-sample t-test against zero (null: no learning)
    if len(df) > 1:
        t_stat, p_value = stats.ttest_1samp(df["total_return"], 0)
        results["t_stat_return"] = float(t_stat)
        results["p_value_return"] = float(p_value)

        t_stat_sr, p_value_sr = stats.ttest_1samp(df["sharpe_ratio"], 0)
        results["t_stat_sharpe"] = float(t_stat_sr)
        results["p_value_sharpe"] = float(p_value_sr)

    return results


def check_learning_criteria(
    significance: dict[str, Any],
    baseline: RandomBaseline | None = None,
) -> dict[str, bool]:
    """Check if algorithm meets learning criteria.

    The win-rate criterion is judged against the measured random baseline when
    one is available, falling back to 0.5 otherwise. The two are not
    interchangeable: the H4 scenario's measured random win rate is 52%, so a
    0.5 threshold passes agents that do not actually beat a random policy.
    """
    criteria = {}

    # Criteria 1: Positive mean return
    criteria["positive_return"] = significance.get("mean_return", 0) > 0

    # Criteria 2: Positive mean Sharpe ratio
    criteria["positive_sharpe"] = significance.get("mean_sharpe", 0) > 0

    # Criteria 3: Statistically significant return (p < 0.05)
    criteria["significant_return"] = significance.get("p_value_return", 1) < 0.05

    # Criteria 4: Win rate above the random baseline
    baseline_win_rate = (
        baseline.win_rate if (baseline is not None and baseline.usable) else 0.5
    )
    criteria["win_rate_above_baseline"] = (
        significance.get("mean_win_rate", 0) > baseline_win_rate
    )

    # Overall conclusion
    passed = sum(criteria.values())
    total = len(criteria)
    criteria["overall"] = passed >= 3  # Require 3/4 criteria met
    criteria["criteria_met"] = f"{passed}/{total}"

    return criteria


def main() -> None:
    from argparse import ArgumentParser

    parser = ArgumentParser(description="H4 Learning progression report")
    parser.add_argument(
        "--scenario",
        default="pooled/td3_hft_lob_state_space_pooled_streaming_selected_dsr",
        help="Scenario configuration name",
    )
    parser.add_argument(
        "--random-scenario",
        default=None,
        help=(
            "Scenario providing the random-policy baseline. "
            "Defaults to the matching random_* scenario."
        ),
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=5,
        help="Number of trials to analyze",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200000,
        help="Max steps per trial (for reference)",
    )
    args = parser.parse_args()

    console.print(
        f"[bold blue]H4: Learning Progression Analysis[/bold blue]\n"
        f"Scenario: {args.scenario}\n"
        f"Trials: {args.n_trials}\n"
        f"Max steps per trial: {args.max_steps:,}\n"
    )

    # Load trial results from MLflow
    trial_results = find_trial_results(args.scenario, args.n_trials, args.max_steps)

    if not trial_results:
        console.print("[red]No trial results found. Have you trained yet?[/red]")
        console.print("Run: uv run thesis-experiments h4")
        return

    console.print(f"[green]Found {len(trial_results)} trial results[/green]\n")

    # Compare to the measured random-policy baseline
    random_scenario = args.random_scenario or _derive_random_scenario(args.scenario)
    baseline = load_random_baseline(random_scenario)
    if baseline is None:
        console.print(
            f"[yellow]No random baseline found for {random_scenario!r}; "
            "'vs Dummy' will show n/a and the win-rate criterion falls back "
            "to 0.5. Run that scenario's evaluate + export to populate it."
            "[/yellow]\n"
        )
    elif not baseline.usable:
        console.print(
            f"[yellow]Random baseline for {random_scenario!r} came from the "
            f"{baseline.source_split!r} split, not val/test; refusing to use it "
            "as a comparison. 'vs Dummy' will show n/a.[/yellow]\n"
        )
    else:
        console.print(
            f"[dim]Random baseline ({random_scenario}, {baseline.source_split} "
            f"split): return={baseline.total_return:.6f} "
            f"win_rate={baseline.win_rate:.2%}[/dim]\n"
        )

    df = compare_to_baseline(trial_results, baseline)

    # Display summary table
    table = Table(title="Trial Performance Summary")
    table.add_column("Trial", justify="right")
    table.add_column("Return", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("Win Rate", justify="right")
    table.add_column("vs Dummy", justify="right")

    for _, row in df.iterrows():
        vs_dummy = row["vs_dummy_return"]
        if pd.isna(vs_dummy):
            vs_dummy_cell = "[dim]n/a[/dim]"
        else:
            vs_dummy_color = "green" if vs_dummy > 0 else "red"
            vs_dummy_cell = f"[{vs_dummy_color}]{vs_dummy:.4f}[/{vs_dummy_color}]"
        table.add_row(
            f"{int(row['trial'])}",
            f"{row['total_return']:.4f}",
            f"{row['sharpe_ratio']:.2f}",
            f"{row['win_rate']:.2%}",
            vs_dummy_cell,
        )

    console.print(table)
    console.print()

    # Compute statistical significance
    significance = compute_learning_significance(df)

    # Display learning criteria
    criteria = check_learning_criteria(significance, baseline)

    criteria_table = Table(title="Learning Criteria Check")
    criteria_table.add_column("Criterion")
    criteria_table.add_column("Status", justify="center")

    criteria_labels = {
        "positive_return": "Positive mean return",
        "positive_sharpe": "Positive mean Sharpe ratio",
        "significant_return": "Statistically significant return (p<0.05)",
        "win_rate_above_baseline": (
            f"Win rate > {baseline.win_rate:.2%} (measured random baseline)"
            if (baseline is not None and baseline.usable)
            else "Win rate > 50% (assumed random baseline)"
        ),
    }

    for key, label in criteria_labels.items():
        status = "[green]PASS[/green]" if criteria[key] else "[red]FAIL[/red]"
        criteria_table.add_row(label, status)

    console.print(criteria_table)
    console.print()

    # Display statistical summary
    stats_table = Table(title="Statistical Summary")
    stats_table.add_column("Metric")
    stats_table.add_column("Value", justify="right")

    stats_table.add_row("N trials", f"{significance['n_trials']}")
    stats_table.add_row("Mean return", f"{significance['mean_return']:.4f}")
    stats_table.add_row("Std return", f"{significance['std_return']:.4f}")
    stats_table.add_row("Mean Sharpe", f"{significance['mean_sharpe']:.2f}")
    stats_table.add_row("Mean win rate", f"{significance['mean_win_rate']:.2%}")

    if "p_value_return" in significance:
        p_color = "green" if significance["p_value_return"] < 0.05 else "red"
        stats_table.add_row(
            "Return p-value",
            f"[{p_color}]{significance['p_value_return']:.4f}[/{p_color}]",
        )

    console.print(stats_table)
    console.print()

    # Final conclusion
    if criteria["overall"]:
        console.print(
            "[bold green]CONCLUSION: Algorithm shows learning capability "
            f"({criteria['criteria_met']} criteria met)[/bold green]"
        )
    else:
        console.print(
            "[bold yellow]CONCLUSION: Learning not clearly established "
            f"({criteria['criteria_met']} criteria met)[/bold yellow]"
        )

    # Save results for reference
    output_dir = Path("eval_results") / args.scenario.replace("/", "_")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / "h4_learning_report.json"
    report_data = {
        "scenario": args.scenario,
        "n_trials": args.n_trials,
        "max_steps": args.max_steps,
        "statistics": significance,
        "criteria": criteria,
        "trials": df.to_dict(orient="records"),
    }

    with results_file.open("w") as f:
        json.dump(_json_safe(report_data), f, indent=2)

    console.print(f"\nReport saved to: {results_file}")

    # Export to thesis snapshot
    console.print("\nExporting to thesis snapshot...")
    thesis_results_root = Path("thesis/qmd/results")
    experiment_name = f"{args.scenario.replace('/', '_')}_h4_n{args.n_trials}"
    experiment_dir = thesis_results_root / experiment_name
    snapshot_dir = experiment_dir / "latest_finished"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # Write evaluation_report.json (key metrics for thesis)
    evaluation_report = {
        "sharpe_ratio": significance.get("mean_sharpe"),
        "total_return": significance.get("mean_return"),
        "win_rate": significance.get("mean_win_rate"),
        "std_return": significance.get("std_return"),
        "p_value_return": significance.get("p_value_return"),
        "n_trials": significance.get("n_trials"),
        "max_steps": args.max_steps,
        "criteria_met": criteria.get("criteria_met"),
        "overall_learning": criteria.get("overall"),
    }

    from datetime import datetime

    now = datetime.now(UTC).isoformat()

    (snapshot_dir / "evaluation_report.json").write_text(
        json.dumps(_json_safe(evaluation_report), indent=2)
    )
    (snapshot_dir / "h4_learning_report.json").write_text(
        json.dumps(_json_safe(report_data), indent=2)
    )

    # Write run.json for consistency
    run_json = {
        "run_id": None,
        "run_name": f"{args.scenario.replace('/', '_')}_h4_n{args.n_trials}",
        "status": "FINISHED",
        "start_time": now,
        "end_time": now,
        "experiment_name": experiment_name,
        "h4_experiment": True,
        "n_trials": args.n_trials,
        "max_steps_per_trial": args.max_steps,
    }
    (snapshot_dir / "run.json").write_text(json.dumps(run_json, indent=2))

    # Write manifest.json
    manifest = {
        "schema_version": 1,
        "experiment_name": experiment_name,
        "exported_at_utc": now,
        "source": {"type": "h4_learning_progression_report"},
        "files": {},
        "runs": {"latest_running": None, "latest_finished": "latest_finished/run.json"},
    }
    (experiment_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    console.print(f"[green]Thesis snapshot exported to: {snapshot_dir}[/green]")


if __name__ == "__main__":
    main()
