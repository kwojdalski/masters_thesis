#!/usr/bin/env python3
"""Export evaluate CLI output into a thesis result snapshot.

Reads results.json (and plots if available) from the evaluate CLI output directory
and writes a thesis/qmd/results/{experiment_name}/latest_finished/ snapshot that
Quarto chapters can render without querying the MLflow database.

Usage:
    # Derive eval dir from scenario name (strips the config prefix, e.g. pooled/)
    uv run python scripts/export_eval_to_thesis.py \\
        --scenario pooled/td3_hft_lob_state_space_pooled_streaming_selected_dsr

    # Explicit paths
    uv run python scripts/export_eval_to_thesis.py \\
        --output-dir logs/td3_hft_lob_state_space_pooled_streaming_selected_dsr \\
        --experiment-name pooled_td3_hft_lob_state_space_pooled_streaming_selected_dsr

    # Include plots from a specific directory (produced by evaluate without --only)
    uv run python scripts/export_eval_to_thesis.py \\
        --scenario pooled/td3_hft_lob_state_space_pooled_streaming_selected_dsr \\
        --plots-dir eval_results
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _sanitise_for_json(obj: object) -> object:
    """Replace NaN/Inf floats with None so json.dumps produces valid JSON."""
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    if isinstance(obj, dict):
        return {k: _sanitise_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitise_for_json(v) for v in obj]
    return obj


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_sanitise_for_json(payload), indent=2, default=str))


def _load_results_json(path: Path) -> dict:
    """Load results.json, tolerating Python-style NaN/Infinity tokens."""
    raw = path.read_text(encoding="utf-8")
    # Python's json module writes bare NaN/Infinity which is not valid JSON.
    raw = re.sub(r"\bNaN\b", "null", raw)
    raw = re.sub(r"\bInfinity\b", "null", raw)
    raw = re.sub(r"\b-Infinity\b", "null", raw)
    return json.loads(raw)


def _aggregate_test_metrics(results: dict) -> dict:
    """Average metrics across all test_* splits in results.json.

    Falls back to val_* then train_* splits if no test entries exist.
    """
    for prefix in ("test", "val", "train"):
        entries = {
            k: v
            for k, v in results.items()
            if k.startswith(prefix) and isinstance(v.get("metrics"), dict)
        }
        if entries:
            break
    else:
        return {}

    all_keys: set[str] = set()
    for entry in entries.values():
        all_keys.update(entry["metrics"].keys())

    aggregated: dict = {}
    for key in sorted(all_keys):
        vals = [
            entry["metrics"][key]
            for entry in entries.values()
            if key in entry["metrics"]
            and isinstance(entry["metrics"][key], (int, float))
            and entry["metrics"][key] is not None
            and math.isfinite(entry["metrics"][key])
        ]
        aggregated[key] = (sum(vals) / len(vals)) if vals else None

    return aggregated


def _per_symbol_summary(results: dict) -> dict:
    """Compact per-symbol summary (split, final_reward, n_steps)."""
    return {
        k: {
            "split": v.get("split"),
            "final_reward": v.get("final_reward"),
            "n_steps": v.get("n_steps"),
        }
        for k, v in results.items()
    }


def _find_plots(search_dirs: list[Path]) -> dict[str, Path]:
    """Find reward and position (action) plots in the given directories.

    Checks directories in order, preferring test_* files.  Returns as soon
    as both 'rewards' and 'positions' are found.
    """
    found: dict[str, Path] = {}
    for d in search_dirs:
        if not d.exists():
            continue
        if "rewards" not in found:
            candidates = sorted(d.glob("test_*_reward_plot.png"))
            if candidates:
                found["rewards"] = candidates[0]
        if "positions" not in found:
            candidates = sorted(d.glob("test_*_action_plot.png"))
            if candidates:
                found["positions"] = candidates[0]
        if len(found) == 2:
            break
    return found


def _copy_plots(plots: dict[str, Path], dest_dir: Path) -> dict[str, str]:
    """Copy plots into dest_dir/plots/ and return relative paths."""
    plots_dir = dest_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    for key, src in plots.items():
        if not src.exists():
            continue
        dst = plots_dir / f"{key}{src.suffix}"
        shutil.copy2(src, dst)
        copied[key] = str(dst.relative_to(dest_dir))
    return copied


def _resolve_eval_dir(scenario: str | None, output_dir: Path | None, repo_root: Path) -> tuple[Path, Path]:
    """Return (primary_eval_dir, fallback_eval_dir) for locating results.json.

    Primary:  logs/{last_component_of_scenario}  (evaluate CLI --output-dir)
    Fallback: logs/{scenario_with_slashes_as_underscores}  (training log dir)
    """
    if scenario:
        log_name = scenario.split("/")[-1]
        primary = repo_root / "logs" / log_name
        fallback = repo_root / "logs" / scenario.replace("/", "_")
    else:
        assert output_dir is not None
        primary = output_dir.resolve()
        fallback = primary
    return primary, fallback


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export evaluate CLI output into a thesis result snapshot.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--scenario",
        metavar="SCENARIO",
        help=(
            "Config scenario path (e.g. pooled/td3_hft_lob_state_space_pooled_streaming_selected_dsr). "
            "The eval output dir is inferred as logs/<last-component-of-scenario>."
        ),
    )
    src.add_argument(
        "--output-dir",
        type=Path,
        metavar="DIR",
        help="Explicit path to the evaluate CLI output directory (contains results.json).",
    )
    p.add_argument(
        "--experiment-name",
        metavar="NAME",
        help=(
            "Override the thesis experiment name written to thesis/qmd/results/. "
            "Defaults to scenario with '/' replaced by '_'."
        ),
    )
    p.add_argument(
        "--plots-dir",
        type=Path,
        metavar="DIR",
        help=(
            "Additional directory to scan for plots. "
            "Checked before the default eval_results/ fallback."
        ),
    )
    p.add_argument(
        "--thesis-results-root",
        type=Path,
        metavar="DIR",
        help="Override the thesis/qmd/results root directory.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    repo_root = _repo_root()

    # --------------------------------------------------------------------------
    # Resolve directories and experiment name
    # --------------------------------------------------------------------------
    primary_dir, fallback_dir = _resolve_eval_dir(args.scenario, args.output_dir, repo_root)

    if args.experiment_name:
        experiment_name = args.experiment_name
    elif args.scenario:
        experiment_name = args.scenario.replace("/", "_")
    else:
        assert args.output_dir is not None
        experiment_name = primary_dir.name

    # --------------------------------------------------------------------------
    # Locate results.json
    # --------------------------------------------------------------------------
    results_file: Path | None = None
    for candidate in (primary_dir / "results.json", fallback_dir / "results.json"):
        if candidate.exists():
            results_file = candidate
            eval_dir = candidate.parent
            break

    if results_file is None:
        print(f"ERROR: results.json not found.")
        print(f"  Checked: {primary_dir / 'results.json'}")
        if fallback_dir != primary_dir:
            print(f"  Checked: {fallback_dir / 'results.json'}")
        print(
            "\nRun evaluate first:\n"
            f"  uv run python src/cli.py evaluate -c {args.scenario or args.output_dir} "
            f"--output-dir {primary_dir}"
        )
        return 1

    print(f"Reading results from: {results_file}")

    try:
        results = _load_results_json(results_file)
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse results.json: {e}")
        return 1

    # --------------------------------------------------------------------------
    # Aggregate metrics across test splits
    # --------------------------------------------------------------------------
    metrics = _aggregate_test_metrics(results)
    if not metrics:
        print("WARNING: No metrics found in results.json")
    else:
        print(f"  Aggregated {len(metrics)} metrics across {sum(1 for k in results if k.startswith('test'))} test split(s)")

    # --------------------------------------------------------------------------
    # Find plots
    # --------------------------------------------------------------------------
    plot_search_dirs: list[Path] = [eval_dir]
    if args.plots_dir:
        plot_search_dirs.insert(0, args.plots_dir.resolve())
    plot_search_dirs.append(repo_root / "eval_results")

    plots = _find_plots(plot_search_dirs)
    if plots:
        print(f"  Found plots: {list(plots)}")
    else:
        print(
            "  WARNING: No plots found. To include plots, run evaluate without --only:\n"
            f"    uv run python src/cli.py evaluate -c {args.scenario or args.output_dir}\n"
            "  (writes to eval_results/ which is automatically picked up)"
        )

    # --------------------------------------------------------------------------
    # Write thesis snapshot
    # --------------------------------------------------------------------------
    thesis_results_root = (
        args.thesis_results_root.resolve()
        if args.thesis_results_root
        else repo_root / "thesis" / "qmd" / "results"
    )
    experiment_dir = thesis_results_root / experiment_name
    snapshot_dir = experiment_dir / "latest_finished"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc).isoformat()

    # evaluation_report.json — aggregated flat metrics consumed by format_key_metrics()
    _write_json(snapshot_dir / "evaluation_report.json", metrics)

    # params.json and latest_metrics.json — not available from eval-only output
    if not (snapshot_dir / "params.json").exists():
        _write_json(snapshot_dir / "params.json", {})
    if not (snapshot_dir / "latest_metrics.json").exists():
        _write_json(snapshot_dir / "latest_metrics.json", {})

    # Copy plots into snapshot
    plot_relpaths = _copy_plots(plots, snapshot_dir) if plots else {}

    # run.json — standard snapshot format read by _load_run_from_export()
    run_json: dict = {
        "run_id": None,
        "run_name": experiment_name,
        "status": "FINISHED",
        "start_time": now,
        "end_time": now,
        "artifact_uri": None,
        "experiment_name": experiment_name,
        "experiment_id": None,
        "source": {
            "type": "evaluate_cli",
            "eval_output_dir": str(eval_dir),
            "results_file": str(results_file),
            "exported_at_utc": now,
        },
        "files": {
            "params": "params.json",
            "latest_metrics": "latest_metrics.json",
            "evaluation_report": "evaluation_report.json",
            "statistical_tests": None,
        },
        "evaluation_plots": plot_relpaths,
        "per_symbol_results": _per_symbol_summary(results),
    }
    _write_json(snapshot_dir / "run.json", run_json)

    # manifest.json at experiment level (keeps manifest schema consistent)
    manifest: dict = {
        "schema_version": 1,
        "experiment_name": experiment_name,
        "exported_at_utc": now,
        "source": {
            "type": "evaluate_cli",
            "eval_output_dir": str(eval_dir),
        },
        "files": {},
        "runs": {
            "latest_running": None,
            "latest_finished": "latest_finished/run.json",
        },
    }
    _write_json(experiment_dir / "manifest.json", manifest)

    print(f"\nExported thesis snapshot for '{experiment_name}'")
    print(f"  Location: {snapshot_dir}")
    print(
        "  QMD chapters will use this snapshot as fallback when MLflow has no data "
        "for this experiment."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
