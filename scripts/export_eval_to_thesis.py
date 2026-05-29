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
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml

from logger import get_logger, setup_logging

logger = get_logger(__name__)


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


def _load_benchmark_table(eval_dir: Path) -> dict | None:
    """Load benchmark table(s) and convert to the statistical_tests format.

    Handles two layouts produced by different versions of the evaluate CLI:

    Old (aggregated):  eval_dir/benchmark_tables/test_benchmark_table.json
    New (per-symbol):  eval_dir/test_AAPL_benchmark_table.json
                       eval_dir/test_AMZN_benchmark_table.json  …

    For the per-symbol layout, numeric metrics are averaged across all symbols.
    format_benchmark_comparison_table() expects a 'strategy' key and a top-level
    'benchmark_comparison_table' list.
    """
    def _convert_rows(rows: list[dict]) -> list[dict]:
        converted = []
        for row in rows:
            new_row = {"strategy": row.get("name", "?")}
            for k, v in row.items():
                if k not in ("name", "is_strategy"):
                    new_row[k] = v
            converted.append(new_row)
        return converted

    def _load_json(path: Path) -> dict | None:
        try:
            raw = path.read_text(encoding="utf-8")
            raw = re.sub(r"\bNaN\b", "null", raw)
            raw = re.sub(r"\bInfinity\b", "null", raw)
            raw = re.sub(r"\b-Infinity\b", "null", raw)
            return json.loads(raw)
        except Exception:
            return None

    # --- Try old aggregated layout first ---
    for split in ("test", "val", "train"):
        bench_path = eval_dir / "benchmark_tables" / f"{split}_benchmark_table.json"
        if bench_path.exists():
            data = _load_json(bench_path)
            if data:
                rows = data.get("rows", [])
                if rows:
                    return {"benchmark_comparison_table": _convert_rows(rows)}

    # --- Try new per-symbol layout (test_AAPL_benchmark_table.json, …) ---
    for split_prefix in ("test", "val", "train"):
        per_symbol_files = sorted(eval_dir.glob(f"{split_prefix}_*_benchmark_table.json"))
        if not per_symbol_files:
            continue

        # Collect rows keyed by strategy name, accumulating numeric values
        strategy_accum: dict[str, dict[str, list[float]]] = {}
        strategy_order: list[str] = []

        for path in per_symbol_files:
            data = _load_json(path)
            if not data:
                continue
            for row in data.get("rows", []):
                name = row.get("name", "?")
                if name not in strategy_accum:
                    strategy_accum[name] = {}
                    strategy_order.append(name)
                for k, v in row.items():
                    if k in ("name", "is_strategy"):
                        continue
                    if isinstance(v, (int, float)) and v is not None and math.isfinite(v):
                        strategy_accum[name].setdefault(k, []).append(float(v))

        if not strategy_accum:
            continue

        converted = []
        for name in strategy_order:
            row_out: dict = {"strategy": name}
            for k, vals in strategy_accum[name].items():
                row_out[k] = sum(vals) / len(vals) if vals else None
            converted.append(row_out)

        return {"benchmark_comparison_table": converted}

    return None


def _find_plots(search_dirs: list[Path]) -> dict[str, Path]:
    """Find reward, position, and portfolio value plots in the given directories.

    Checks directories in order, preferring test_* files.  Returns as soon
    as both mandatory plots ('rewards' and 'positions') are found.
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
        if "portfolio_value" not in found:
            candidates = sorted(d.glob("test_*_portfolio_value_plot.png"))
            if candidates:
                found["portfolio_value"] = candidates[0]
        if "rewards" in found and "positions" in found:
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


def _load_scenario_hyperparams(scenario: str | None, repo_root: Path) -> dict | None:
    """Extract training hyperparameters from the scenario train.yaml.

    Returns a flat dict of the key parameters used in the main experiment table,
    or None if the YAML cannot be found or read.
    """
    if scenario is None:
        return None
    yaml_path = repo_root / "src" / "configs" / "scenarios" / scenario / "train.yaml"
    if not yaml_path.exists():
        return None
    try:
        raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    training = raw.get("training", {})
    env = raw.get("env", {})
    network = raw.get("network", {})
    data = raw.get("data", {})

    return {
        "algorithm": training.get("algorithm"),
        "actor_hidden_dims": network.get("actor_hidden_dims"),
        "value_hidden_dims": network.get("value_hidden_dims"),
        "actor_lr": training.get("actor_lr"),
        "value_lr": training.get("value_lr"),
        "actor_weight_decay": training.get("actor_weight_decay"),
        "value_weight_decay": training.get("value_weight_decay"),
        "loss_function": training.get("loss_function"),
        "gamma": training.get("gamma"),
        "tau": training.get("tau"),
        "policy_delay": training.get("policy_delay"),
        "policy_noise": training.get("policy_noise"),
        "noise_clip": training.get("noise_clip"),
        "exploration_noise_std": training.get("exploration_noise_std"),
        "max_steps": training.get("max_steps"),
        "init_rand_steps": training.get("init_rand_steps"),
        "frames_per_batch": training.get("frames_per_batch"),
        "optim_steps_per_batch": training.get("optim_steps_per_batch"),
        "sample_size": training.get("sample_size"),
        "buffer_size": training.get("buffer_size"),
        "reward_type": env.get("reward_type"),
        "reward_eta": env.get("reward_eta"),
        "trading_fees": env.get("trading_fees"),
        "streaming_episode_length": env.get("streaming_episode_length"),
        "train_size": data.get("train_size"),
    }


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
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    env_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    if env_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        env_level = "INFO"
    level = "DEBUG" if args.verbose else env_level
    setup_logging(level=level)
    os.environ["LOG_LEVEL"] = level

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

    logger.info("exporting scenario={} experiment={}", args.scenario or args.output_dir, experiment_name)

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
        logger.error("results.json not found")
        logger.error("  checked: {}", primary_dir / "results.json")
        if fallback_dir != primary_dir:
            logger.error("  checked: {}", fallback_dir / "results.json")
        logger.error(
            "run evaluate first:  uv run python src/cli.py evaluate -c {} "
            "--output-dir {} --only metrics --only benchmarks --only plots",
            args.scenario or args.output_dir,
            primary_dir,
        )
        return 1

    logger.info("reading results from {}", results_file)

    try:
        results = _load_results_json(results_file)
    except json.JSONDecodeError as e:
        logger.error("failed to parse results.json: {}", e)
        return 1

    # --------------------------------------------------------------------------
    # Aggregate metrics across test splits
    # --------------------------------------------------------------------------
    metrics = _aggregate_test_metrics(results)
    if not metrics:
        logger.warning("no metrics found in results.json")
    else:
        n_test = sum(1 for k in results if k.startswith("test"))
        logger.info("aggregated {} metrics across {} test split(s)", len(metrics), n_test)

    # --------------------------------------------------------------------------
    # Find plots
    # --------------------------------------------------------------------------
    plot_search_dirs: list[Path] = [eval_dir]
    if args.plots_dir:
        plot_search_dirs.insert(0, args.plots_dir.resolve())
    plot_search_dirs.append(repo_root / "eval_results")

    plots = _find_plots(plot_search_dirs)
    if plots:
        logger.info("found plots: {}", list(plots))
    else:
        logger.warning(
            "no plots found — run evaluate without --only to include them: "
            "uv run python src/cli.py evaluate -c %s",
            args.scenario or args.output_dir,
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

    _write_json(snapshot_dir / "evaluation_report.json", metrics)

    if not (snapshot_dir / "params.json").exists():
        _write_json(snapshot_dir / "params.json", {})
    if not (snapshot_dir / "latest_metrics.json").exists():
        _write_json(snapshot_dir / "latest_metrics.json", {})

    hyperparams = _load_scenario_hyperparams(args.scenario, repo_root)
    hyperparams_file: str | None = None
    if hyperparams is not None:
        _write_json(snapshot_dir / "hyperparams.json", hyperparams)
        hyperparams_file = "hyperparams.json"
        logger.info("exported hyperparams from train.yaml")

    plot_relpaths = _copy_plots(plots, snapshot_dir) if plots else {}

    benchmark_table = _load_benchmark_table(eval_dir)
    statistical_tests_file: str | None = None
    if benchmark_table is not None:
        _write_json(snapshot_dir / "statistical_tests.json", benchmark_table)
        statistical_tests_file = "statistical_tests.json"
        n_rows = len(benchmark_table.get("benchmark_comparison_table", []))
        logger.info("benchmark table: {} strategies", n_rows)

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
            "statistical_tests": statistical_tests_file,
            "hyperparams": hyperparams_file,
        },
        "evaluation_plots": plot_relpaths,
        "per_symbol_results": _per_symbol_summary(results),
    }
    _write_json(snapshot_dir / "run.json", run_json)

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

    logger.info("exported thesis snapshot  experiment={}  location={}", experiment_name, snapshot_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
