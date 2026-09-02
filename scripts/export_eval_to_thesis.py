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
from datetime import UTC, datetime
from pathlib import Path

import yaml

from logger import get_logger, setup_logging
from trading_rl.config import EXPERIMENT_OUTPUT_DIR

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


# Split preference, most-preferred first.  Anything other than "test" is a
# fallback that must be reported, never silently published as an out-of-sample
# result.
_SPLIT_PREFERENCE: tuple[str, ...] = ("test", "val", "train")


def _split_entries(results: dict, prefix: str) -> tuple[dict, dict]:
    """Return ``(pooled, per_symbol)`` metric entries for one split prefix.

    A pooled entry is keyed by the bare split name (``"test"``).  Per-symbol
    entries carry a symbol suffix and appear in two shapes depending on which
    code path produced them: ``"test_AAPL"`` from
    ``EvaluateCommand._resolve_per_symbol_splits`` and ``"val__AAPL"`` from
    ``pipeline.evaluation.evaluate_per_symbol``.  Matching on the ``_``
    boundary covers both without letting ``"test"`` swallow unrelated keys.
    """
    pooled: dict = {}
    per_symbol: dict = {}
    for key, entry in results.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("metrics"), dict):
            continue
        if key == prefix:
            pooled[key] = entry
        elif key.startswith(f"{prefix}_"):
            per_symbol[key] = entry
    return pooled, per_symbol


def _aggregate_split_metrics(results: dict) -> tuple[dict, str | None, list[str]]:
    """Average metrics for the most preferred split present in results.json.

    Returns ``(metrics, source_split, source_keys)``.  ``source_split`` is
    None when nothing usable was found; callers must check it rather than
    assuming the numbers are test-split metrics.

    Per-symbol entries are averaged among themselves and a pooled entry is
    used on its own -- the two are never mixed, because a pooled figure is
    already an aggregate over the same symbols and averaging it alongside its
    own components double-counts them.
    """
    entries: dict = {}
    source_split: str | None = None
    for prefix in _SPLIT_PREFERENCE:
        pooled, per_symbol = _split_entries(results, prefix)
        # Prefer the disaggregated per-symbol entries when both are present.
        entries = per_symbol or pooled
        if entries:
            source_split = prefix
            break

    if not entries or source_split is None:
        return {}, None, []

    all_keys: set[str] = set()
    for entry in entries.values():
        all_keys.update(entry["metrics"].keys())

    aggregated: dict = {}
    for key in sorted(all_keys):
        vals = [
            entry["metrics"][key]
            for entry in entries.values()
            if key in entry["metrics"]
            and isinstance(entry["metrics"][key], int | float)
            and entry["metrics"][key] is not None
            and math.isfinite(entry["metrics"][key])
        ]
        aggregated[key] = (sum(vals) / len(vals)) if vals else None

    return aggregated, source_split, sorted(entries)


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


def _load_benchmark_table(eval_dir: Path) -> tuple[dict | None, str | None]:
    """Load benchmark table(s) and convert to the statistical_tests format.

    Handles two layouts produced by different versions of the evaluate CLI:

    Old (aggregated):  eval_dir/benchmark_tables/test_benchmark_table.json
    New (per-symbol):  eval_dir/test_AAPL_benchmark_table.json
                       eval_dir/test_AMZN_benchmark_table.json  …

    For the per-symbol layout, numeric metrics are averaged across all symbols.
    format_benchmark_comparison_table() expects a 'strategy' key and a top-level
    'benchmark_comparison_table' list.

    Returns ``(table, source_split)``.  ``source_split`` lets the caller refuse
    to publish a train-split benchmark comparison as an out-of-sample one.
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

    # --- Try the per-symbol layout first: it is what `evaluate --per-symbol`
    # produces now, and a stale aggregated benchmark_tables/ file from an
    # earlier run must not shadow a fresh per-symbol re-evaluation. Files land
    # either at the eval-dir top level or under per_symbol/<SYMBOL>/.
    for split_prefix in _SPLIT_PREFERENCE:
        per_symbol_files = sorted(
            [
                *eval_dir.glob(f"{split_prefix}_*_benchmark_table.json"),
                *eval_dir.glob(f"per_symbol/*/{split_prefix}_*_benchmark_table.json"),
            ]
        )
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
                    if (
                        isinstance(v, int | float)
                        and v is not None
                        and math.isfinite(v)
                    ):
                        strategy_accum[name].setdefault(k, []).append(float(v))

        if not strategy_accum:
            continue

        converted = []
        for name in strategy_order:
            row_out: dict = {"strategy": name}
            for k, vals in strategy_accum[name].items():
                row_out[k] = sum(vals) / len(vals) if vals else None
            converted.append(row_out)

        return {"benchmark_comparison_table": converted}, split_prefix

    # --- Fall back to the old aggregated layout (benchmark_tables/<split>_…) ---
    for split in _SPLIT_PREFERENCE:
        bench_path = eval_dir / "benchmark_tables" / f"{split}_benchmark_table.json"
        if bench_path.exists():
            data = _load_json(bench_path)
            if data:
                rows = data.get("rows", [])
                if rows:
                    return {"benchmark_comparison_table": _convert_rows(rows)}, split

    return None, None


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


def _resolve_eval_dir(
    scenario: str | None, output_dir: Path | None, repo_root: Path
) -> tuple[Path, Path]:
    """Return (primary_eval_dir, fallback_eval_dir) for locating results.json.

    Primary:  logs/{last_component_of_scenario}  (evaluate CLI --output-dir)
    Fallback: logs/{scenario_with_slashes_as_underscores}  (training log dir)
    """
    if scenario:
        log_name = scenario.split("/")[-1]
        primary = repo_root / EXPERIMENT_OUTPUT_DIR / log_name
        fallback = repo_root / EXPERIMENT_OUTPUT_DIR / scenario.replace("/", "_")
    else:
        assert output_dir is not None
        primary = output_dir.resolve()
        fallback = primary
    return primary, fallback


def _newer_checkpoint(results_file: Path) -> Path | None:
    """Return the newest checkpoint when it is newer than ``results.json``."""
    checkpoints = list(results_file.parent.rglob("*_checkpoint*.pt"))
    if not checkpoints:
        return None
    newest = max(checkpoints, key=lambda path: path.stat().st_mtime_ns)
    return (
        newest if newest.stat().st_mtime_ns > results_file.stat().st_mtime_ns else None
    )


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
        "--allow-split-fallback",
        action="store_true",
        help=(
            "Publish val- or train-split metrics when no test split exists. "
            "Off by default: the thesis presents these snapshots as "
            "out-of-sample results, so a substituted split must be deliberate."
        ),
    )
    p.add_argument(
        "--allow-stale-results",
        action="store_true",
        help=(
            "Export even when results.json predates a checkpoint in its directory. "
            "Use only for a deliberate historical-checkpoint export."
        ),
    )
    p.add_argument(
        "--verbose",
        "-v",
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
    primary_dir, fallback_dir = _resolve_eval_dir(
        args.scenario, args.output_dir, repo_root
    )

    if args.experiment_name:
        experiment_name = args.experiment_name
    elif args.scenario:
        experiment_name = args.scenario.replace("/", "_")
    else:
        assert args.output_dir is not None
        experiment_name = primary_dir.name

    logger.info(
        "exporting scenario={} experiment={}",
        args.scenario or args.output_dir,
        experiment_name,
    )

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

    newer_checkpoint = _newer_checkpoint(results_file)
    if newer_checkpoint is not None:
        results_time = datetime.fromtimestamp(
            results_file.stat().st_mtime, UTC
        ).isoformat()
        checkpoint_time = datetime.fromtimestamp(
            newer_checkpoint.stat().st_mtime, UTC
        ).isoformat()
        if not args.allow_stale_results:
            logger.error("results.json predates a checkpoint; refusing stale export")
            logger.error("  results: {} ({})", results_file, results_time)
            logger.error("  checkpoint: {} ({})", newer_checkpoint, checkpoint_time)
            logger.error(
                "re-run evaluate for the current checkpoint, or pass "
                "--allow-stale-results for a deliberate historical export"
            )
            return 1
        logger.warning(
            "exporting stale results by explicit override: results={} ({}) "
            "newer_checkpoint={} ({})",
            results_file,
            results_time,
            newer_checkpoint,
            checkpoint_time,
        )

    try:
        results = _load_results_json(results_file)
    except json.JSONDecodeError as e:
        logger.error("failed to parse results.json: {}", e)
        return 1

    # --------------------------------------------------------------------------
    # Aggregate metrics across test splits
    # --------------------------------------------------------------------------
    metrics, source_split, source_keys = _aggregate_split_metrics(results)
    if not metrics:
        logger.warning("no metrics found in results.json")
    else:
        logger.info(
            "aggregated {} metrics from {} {!r} entr{} ({})",
            len(metrics),
            len(source_keys),
            source_split,
            "y" if len(source_keys) == 1 else "ies",
            ", ".join(source_keys),
        )

    if source_split is not None and source_split != "test":
        if not args.allow_split_fallback:
            logger.error(
                "results.json has no 'test' entry; the best available split is {!r} ({}).",
                source_split,
                ", ".join(source_keys),
            )
            logger.error(
                "refusing to publish {}-split numbers as the out-of-sample thesis "
                "snapshot for {!r}.",
                source_split,
                experiment_name,
            )
            logger.error(
                "produce a test split first:  uv run python src/cli.py evaluate -c {} "
                "--output-dir {} --per-symbol",
                args.scenario or args.output_dir,
                primary_dir,
            )
            logger.error("or pass --allow-split-fallback to publish it deliberately.")
            return 1
        logger.warning(
            "publishing {!r}-split metrics as the thesis snapshot for {!r} "
            "(--allow-split-fallback). These are NOT out-of-sample results.",
            source_split,
            experiment_name,
        )

    # Provenance travels with the numbers so downstream readers can tell which
    # split produced them. Consumers address metrics by explicit key, so the
    # extra entries are inert for the result tables.
    if metrics:
        metrics["__source_split__"] = source_split
        metrics["__source_keys__"] = source_keys
        # _aggregate_split_metrics collapses the per-symbol splits into one
        # unweighted mean, so the exported snapshot held only pooled numbers.
        # Table 5 needs the splits intact: locally it fell through to
        # logs/<scenario>/results.json and rendered, but logs/ is gitignored,
        # so on CI the loader found nothing and the table printed "No
        # per-instrument Hypothesis 1 results available". Carry the raw
        # per-symbol entries alongside the aggregate; they are addressed by
        # explicit key, so they are inert for every other consumer.
        per_symbol = {
            key: value
            for key, value in results.items()
            if key.startswith(f"{source_split}_") and isinstance(value, dict)
        }
        if per_symbol:
            metrics["__per_symbol__"] = per_symbol

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
            "uv run python src/cli.py evaluate -c {}",
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

    now = datetime.now(UTC).isoformat()
    # The eval CLI wrote results.json when the rollout finished; that mtime is
    # the only real run timing this script can see. Do not synthesise a run
    # start time or a FINISHED status we cannot verify (experiment-audit #11).
    results_mtime = datetime.fromtimestamp(
        results_file.stat().st_mtime, UTC
    ).isoformat()

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

    benchmark_table, benchmark_split = _load_benchmark_table(eval_dir)
    statistical_tests_file: str | None = None
    if benchmark_table is not None:
        if benchmark_split != "test" and not args.allow_split_fallback:
            logger.error(
                "benchmark tables in {} are {!r}-split; refusing to publish them as "
                "the out-of-sample comparison. Re-run evaluate with --per-symbol, "
                "or pass --allow-split-fallback.",
                eval_dir,
                benchmark_split,
            )
            return 1
        if benchmark_split != source_split:
            logger.warning(
                "benchmark table split {!r} does not match the metrics split {!r} — "
                "the comparison table and the metric row describe different data.",
                benchmark_split,
                source_split,
            )
        benchmark_table["__source_split__"] = benchmark_split
        _write_json(snapshot_dir / "statistical_tests.json", benchmark_table)
        statistical_tests_file = "statistical_tests.json"
        n_rows = len(benchmark_table.get("benchmark_comparison_table", []))
        logger.info(
            "benchmark table: {} strategies from {!r} split", n_rows, benchmark_split
        )

    run_json: dict = {
        "run_id": None,
        "run_name": experiment_name,
        # Not "FINISHED": this script exports an evaluate-CLI results file and
        # has no way to confirm the training run completed. start_time is
        # unknown; end_time is when results.json was written (experiment-audit #11).
        "status": "EXPORTED",
        "start_time": None,
        "end_time": results_mtime,
        "artifact_uri": None,
        "experiment_name": experiment_name,
        "experiment_id": None,
        "source": {
            "type": "evaluate_cli",
            "eval_output_dir": str(eval_dir),
            "results_file": str(results_file),
            "results_file_mtime_utc": results_mtime,
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

    logger.info(
        "exported thesis snapshot  experiment={}  location={}",
        experiment_name,
        snapshot_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
