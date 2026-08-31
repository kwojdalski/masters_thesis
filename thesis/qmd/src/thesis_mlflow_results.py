"""Utilities for loading thesis experiment outputs into Quarto chapters.

This module prefers exported thesis result snapshots (JSON / Parquet / PNG)
stored under ``thesis/qmd/results``. If no snapshot is available, it falls back
to read-only queries against the local MLflow SQLite backend and artifact store.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from trading_rl.evaluation.asset_meta import write_asset_meta as _write_asset_meta
except Exception:
    _write_asset_meta = None

try:
    from logger import get_logger

    _logger = get_logger(__name__)
except Exception:
    _logger = None


def _log_fallback(context: str, exc: Exception) -> None:
    """Debug-log a swallowed exception from a fallback data-loading attempt.

    Failures here are expected (a source may simply not be populated yet) and
    the caller always has another fallback to try, so this is deliberately
    debug-level rather than a warning -- but silent `except: pass` makes a
    genuinely broken source indistinguishable from a merely-absent one.
    """
    if _logger is not None:
        _logger.debug(f"{context}: {type(exc).__name__}: {exc}")


try:
    from trading_rl.config import EXPERIMENT_OUTPUT_DIR as _EXPERIMENT_OUTPUT_DIR
except Exception:
    _EXPERIMENT_OUTPUT_DIR = Path("logs")


def _repo_root() -> Path:
    # thesis/qmd/src/thesis_mlflow_results.py -> repo root is 3 levels up
    return Path(__file__).resolve().parents[3]


def mlflow_db_path() -> Path:
    return _repo_root() / "mlflow.db"


def thesis_results_root() -> Path:
    return _repo_root() / "thesis" / "qmd" / "results"


def _experiment_snapshot_dir(
    experiment_name: str, output_root: Path | None = None
) -> Path:
    root = output_root if output_root is not None else thesis_results_root()
    return root / experiment_name


def _connect() -> sqlite3.Connection:
    # Open read-only to avoid contention with active training / dashboard.
    return sqlite3.connect(f"file:{mlflow_db_path()}?mode=ro", uri=True, timeout=1)


def get_experiment_by_name(name: str) -> dict[str, Any] | None:
    with _connect() as con:
        q = """
        SELECT experiment_id, name, artifact_location
        FROM experiments
        WHERE name = ?
        LIMIT 1
        """
        row = pd.read_sql_query(q, con, params=[name])
    if row.empty:
        return None
    return row.iloc[0].to_dict()


def get_runs(experiment_id: int | str) -> pd.DataFrame:
    with _connect() as con:
        q = """
        SELECT
            run_uuid AS run_id,
            name AS run_name,
            status,
            start_time,
            end_time,
            artifact_uri
        FROM runs
        WHERE experiment_id = ? AND lifecycle_stage = 'active'
        ORDER BY start_time DESC
        """
        df = pd.read_sql_query(q, con, params=[int(experiment_id)])
    if not df.empty:
        for col in ["start_time", "end_time"]:
            df[col] = pd.to_datetime(df[col], unit="ms", errors="coerce")
    return df


def get_latest_metrics(run_id: str) -> pd.Series:
    with _connect() as con:
        q = """
        SELECT key, value, step, timestamp
        FROM latest_metrics
        WHERE run_uuid = ?
        """
        df = pd.read_sql_query(q, con, params=[run_id])
    if df.empty:
        return pd.Series(dtype=float)
    # Convert to a simple series key -> value
    return pd.Series(df["value"].values, index=df["key"].values, dtype="float64")


def get_params(run_id: str) -> dict[str, str]:
    with _connect() as con:
        q = "SELECT key, value FROM params WHERE run_uuid = ?"
        df = pd.read_sql_query(q, con, params=[run_id])
    if df.empty:
        return {}
    return dict(zip(df["key"], df["value"], strict=False))


def _artifact_dir_from_uri(artifact_uri: str | None) -> Path | None:
    if not artifact_uri:
        return None
    p = Path(artifact_uri)
    if p.exists():
        return p
    return None


def _latest_file(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    return max(paths, key=lambda p: p.stat().st_mtime)


def _load_latest_json_artifact(
    artifact_uri: str | None,
    artifact_subdir: str,
    *,
    preferred_splits: tuple[str, ...] = ("test", "val", "train"),
) -> dict[str, Any] | None:
    artifact_dir = _artifact_dir_from_uri(artifact_uri)
    if artifact_dir is None:
        return None

    root = artifact_dir / artifact_subdir
    if not root.exists():
        return None

    candidate_paths: list[Path] = []
    for split in preferred_splits:
        split_dir = root / split
        if split_dir.exists():
            candidate_paths.extend(split_dir.glob("*.json"))
    if not candidate_paths:
        candidate_paths.extend(root.glob("*.json"))
    if not candidate_paths:
        candidate_paths.extend(root.glob("*/*.json"))

    path = _latest_file(sorted(candidate_paths))
    if path is None:
        return None
    return json.loads(path.read_text())


def load_latest_evaluation_report(artifact_uri: str | None) -> dict[str, Any] | None:
    result = _load_latest_json_artifact(artifact_uri, "evaluation_data")
    if result is None:
        # Fall back to old path for runs logged before the rename.
        result = _load_latest_json_artifact(artifact_uri, "evaluation_metrics")
    return result


def load_latest_statistical_tests(artifact_uri: str | None) -> dict[str, Any] | None:
    return _load_latest_json_artifact(artifact_uri, "statistical_tests")


def _scenario_log_dirs(experiment_name: str) -> list[Path]:
    """Candidate log directories for a given experiment name.

    The run scripts use LOG_NAME="${SCENARIO##*/}" which strips the group prefix
    (e.g. "pooled/td3_..." → "td3_...").  To cover both the training dir
    (pooled_td3_...) and the evaluate output dir (td3_...) we probe both.
    """
    logs_root = _repo_root() / _EXPERIMENT_OUTPUT_DIR
    candidates: list[Path] = [logs_root / experiment_name]
    # Strip the first underscore-delimited component to derive the LOG_NAME.
    parts = experiment_name.split("_", 1)
    if len(parts) == 2:
        candidates.append(logs_root / parts[1])
    return [p for p in candidates if p.exists()]


def find_evaluation_plot_data(
    artifact_uri: str | None,
    artifact_subdir: str = "evaluation_plots",
) -> dict[str, Any]:
    """Return plot DataFrames and metadata saved by log_evaluation_plots.

    Loads parquet files (<timestamp>_{rewards,actions,actions_ma,equity}_data.parquet)
    and the companion JSON metadata file from the given artifact subdirectory.

    Returns a dict with keys: rewards, actions, actions_ma (optional), equity (optional),
    plus metadata keys: stride, date_str, reward_type, is_portfolio, training_steps,
    training_episodes, n_obs, allocation_ma_window, initial_portfolio_value,
    policy_mode, symbols, n_total_symbols.
    Returns an empty dict if no parquet files are found.
    """
    import json

    artifact_dir = _artifact_dir_from_uri(artifact_uri)
    if artifact_dir is None:
        return {}

    plot_dir = artifact_dir / artifact_subdir
    if not plot_dir.exists():
        return {}

    from trading_rl.evaluation.asset_meta import load_asset_meta

    result: dict[str, Any] = {}

    def _find_parquets(name: str) -> list[Path]:
        # Prefer split subdirs (test > val > train) then root then any subdir.
        for split in ("test", "val", "train"):
            hits = sorted((plot_dir / split).glob(f"*_{name}_data.parquet"))
            if hits:
                return hits
        hits = sorted(plot_dir.glob(f"*_{name}_data.parquet"))
        if hits:
            return hits
        return sorted(plot_dir.glob(f"**/*_{name}_data.parquet"))

    for frame_name in ("rewards", "actions", "actions_ma", "equity"):
        files = _find_parquets(frame_name)
        if files:
            try:
                result[frame_name] = pd.read_parquet(files[-1])
                sidecar = load_asset_meta(files[-1])
                if sidecar:
                    result.setdefault("asset_meta", {})[frame_name] = sidecar
            except Exception as exc:
                _log_fallback(f"reading rollout parquet for {frame_name!r}", exc)

    def _find_meta_json() -> list[Path]:
        for split in ("test", "val", "train"):
            hits = sorted((plot_dir / split).glob("*_plot_meta.json"))
            if hits:
                return hits
        hits = sorted(plot_dir.glob("*_plot_meta.json"))
        return hits or sorted(plot_dir.glob("**/*_plot_meta.json"))

    meta_files = _find_meta_json()
    if meta_files:
        try:
            with open(meta_files[-1], encoding="utf-8") as f:
                result.update(json.load(f))
        except Exception as exc:
            _log_fallback("reading plot metadata json", exc)

    return result


def find_evaluation_plots(
    artifact_uri: str | None,
    *,
    log_dirs: list[Path] | None = None,
) -> dict[str, Path]:
    """Return latest available evaluation plots for a run.

    Preference order:
    1. final evaluation_plots/* inside the MLflow artifact directory
    2. temporary evaluation_plots_temp/**/* inside the MLflow artifact directory
    3. scenario-specific log dirs (logs/{log_name}/) — written by the evaluate CLI
       when --only plots is included; checked live on every render
    4. eval_results/ at the repo root (non-scenario-specific fallback)
    """
    plot_keys = {
        "merged_comparison": "*_merged_comparison.png",
        "rewards": "*_rewards.png",
        "positions": "*_positions.png",
        "actual_returns": "*_actual_returns.png",
    }
    # Patterns used by the evaluate CLI command in its --output-dir
    cli_plot_patterns = {
        "rewards": "*_reward_plot.png",
        "positions": "*_action_plot.png",
    }
    found: dict[str, Path] = {}

    artifact_dir = _artifact_dir_from_uri(artifact_uri)
    if artifact_dir is not None:
        final_dir = artifact_dir / "evaluation_plots"
        if final_dir.exists():
            for key, pattern in plot_keys.items():
                p = _latest_file(list(final_dir.glob(pattern)))
                if p is not None:
                    found[key] = p

        if len(found) < len(plot_keys):
            temp_dir = artifact_dir / "evaluation_plots_temp"
            if temp_dir.exists():
                for key, pattern in plot_keys.items():
                    if key in found:
                        continue
                    p = _latest_file(list(temp_dir.glob(f"**/{pattern}")))
                    if p is not None:
                        found[key] = p

    # Scenario-specific log dirs (checked live — always reflects the latest evaluate run)
    for log_dir in log_dirs or []:
        if not log_dir.exists():
            continue
        for key, pattern in cli_plot_patterns.items():
            if key in found:
                continue
            p = _latest_file(list(log_dir.glob(pattern)))
            if p is not None:
                found[key] = p

    # Non-scenario-specific fallback: eval_results/ at the repo root
    eval_results_dir = _repo_root() / "eval_results"
    if eval_results_dir.exists():
        for key, pattern in cli_plot_patterns.items():
            if key in found:
                continue
            p = _latest_file(list(eval_results_dir.glob(pattern)))
            if p is not None:
                found[key] = p

    return found


def _find_static_export_plots(experiment_name: str) -> dict[str, Path]:
    """Return plots from a previously exported thesis snapshot, if any."""
    snap_dir = _experiment_snapshot_dir(experiment_name) / "latest_finished"
    run_json_path = snap_dir / "run.json"
    if not run_json_path.exists():
        return {}
    try:
        raw = json.loads(run_json_path.read_text())
        return {
            key: (snap_dir / rel_path)
            for key, rel_path in (raw.get("evaluation_plots") or {}).items()
            if (snap_dir / rel_path).exists()
        }
    except Exception:
        return {}


def latest_run_for_experiment(
    experiment_name: str, status: str | None = None
) -> dict[str, Any] | None:
    exp = get_experiment_by_name(experiment_name)
    if exp is None:
        return None
    runs = get_runs(int(exp["experiment_id"]))
    if runs.empty:
        return None
    if status is not None:
        runs = runs[runs["status"] == status]
        if runs.empty:
            return None
    row = runs.iloc[0].to_dict()
    row["experiment_name"] = exp["name"]
    row["experiment_id"] = int(exp["experiment_id"])
    row["latest_metrics"] = get_latest_metrics(row["run_id"])
    row["params"] = get_params(row["run_id"])
    row["evaluation_report"] = load_latest_evaluation_report(row["artifact_uri"])
    row["statistical_tests"] = load_latest_statistical_tests(row["artifact_uri"])
    log_dirs = _scenario_log_dirs(experiment_name)
    row["evaluation_plots"] = find_evaluation_plots(
        row["artifact_uri"], log_dirs=log_dirs
    )
    # Supplement with static export plots when neither MLflow nor log dirs have any
    # (e.g. evaluate was run with --only metrics before plots were ever generated).
    if not row["evaluation_plots"]:
        row["evaluation_plots"] = _find_static_export_plots(experiment_name)
    return row


def _runs_overview_table_from_mlflow(experiment_name: str) -> pd.DataFrame:
    exp = get_experiment_by_name(experiment_name)
    if exp is None:
        return pd.DataFrame()
    runs = get_runs(int(exp["experiment_id"]))
    if runs.empty:
        return runs

    rows: list[dict[str, Any]] = []
    for _, run in runs.iterrows():
        latest = get_latest_metrics(str(run["run_id"]))
        rows.append(
            {
                "run_name": run["run_name"],
                "status": run["status"],
                "start_time": run["start_time"],
                "end_time": run["end_time"],
                "final_reward": latest.get("final_reward"),
                "training_steps": latest.get("training_steps"),
                "episode_reward": latest.get("episode_reward"),
                "value_loss": latest.get("value_loss"),
            }
        )
    return pd.DataFrame(rows)


def format_key_metrics(report: dict[str, Any] | None) -> pd.DataFrame:
    if not report:
        return pd.DataFrame(columns=["Metric", "Value"])

    rows: list[tuple[str, str]] = []
    pct_keys = {"annualized_return_cagr", "annualized_volatility", "win_rate"}
    small_return_keys = {"total_return", "max_drawdown"}
    key_order = [
        ("total_return", "Total Return"),
        ("annualized_volatility", "Annualized Volatility"),
        ("sharpe_ratio", "Sharpe Ratio"),
        ("sortino_ratio", "Sortino Ratio"),
        ("max_drawdown", "Max Drawdown"),
        ("win_rate", "Win Rate"),
        ("profit_factor", "Profit Factor"),
        ("omega_ratio", "Omega Ratio"),
        ("var_95", "VaR (95%)"),
        ("cvar_95", "CVaR (95%)"),
    ]
    for key, label in key_order:
        if key not in report:
            continue
        val = report[key]
        if isinstance(val, int | float):
            if key in small_return_keys:
                rows.append((label, f"{val:.2e}"))
            elif key in pct_keys:
                rows.append((label, f"{val * 100:.2f}%"))
            else:
                rows.append((label, f"{val:.4f}"))
        else:
            rows.append((label, str(val)))
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def format_benchmark_comparison_table(
    statistical_tests: dict[str, Any] | None,
) -> pd.DataFrame:
    if not statistical_tests:
        return pd.DataFrame()

    benchmark_table = statistical_tests.get("benchmark_comparison_table")
    if not isinstance(benchmark_table, list) or not benchmark_table:
        return pd.DataFrame()

    frame = pd.DataFrame(benchmark_table)
    display_frame = frame.copy()
    pct_columns = {"annualized_return_cagr", "annualized_volatility", "win_rate"}
    small_return_columns = {"total_return", "max_drawdown"}
    ratio_columns = {"sharpe_ratio", "sortino_ratio", "turnover"}
    ordered_columns = [
        "strategy",
        "total_return",
        "annualized_volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "win_rate",
        "turnover",
    ]
    labels = {
        "strategy": "Strategy",
        "total_return": "TR",
        "annualized_volatility": "Volatility",
        "sharpe_ratio": "SR",
        "sortino_ratio": "Sortino",
        "max_drawdown": "Max DD",
        "win_rate": "Win Rate",
        "turnover": "Turnover",
    }

    display_frame = display_frame[
        [c for c in ordered_columns if c in display_frame.columns]
    ]
    for column in display_frame.columns:
        if column == "strategy":
            continue
        if column in small_return_columns:
            display_frame[column] = display_frame[column].apply(
                lambda x: f"{float(x):.2e}" if pd.notna(x) else "N/A"
            )
        elif column in pct_columns:
            display_frame[column] = display_frame[column].apply(
                lambda x: f"{float(x) * 100:.2f}%" if pd.notna(x) else "N/A"
            )
        elif column in ratio_columns:
            display_frame[column] = display_frame[column].apply(
                lambda x: f"{float(x):.4f}" if pd.notna(x) else "N/A"
            )
        else:
            display_frame[column] = display_frame[column].apply(
                lambda x: f"{float(x):.4f}" if pd.notna(x) else "N/A"
            )

    return display_frame.rename(columns=labels)


def format_statistical_significance_summary(
    statistical_tests: dict[str, Any] | None,
) -> pd.DataFrame:
    if not statistical_tests:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    baselines = statistical_tests.get("baselines", [])
    for baseline_result in baselines:
        if not isinstance(baseline_result, dict) or "error" in baseline_result:
            continue
        baseline_name = str(baseline_result.get("baseline", "unknown"))
        for test_name, test_result in baseline_result.items():
            if not isinstance(test_result, dict):
                continue
            p_value = test_result.get("p_value")
            significant = test_result.get("significant")
            if p_value is None:
                continue
            rows.append(
                {
                    "Test": str(test_name).replace("_", " ").title(),
                    "Benchmark": baseline_name.replace("_", " ").title(),
                    "p-value": float(p_value),
                    "Significant (p < 0.05)": ("Yes" if bool(significant) else "No"),
                }
            )

    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    frame = frame.sort_values(["Benchmark", "Test"], ignore_index=True)
    frame["p-value"] = frame["p-value"].map(lambda x: f"{x:.4f}")
    return frame


@dataclass
class ExperimentSnapshot:
    experiment_name: str
    latest_running: dict[str, Any] | None
    latest_finished: dict[str, Any] | None


def _load_experiment_snapshot_from_mlflow(experiment_name: str) -> ExperimentSnapshot:
    return ExperimentSnapshot(
        experiment_name=experiment_name,
        latest_running=latest_run_for_experiment(experiment_name, status="RUNNING"),
        latest_finished=latest_run_for_experiment(experiment_name, status="FINISHED"),
    )


def _sanitise_for_json(obj: Any) -> Any:
    """Recursively replace NaN/Inf floats with None so json.dumps produces valid JSON."""
    import math

    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    if isinstance(obj, dict):
        return {k: _sanitise_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitise_for_json(v) for v in obj]
    return obj


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_sanitise_for_json(payload), indent=2, default=str))
    if _write_asset_meta:
        _write_asset_meta(path, generator="thesis_mlflow_results.py")


def _iso_or_none(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _json_number_or_none(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception as exc:
        _log_fallback(f"pd.isna check on {type(value).__name__} value", exc)
    if isinstance(value, int | float | str | bool):
        return value
    return str(value)


def _copy_plots_to_snapshot(
    plots: dict[str, Path], destination_dir: Path
) -> dict[str, str]:
    copied: dict[str, str] = {}
    plot_dir = destination_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for key, src in plots.items():
        src = Path(src)
        if not src.exists():
            continue
        filename = f"{key}{src.suffix.lower() or '.png'}"
        dst = plot_dir / filename
        shutil.copy2(src, dst)
        copied[key] = str(dst.relative_to(destination_dir))
        sidecar = src.with_name(src.name + ".meta.json")
        if sidecar.exists():
            shutil.copy2(sidecar, plot_dir / (filename + ".meta.json"))
    return copied


def _serialize_run_payload_for_export(
    run: dict[str, Any], destination_dir: Path
) -> dict[str, Any]:
    destination_dir.mkdir(parents=True, exist_ok=True)

    params = dict(run.get("params", {}) or {})
    latest_metrics_obj = run.get("latest_metrics")
    latest_metrics_dict: dict[str, Any]
    if isinstance(latest_metrics_obj, pd.Series):
        latest_metrics_dict = {
            str(k): _json_number_or_none(v)
            for k, v in latest_metrics_obj.to_dict().items()
        }
    elif isinstance(latest_metrics_obj, dict):
        latest_metrics_dict = {
            str(k): _json_number_or_none(v) for k, v in latest_metrics_obj.items()
        }
    else:
        latest_metrics_dict = {}

    _write_json(destination_dir / "params.json", params)
    _write_json(destination_dir / "latest_metrics.json", latest_metrics_dict)

    evaluation_report_file: str | None = None
    evaluation_report = run.get("evaluation_report")
    if isinstance(evaluation_report, dict):
        evaluation_report_file = "evaluation_report.json"
        _write_json(destination_dir / evaluation_report_file, evaluation_report)

    statistical_tests_file: str | None = None
    statistical_tests = run.get("statistical_tests")
    if isinstance(statistical_tests, dict):
        statistical_tests_file = "statistical_tests.json"
        _write_json(destination_dir / statistical_tests_file, statistical_tests)

    plots = run.get("evaluation_plots", {}) or {}
    plot_relpaths = _copy_plots_to_snapshot(plots, destination_dir) if plots else {}

    run_json = {
        "run_id": run.get("run_id"),
        "run_name": run.get("run_name"),
        "status": run.get("status"),
        "start_time": _iso_or_none(run.get("start_time")),
        "end_time": _iso_or_none(run.get("end_time")),
        "artifact_uri": run.get("artifact_uri"),
        "experiment_name": run.get("experiment_name"),
        "experiment_id": run.get("experiment_id"),
        "files": {
            "params": "params.json",
            "latest_metrics": "latest_metrics.json",
            "evaluation_report": evaluation_report_file,
            "statistical_tests": statistical_tests_file,
        },
        "evaluation_plots": plot_relpaths,
    }
    _write_json(destination_dir / "run.json", run_json)
    return run_json


def export_experiment_snapshot(
    experiment_name: str, output_root: Path | None = None
) -> Path:
    """Export a thesis-friendly snapshot for an MLflow experiment.

    The export contains JSON/Parquet/PNG artifacts so Quarto can render without
    querying the live MLflow database.
    """
    output_dir = _experiment_snapshot_dir(experiment_name, output_root=output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot = _load_experiment_snapshot_from_mlflow(experiment_name)
    runs_df = _runs_overview_table_from_mlflow(experiment_name)

    # Refresh directory contents while preserving the top-level directory itself.
    for child in output_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()

    if not runs_df.empty:
        _parquet_path = output_dir / "runs_overview.parquet"
        runs_df.to_parquet(_parquet_path, index=False)
        if _write_asset_meta:
            _write_asset_meta(_parquet_path, generator="thesis_mlflow_results.py")
        _json_path = output_dir / "runs_overview.json"
        runs_df.assign(
            start_time=runs_df["start_time"].astype(str),
            end_time=runs_df["end_time"].astype(str),
        ).to_json(_json_path, orient="records", indent=2)
        if _write_asset_meta:
            _write_asset_meta(_json_path, generator="thesis_mlflow_results.py")
    else:
        pd.DataFrame().to_parquet(output_dir / "runs_overview.parquet", index=False)
        _write_json(output_dir / "runs_overview.json", [])

    exported_runs: dict[str, Any] = {}
    for slot_name in ("latest_running", "latest_finished"):
        run = getattr(snapshot, slot_name)
        if run is None:
            exported_runs[slot_name] = None
            continue
        slot_dir = output_dir / slot_name
        exported_runs[slot_name] = _serialize_run_payload_for_export(run, slot_dir)

    manifest = {
        "schema_version": 1,
        "experiment_name": experiment_name,
        "exported_at_utc": datetime.now(UTC).isoformat(),
        "source": {
            "type": "mlflow",
            "mlflow_db_path": str(mlflow_db_path()),
        },
        "files": {
            "runs_overview_parquet": "runs_overview.parquet",
            "runs_overview_json": "runs_overview.json",
        },
        "runs": {
            "latest_running": None
            if exported_runs["latest_running"] is None
            else "latest_running/run.json",
            "latest_finished": None
            if exported_runs["latest_finished"] is None
            else "latest_finished/run.json",
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    return output_dir


def _load_run_from_export(run_json_path: Path) -> dict[str, Any] | None:
    if not run_json_path.exists():
        return None
    raw = json.loads(run_json_path.read_text())
    base_dir = run_json_path.parent

    params_file = raw.get("files", {}).get("params")
    latest_metrics_file = raw.get("files", {}).get("latest_metrics")
    evaluation_report_file = raw.get("files", {}).get("evaluation_report")
    statistical_tests_file = raw.get("files", {}).get("statistical_tests")

    params = {}
    if params_file:
        p = base_dir / params_file
        if p.exists():
            params = json.loads(p.read_text())

    latest_metrics = pd.Series(dtype=float)
    if latest_metrics_file:
        p = base_dir / latest_metrics_file
        if p.exists():
            latest_metrics_dict = json.loads(p.read_text())
            latest_metrics = pd.Series(latest_metrics_dict, dtype="float64")

    evaluation_report = None
    if evaluation_report_file:
        p = base_dir / evaluation_report_file
        if p.exists():
            evaluation_report = json.loads(p.read_text())

    statistical_tests = None
    if statistical_tests_file:
        p = base_dir / statistical_tests_file
        if p.exists():
            statistical_tests = json.loads(p.read_text())
    if statistical_tests is None:
        statistical_tests = load_latest_statistical_tests(raw.get("artifact_uri"))

    evaluation_plots = {
        key: (base_dir / rel_path)
        for key, rel_path in (raw.get("evaluation_plots") or {}).items()
    }

    loaded = dict(raw)
    loaded["start_time"] = pd.to_datetime(raw.get("start_time"), errors="coerce")
    loaded["end_time"] = pd.to_datetime(raw.get("end_time"), errors="coerce")
    loaded["params"] = params
    loaded["latest_metrics"] = latest_metrics
    loaded["evaluation_report"] = evaluation_report
    loaded["statistical_tests"] = statistical_tests
    loaded["evaluation_plots"] = evaluation_plots
    return loaded


def _load_experiment_snapshot_from_export(
    experiment_name: str,
) -> ExperimentSnapshot | None:
    snapshot_dir = _experiment_snapshot_dir(experiment_name)
    manifest_path = snapshot_dir / "manifest.json"
    if not manifest_path.exists():
        return None

    manifest = json.loads(manifest_path.read_text())
    runs = manifest.get("runs", {})
    latest_running = None
    latest_finished = None
    if runs.get("latest_running"):
        latest_running = _load_run_from_export(snapshot_dir / runs["latest_running"])
    if runs.get("latest_finished"):
        latest_finished = _load_run_from_export(snapshot_dir / runs["latest_finished"])

    return ExperimentSnapshot(
        experiment_name=experiment_name,
        latest_running=latest_running,
        latest_finished=latest_finished,
    )


def _supplement_run_from_export(
    live_run: dict[str, Any], export_run: dict[str, Any]
) -> None:
    """Fill gaps in a live-MLflow run dict using a static-export run dict.

    The live MLflow path may be missing artifacts that were only written to the
    thesis snapshot (e.g. statistical_tests, final evaluation plots).  Supplement
    rather than replace so live data always takes precedence.
    """
    if (
        live_run.get("statistical_tests") is None
        and export_run.get("statistical_tests") is not None
    ):
        live_run["statistical_tests"] = export_run["statistical_tests"]
    if not live_run.get("evaluation_plots") and export_run.get("evaluation_plots"):
        live_run["evaluation_plots"] = export_run["evaluation_plots"]


def load_experiment_snapshot(experiment_name: str) -> ExperimentSnapshot:
    # Prefer live MLflow so renders always reflect the latest finished run.
    # Fall back to the static export when the database is unavailable (CI, offline).
    try:
        live = _load_experiment_snapshot_from_mlflow(experiment_name)
        if live.latest_finished is not None or live.latest_running is not None:
            # Supplement any missing artifacts (e.g. statistical_tests) from the
            # static export, which may contain data not stored in the MLflow
            # artifact directory.
            exported = _load_experiment_snapshot_from_export(experiment_name)
            if exported is not None:
                for slot in ("latest_finished", "latest_running"):
                    live_run = getattr(live, slot)
                    exp_run = getattr(exported, slot)
                    if live_run is not None and exp_run is not None:
                        _supplement_run_from_export(live_run, exp_run)
            return live
    except Exception as exc:
        _log_fallback(f"loading live MLflow snapshot for {experiment_name!r}", exc)
    exported = _load_experiment_snapshot_from_export(experiment_name)
    if exported is not None:
        return exported
    return ExperimentSnapshot(
        experiment_name=experiment_name,
        latest_running=None,
        latest_finished=None,
    )


def runs_overview_table(experiment_name: str) -> pd.DataFrame:
    # Same preference: live MLflow first, static export as fallback.
    try:
        df = _runs_overview_table_from_mlflow(experiment_name)
        if not df.empty:
            return df
    except Exception as exc:
        _log_fallback(
            f"loading runs overview from live MLflow for {experiment_name!r}", exc
        )
    snapshot_dir = _experiment_snapshot_dir(experiment_name)
    json_path = snapshot_dir / "runs_overview.json"
    parquet_path = snapshot_dir / "runs_overview.parquet"
    if json_path.exists():
        df = pd.read_json(json_path)
        for col in ("start_time", "end_time"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        return df
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Multi-scenario comparison helpers
# ---------------------------------------------------------------------------


def _load_results_json_tolerant(path: Path) -> dict:
    """Parse results.json that may contain bare NaN/Infinity tokens."""
    raw = path.read_text(encoding="utf-8")
    raw = re.sub(r"\bNaN\b", "null", raw)
    raw = re.sub(r"\bInfinity\b", "null", raw)
    raw = re.sub(r"\b-Infinity\b", "null", raw)
    return json.loads(raw)


def _results_split_entries(results: dict, prefix: str) -> tuple[dict, dict]:
    """Return ``(pooled, per_symbol)`` metric entries for one split prefix.

    A pooled entry is keyed by the bare split name (``"test"``). Per-symbol
    entries carry a symbol suffix in one of two shapes depending on which code
    path wrote them: ``"test_AAPL"`` from ``EvaluateCommand`` and
    ``"val__AAPL"`` from ``pipeline.evaluation.evaluate_per_symbol``. Matching
    on the ``_`` boundary covers both without letting ``"test"`` also swallow
    its own per-symbol components.

    Mirrors ``_split_entries`` in scripts/export_eval_to_thesis.py.
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


def _aggregate_from_results_json(results: dict, split: str = "test") -> dict[str, Any]:
    """Average metrics for the requested split in a results.json dict.

    Falls back to val then train when the requested split is absent, but stamps
    ``__source_split__`` on the result so callers can tell that a substitution
    happened -- these numbers are rendered by the results chapters as
    out-of-sample performance, and a silent train-for-test swap presents
    in-sample figures as held-out ones.

    Pooled and per-symbol entries are never averaged together: a pooled figure
    is already an aggregate over the same symbols, so mixing it with its own
    components double-counts them.
    """
    # dict.fromkeys keeps order while dropping the duplicate that appears when
    # split is itself "val" or "train".
    entries: dict = {}
    source_split: str | None = None
    for prefix in dict.fromkeys((split, "test", "val", "train")):
        pooled, per_symbol = _results_split_entries(results, prefix)
        # Prefer the disaggregated per-symbol entries when both are present.
        entries = per_symbol or pooled
        if entries:
            source_split = prefix
            break

    if not entries or source_split is None:
        return {}

    all_keys: set[str] = set()
    for entry in entries.values():
        all_keys.update(entry["metrics"].keys())

    aggregated: dict[str, Any] = {}
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

    # Provenance travels with the numbers; load_scenario_metrics warns on a
    # mismatch. Consumers address metrics by explicit key, so this is inert
    # for the result tables.
    aggregated["__source_split__"] = source_split
    aggregated["__source_keys__"] = sorted(entries)
    return aggregated


def _warn_on_split_mismatch(
    scenario_name: str, report: dict[str, Any], requested_split: str, source: str
) -> None:
    """Warn when a snapshot's metrics did not come from the requested split.

    export_eval_to_thesis.py stamps ``__source_split__`` on every snapshot it
    writes.  A snapshot exported with --allow-split-fallback carries val- or
    train-split numbers, which the results chapters would otherwise render as
    out-of-sample performance without any visible signal.  Snapshots written
    before the stamp existed have no key and cannot be checked here -- they
    need re-exporting.
    """
    source_split = report.get("__source_split__")
    if source_split is None or source_split == requested_split:
        return
    message = (
        f"{scenario_name}: metrics come from the {source_split!r} split, not "
        f"{requested_split!r} ({source}). These are NOT out-of-sample results."
    )
    if _logger is not None:
        _logger.warning(message)
    else:  # pragma: no cover - logger is configured in normal thesis builds
        print(f"WARNING: {message}")


def load_scenario_metrics(scenario_name: str, *, split: str = "test") -> dict[str, Any]:
    """Load aggregated evaluation metrics for a scenario.

    Preference order:
    1. Thesis snapshot: thesis/qmd/results/{scenario_name}/latest_finished/evaluation_report.json
    2. MLflow artifact store (if experiment exists in the tracking DB)
    3. logs/{log_name}/results.json read directly (strips first '_'-delimited component
       so "pooled_td3_..." maps to logs/td3_.../results.json)
    """
    # 1. Thesis snapshot (fastest — no DB round-trip)
    snap_path = (
        thesis_results_root()
        / scenario_name
        / "latest_finished"
        / "evaluation_report.json"
    )
    if snap_path.exists():
        try:
            data = json.loads(snap_path.read_text())
            if isinstance(data, dict) and data:
                _warn_on_split_mismatch(scenario_name, data, split, str(snap_path))
                return data
        except Exception as exc:
            _log_fallback(
                f"reading thesis snapshot evaluation_report for {scenario_name!r}", exc
            )

    # 2. MLflow
    try:
        live = _load_experiment_snapshot_from_mlflow(scenario_name)
        if live.latest_finished is not None:
            report = live.latest_finished.get("evaluation_report")
            if isinstance(report, dict) and report:
                return report
    except Exception as exc:
        _log_fallback(
            f"loading live MLflow evaluation_report for {scenario_name!r}", exc
        )

    # 3. logs/{log_name}/results.json
    logs_root = _repo_root() / _EXPERIMENT_OUTPUT_DIR
    parts = scenario_name.split("_", 1)
    log_name = parts[1] if len(parts) == 2 else scenario_name
    for candidate in (log_name, scenario_name):
        results_path = logs_root / candidate / "results.json"
        if results_path.exists():
            try:
                results = _load_results_json_tolerant(results_path)
                aggregated = _aggregate_from_results_json(results, split)
                if aggregated:
                    _warn_on_split_mismatch(
                        scenario_name, aggregated, split, str(results_path)
                    )
                    return aggregated
            except Exception as exc:
                _log_fallback(f"reading logs results.json at {results_path}", exc)

    return {}


def load_experiment_hyperparams(experiment_name: str) -> dict[str, Any]:
    """Load training hyperparameters from the static export snapshot.

    Reads hyperparams.json written by export_eval_to_thesis.py, which reflects
    the scenario train.yaml used for the run. Returns an empty dict when no
    snapshot is available.
    """
    hp_path = (
        thesis_results_root() / experiment_name / "latest_finished" / "hyperparams.json"
    )
    if hp_path.exists():
        try:
            return json.loads(hp_path.read_text())
        except Exception as exc:
            _log_fallback(f"reading hyperparams.json for {experiment_name!r}", exc)
    return {}


# ---------------------------------------------------------------------------
# Scenario observation config helpers
# ---------------------------------------------------------------------------

_FEATURE_GROUP_PATTERNS: list[tuple[str, str]] = [
    ("book_pressure", "imbalance"),
    ("order_book_imbalance", "imbalance"),
    ("order_count_imbalance", "imbalance"),
    ("microprice", "fair_value"),
    ("vwmp_skew", "fair_value"),
    ("price_vamp", "fair_value"),
    ("spread", "spread"),
    ("bid_convexity", "spread"),
    ("ask_convexity", "spread"),
    ("bid_slope", "spread"),
    ("ask_slope", "spread"),
    ("depth_ratio", "spread"),
    ("ofi", "flow"),
    ("queue_depletion", "flow"),
    ("signed_trade_flow", "flow"),
    ("vpin", "flow"),
    ("large_trade_ratio", "flow"),
    ("cancel_to_trade", "flow"),
    ("trade_arrival_rate", "flow"),
    ("odd_lot", "flow"),
    ("inter_event_time", "regime"),
    ("mid_price_acceleration", "regime"),
    ("hour_sin", "regime"),
    ("hour_cos", "regime"),
]


def _classify_feature_group(feature_name: str) -> str:
    """Map a feature column name to its microstructure group.

    Uses ordered substring matching against _FEATURE_GROUP_PATTERNS.
    Returns "other" if no pattern matches.
    """
    lower = feature_name.lower()
    for pattern, group in _FEATURE_GROUP_PATTERNS:
        if pattern in lower:
            return group
    return "other"


def load_scenario_feature_info(scenario_path: str) -> dict[str, Any]:
    """Load feature metadata from a scenario's observation.yaml.

    ``scenario_path`` is relative to ``src/configs/scenarios/``, e.g.
    ``"pooled/td3_h3_features_full"``.

    Returns a dict with:
      - ``feature_columns`` — full list of feature column names
      - ``n_lob_features`` — count of non-position LOB features
      - ``n_total_features`` — total observation dimension (LOB + position)
      - ``has_position`` — whether a runtime position feature is included
      - ``group_counts`` — mapping of group name → feature count (LOB features only)
    """
    obs_yaml = (
        _repo_root()
        / "src"
        / "configs"
        / "scenarios"
        / scenario_path
        / "observation.yaml"
    )
    if not obs_yaml.exists():
        return {}
    try:
        import yaml

        raw = yaml.safe_load(obs_yaml.read_text(encoding="utf-8"))
    except Exception:
        return {}

    feature_columns: list[str] = raw.get("env", {}).get("feature_columns", [])
    position_cols = [f for f in feature_columns if "position" in f.lower()]
    lob_cols = [f for f in feature_columns if "position" not in f.lower()]

    group_counts: dict[str, int] = {}
    for col in lob_cols:
        group = _classify_feature_group(col)
        group_counts[group] = group_counts.get(group, 0) + 1

    return {
        "feature_columns": feature_columns,
        "n_lob_features": len(lob_cols),
        "n_total_features": len(feature_columns),
        "has_position": len(position_cols) > 0,
        "group_counts": group_counts,
    }


def compute_mlp_parameter_count(
    input_dim: int,
    hidden_dims: list[int],
    output_dim: int = 1,
) -> dict[str, Any]:
    """Compute the trainable parameter count for a fully-connected MLP.

    Each Linear layer has (in_features + 1) * out_features parameters
    (weights + bias). Returns a dict with per-layer counts and a total.
    """
    layers = []
    in_dim = input_dim
    for h in hidden_dims:
        params = (in_dim + 1) * h
        layers.append({"in": in_dim, "out": h, "params": params})
        in_dim = h
    # Output layer
    params = (in_dim + 1) * output_dim
    layers.append({"in": in_dim, "out": output_dim, "params": params})
    total = sum(layer["params"] for layer in layers)
    return {"layers": layers, "total": total}


# ---------------------------------------------------------------------------
# Public repo root accessor
# ---------------------------------------------------------------------------


def repo_root() -> Path:
    """Return the repository root as an absolute Path."""
    return _repo_root()


def find_observation_sample(
    split: str = "test",
    symbol: str | None = None,
    artifact_uri: str | None = None,
) -> pd.DataFrame | None:
    """Load an observation-sample parquet saved by save_observation_sample_artifact.

    Search order:
    1. ``eval_results/evaluation_data/`` at the repo root — prefers symbol-qualified
       files (``{split}_{symbol}_observations_head_*.parquet``) when *symbol* is given,
       then falls back to plain ``{split}_observations_head_*.parquet``.
    2. The MLflow artifact dir pointed to by *artifact_uri* (same subdir pattern).
    3. ``thesis/qmd/results/observation_samples/`` — a small trimmed snapshot
       committed to the repo (see scripts/export_observation_sample_to_thesis.py)
       so a CI checkout, which never runs `evaluate`, still has real data to render.

    Returns the largest matching file as a DataFrame, or ``None`` if nothing is found.
    """

    def _find_in_dir(directory: Path) -> Path | None:
        if not directory.exists():
            return None
        candidates: list[Path] = []
        if symbol:
            candidates = list(
                directory.glob(f"{split}_{symbol}_observations_head_*.parquet")
            )
        if not candidates:
            candidates = list(directory.glob(f"{split}_observations_head_*.parquet"))
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.stat().st_size)

    eval_results_dir = _repo_root() / "eval_results" / "evaluation_data"
    path = _find_in_dir(eval_results_dir)

    if path is None and artifact_uri is not None:
        artifact_dir = _artifact_dir_from_uri(artifact_uri)
        if artifact_dir is not None:
            path = _find_in_dir(artifact_dir / "evaluation_data" / split)
            if path is None:
                path = _find_in_dir(artifact_dir / "evaluation_data")

    if path is None:
        snapshot_dir = (
            _repo_root() / "thesis" / "qmd" / "results" / "observation_samples"
        )
        path = _find_in_dir(snapshot_dir)

    if path is None:
        return None
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# H4 learning report loader
# ---------------------------------------------------------------------------


def load_h4_report(scenario_name: str) -> dict[str, Any]:
    """Load the H4 learning report JSON for a given scenario.

    Looks for ``thesis/qmd/results/<scenario_name>/latest_finished/h4_learning_report.json``.
    Returns an empty dict when the file is missing or unparseable.
    """
    path = (
        thesis_results_root()
        / scenario_name
        / "latest_finished"
        / "h4_learning_report.json"
    )
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception as exc:
            _log_fallback(f"reading h4_learning_report.json for {scenario_name!r}", exc)
    return {}


# ---------------------------------------------------------------------------
# Experiment specification row builder (appendix table)
# ---------------------------------------------------------------------------


def build_experiment_specification_rows(experiment_name: str) -> list[tuple[str, str]]:
    """Build (Component, Specification) rows for the appendix experiment spec table.

    Loads hyperparameters from the exported snapshot and assembles human-readable
    rows describing the full experimental configuration. Returns an empty list when
    no snapshot is available.
    """
    from thesis_tables import (
        fmt_loss_fn,
        fmt_network_dims,
        fmt_reward_type,
        fmt_scientific,
    )

    hp = load_experiment_hyperparams(experiment_name)
    if not hp:
        return []

    def _get(key: str, default: Any) -> Any:
        val = hp.get(key)
        return val if val is not None else default

    actor_dims = fmt_network_dims(_get("actor_hidden_dims", [128, 64]))
    value_dims = fmt_network_dims(_get("value_hidden_dims", [128, 64]))
    network_dims = (
        f"Actor and critic hidden layers {actor_dims}"
        if actor_dims == value_dims
        else f"Actor {actor_dims}; critic {value_dims}"
    )

    actor_lr = _get("actor_lr", 0.0001)
    value_lr = _get("value_lr", 0.0001)
    lr_str = (
        f"Actor and critic {actor_lr}"
        if actor_lr == value_lr
        else f"Actor {actor_lr}; critic {value_lr}"
    )

    _get("actor_weight_decay", 0.0)
    value_wd = _get("value_weight_decay", 2e-6)
    wd_str = f"Actor 0.0; critic {fmt_scientific(value_wd)}" if value_wd else "0.0"

    episode_len = _get("streaming_episode_length", 10_000)
    reward_type = _get("reward_type", "differential_sharpe")
    reward_eta = _get("reward_eta", 0.01)
    trading_fees = _get("trading_fees", 0.0)
    algorithm = _get("algorithm", "TD3")
    gamma = _get("gamma", 0.9)
    tau = _get("tau", 0.005)
    policy_delay = _get("policy_delay", 2)
    policy_noise = _get("policy_noise", 0.2)
    noise_clip = _get("noise_clip", 0.3)
    exploration = _get("exploration_noise_std", 0.3)
    max_steps = _get("max_steps", 3_000_000)
    init_rand = _get("init_rand_steps", 5_000)
    fpb = _get("frames_per_batch", 200)
    optim_steps = _get("optim_steps_per_batch", 5)
    sample_size = _get("sample_size", 128)
    buffer_size = _get("buffer_size", 100_000)
    loss_fn = _get("loss_function", "smooth_l1")

    return [
        (
            "Asset and data",
            "AAPL, MSFT, TSLA, META, AMZN, and AVGO; ten-level limit order book observations during regular U.S. trading hours",
        ),
        (
            "Training data",
            "18 pooled symbol-day files: AAPL, AMZN, AVGO, META, MSFT, and TSLA over February 25–27, 2026",
        ),
        (
            "Validation/test data",
            "March 2, 2026 symbol-day files, capped at 50,000 rows per split",
        ),
        (
            "Feature set",
            "Selected causal LOB microstructure set: book pressure L0, three-level OBI, order-count imbalance L0, microprice, microprice divergence, bid/ask slope, OFI, 50-event rolling OFI, 50-event signed trade flow, and runtime position",
        ),
        ("Environment backend", "Continuous single-asset trading environment"),
        ("Episode length", f"{episode_len:,} event-time steps (streaming)"),
        ("Action", "Target portfolio exposure in [-1, 1]"),
        ("Reward", f"{fmt_reward_type(reward_type)} with eta = {reward_eta}"),
        ("Transaction fee", f"{trading_fees:.3f} in the baseline setting"),
        ("Algorithm", algorithm),
        ("Network widths", network_dims),
        ("Learning rates", lr_str),
        ("Weight decay", wd_str),
        ("Discount factor", f"gamma = {gamma}"),
        ("Target update", f"tau = {tau}"),
        ("Policy delay", f"{policy_delay} critic updates per actor update"),
        ("Target policy noise", f"sigma = {policy_noise}, clipped at ±{noise_clip}"),
        ("Exploration noise", f"sigma = {exploration} during training"),
        ("Total training steps", f"{max_steps:,}"),
        ("Initial random steps", f"{init_rand:,}"),
        ("Frames per batch", f"{fpb:,}"),
        ("Optimisation steps per batch", str(optim_steps)),
        ("Mini-batch size", f"{sample_size:,}"),
        ("Replay buffer size", f"{buffer_size:,}"),
        ("Loss function", fmt_loss_fn(loss_fn)),
        ("Evaluation random seed", "42"),
        (
            "Compute environment",
            "Apple M3 MacBook (November 2023) with 18 GB unified memory",
        ),
    ]


def audit_plots_enabled() -> bool:
    """Return True when AUDIT_PLOTS=1/true/yes is set in the environment.

    Use at the top of a QMD setup chunk:
        audit = audit_plots_enabled()

    Enable at render time:
        AUDIT_PLOTS=1 uv run quarto render masters-thesis.qmd --to pdf
    """
    import os

    return os.environ.get("AUDIT_PLOTS", "").lower() in {"1", "true", "yes"}


def _figures_dir() -> Path:
    """Return (and create) the _figures/ subdir next to this file."""
    from pathlib import Path

    d = Path(__file__).parent / "_figures"
    d.mkdir(exist_ok=True)
    return d


def show_plot(
    plot: Any,
    data: dict[str, Any],
    frame: str = "rewards",
    *,
    width: float | None = None,
    height: float | None = None,
    audit: bool = False,
    fig_label: str | None = None,
    fig_cap: str | None = None,
    render_base_size: int | None = None,
) -> None:
    """Draw a plotnine plot and optionally print asset provenance.

    Two rendering modes depending on whether *fig_label* is provided:

    **Inline mode** (fig_label=None, default)
        Calls plot.draw() so Quarto captures the image inline.  The plot title,
        subtitle, and caption are always stripped from the PNG and emitted as
        adjacent Markdown paragraphs (LaTeX-typeset for PDF, rendered text for
        HTML), but they appear outside the figure environment.

    **Figure-env mode** (fig_label="fig-xxx")
        Saves the stripped PNG to _figures/<label>.png and emits a Quarto
        cross-reference div via Markdown so the title/caption land inside
        \\begin{figure}...\\end{figure} and @fig-xxx cross-references resolve.

        The calling cell MUST have ``#| output: asis`` for Quarto to treat the
        emitted Markdown as raw markup rather than literal text.

        Example cell header::

            ```{python}
            #| output: asis
            #| echo: false
            show_plot(p, data, fig_label="fig-rewards", fig_cap="Cumulative rewards.")
            ```

    Args:
        plot: plotnine ggplot object.
        data: dict returned by find_evaluation_plot_data.
        frame: which DataFrame key to look up provenance for ("rewards", "actions", "equity").
        width: figure width in inches. Overrides the plot's existing figure_size.
        height: figure height in inches.
        audit: print commit hash and generation datetime below the plot.
        fig_label: Quarto cross-reference label (e.g. "fig-rewards"). Enables
                   figure-env mode; requires ``#| output: asis`` on the chunk.
        fig_cap: explicit figure caption for the LaTeX figure environment.
                 Defaults to the plot's labs(caption=...) if omitted.
        render_base_size: if set, overrides the theme base font size just before
                 drawing — does not affect MLflow artifacts saved during training.
    """
    import matplotlib.pyplot as plt
    from IPython.display import Markdown, display
    from plotnine import element_text, theme

    if width is not None or height is not None:
        current = plot.theme.themeables.get("figure_size")
        cur_w, cur_h = current.properties["value"] if current else (8, 5)
        plot = plot + theme(figure_size=(width or cur_w, height or cur_h))

    if render_base_size is not None:
        plot = plot + theme(
            text=element_text(size=render_base_size),
            axis_title=element_text(size=render_base_size),
            axis_text=element_text(size=render_base_size - 1),
            legend_title=element_text(size=render_base_size - 1),
            legend_text=element_text(size=render_base_size - 1),
            plot_title=element_text(size=render_base_size + 1),
            plot_caption=element_text(size=round(render_base_size * 0.65)),
        )

    # Extract and strip text labels from the plot.
    # labels_view is a dataclass — use attribute access, not dict methods.
    title = plot.labels.title or ""
    subtitle = plot.labels.subtitle or ""
    caption = fig_cap or plot.labels.caption or ""
    plot.labels.title = None
    plot.labels.subtitle = None
    plot.labels.caption = None

    if fig_label:
        # --- Figure-env mode: save to file, emit Quarto cross-ref div ---
        path = _figures_dir() / f"{fig_label}.png"
        mpl_fig = plot.draw()
        mpl_fig.savefig(str(path), dpi=150, bbox_inches="tight")
        plt.close(mpl_fig)

        # Build the caption that goes inside \caption{}.
        parts: list[str] = []
        if title:
            t = f"**{title}**"
            if subtitle:
                t += f" — *{subtitle}*"
            parts.append(t)
        if caption:
            parts.append(caption)
        cap_md = "\n\n".join(parts)

        display(
            Markdown(
                f"::: {{#{fig_label}}}\n\n"
                f"![](_figures/{fig_label}.png)\n\n"
                f"{cap_md}\n\n"
                f":::"
            )
        )
    else:
        # --- Inline mode: title/caption always emitted as Markdown ---
        if title:
            display(Markdown(f"**{title}**\n"))
        if subtitle:
            display(Markdown(f"*{subtitle}*\n"))

        plot.draw()

        if caption:
            display(Markdown(f"\n*{caption}*"))

    if audit and data:
        meta = data.get("asset_meta", {}).get(frame, {})
        if meta:
            commit = meta.get("commit", "unknown")[:8]
            dt = meta.get("datetime", "unknown")
            generator = meta.get("generator", "")
            parts_a = [f"commit: {commit}", f"generated: {dt}"]
            if generator:
                parts_a.append(f"source: {generator}")
            print("  |  ".join(parts_a))


def show_table_meta(data: dict[str, Any], *, audit: bool = False) -> None:
    """Emit commit/run provenance below a table, mirroring show_plot's audit block.

    *data* can be either:
    - a *plot_data* dict (returned by find_evaluation_plot_data) — uses the
      ``asset_meta`` sidecar commit hash and datetime, or
    - a *finished* dict (from load_experiment_snapshot) — falls back to
      ``run_name`` and ``start_time``.

    Usage in QMD (same audit flag as show_plot)::

        show_table_meta(_plot_data, audit=_audit)
        # or
        show_table_meta(finished, audit=_audit)
    """
    if not audit:
        return
    if not data:
        # No snapshot for this experiment (e.g. results not exported yet). The
        # table itself may still render from a logs/ fallback; the provenance
        # footnote just has nothing to show.
        return
    from thesis_tables import table_note

    # Prefer sidecar metadata (has git commit hash)
    asset_meta = data.get("asset_meta", {})
    sidecar = next(iter(asset_meta.values()), None) if asset_meta else None

    if sidecar:
        commit = sidecar.get("commit", "unknown")[:8]
        dt = sidecar.get("datetime", "unknown")
        generator = sidecar.get("generator", "")
        parts: list[str] = [f"commit: {commit}", f"generated: {dt}"]
        if generator:
            parts.append(f"source: {generator}")
        table_note(note="  |  ".join(parts))
    else:
        # Fallback: render-time commit + export timestamp from finished dict
        try:
            from trading_rl.evaluation.asset_meta import _git_commit

            commit = _git_commit()[:8]
        except Exception:
            commit = "unknown"
        source = data.get("source") or {}
        exported_at = (
            source.get("exported_at_utc") or str(data.get("start_time", ""))[:19]
        )
        parts = [f"commit: {commit}"]
        if exported_at:
            parts.append(f"exported: {exported_at[:19]}")
        table_note(note="  |  ".join(parts))
