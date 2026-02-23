"""Utilities for loading thesis experiment outputs into Quarto chapters.

This module prefers exported thesis result snapshots (JSON / Parquet / PNG)
stored under ``thesis/qmd/results``. If no snapshot is available, it falls back
to read-only queries against the local MLflow SQLite backend and artifact store.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Any

import pandas as pd


def _repo_root() -> Path:
    # thesis/qmd/src/thesis_mlflow_results.py -> repo root is 3 levels up
    return Path(__file__).resolve().parents[3]


def mlflow_db_path() -> Path:
    return _repo_root() / "mlflow.db"


def thesis_results_root() -> Path:
    return _repo_root() / "thesis" / "qmd" / "results"


def _experiment_snapshot_dir(experiment_name: str, output_root: Path | None = None) -> Path:
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


def load_latest_evaluation_report(artifact_uri: str | None) -> dict[str, Any] | None:
    artifact_dir = _artifact_dir_from_uri(artifact_uri)
    if artifact_dir is None:
        return None
    candidates = sorted((artifact_dir / "evaluation_metrics").glob("*.json"))
    if not candidates:
        return None
    path = _latest_file(candidates)
    if path is None:
        return None
    return json.loads(path.read_text())


def find_evaluation_plots(artifact_uri: str | None) -> dict[str, Path]:
    """Return latest available evaluation plots for a run.

    Preference order:
    1. final evaluation_plots/*
    2. temporary evaluation_plots_temp/**/*
    """
    artifact_dir = _artifact_dir_from_uri(artifact_uri)
    if artifact_dir is None:
        return {}

    plot_keys = {
        "merged_comparison": "*_merged_comparison.png",
        "rewards": "*_rewards.png",
        "positions": "*_positions.png",
        "actual_returns": "*_actual_returns.png",
    }
    found: dict[str, Path] = {}

    final_dir = artifact_dir / "evaluation_plots"
    if final_dir.exists():
        for key, pattern in plot_keys.items():
            p = _latest_file(list(final_dir.glob(pattern)))
            if p is not None:
                found[key] = p

    if len(found) == len(plot_keys):
        return found

    # Fallback to temp periodic eval plots
    temp_dir = artifact_dir / "evaluation_plots_temp"
    if temp_dir.exists():
        for key, pattern in plot_keys.items():
            if key in found:
                continue
            p = _latest_file(list(temp_dir.glob(f"**/{pattern}")))
            if p is not None:
                found[key] = p

    return found


def latest_run_for_experiment(experiment_name: str, status: str | None = None) -> dict[str, Any] | None:
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
    row["evaluation_plots"] = find_evaluation_plots(row["artifact_uri"])
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
    pct_keys = {"total_return", "annualized_return_cagr", "annualized_volatility", "max_drawdown", "win_rate"}
    key_order = [
        ("total_return", "Total Return"),
        ("annualized_return_cagr", "CAGR"),
        ("annualized_volatility", "Annualized Volatility"),
        ("sharpe_ratio", "Sharpe Ratio"),
        ("sortino_ratio", "Sortino Ratio"),
        ("max_drawdown", "Max Drawdown"),
        ("win_rate", "Win Rate"),
        ("profit_factor", "Profit Factor"),
        ("var_95", "VaR (95%)"),
        ("cvar_95", "CVaR (95%)"),
    ]
    for key, label in key_order:
        if key not in report:
            continue
        val = report[key]
        if isinstance(val, (int, float)):
            if key in pct_keys:
                rows.append((label, f"{val*100:.2f}%"))
            else:
                rows.append((label, f"{val:.4f}"))
        else:
            rows.append((label, str(val)))
    return pd.DataFrame(rows, columns=["Metric", "Value"])


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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


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
    except Exception:
        pass
    if isinstance(value, (int, float, str, bool)):
        return value
    return str(value)


def _copy_plots_to_snapshot(plots: dict[str, Path], destination_dir: Path) -> dict[str, str]:
    copied: dict[str, str] = {}
    plot_dir = destination_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for key, src in plots.items():
        if not Path(src).exists():
            continue
        filename = f"{key}{Path(src).suffix.lower() or '.png'}"
        dst = plot_dir / filename
        shutil.copy2(src, dst)
        copied[key] = str(dst.relative_to(destination_dir))
    return copied


def _serialize_run_payload_for_export(run: dict[str, Any], destination_dir: Path) -> dict[str, Any]:
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
            str(k): _json_number_or_none(v)
            for k, v in latest_metrics_obj.items()
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
        },
        "evaluation_plots": plot_relpaths,
    }
    _write_json(destination_dir / "run.json", run_json)
    return run_json


def export_experiment_snapshot(experiment_name: str, output_root: Path | None = None) -> Path:
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
        runs_df.to_parquet(output_dir / "runs_overview.parquet", index=False)
        runs_df.assign(
            start_time=runs_df["start_time"].astype(str),
            end_time=runs_df["end_time"].astype(str),
        ).to_json(output_dir / "runs_overview.json", orient="records", indent=2)
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
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "type": "mlflow",
            "mlflow_db_path": str(mlflow_db_path()),
        },
        "files": {
            "runs_overview_parquet": "runs_overview.parquet",
            "runs_overview_json": "runs_overview.json",
        },
        "runs": {
            "latest_running": None if exported_runs["latest_running"] is None else "latest_running/run.json",
            "latest_finished": None if exported_runs["latest_finished"] is None else "latest_finished/run.json",
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
    loaded["evaluation_plots"] = evaluation_plots
    return loaded


def _load_experiment_snapshot_from_export(experiment_name: str) -> ExperimentSnapshot | None:
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


def load_experiment_snapshot(experiment_name: str) -> ExperimentSnapshot:
    exported = _load_experiment_snapshot_from_export(experiment_name)
    if exported is not None:
        return exported
    return _load_experiment_snapshot_from_mlflow(experiment_name)


def runs_overview_table(experiment_name: str) -> pd.DataFrame:
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
    return _runs_overview_table_from_mlflow(experiment_name)
