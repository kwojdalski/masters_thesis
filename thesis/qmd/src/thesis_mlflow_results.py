"""Utilities for loading MLflow experiment outputs into Quarto chapters.

This module is intentionally lightweight and read-only. It reads run metadata
from the local MLflow SQLite backend and loads selected artifacts (JSON reports
and evaluation plot images) from the artifact store.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


def _repo_root() -> Path:
    # thesis/qmd/src/thesis_mlflow_results.py -> repo root is 3 levels up
    return Path(__file__).resolve().parents[3]


def mlflow_db_path() -> Path:
    return _repo_root() / "mlflow.db"


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


def runs_overview_table(experiment_name: str) -> pd.DataFrame:
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


def load_experiment_snapshot(experiment_name: str) -> ExperimentSnapshot:
    return ExperimentSnapshot(
        experiment_name=experiment_name,
        latest_running=latest_run_for_experiment(experiment_name, status="RUNNING"),
        latest_finished=latest_run_for_experiment(experiment_name, status="FINISHED"),
    )
