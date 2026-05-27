"""Artifacts command — list and delete MLflow artifacts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import typer
from rich.table import Table

from .base_command import BaseCommand


@dataclass
class ArtifactsParams:
    tracking_uri: str | None = None
    experiment: str | None = None
    run_id: str | None = None
    prefix: str | None = None
    delete: str | None = None
    delete_all: bool = False
    force: bool = False
    dry_run: bool = False
    max_runs: int = 50


class ArtifactsCommand(BaseCommand):
    """List MLflow artifacts grouped by experiment and run."""

    def execute(self, params: ArtifactsParams) -> None:
        import mlflow
        from mlflow.tracking import MlflowClient

        if params.tracking_uri:
            mlflow.set_tracking_uri(params.tracking_uri)

        client = MlflowClient()

        if params.run_id:
            self._handle_single_run(client, mlflow, params)
            return

        self._handle_all_experiments(client, mlflow, params)

    def _handle_single_run(self, client: Any, mlflow: Any, params: ArtifactsParams) -> None:
        run = client.get_run(params.run_id)
        experiment_obj = client.get_experiment(run.info.experiment_id)
        exp_name = experiment_obj.name if experiment_obj else run.info.experiment_id
        artifacts_list = self._list_run_artifacts(client, params.run_id, params.prefix)
        if params.delete_all or params.delete:
            pattern = re.compile(params.delete) if params.delete else None
            targets = [
                entry
                for entry in artifacts_list
                if params.delete_all or (pattern and pattern.search(entry.path))
            ]
            if not targets:
                self.console.print("[yellow]No artifacts matched for deletion.[/yellow]")
                raise typer.Exit(0)
            if params.dry_run:
                self.console.print("[yellow]Dry run: artifacts to delete[/yellow]")
                for entry in targets:
                    self.console.print(f"  {params.run_id}:{entry.path}")
                raise typer.Exit(0)
            if not self._confirm_delete([f"{params.run_id}:{e.path}" for e in targets], params.force):
                self.console.print("[yellow]Deletion cancelled.[/yellow]")
                raise typer.Exit(0)
            for entry in targets:
                self._delete_artifact_path(
                    client,
                    params.run_id,
                    entry.path,
                    getattr(run.info, "artifact_uri", None),
                )
            self.console.print(f"[green]Deleted {len(targets)} artifacts.[/green]")
            return
        self._print_run_artifacts(exp_name, run, artifacts_list)

    def _handle_all_experiments(self, client: Any, mlflow: Any, params: ArtifactsParams) -> None:
        experiments_list, file_store_fallback, mlruns_dir = self._safe_search_experiments(
            params.tracking_uri
        )
        exp_pattern = re.compile(params.experiment) if params.experiment else None
        targets = [
            exp for exp in experiments_list
            if not exp_pattern or exp_pattern.search(exp["name"])
        ]
        if not targets:
            self.console.print("[yellow]No experiments matched.[/yellow]")
            raise typer.Exit(0)

        delete_any = params.delete_all or params.delete
        for exp in sorted(targets, key=lambda e: e["name"]):
            delete_targets = []
            if file_store_fallback:
                runs = self._safe_list_runs_file_store(mlruns_dir, exp["experiment_id"])
                if not runs:
                    self.console.print(f"[yellow]No runs for experiment {exp['name']}.[/yellow]")
                    continue
                for run in runs[:params.max_runs]:
                    artifacts_list = self._list_artifacts_file_store(
                        run.get("artifact_uri"), params.prefix
                    )
                    if delete_any:
                        pattern = re.compile(params.delete) if params.delete else None
                        delete_targets.extend(
                            (run["run_id"], run.get("artifact_uri"), entry)
                            for entry in artifacts_list
                            if params.delete_all or (pattern and pattern.search(entry["path"]))
                        )
                    else:
                        title = f"Run: {run['run_id']}"
                        if run.get("run_name"):
                            title += f" ({run['run_name']})"
                        table = Table(title=f"Experiment: {exp['name']} | {title}")
                        table.add_column("Artifact")
                        table.add_column("Size", justify="right")
                        if not artifacts_list:
                            self.console.print(f"[yellow]No artifacts for run {run['run_id']}[/yellow]")
                        else:
                            for entry in sorted(artifacts_list, key=lambda e: e["path"]):
                                size = (
                                    f"{entry['file_size'] / 1024:.1f} KB"
                                    if entry["file_size"]
                                    else "-"
                                )
                                table.add_row(entry["path"], size)
                            self.console.print(table)
            else:
                runs = mlflow.search_runs(
                    experiment_ids=[exp["experiment_id"]],
                    max_results=params.max_runs,
                    order_by=["start_time DESC"],
                )
                if runs.empty:
                    self.console.print(f"[yellow]No runs for experiment {exp['name']}.[/yellow]")
                    continue
                for _, row in runs.iterrows():
                    run = client.get_run(row["run_id"])
                    artifacts_list = self._list_run_artifacts(client, run.info.run_id, params.prefix)
                    if delete_any:
                        pattern = re.compile(params.delete) if params.delete else None
                        delete_targets.extend(
                            (run.info.run_id, None, entry)
                            for entry in artifacts_list
                            if params.delete_all or (pattern and pattern.search(entry.path))
                        )
                    else:
                        self._print_run_artifacts(exp["name"], run, artifacts_list)
            if delete_any:
                if not delete_targets:
                    self.console.print("[yellow]No artifacts matched for deletion.[/yellow]")
                    raise typer.Exit(0)
                if params.dry_run:
                    self.console.print("[yellow]Dry run: artifacts to delete[/yellow]")
                    for run_id_val, _artifact_uri, entry in delete_targets:
                        path = entry["path"] if isinstance(entry, dict) else entry.path
                        self.console.print(f"  {run_id_val}:{path}")
                    raise typer.Exit(0)
                if not self._confirm_delete(
                    [
                        f"{rid}:{(e['path'] if isinstance(e, dict) else e.path)}"
                        for rid, _artifact_uri, e in delete_targets
                    ],
                    params.force,
                ):
                    self.console.print("[yellow]Deletion cancelled.[/yellow]")
                    raise typer.Exit(0)
                for run_id_val, artifact_uri, entry in delete_targets:
                    if not artifact_uri:
                        run_info = client.get_run(run_id_val)
                        artifact_uri = getattr(run_info.info, "artifact_uri", None)
                    if isinstance(entry, dict):
                        self._delete_artifact_path(client, run_id_val, entry["path"], artifact_uri)
                    else:
                        self._delete_artifact_path(client, run_id_val, entry.path, artifact_uri)
                self.console.print(f"[green]Deleted {len(delete_targets)} artifacts.[/green]")

    def _print_run_artifacts(self, exp_name: str, run: Any, artifacts_list: list) -> None:
        run_name = run.data.tags.get("mlflow.runName", "") if run else ""
        title = f"Run: {run.info.run_id}"
        if run_name:
            title += f" ({run_name})"
        table = Table(title=f"Experiment: {exp_name} | {title}")
        table.add_column("Artifact")
        table.add_column("Size", justify="right")
        if not artifacts_list:
            self.console.print(f"[yellow]No artifacts for run {run.info.run_id}[/yellow]")
            return
        for entry in sorted(artifacts_list, key=lambda e: e.path):
            size = f"{entry.file_size / 1024:.1f} KB" if entry.file_size else "-"
            table.add_row(entry.path, size)
        self.console.print(table)

    def _list_run_artifacts(self, client: Any, run_id: str, prefix: str | None = None) -> list:
        artifacts = []
        stack = [prefix or ""]
        while stack:
            path = stack.pop()
            for entry in client.list_artifacts(run_id, path):
                if entry.is_dir:
                    stack.append(entry.path)
                else:
                    artifacts.append(entry)
        return artifacts

    def _safe_search_experiments(
        self,
        tracking_uri: str | None = None,
    ) -> tuple[list[dict[str, str]], bool, Path | None]:
        import mlflow
        import yaml

        try:
            experiments_list = mlflow.search_experiments()
            return (
                [
                    {"experiment_id": exp.experiment_id, "name": exp.name}
                    for exp in experiments_list
                ],
                False,
                None,
            )
        except Exception as exc:  # pragma: no cover - fallback for malformed stores
            self.console.print(
                f"[yellow]Warning: failed to list experiments via MLflow ({exc}). "
                "Falling back to scanning the file store.[/yellow]"
            )

        uri = tracking_uri or mlflow.get_tracking_uri()
        parsed = urlparse(uri)
        if parsed.scheme in ("", "file"):
            base_path = Path(parsed.path or uri)
        else:
            self.console.print(
                "[red]Unable to recover experiments from non-file tracking URI.[/red]"
            )
            raise typer.Exit(1)

        mlruns_dir = base_path if base_path.name == "mlruns" else base_path / "mlruns"
        if not mlruns_dir.exists():
            self.console.print(f"[red]MLflow directory not found: {mlruns_dir}[/red]")
            raise typer.Exit(1)

        experiments = []
        for exp_dir in sorted(mlruns_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            meta_path = exp_dir / "meta.yaml"
            if not meta_path.exists():
                continue
            try:
                meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
                exp_id = meta.get("experiment_id")
                exp_name = meta.get("name")
                if exp_id and exp_name:
                    experiments.append({"experiment_id": exp_id, "name": exp_name})
            except Exception:
                continue
        return experiments, True, mlruns_dir

    def _safe_list_runs_file_store(
        self, mlruns_dir: Path, experiment_id: str
    ) -> list[dict[str, str]]:
        import yaml

        runs = []
        exp_dir = mlruns_dir / experiment_id
        if not exp_dir.exists():
            return runs
        for run_dir in exp_dir.iterdir():
            if not run_dir.is_dir():
                continue
            meta_path = run_dir / "meta.yaml"
            if not meta_path.exists():
                continue
            try:
                meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
                run_id = meta.get("run_id") or meta.get("run_uuid")
                if not run_id:
                    continue
                run_name = ""
                tags = meta.get("tags") or []
                for tag in tags:
                    if tag.get("key") == "mlflow.runName":
                        run_name = tag.get("value") or ""
                        break
                runs.append(
                    {
                        "run_id": run_id,
                        "run_name": run_name,
                        "artifact_uri": meta.get("artifact_uri"),
                    }
                )
            except Exception:
                continue
        return runs

    def _list_artifacts_file_store(
        self, artifact_uri: str | None, prefix: str | None
    ) -> list[dict[str, str | int]]:
        if not artifact_uri:
            return []
        parsed = urlparse(artifact_uri)
        if parsed.scheme not in ("", "file"):
            return []
        artifact_root = Path(parsed.path or artifact_uri)
        scan_root = artifact_root / prefix if prefix else artifact_root
        if not scan_root.exists():
            return []
        artifacts = []
        for path in scan_root.rglob("*"):
            if path.is_dir():
                continue
            rel_path = path.relative_to(artifact_root)
            artifacts.append({"path": str(rel_path), "file_size": path.stat().st_size})
        return artifacts

    def _delete_artifacts_file_store(
        self, artifact_uri: str | None, entries: list[dict[str, str | int]]
    ) -> None:
        if not artifact_uri:
            return
        parsed = urlparse(artifact_uri)
        if parsed.scheme not in ("", "file"):
            return
        base_path = Path(parsed.path or artifact_uri)
        for entry in entries:
            path = base_path / entry["path"]
            path.unlink(missing_ok=True)
        for path in sorted(base_path.rglob("*"), reverse=True):
            if path.is_dir():
                try:
                    path.rmdir()
                except OSError:
                    continue

    def _delete_artifact_path(
        self, client: Any, run_id: str, artifact_path: str, artifact_uri: str | None = None
    ) -> None:
        import mlflow

        if hasattr(client, "delete_artifacts"):
            client.delete_artifacts(run_id, artifact_path)
            return
        if hasattr(mlflow, "artifacts") and hasattr(mlflow.artifacts, "delete_artifacts"):
            mlflow.artifacts.delete_artifacts(run_id, artifact_path)
            return
        if artifact_uri:
            self._delete_artifacts_file_store(artifact_uri, [{"path": artifact_path}])
            return
        self.console.print(
            f"[yellow]Skipping delete for {run_id}:{artifact_path} (unsupported backend).[/yellow]"
        )

    def _confirm_delete(self, items: list[str], force: bool) -> bool:
        if force:
            return True
        self.console.print("[yellow]Delete the following items?[/yellow]")
        for item in items:
            self.console.print(f"  {item}")
        return typer.confirm("Proceed with deletion?", default=False)
