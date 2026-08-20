"""MCP server exposing this project's CLI commands as tools.

Wraps the same command classes used by `cli.py` (see `cli/commands/`) so an
agent can drive training/evaluation/data pipelines through MCP instead of
shelling out to the CLI. Long-running commands (train, evaluate, experiment,
feature-research, collect-results) run as background jobs started by a
`*_start` tool and polled with `job_status` / `job_logs`, since MCP clients
typically enforce call timeouts that a multi-hour training run would blow
past.

`dashboard` is intentionally NOT exposed here: it launches `mlflow ui` via
`subprocess.run` with inherited stdio, which would write directly to this
process's stdout file descriptor and corrupt the MCP stdio protocol stream.
Run `uv run python src/cli.py dashboard` directly instead.

Any output this process sends to stdout other than MCP protocol frames
corrupts the stdio transport, so logging is redirected to a file below and
every CLI command is invoked with its own captured `rich.Console` rather than
the default stdout-backed one.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

from cli.commands import (
    ArtifactsCommand,
    ArtifactsParams,
    CheckpointsCommand,
    CheckpointsParams,
    CollectResultsCommand,
    CollectResultsParams,
    DataGenerationParams,
    DataGeneratorCommand,
    EvaluateCommand,
    EvaluateParams,
    ExperimentCommand,
    ExperimentParams,
    ExperimentsCommand,
    ExperimentsParams,
    FeatureResearchCommand,
    FeatureResearchParams,
    PeekCommand,
    PeekParams,
    ScenariosCommand,
    ScenariosParams,
    SineWaveParams,
    TrainingCommand,
    TrainingParams,
    UpwardDriftParams,
    ValidateDataCommand,
    ValidateDataParams,
    ValidationCommand,
    ValidationParams,
)
from logger import setup_logging as _setup_root_logging

from .jobs import JOBS
from .support import new_capture_console, require_force_for_delete, run_command

# Ensure matplotlib can cache fonts to a writable directory (mirrors cli.py).
if "MPLCONFIGDIR" not in os.environ:
    mpl_cache_dir = Path(".cache/matplotlib")
    mpl_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_cache_dir.resolve())

mcp_server = MCPServer(
    "trading-rl",
    instructions=(
        "Tools for running and inspecting this project's trading-RL CLI: "
        "training, evaluation, multi-trial experiments, feature research, "
        "data generation/validation, and MLflow/checkpoint housekeeping. "
        "Long-running commands use a *_start tool that returns a job_id; "
        "poll it with job_status/job_logs."
    ),
)


def _run_sync(build: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
    console, buffer = new_capture_console()
    return run_command(console, buffer, build, *args, **kwargs)


# ---------------------------------------------------------------------------
# Background jobs: train / evaluate / experiment / feature-research / collect
# ---------------------------------------------------------------------------


@mcp_server.tool()
def train_start(
    scenario: str | None = None,
    config_file: str | None = None,
    experiment_name: str | None = None,
    config_overrides: list[str] | None = None,
    seed: int | None = None,
    max_steps: int | None = None,
    from_checkpoint: str | None = None,
    from_last_checkpoint: bool = False,
    mlflow_run_id: str | None = None,
    additional_steps: int | None = None,
) -> dict[str, Any]:
    """Start a single training run in the background; poll with job_status/job_logs."""
    params = TrainingParams(
        experiment_name=experiment_name,
        config_file=Path(config_file) if config_file else None,
        scenario=scenario,
        config_overrides=config_overrides,
        seed=seed,
        max_steps=max_steps,
        additional_steps=additional_steps,
        from_checkpoint=Path(from_checkpoint) if from_checkpoint else None,
        from_last_checkpoint=from_last_checkpoint,
        mlflow_run_id=mlflow_run_id,
        interactive=False,
    )
    job_id = JOBS.start(
        "train", lambda c, b: run_command(c, b, TrainingCommand, params)
    )
    return {"job_id": job_id, "status": "started", "kind": "train"}


@mcp_server.tool()
def evaluate_start(
    config: str,
    checkpoint: str | None = None,
    split: str = "all",
    only: list[str] | None = None,
    output_dir: str = "./eval_results",
    config_overrides: list[str] | None = None,
    tracking_uri: str = "sqlite:///mlflow.db",
    no_mlflow: bool = False,
    data_path: str | None = None,
    save_rollout: bool = False,
    save_trades: bool = False,
    per_symbol: bool = False,
) -> dict[str, Any]:
    """Start evaluating a trained checkpoint in the background; poll with job_status/job_logs.

    `config` accepts a scenario name (e.g. "sine_wave/ppo_no_trend") or a config file path.
    """
    params = EvaluateParams(
        config_file=Path(config),
        checkpoint=Path(checkpoint) if checkpoint else None,
        split=split,
        only=only,
        output_dir=Path(output_dir),
        config_overrides=config_overrides,
        tracking_uri=tracking_uri,
        no_mlflow=no_mlflow,
        data_path=Path(data_path) if data_path else None,
        save_rollout=save_rollout,
        save_trades=save_trades,
        per_symbol=per_symbol,
    )
    job_id = JOBS.start(
        "evaluate", lambda c, b: run_command(c, b, EvaluateCommand, params)
    )
    return {"job_id": job_id, "status": "started", "kind": "evaluate"}


@mcp_server.tool()
def experiment_start(
    scenario: str | None = None,
    config_file: str | None = None,
    experiment_name: str | None = None,
    n_trials: int = 5,
    config_overrides: list[str] | None = None,
    clear_cache: bool = False,
) -> dict[str, Any]:
    """Start a multi-trial experiment (repeated training runs, MLflow-tracked) in the background."""
    params = ExperimentParams(
        experiment_name=experiment_name,
        n_trials=n_trials,
        config_file=Path(config_file) if config_file else None,
        scenario=scenario,
        config_overrides=config_overrides,
        clear_cache=clear_cache,
    )
    job_id = JOBS.start(
        "experiment", lambda c, b: run_command(c, b, ExperimentCommand, params)
    )
    return {"job_id": job_id, "status": "started", "kind": "experiment"}


@mcp_server.tool()
def feature_research_start(
    scenario: str | None = None,
    config_file: str | None = None,
    experiment_config_file: str | None = None,
    config_overrides: list[str] | None = None,
) -> dict[str, Any]:
    """Start offline feature research (IC/ICIR scoring, feature selection) in the background."""
    params = FeatureResearchParams(
        config_file=Path(config_file) if config_file else None,
        experiment_config_file=(
            Path(experiment_config_file) if experiment_config_file else None
        ),
        scenario=scenario,
        config_overrides=config_overrides,
    )
    job_id = JOBS.start(
        "feature_research",
        lambda c, b: run_command(c, b, FeatureResearchCommand, params),
    )
    return {"job_id": job_id, "status": "started", "kind": "feature_research"}


@mcp_server.tool()
def collect_results_start(
    algorithms: list[str],
    dirs: list[str],
    output_dir: str = "masters_thesis_results",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Merge per-algorithm evaluation results into a unified thesis results directory."""
    params = CollectResultsParams(
        algorithms=algorithms,
        dirs=dirs,
        output_dir=Path(output_dir),
        overwrite=overwrite,
    )
    job_id = JOBS.start(
        "collect_results",
        lambda c, b: run_command(c, b, CollectResultsCommand, params),
    )
    return {"job_id": job_id, "status": "started", "kind": "collect_results"}


@mcp_server.tool()
def job_status(job_id: str) -> dict[str, Any]:
    """Get the status/result summary of a background job started by a *_start tool."""
    job = JOBS.get(job_id)
    if job is None:
        return {"ok": False, "error": f"Unknown job_id: {job_id}"}
    return job.summary()


@mcp_server.tool()
def job_logs(job_id: str) -> dict[str, Any]:
    """Get the captured console output of a background job (partial while it's still running)."""
    job = JOBS.get(job_id)
    if job is None:
        return {"ok": False, "error": f"Unknown job_id: {job_id}"}
    return {"job_id": job_id, "status": job.status.value, "logs": job.logs()}


@mcp_server.tool()
def job_list() -> list[dict[str, Any]]:
    """List all background jobs started in this server session, newest first."""
    return [job.summary() for job in JOBS.list()]


# ---------------------------------------------------------------------------
# Synchronous tools: fast enough to return within a single tool call
# ---------------------------------------------------------------------------


@mcp_server.tool()
def checkpoints(
    log_dir: str = "logs",
    delete: str | None = None,
    delete_all: bool = False,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """List training checkpoints, or delete ones matching a regex (deletion requires force=True)."""
    require_force_for_delete(delete, delete_all, force, dry_run)
    params = CheckpointsParams(
        log_dir=Path(log_dir),
        delete=delete,
        delete_all=delete_all,
        force=force,
        dry_run=dry_run,
    )
    return _run_sync(CheckpointsCommand, params)


@mcp_server.tool()
def scenarios(
    delete: str | None = None,
    delete_all: bool = False,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """List scenario config files under src/configs/scenarios, or delete matches (requires force=True)."""
    require_force_for_delete(delete, delete_all, force, dry_run)
    params = ScenariosParams(
        delete=delete, delete_all=delete_all, force=force, dry_run=dry_run
    )
    return _run_sync(ScenariosCommand, params)


@mcp_server.tool()
def artifacts(
    tracking_uri: str | None = None,
    experiment: str | None = None,
    run_id: str | None = None,
    prefix: str | None = None,
    delete: str | None = None,
    delete_all: bool = False,
    force: bool = False,
    dry_run: bool = False,
    max_runs: int = 50,
) -> dict[str, Any]:
    """List or delete MLflow run artifacts (deletion requires force=True)."""
    require_force_for_delete(delete, delete_all, force, dry_run)
    params = ArtifactsParams(
        tracking_uri=tracking_uri,
        experiment=experiment,
        run_id=run_id,
        prefix=prefix,
        delete=delete,
        delete_all=delete_all,
        force=force,
        dry_run=dry_run,
        max_runs=max_runs,
    )
    return _run_sync(ArtifactsCommand, params)


@mcp_server.tool()
def experiments(
    tracking_uri: str | None = None,
    delete: str | None = None,
    delete_all: bool = False,
    purge: bool = False,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """List or delete MLflow experiments (deletion requires force=True)."""
    require_force_for_delete(delete, delete_all, force, dry_run)
    params = ExperimentsParams(
        tracking_uri=tracking_uri,
        delete=delete,
        delete_all=delete_all,
        purge=purge,
        force=force,
        dry_run=dry_run,
    )
    return _run_sync(ExperimentsCommand, params)


@mcp_server.tool()
def validate_config(
    config_file: str | None = None,
    scenario: str | None = None,
    config_overrides: list[str] | None = None,
) -> dict[str, Any]:
    """Validate an experiment config (by scenario name or file path) and report issues."""
    params = ValidationParams(
        config_file=Path(config_file) if config_file else None,
        scenario=scenario,
        config_overrides=config_overrides,
    )
    return _run_sync(ValidationCommand, params)


@mcp_server.tool()
def validate_data(
    scenario: str | None = None,
    config_file: str | None = None,
    config_override: list[str] | None = None,
    check_nan: bool = True,
    check_inf: bool = True,
    check_duplicates: bool = True,
    check_zero_variance: bool = True,
    check_lob_deltas: bool = True,
    check_temporal_order: bool = True,
    check_overlap: bool = True,
    check_sizes: bool = True,
    lob_levels: int = 5,
    verbose: bool = False,
    transpose: bool = False,
) -> dict[str, Any]:
    """Build the prepared dataset for a scenario/config and run data quality checks on it."""
    params = ValidateDataParams(
        scenario=scenario,
        config_file=Path(config_file) if config_file else None,
        config_override=config_override,
        check_nan=check_nan,
        check_inf=check_inf,
        check_duplicates=check_duplicates,
        check_zero_variance=check_zero_variance,
        check_lob_deltas=check_lob_deltas,
        check_temporal_order=check_temporal_order,
        check_overlap=check_overlap,
        check_sizes=check_sizes,
        lob_levels=lob_levels,
        verbose=verbose,
        transpose=transpose,
    )
    return _run_sync(ValidateDataCommand, params)


@mcp_server.tool()
def peek(
    scenario: str | None = None,
    config_file: str | None = None,
    config_override: list[str] | None = None,
    n_features: int = 20,
    skip_rows: int = 0,
    show_correlations: bool = False,
    export: bool = False,
) -> dict[str, Any]:
    """Preview the prepared dataset's features for a scenario/config."""
    params = PeekParams(
        scenario=scenario,
        config_file=Path(config_file) if config_file else None,
        config_override=config_override,
        n_features=n_features,
        skip_rows=skip_rows,
        show_correlations=show_correlations,
        export=export,
    )
    return _run_sync(PeekCommand, params)


@mcp_server.tool()
def generate_data(
    scenario: str | None = None,
    source_dir: str | None = None,
    output_dir: str | None = None,
    source_file: str | None = None,
    output_file: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    sample_size: int | None = None,
    copy: bool = False,
    list_files: bool = False,
    sine_wave: bool = False,
    sine_n_periods: int | None = None,
    sine_samples_per_period: int | None = None,
    sine_base_price: float | None = None,
    sine_amplitude: float | None = None,
    sine_trend_slope: float | None = None,
    sine_volatility: float | None = None,
    upward_drift: bool = False,
    drift_samples: int | None = None,
    drift_rate: float | None = None,
    drift_volatility: float | None = None,
    drift_floor: float | None = None,
) -> dict[str, Any]:
    """Generate or copy synthetic price data, or list available generated files."""
    params = DataGenerationParams(
        scenario=scenario,
        source_dir=source_dir,
        output_dir=output_dir,
        source_file=source_file,
        output_file=output_file,
        start_date=start_date,
        end_date=end_date,
        sample_size=sample_size,
        copy=copy,
        list_files=list_files,
    )
    sine_params = SineWaveParams(
        enabled=sine_wave,
        n_periods=sine_n_periods,
        samples_per_period=sine_samples_per_period,
        base_price=sine_base_price,
        amplitude=sine_amplitude,
        trend_slope=sine_trend_slope,
        volatility=sine_volatility,
    )
    drift_params = UpwardDriftParams(
        enabled=upward_drift,
        drift_samples=drift_samples,
        drift_rate=drift_rate,
        drift_volatility=drift_volatility,
        drift_floor=drift_floor,
    )
    return _run_sync(
        DataGeneratorCommand, params, sine_params, drift_params, start_date
    )


def main() -> None:
    """Entrypoint for the `trading-rl-mcp` console script."""
    _setup_root_logging(
        level="INFO", console_output=False, log_file="logs/mcp_server.log"
    )
    mcp_server.run()


if __name__ == "__main__":
    main()
