#!/usr/bin/env python3
"""
Refactored command-line interface using command classes.
"""

import os
from datetime import UTC, datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import ProgressColumn
from rich.table import Table
from rich.text import Text

# Command classes and `trading_rl` are imported lazily inside each callback, so
# a lightweight subcommand (dashboard, checkpoints, scenarios, ...) does not load
# the training/evaluation stack (torch, torchrl, the statistical-test registry).
from logger import setup_logging as _setup_root_logging

# Mirror of trading_rl.config.EXPERIMENT_OUTPUT_DIR, inlined to avoid importing
# the trading_rl package (and its heavy transitive deps) at CLI startup.
_EXPERIMENT_OUTPUT_DIR = Path(os.environ.get("EXPERIMENT_OUTPUT_DIR", "logs"))

# Ensure matplotlib can cache fonts to a writable directory
if "MPLCONFIGDIR" in os.environ:
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
else:
    mpl_cache_dir = Path(".cache/matplotlib")
    mpl_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_cache_dir.resolve())

# Configure project logging for the CLI component before creating loggers
app = typer.Typer(
    help="CLI tools for trading data science project",
    add_completion=True,
)
data_app = typer.Typer(help="Data generation utilities")
app.add_typer(data_app, name="data")

validate_app = typer.Typer(help="Validation utilities")
app.add_typer(validate_app, name="validate")


class _HHMMSSElapsedColumn(ProgressColumn):
    """Time-elapsed column that always zero-pads hours (HH:MM:SS, not H:MM:SS)."""

    def render(self, task) -> Text:
        elapsed = task.finished_time if task.finished else task.elapsed
        if elapsed is None:
            return Text("--:--:--", style="progress.elapsed")
        hours, remainder = divmod(max(0, int(elapsed)), 3600)
        minutes, seconds = divmod(remainder, 60)
        return Text(
            f"{hours:02d}:{minutes:02d}:{seconds:02d}", style="progress.elapsed"
        )


def _configure_logging(verbose: bool, log_regex: str | None) -> None:
    """Configure logging based on CLI context."""
    env_level = os.environ.get("LOG_LEVEL", "").upper()
    if env_level not in {"TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        env_level = ""
    if verbose:
        level = "DEBUG"
    elif env_level:
        level = env_level
    else:
        level = "INFO"
    if log_regex:
        os.environ["LOG_REGEX"] = log_regex
    else:
        os.environ.pop("LOG_REGEX", None)
    _setup_root_logging(
        level=level,
        log_regex=log_regex,
    )
    os.environ["LOG_LEVEL"] = level


# Add global options
@app.callback()
def main(
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
    log_regex: str | None = typer.Option(
        None,
        "--log-regex",
        help="Only show log lines that match this regex",
    ),
):
    """Global CLI options."""
    _configure_logging(verbose, log_regex)


console = Console()


@app.command(name="checkpoints")
def checkpoints(
    log_dir: Path = typer.Option(  # noqa: B008
        _EXPERIMENT_OUTPUT_DIR,
        "--log-dir",
        help="Root directory to scan for checkpoints",
    ),
    delete: str | None = typer.Option(
        None, "--delete", help="Delete checkpoints matching regex"
    ),
    delete_all: bool = typer.Option(
        False, "--delete-all", help="Delete all checkpoints"
    ),
    force: bool = typer.Option(False, "--force", help="Delete without confirmation"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be deleted"),
):
    """List checkpoints grouped by experiment with size, mtime, and step."""
    from cli.commands import CheckpointsCommand, CheckpointsParams

    CheckpointsCommand(console).execute(
        CheckpointsParams(
            log_dir=log_dir,
            delete=delete,
            delete_all=delete_all,
            force=force,
            dry_run=dry_run,
        )
    )


@app.command(name="ps")
def ps():
    """List live trainer processes started with -o training.ipc_enabled=true."""
    from cli.commands import PsCommand, PsParams

    PsCommand(console).execute(PsParams())


@app.command(name="attach")
def attach(
    run_id: str = typer.Argument(..., help="run_id from `ps` (prefix accepted)"),
    watch: bool = typer.Option(
        False, "--watch", help="Live-refresh instead of printing once"
    ),
    interval: float = typer.Option(
        1.0, "--interval", help="Seconds between refreshes with --watch"
    ),
    path: str | None = typer.Option(
        None, "--path", help="Dotted attribute path to fetch instead of the status view"
    ),
):
    """Inspect one live trainer's status view (or a specific attribute with --path)."""
    from cli.commands import AttachCommand, AttachParams

    AttachCommand(console).execute(
        AttachParams(run_id=run_id, watch=watch, interval=interval, path=path)
    )


@data_app.command(name="generate")
def generate_data(
    scenario: str | None = typer.Option(
        None,
        "--scenario",
        "-s",
        help="Scenario config name (e.g., 'sine_wave') or path to YAML file defining defaults",
        show_default=False,
    ),
    source_dir: str | None = typer.Option(
        None,
        "--source-dir",
        help="Source directory containing parquet files",
        show_default=False,
    ),
    output_dir: str | None = typer.Option(
        None,
        "--output-dir",
        help="Output directory for synthetic data",
        show_default=False,
    ),
    list_files: bool = typer.Option(
        False, "--list", help="List available source files"
    ),
    source_file: str | None = typer.Option(
        None, "--source-file", help="Source parquet file name"
    ),
    output_file: str | None = typer.Option(
        None, "--output-file", help="Output file name"
    ),
    start_date: str | None = typer.Option(
        None, "--start-date", help="Start date for filtering (YYYY-MM-DD)"
    ),
    end_date: str | None = typer.Option(
        None, "--end-date", help="End date for filtering (YYYY-MM-DD)"
    ),
    sample_size: int | None = typer.Option(
        None, "--sample-size", help="Number of rows to sample randomly"
    ),
    copy: bool = typer.Option(
        False, "--copy", help="Copy source file without modifications"
    ),
    sine_wave: bool = typer.Option(
        False, "--sine-wave", help="Generate sine wave pattern with trend"
    ),
    n_periods: int | None = typer.Option(
        None, "--n-periods", help="Number of sine wave periods", show_default=False
    ),
    samples_per_period: int | None = typer.Option(
        None,
        "--samples-per_period",
        help="Samples per sine wave period",
        show_default=False,
    ),
    base_price: float | None = typer.Option(
        None, "--base-price", help="Base price level", show_default=False
    ),
    amplitude: float | None = typer.Option(
        None, "--amplitude", help="Sine wave amplitude", show_default=False
    ),
    trend_slope: float | None = typer.Option(
        None, "--trend-slope", help="Linear trend slope per step", show_default=False
    ),
    volatility: float | None = typer.Option(
        None, "--volatility", help="Random noise factor", show_default=False
    ),
    upward_drift: bool = typer.Option(
        False, "--upward-drift", help="Generate upward drift pattern"
    ),
    drift_samples: int | None = typer.Option(
        None,
        "--drift-samples",
        help="Number of samples for drift pattern",
        show_default=False,
    ),
    drift_rate: float | None = typer.Option(
        None, "--drift-rate", help="Exponential drift rate per step", show_default=False
    ),
    drift_volatility: float | None = typer.Option(
        None,
        "--drift-volatility",
        help="Volatility factor for drift pattern",
        show_default=False,
    ),
    drift_floor: float | None = typer.Option(
        None,
        "--drift-floor",
        help="Pullback floor multiplier for drift pattern",
        show_default=False,
    ),
):
    """Generate synthetic price data from existing parquet files."""
    from cli.commands import (
        DataGenerationParams,
        DataGeneratorCommand,
        SineWaveParams,
        UpwardDriftParams,
    )

    # Create parameter objects
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

    sine_wave_params = SineWaveParams(
        enabled=sine_wave,
        n_periods=n_periods,
        samples_per_period=samples_per_period,
        base_price=base_price,
        amplitude=amplitude,
        trend_slope=trend_slope,
        volatility=volatility,
    )

    upward_drift_params = UpwardDriftParams(
        enabled=upward_drift,
        drift_samples=drift_samples,
        drift_rate=drift_rate,
        drift_volatility=drift_volatility,
        drift_floor=drift_floor,
    )

    # Execute command
    DataGeneratorCommand(console).execute(
        params, sine_wave_params, upward_drift_params, start_date
    )


@app.command()
def train(
    trials: int = typer.Option(
        1, "--trials", "-t", help="Number of trials to run (1 = single run)"
    ),
    experiment_name: str | None = typer.Option(
        None,
        "--name",
        "-n",
        help="MLflow experiment name (defaults to config's experiment_name)",
    ),
    config_file: Path | None = typer.Option(  # noqa: B008
        None, "--config", "-c", help="Path to custom config file"
    ),
    config_override: list[str] | None = typer.Option(  # noqa: B008
        None,
        "--config-override",
        "-o",
        help="OmegaConf override in dotlist format (repeatable)",
    ),
    clear_cache: bool = typer.Option(
        False,
        "--clear-cache",
        help="Clear cached datasets and models before running experiments",
    ),
    from_checkpoint: Path | None = typer.Option(  # noqa: B008
        None,
        "--from-checkpoint",
        help="Path to checkpoint file to resume training from",
    ),
    from_last_checkpoint: bool = typer.Option(
        False,
        "--from-last-checkpoint",
        help="Resume from the most recent checkpoint for the experiment",
    ),
    mlflow_run_id: str | None = typer.Option(
        None,
        "--mlflow-run-id",
        help="Resume training into an existing MLflow run ID",
    ),
    additional_steps: int | None = typer.Option(
        None,
        "--additional-steps",
        help="Additional steps to train when resuming (requires --checkpoint)",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
    log_regex: str | None = typer.Option(
        None,
        "--log-regex",
        help="Only show log lines that match this regex",
    ),
    interactive: bool = typer.Option(
        False, "--interactive", "-i", help="Ask setup questions before training starts"
    ),
):
    """Train trading agents (single run or multiple trials).

    Single run is the default. Use --trials > 1 to run multiple trials.
    You can resume single-run training from a checkpoint by providing
    --from-checkpoint flag.

    Examples:
        # Basic training
        python src/cli.py train

        # With custom config (full path or group/name shorthand)
        python src/cli.py train --config src/configs/scenarios/sine_wave/ppo_no_trend.yaml
        python src/cli.py train --config sine_wave/ppo_no_trend

        # Override config parameters
        python src/cli.py train -c sine_wave/ppo_no_trend -o training.max_steps=10000

        # Custom experiment name
        python src/cli.py train -c sine_wave/ppo_no_trend --name my_experiment

        # Multiple trials
        python src/cli.py train -c sine_wave/ppo_no_trend --trials 5

        # Resume from last checkpoint
        python src/cli.py train -c sine_wave/ppo_no_trend --from-last-checkpoint --additional-steps 5000

        # Resume from specific checkpoint
        python src/cli.py train --from-checkpoint logs/my_exp/my_exp_checkpoint_step_1000.pt --additional-steps 10000

        # Verbose logging
        python src/cli.py train --from-last-checkpoint --additional-steps 5000 --verbose

        # Interactive setup (guided prompts)
        python src/cli.py train --interactive
        python src/cli.py train -c sine_wave/ppo_no_trend --interactive
    """

    if verbose or log_regex:
        _configure_logging(verbose, log_regex)

    if trials > 1:
        if from_checkpoint or from_last_checkpoint or mlflow_run_id or additional_steps:
            raise typer.BadParameter(
                "Checkpoint resume options are only supported for single-run training."
            )
        from cli.commands import ExperimentCommand, ExperimentParams

        params = ExperimentParams(
            experiment_name=experiment_name,
            n_trials=trials,
            config_file=config_file,
            clear_cache=clear_cache,
            config_overrides=config_override,
        )
        ExperimentCommand(console).execute(params)
        return

    from cli.commands import TrainingCommand, TrainingParams

    params = TrainingParams(
        config_file=config_file,
        config_overrides=config_override,
        experiment_name=experiment_name,
        from_checkpoint=from_checkpoint,
        from_last_checkpoint=from_last_checkpoint,
        mlflow_run_id=mlflow_run_id,
        additional_steps=additional_steps,
        interactive=interactive,
    )

    TrainingCommand(console).execute(params)


@app.command()
def evaluate(
    config_file: Path | None = typer.Option(  # noqa: B008
        None,
        "--config",
        "-c",
        help="Path or scenario shorthand for the experiment config YAML",
    ),
    checkpoint: Path | None = typer.Option(  # noqa: B008
        None,
        "--checkpoint",
        help=(
            "Path to a .pt checkpoint file. "
            "When omitted, the most recent checkpoint for the experiment is used."
        ),
    ),
    split: str = typer.Option(
        "all",
        "--split",
        "-s",
        help="Data split(s) to evaluate: train, val, test, or all",
    ),
    only: list[str] | None = typer.Option(  # noqa: B008
        None,
        "--only",
        help=(
            "Components to run (repeatable): metrics, benchmarks, plots, stats. "
            "Default: all components."
        ),
    ),
    output_dir: Path = typer.Option(  # noqa: B008
        Path("./eval_results"),
        "--output-dir",
        help="Directory to write results.json and plot PNGs",
    ),
    config_override: list[str] | None = typer.Option(  # noqa: B008
        None,
        "--config-override",
        "-o",
        help="OmegaConf dotlist override (repeatable)",
    ),
    tracking_uri: str = typer.Option(
        "sqlite:///mlflow.db",
        "--tracking-uri",
        help="MLflow tracking URI for logging results",
    ),
    no_mlflow: bool = typer.Option(
        False,
        "--no-mlflow",
        help="Skip MLflow logging entirely (results only written to --output-dir)",
    ),
    data_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--data-path",
        help=(
            "Path to an arbitrary raw parquet file to evaluate on. "
            "When provided, --split is ignored and the feature pipeline from "
            "the scenario config is applied to this file before evaluation."
        ),
    ),
    save_rollout: bool = typer.Option(
        False,
        "--save-rollout",
        help=(
            "Save per-step rollout data (action, simple_return, cumulative_log_return) "
            "to <output-dir>/<split>_rollout.csv for offline plotting or analysis."
        ),
    ),
    save_trades: bool = typer.Option(
        False,
        "--save-trades",
        help=(
            "Export the broker trade log (one row per trade) to "
            "<output-dir>/<split>_trades.csv. Useful for debugging execution quality."
        ),
    ),
    per_symbol: bool = typer.Option(
        False,
        "--per-symbol",
        help=(
            "For multi-symbol scenarios: evaluate each symbol independently and write "
            "artifacts to <output-dir>/per_symbol/<SYMBOL>/. "
            "Without this flag, val/test evaluation is skipped for multi-symbol configs."
        ),
    ),
):
    """Evaluate a trained policy from a checkpoint without re-running training.

    Loads the experiment config to locate data and build the evaluation
    environment, then loads the actor from a checkpoint (auto-discovered
    from the config's log directory when --checkpoint is omitted).

    Results are written to --output-dir as results.json + plot PNGs and are
    also logged to MLflow under the same experiment as the training run.
    Use --no-mlflow to skip MLflow logging.

    Examples:
        # Evaluate the latest checkpoint on val and test splits (default)
        python src/cli.py evaluate --config sine_wave/ppo_no_trend

        # Evaluate only one split
        python src/cli.py evaluate -c sine_wave/ppo_no_trend --split test

        # Only compute metrics and plots, skip benchmarks and stats
        python src/cli.py evaluate -c sine_wave/ppo_no_trend --only metrics --only plots

        # Use a specific checkpoint
        python src/cli.py evaluate -c sine_wave/ppo_no_trend \\
            --checkpoint logs/my_exp/my_exp_checkpoint_step_5000.pt

        # Skip MLflow, write results locally only
        python src/cli.py evaluate -c sine_wave/ppo_no_trend --no-mlflow

        # Use a remote MLflow server
        python src/cli.py evaluate -c sine_wave/ppo_no_trend \\
            --tracking-uri http://localhost:5000

        # Evaluate on an arbitrary parquet file (feature pipeline applied automatically)
        python src/cli.py evaluate -c pooled/td3_hft_lob_state_space_pooled_streaming_selected \\
            --checkpoint logs/my_exp/checkpoint.pt \\
            --data-path data/raw/stocks/daily/AAPL/AAPL_2026-03-10_raw_mbp-10_us_hours.parquet \\
            --no-mlflow

        # Evaluate each symbol independently (results in eval_results/per_symbol/<SYMBOL>/)
        python src/cli.py evaluate -c pooled/td3_hft_lob_state_space_pooled_streaming_selected \\
            --per-symbol --split test --no-mlflow
    """
    from cli.commands import EvaluateCommand, EvaluateParams

    params = EvaluateParams(
        config_file=config_file,
        checkpoint=checkpoint,
        split=split,
        only=only or None,
        output_dir=output_dir,
        config_overrides=config_override,
        tracking_uri=tracking_uri,
        no_mlflow=no_mlflow,
        data_path=data_path,
        save_rollout=save_rollout,
        save_trades=save_trades,
        per_symbol=per_symbol,
    )
    EvaluateCommand(console).execute(params)


@app.command(name="collect-results")
def collect_results(
    algorithms: list[str] = typer.Option(  # noqa: B008
        ...,
        "--algorithm",
        "-a",
        help="Algorithm name (repeat for each, e.g. -a TD3 -a DDPG -a PPO)",
    ),
    dirs: list[str] = typer.Option(  # noqa: B008
        ...,
        "--dir",
        "-d",
        help="Path to eval_results directory for the corresponding algorithm (same order as --algorithm)",
    ),
    output_dir: Path = typer.Option(  # noqa: B008
        Path("masters_thesis_results"),
        "--output-dir",
        "-o",
        help="Destination directory for aggregated thesis results",
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Overwrite existing output"
    ),
) -> None:
    """Merge per-algorithm evaluation results into a single thesis results directory.

    Example:

        python src/cli.py collect-results \\
            -a TD3 -d ./eval_results/td3 \\
            -a DDPG -d ./eval_results/ddpg \\
            -a PPO -d ./eval_results/ppo \\
            -o masters_thesis_results/
    """
    from cli.commands import CollectResultsCommand, CollectResultsParams

    params = CollectResultsParams(
        algorithms=algorithms,
        dirs=dirs,
        output_dir=output_dir,
        overwrite=overwrite,
    )
    CollectResultsCommand(console).execute(params)


@app.command(name="prepare-data")
def prepare_data(
    scenario: str | None = typer.Option(
        None,
        "--scenario",
        "-s",
        help="Scenario name or path under src/configs/scenarios",
    ),
    config_file: Path | None = typer.Option(  # noqa: B008
        None, "--config", "-c", help="Path to experiment config YAML"
    ),
    config_override: list[str] | None = typer.Option(  # noqa: B008
        None, "--config-override", "-o", help="OmegaConf dotlist override (repeatable)"
    ),
):
    """Materialise features and populate the data cache without training.

    Loads the experiment config, runs the full data-preparation pipeline
    (raw parquet → LOB filter → feature computation → memmap/parquet cache),
    then exits.  Run this once after changing hft_lob_features_all.yaml so
    that subsequent feature-research and train runs hit the cache immediately.
    """
    from logger import get_logger as _get_logger
    from trading_rl import ExperimentConfig
    from trading_rl.data_utils import build_prepared_dataset

    _log = _get_logger(__name__)
    console = Console()

    if scenario and config_file:
        raise typer.BadParameter("Cannot specify both --config and --scenario.")
    if not scenario and not config_file:
        raise typer.BadParameter("Provide --scenario or --config.")

    if scenario:
        search = [
            Path(scenario),
            Path("src/configs/scenarios") / scenario,
            Path("src/configs/scenarios") / f"{scenario}.yaml",
        ]
        config_path = next((p for p in search if p.exists()), None)
        if config_path is None:
            raise typer.BadParameter(f"Scenario '{scenario}' not found.")
    else:
        config_path = config_file

    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
    )

    config = ExperimentConfig.load(config_path, overrides=config_override)
    console.print(f"[blue]Config loaded: {config_path}[/blue]")

    data_paths = getattr(config.data, "data_paths", None) or []
    val_data_paths = getattr(config.data, "val_data_paths", None) or []
    n_symbols = len({Path(p).name.split("_")[0] for p in data_paths})
    total_steps = n_symbols + len(data_paths) + len(val_data_paths)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}", justify="left"),
        BarColumn(),
        MofNCompleteColumn(),
        _HHMMSSElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Preparing data…", total=total_steps)

        def _on_step(label: str) -> None:
            progress.update(task, advance=1, description=label)

        dataset = build_prepared_dataset(config, _log, progress_callback=_on_step)

    console.print(
        f"[green]Done.[/green] "
        f"train={len(dataset.train_df):,} val={len(dataset.val_df):,} "
        f"test={len(dataset.test_df):,} rows, "
        f"{len(dataset.feature_columns)} feature columns"
    )


peek_app = typer.Typer(help="Peek utilities: dataset summaries and config listings")
app.add_typer(peek_app, name="peek")


@peek_app.command(name="dataset")
def peek_dataset(
    scenario: str | None = typer.Option(
        None,
        "--scenario",
        "-s",
        help="Scenario name or path under src/configs/scenarios",
    ),
    config_file: Path | None = typer.Option(  # noqa: B008
        None, "--config", "-c", help="Path to experiment config YAML"
    ),
    config_override: list[str] | None = typer.Option(  # noqa: B008
        None, "--config-override", "-o", help="OmegaConf dotlist override (repeatable)"
    ),
    n_features: int = typer.Option(
        20, "--top", "-n", help="Max feature rows to show in stats table"
    ),
    skip_rows: int = typer.Option(
        0,
        "--skip",
        help="Skip first N rows before computing feature stats (excludes indicator warm-up)",
    ),
    show_correlations: bool = typer.Option(
        False,
        "--corr",
        help="Show feature–reward correlation table (Pearson + Spearman)",
    ),
    export: bool = typer.Option(
        False, "--export", help="Export tables as CSV files to reports/peek/<scenario>/"
    ),
):
    """Show a summary of the prepared dataset for a scenario.

    Loads from cache when available (instant after prepare-data).  Displays
    split sizes, date ranges, per-feature statistics, and memmap file inventory.
    """
    from cli.commands import PeekCommand, PeekParams

    PeekCommand(console).execute(
        PeekParams(
            scenario=scenario,
            config_file=config_file,
            config_override=config_override,
            n_features=n_features,
            skip_rows=skip_rows,
            show_correlations=show_correlations,
            export=export,
        )
    )


@peek_app.command(name="configs")
def peek_configs(
    n: int = typer.Option(20, "--top", "-n", help="Number of configs to show"),
    search_dir: Path | None = typer.Option(  # noqa: B008
        None, "--dir", "-d", help="Directory to search (default: src/configs)"
    ),
    pattern: str | None = typer.Option(
        None, "--filter", "-f", help="Filter by substring in path"
    ),
) -> None:
    """List config YAML files sorted by most recently modified.

    Shows algorithm, reward type, and other key fields so you know which
    scenario to run next.
    """
    import yaml

    base = search_dir or Path("src/configs")
    yamls = sorted(base.rglob("*.yaml"), key=lambda p: p.stat().st_mtime, reverse=True)

    if pattern:
        yamls = [p for p in yamls if pattern in str(p)]

    yamls = yamls[:n]

    tbl = Table(
        title=f"Recent configs (top {len(yamls)} of {base})",
        show_header=True,
        header_style="bold",
    )
    tbl.add_column("modified", style="dim")
    tbl.add_column("scenario")
    tbl.add_column("algorithm")
    tbl.add_column("reward")
    tbl.add_column("backend")

    for p in yamls:
        mtime = (
            datetime.fromtimestamp(p.stat().st_mtime, tz=UTC)
            .astimezone()
            .strftime("%m-%d %H:%M")
        )
        rel = p.relative_to(base)
        # For train.yaml show just the parent dir; strip leading scenarios/ prefix
        label = rel.parent if p.name == "train.yaml" else rel
        parts = label.parts
        scenario_label = (
            str(Path(*parts[1:])) if parts and parts[0] == "scenarios" else str(label)
        )
        try:
            with p.open() as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            cfg = {}

        algorithm = (cfg.get("training") or {}).get("algorithm", "")
        reward = (cfg.get("env") or {}).get("reward_type", "")
        backend = (cfg.get("env") or {}).get("backend", "")
        tbl.add_row(mtime, scenario_label, algorithm, reward, backend)

    console.print(tbl)


@validate_app.command(name="config")
def validate_config(
    config_file: Path | None = typer.Option(  # noqa: B008
        None, "--config", "-c", help="Path to config file"
    ),
    scenario: str | None = typer.Option(
        None, "--scenario", "-s", help="Scenario name or path to scenario file"
    ),
    config_override: list[str] | None = typer.Option(  # noqa: B008
        None,
        "--config-override",
        "-o",
        help="OmegaConf override in dotlist format (repeatable)",
    ),
):
    """Validate experiment config, data dependencies, and feature wiring."""
    from cli.commands import ValidationCommand, ValidationParams

    ValidationCommand(console).execute(
        ValidationParams(
            config_file=config_file,
            scenario=scenario,
            config_overrides=config_override,
        )
    )


@validate_app.command(name="data")
def validate_data(
    config_file: Path | None = typer.Option(  # noqa: B008
        None, "--config", "-c", help="Path to config file"
    ),
    scenario: str | None = typer.Option(
        None, "--scenario", "-s", help="Scenario name or path to scenario file"
    ),
    config_override: list[str] | None = typer.Option(  # noqa: B008
        None,
        "--config-override",
        "-o",
        help="OmegaConf override in dotlist format (repeatable)",
    ),
    no_nan: bool = typer.Option(False, "--no-nan", help="Skip NaN check"),
    no_inf: bool = typer.Option(False, "--no-inf", help="Skip inf check"),
    no_duplicates: bool = typer.Option(
        False, "--no-duplicates", help="Skip duplicate index check"
    ),
    no_zero_variance: bool = typer.Option(
        False, "--no-zero-variance", help="Skip zero-variance feature check"
    ),
    no_lob_deltas: bool = typer.Option(
        False, "--no-lob-deltas", help="Skip LOB delta check"
    ),
    no_temporal_order: bool = typer.Option(
        False, "--no-temporal-order", help="Skip temporal ordering check"
    ),
    no_overlap: bool = typer.Option(
        False, "--no-overlap", help="Skip index overlap / data-leakage check"
    ),
    no_sizes: bool = typer.Option(
        False, "--no-sizes", help="Skip split-size vs config check"
    ),
    lob_levels: int = typer.Option(
        5, "--lob-levels", help="Number of LOB levels to check for deltas"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show description for each check"
    ),
    transpose: bool = typer.Option(
        False,
        "--transpose",
        "-t",
        help="Transpose data glimpse table (shows all columns)",
    ),
):
    """Validate the prepared dataset for a scenario using DataValidator."""
    from cli.commands import ValidateDataCommand, ValidateDataParams

    ValidateDataCommand(console).execute(
        ValidateDataParams(
            scenario=scenario,
            config_file=config_file,
            config_override=config_override,
            check_nan=not no_nan,
            check_inf=not no_inf,
            check_duplicates=not no_duplicates,
            check_zero_variance=not no_zero_variance,
            check_lob_deltas=not no_lob_deltas,
            check_temporal_order=not no_temporal_order,
            check_overlap=not no_overlap,
            check_sizes=not no_sizes,
            lob_levels=lob_levels,
            verbose=verbose,
            transpose=transpose,
        )
    )


@validate_app.command(name="guardrails")
def validate_guardrails(
    config_file: Path | None = typer.Option(  # noqa: B008
        None, "--config", "-c", help="Path to config file"
    ),
    scenario: str | None = typer.Option(
        None, "--scenario", "-s", help="Scenario name or path to scenario file"
    ),
    config_override: list[str] | None = typer.Option(  # noqa: B008
        None,
        "--config-override",
        "-o",
        help="OmegaConf override in dotlist format (repeatable)",
    ),
    all_scenarios: bool = typer.Option(
        False,
        "--all",
        help="Check every scenario directory under src/configs/scenarios",
    ),
):
    """Run pre-flight config guardrails (parameter consistency checks).

    Checks training parameter relationships like sample_size vs buffer_size,
    sample_size vs init_rand_steps, and algorithm-specific constraints.
    Reports FATAL and WARN findings without running training.

    Use --all to validate every scenario in src/configs/scenarios at once.
    """
    from rich.markup import escape as _escape

    from trading_rl import ExperimentConfig
    from trading_rl.config_guardrails_checks import Severity, check_config_guardrails

    def _check_one(config_path: Path) -> tuple[list, list, str]:
        """Return (fatals, warns, scenario_name) for a single config path."""
        command = "train" if config_path.is_dir() else None
        try:
            cfg = ExperimentConfig.load(
                config_path, command=command, overrides=config_override
            )
        except Exception as exc:
            return [], [], f"[red]LOAD ERROR[/red] {_escape(str(exc))}"
        findings = check_config_guardrails(cfg)
        fatals = [f for f in findings if f.severity == Severity.FATAL]
        warns = [f for f in findings if f.severity == Severity.WARN]
        name = getattr(cfg, "experiment_name", None) or str(config_path)
        return fatals, warns, name

    def _print_findings(fatals: list, warns: list, scenario_name: str) -> None:
        tag = f"  [dim]{_escape(f'[{scenario_name}]')}[/dim]"
        if fatals:
            console.print(f"\n[bold red]FATAL ({len(fatals)})[/bold red]{tag}")
            for i, f in enumerate(fatals, 1):
                console.print(
                    f"  [cyan][{i}][/cyan] [yellow]{_escape(f.parameter)}[/yellow]"
                )
                console.print(f"       Problem: {_escape(f.message)}")
                console.print(f"       Fix:     [cyan]{_escape(f.suggestion)}[/cyan]")
        if warns:
            console.print(f"\n[bold yellow]WARN ({len(warns)})[/bold yellow]{tag}")
            for i, f in enumerate(warns, 1):
                console.print(
                    f"  [cyan][{i}][/cyan] [yellow]{_escape(f.parameter)}[/yellow]"
                )
                console.print(f"       Problem:    {_escape(f.message)}")
                console.print(
                    f"       Suggestion: [cyan]{_escape(f.suggestion)}[/cyan]"
                )

    if all_scenarios:
        scenarios_root = Path("src/configs/scenarios")
        # Every directory that contains a train.yaml is a scenario
        scenario_dirs = sorted(p.parent for p in scenarios_root.rglob("train.yaml"))
        if not scenario_dirs:
            console.print(
                "[yellow]No scenarios found under src/configs/scenarios[/yellow]"
            )
            raise typer.Exit(0)

        console.print(f"Checking [bold]{len(scenario_dirs)}[/bold] scenario(s)...\n")
        passed = failed = warned = 0
        for sdir in scenario_dirs:
            fatals, warns, name = _check_one(sdir)
            rel = sdir.relative_to(scenarios_root)
            if fatals:
                console.print(f"[red][FAIL][/red] {_escape(str(rel))}")
                _print_findings(fatals, warns, name)
                failed += 1
            elif warns:
                console.print(f"[yellow][WARN][/yellow] {_escape(str(rel))}")
                _print_findings([], warns, name)
                warned += 1
            else:
                console.print(f"[green][PASS][/green] {_escape(str(rel))}")
                passed += 1

        console.print(
            f"\nSummary: "
            f"[green]passed={passed}[/green]  "
            f"[yellow]warned={warned}[/yellow]  "
            f"[{'red' if failed else 'green'}]failed={failed}[/{'red' if failed else 'green'}]"
        )
        if failed:
            raise typer.Exit(code=1)
        return

    if scenario and config_file:
        raise typer.BadParameter("Cannot specify both --config and --scenario.")
    if not scenario and not config_file:
        raise typer.BadParameter("Provide --scenario, --config, or --all.")

    input_path = scenario if scenario else str(config_file)

    path = Path(input_path)
    if not path.exists():
        candidate = Path("src/configs/scenarios") / input_path
        if candidate.exists():
            path = candidate

    if not path.exists():
        search = [
            Path(input_path),
            Path("src/configs/scenarios") / input_path,
            Path("src/configs/scenarios") / f"{input_path}.yaml",
        ]
        config_path = next((p for p in search if p.exists()), None)
        if config_path is None:
            raise typer.BadParameter(f"Config '{input_path}' not found.")
    else:
        config_path = path

    fatals, warns, scenario_name = _check_one(config_path)

    if not fatals and not warns:
        console.print("[green]Guardrails passed — no issues found[/green]")
        return

    _print_findings(fatals, warns, scenario_name)

    if fatals:
        raise typer.Exit(code=1)


@app.command(name="feature-research")
def feature_research(
    config_file: Path | None = typer.Option(  # noqa: B008
        None,
        "--config",
        help="Path to feature research config YAML",
        show_default=False,
    ),
    experiment_config_file: Path | None = typer.Option(  # noqa: B008
        None,
        "--experiment-config",
        help="Path to experiment scenario YAML to derive research settings from",
        show_default=False,
    ),
    scenario: str | None = typer.Option(
        None,
        "--scenario",
        help="Scenario config name or path under src/configs/scenarios",
        show_default=False,
    ),
    config_override: list[str] | None = typer.Option(  # noqa: B008
        None,
        "--config-override",
        help="OmegaConf override in dotlist format (repeatable)",
        show_default=False,
    ),
):
    """Run offline feature scoring and shortlist generation."""
    from cli.commands import FeatureResearchCommand, FeatureResearchParams

    params = FeatureResearchParams(
        config_file=config_file,
        experiment_config_file=experiment_config_file,
        scenario=scenario,
        config_overrides=config_override,
    )
    FeatureResearchCommand(console).execute(params)


@app.command()
def dashboard(
    port: int = typer.Option(5000, "--port", "-p", help="Port for MLflow UI"),
    host: str = typer.Option("localhost", "--host", help="Host for MLflow UI"),
    tracking_uri: str | None = typer.Option(
        None,
        "--tracking-uri",
        help="MLflow tracking URI (default sqlite:///mlflow.db)",
        show_default=False,
    ),
):
    """Launch MLflow UI for viewing experiments."""
    from cli.commands import DashboardCommand, DashboardParams

    params = DashboardParams(port=port, host=host, tracking_uri=tracking_uri)
    DashboardCommand(console, default_tracking_uri="sqlite:///mlflow.db").execute(
        params
    )


@app.command(name="experiments")
def experiments(
    tracking_uri: str | None = typer.Option(
        None,
        "--tracking-uri",
        help="MLflow tracking URI (default sqlite:///mlflow.db)",
        show_default=False,
    ),
    delete: str | None = typer.Option(
        None, "--delete", help="Permanently delete experiments matching regex"
    ),
    delete_all: bool = typer.Option(
        False, "--delete-all", help="Permanently delete all experiments"
    ),
    purge: bool = typer.Option(
        False,
        "--purge",
        help=(
            "Permanently remove soft-deleted experiments from the SQLite DB. "
            "Combine with --delete <regex> or --delete-all to select targets. "
            "Requires sqlite:/// tracking URI."
        ),
    ),
    force: bool = typer.Option(False, "--force", help="Skip confirmation prompt"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be deleted"),
):
    """List available MLflow experiments or permanently delete them."""
    from cli.commands import ExperimentsCommand, ExperimentsParams

    ExperimentsCommand(console).execute(
        ExperimentsParams(
            tracking_uri=tracking_uri,
            delete=delete,
            delete_all=delete_all,
            purge=purge,
            force=force,
            dry_run=dry_run,
        )
    )


@app.command(name="scenarios")
def scenarios(
    delete: str | None = typer.Option(
        None, "--delete", help="Delete scenarios matching regex"
    ),
    delete_all: bool = typer.Option(False, "--delete-all", help="Delete all scenarios"),
    force: bool = typer.Option(False, "--force", help="Delete without confirmation"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be deleted"),
):
    """List available scenario configurations."""
    from cli.commands import ScenariosCommand, ScenariosParams

    ScenariosCommand(console).execute(
        ScenariosParams(
            delete=delete,
            delete_all=delete_all,
            force=force,
            dry_run=dry_run,
        )
    )


@app.command(name="artifacts")
def artifacts(
    tracking_uri: str | None = typer.Option(
        None,
        "--tracking-uri",
        help="MLflow tracking URI (default sqlite:///mlflow.db)",
        show_default=False,
    ),
    experiment: str | None = typer.Option(
        None, "--experiment", help="Filter experiments by regex"
    ),
    run_id: str | None = typer.Option(
        None, "--run-id", help="List artifacts for a specific run id"
    ),
    prefix: str | None = typer.Option(
        None, "--prefix", help="Only list artifacts under this path prefix"
    ),
    delete: str | None = typer.Option(
        None, "--delete", help="Delete artifacts matching regex"
    ),
    delete_all: bool = typer.Option(
        False, "--delete-all", help="Delete all artifacts for selected runs"
    ),
    force: bool = typer.Option(False, "--force", help="Delete without confirmation"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be deleted"),
    max_runs: int = typer.Option(
        50, "--max-runs", help="Maximum runs to show per experiment"
    ),
):
    """List MLflow artifacts grouped by experiment and run."""
    from cli.commands import ArtifactsCommand, ArtifactsParams

    ArtifactsCommand(console).execute(
        ArtifactsParams(
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
    )


if __name__ == "__main__":
    app()
