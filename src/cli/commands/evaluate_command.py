"""Evaluate a trained policy from a checkpoint without running training."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.table import Table

from .base_command import BaseCommand


_ALL_COMPONENTS = frozenset({"metrics", "benchmarks", "plots", "stats"})

_PERF_ROWS = [
    ("total_return", "Total Return", ".2%"),
    ("annualized_return_cagr", "CAGR", ".2%"),
    ("sharpe_ratio", "Sharpe Ratio", ".3f"),
    ("sortino_ratio", "Sortino Ratio", ".3f"),
    ("max_drawdown", "Max Drawdown", ".2%"),
    ("win_rate", "Win Rate", ".2%"),
    ("profit_factor", "Profit Factor", ".3f"),
]


@dataclass
class EvaluateParams:
    """Parameters for the standalone evaluate command."""

    config_file: Path | None = None
    checkpoint: Path | None = None
    split: str = "test"
    only: list[str] | None = None
    output_dir: Path = field(default_factory=lambda: Path("./eval_results"))
    config_overrides: list[str] | None = None
    tracking_uri: str = "sqlite:///mlflow.db"
    no_mlflow: bool = False


class EvaluateCommand(BaseCommand):
    """Evaluate a saved checkpoint without re-running training."""

    def execute(self, params: EvaluateParams) -> None:
        try:
            self._run(params)
        except Exception as e:
            self.handle_error(e, "Evaluation")

    # ------------------------------------------------------------------
    # Main flow
    # ------------------------------------------------------------------

    def _run(self, params: EvaluateParams) -> None:
        from trading_rl.data_utils import build_prepared_dataset
        from trading_rl.evaluation import (
            EvaluationConfig,
            PolicyLoader,
            StrategyEvaluator,
            periods_per_year_from_timeframe,
            run_all_statistical_tests,
        )
        from trading_rl.evaluation.benchmarks import BenchmarkEngine
        from trading_rl.evaluation.metrics import build_metric_report
        from trading_rl.pipeline.evaluation import build_evaluation_context_for_split

        components = frozenset(params.only) if params.only else _ALL_COMPONENTS

        config = self._load_config(params)
        self.console.print(f"[blue]Experiment: {config.experiment_name}[/blue]")

        checkpoint_path = self._resolve_checkpoint(config, params)
        self.console.print(f"[blue]Checkpoint: {checkpoint_path}[/blue]")

        self.console.print("[dim]Loading policy...[/dim]")
        policy = PolicyLoader.from_checkpoint(str(checkpoint_path))
        meta = PolicyLoader.inspect(str(checkpoint_path))
        self.console.print(
            f"[dim]Algorithm: {meta.get('algorithm')}  "
            f"n_obs={meta.get('n_obs')}  n_act={meta.get('n_act')}[/dim]"
        )

        self.console.print("[dim]Loading data...[/dim]")
        dataset = build_prepared_dataset(config, self.logger)

        splits_to_eval = (
            ["train", "val", "test"] if params.split == "all" else [params.split]
        )
        params.output_dir.mkdir(parents=True, exist_ok=True)

        price_column = getattr(config.env, "price_column", None) or "close"
        backend = str(getattr(config.env, "backend", "tradingenv")).lower()
        reward_type = str(getattr(config.env, "reward_type", "log_return"))
        timeframe = getattr(config.data, "timeframe", "1d")
        periods_py = periods_per_year_from_timeframe(timeframe)

        split_dfs = {
            "train": dataset.train_df,
            "val": dataset.val_df,
            "test": dataset.test_df,
        }

        all_results: dict[str, Any] = {}

        mlflow_ctx = self._start_mlflow_run(config, meta, checkpoint_path, splits_to_eval, params)

        with mlflow_ctx as mlflow_run_id:
            for split in splits_to_eval:
                split_df = split_dfs[split]
                if len(split_df) < 2:
                    self.console.print(
                        f"[yellow]Skipping {split} split: insufficient data[/yellow]"
                    )
                    continue

                self.console.print(
                    f"[bold]Evaluating {split} split ({len(split_df):,} rows)...[/bold]"
                )

                split_ctx = build_evaluation_context_for_split(
                    split=split,
                    df=split_df,
                    config=config,
                )

                eval_config = EvaluationConfig(
                    reward_type=reward_type,
                    backend=backend,
                    max_steps=split_ctx.max_steps,
                    price_column=price_column,
                    enable_plots="plots" in components,
                    enable_metrics="metrics" in components,
                )

                evaluator = StrategyEvaluator(
                    env_factory=lambda df, cfg: split_ctx.env,
                    policy=policy,
                    config=eval_config,
                )

                result = evaluator.evaluate_split(split, split_df, env=split_ctx.env)

                split_output: dict[str, Any] = {
                    "split": split,
                    "final_reward": result.final_reward,
                    "n_steps": int(len(result.simple_returns)),
                }

                if "metrics" in components and result.metrics:
                    split_output["metrics"] = result.metrics
                    self._print_metrics_table(split, result.metrics)
                    if mlflow_run_id:
                        from trading_rl.callbacks.artifacts import log_evaluation_report
                        log_evaluation_report(result.metrics, split_prefix=split)

                bench_returns_map: dict[str, Any] = {}
                if "benchmarks" in components or "stats" in components:
                    benchmarks, _ = BenchmarkEngine.build(
                        split_df, config.benchmarks, price_column
                    )

                    if "benchmarks" in components and benchmarks:
                        bench_out: dict[str, Any] = {}
                        for spec in benchmarks:
                            bench_returns = spec.compute_returns(split_ctx.max_steps)
                            bench_returns_map[spec.name] = bench_returns
                            n = min(len(result.simple_returns), len(bench_returns))
                            bench_report = build_metric_report(
                                strategy_simple_returns=result.simple_returns[:n],
                                benchmark_simple_returns=bench_returns[:n],
                                actions=None,
                                periods_per_year=periods_py,
                                risk_free_rate_annual=0.0,
                            )
                            bench_out[spec.name] = bench_report
                        split_output["benchmarks"] = bench_out
                        self._print_benchmark_table(split, bench_out)
                        if mlflow_run_id:
                            self._log_benchmarks_to_mlflow(bench_out, split)

                    if "stats" in components and getattr(config, "statistical_testing", None):
                        stat_results = run_all_statistical_tests(
                            strategy_returns=result.simple_returns,
                            benchmarks=benchmarks,
                            max_steps=split_ctx.max_steps,
                            config=config.statistical_testing,
                            periods_per_year=periods_py,
                        )
                        split_output["statistical_tests"] = stat_results
                        if mlflow_run_id:
                            from trading_rl.callbacks.artifacts import log_statistical_tests
                            log_statistical_tests(stat_results, split_prefix=split)

                if "plots" in components and result.plots:
                    self._save_plots(result.plots, split, params.output_dir)
                    if mlflow_run_id:
                        from trading_rl.callbacks.artifacts import log_evaluation_plots
                        log_evaluation_plots(
                            reward_plot=result.plots.get("reward_plot"),
                            action_plot=result.plots.get("action_plot"),
                            artifact_path_prefix=f"evaluation_plots/{split}",
                        )

                all_results[split] = split_output

            # Write and upload results.json
            out_json = params.output_dir / "results.json"
            with out_json.open("w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, default=_json_default)
            self.console.print(f"[green]Results written to {out_json}[/green]")

            if mlflow_run_id:
                import mlflow
                mlflow.log_artifact(str(out_json), "evaluation_summary")
                run_url = self._mlflow_run_url(params.tracking_uri, mlflow_run_id)
                self.console.print(f"[green]MLflow run: {run_url}[/green]")

    # ------------------------------------------------------------------
    # MLflow helpers
    # ------------------------------------------------------------------

    def _start_mlflow_run(
        self,
        config: Any,
        meta: dict,
        checkpoint_path: Path,
        splits_to_eval: list[str],
        params: EvaluateParams,
    ):
        """Return a context manager that either starts a real MLflow run or is a no-op."""
        if params.no_mlflow:
            return _noop_context()

        try:
            import mlflow
            from datetime import UTC, datetime

            mlflow.set_tracking_uri(params.tracking_uri)
            mlflow.set_experiment(config.experiment_name)

            source_run_id = meta.get("mlflow_run_id") or "unknown"
            timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
            run_name = f"{config.experiment_name}_eval_{timestamp}"

            tags = {
                "eval_checkpoint": str(checkpoint_path),
                "eval_splits": ",".join(splits_to_eval),
                "source_run_id": source_run_id,
                "mlflow.runName": run_name,
                "run_type": "evaluation",
            }

            ctx = mlflow.start_run(run_name=run_name, tags=tags)

            # We need to inject param logging before entering the context —
            # do it lazily in a wrapper so params are logged inside the run.
            return _MlflowRunContext(ctx, meta, checkpoint_path, splits_to_eval)

        except Exception as exc:
            self.logger.warning("MLflow unavailable, skipping run creation: %s", exc)
            return _noop_context()

    def _log_benchmarks_to_mlflow(
        self, bench_out: dict[str, dict[str, float]], split: str
    ) -> None:
        try:
            import mlflow
            import numpy as np

            for bench_name, report in bench_out.items():
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False
                ) as f:
                    json.dump(report, f, indent=2, default=_json_default)
                    f.flush()
                    mlflow.log_artifact(f.name, f"benchmarks/{split}/{bench_name}")
                    os.unlink(f.name)

                for key, _, _ in _PERF_ROWS:
                    val = report.get(key)
                    if val is not None and np.isfinite(float(val)):
                        mlflow.log_metric(
                            f"bench_{split}_{bench_name}_{key}", float(val)
                        )
        except Exception as exc:
            self.logger.warning("Failed to log benchmarks to MLflow: %s", exc)

    @staticmethod
    def _mlflow_run_url(tracking_uri: str, run_id: str) -> str:
        if tracking_uri.startswith("sqlite:///"):
            db_path = tracking_uri[len("sqlite:///"):]
            abs_db = Path(db_path).resolve()
            return f"mlflow ui --backend-store-uri sqlite:///{abs_db}  →  run {run_id}"
        return f"{tracking_uri}  →  run {run_id}"

    # ------------------------------------------------------------------
    # Config and checkpoint helpers
    # ------------------------------------------------------------------

    def _load_config(self, params: EvaluateParams) -> Any:
        from trading_rl import ExperimentConfig

        if params.config_file is None:
            raise ValueError("--config is required for the evaluate command.")

        config_path = params.config_file
        if not config_path.exists():
            config_path = self._resolve_scenario_config_path(
                str(params.config_file), command_file="evaluate.yaml"
            )

        config = ExperimentConfig.from_yaml(
            config_path, overrides=params.config_overrides
        )
        self.console.print(f"[dim]Config: {config_path}[/dim]")
        return config


    def _resolve_checkpoint(self, config: Any, params: EvaluateParams) -> Path:
        if params.checkpoint is not None:
            if not params.checkpoint.exists():
                raise FileNotFoundError(f"Checkpoint not found: {params.checkpoint}")
            return params.checkpoint

        log_dir = Path(config.logging.log_dir)
        pattern = f"{config.experiment_name}_checkpoint*.pt"
        matches = list(log_dir.rglob(pattern))
        if not matches:
            raise FileNotFoundError(
                f"No checkpoints found for '{config.experiment_name}' in {log_dir}. "
                "Provide --checkpoint explicitly."
            )
        latest = max(matches, key=lambda p: p.stat().st_mtime)
        self.console.print(f"[dim]Auto-selected checkpoint: {latest}[/dim]")
        return latest

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def _print_metrics_table(self, split: str, metrics: dict[str, float]) -> None:
        table = Table(title=f"Metrics ({split})", show_header=True, header_style="bold")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")
        for key, label, fmt in _PERF_ROWS:
            if key in metrics:
                table.add_row(label, f"{metrics[key]:{fmt}}")
        self.console.print(table)

    def _print_benchmark_table(
        self, split: str, bench_out: dict[str, dict[str, float]]
    ) -> None:
        table = Table(
            title=f"Benchmark comparison ({split})", show_header=True, header_style="bold"
        )
        table.add_column("Benchmark", style="cyan")
        for _, label, _ in _PERF_ROWS:
            table.add_column(label, justify="right")
        for bench_name, report in bench_out.items():
            row = [bench_name]
            for key, _, fmt in _PERF_ROWS:
                val = report.get(key)
                row.append(f"{val:{fmt}}" if val is not None else "—")
            table.add_row(*row)
        self.console.print(table)

    def _save_plots(
        self, plots: dict[str, Any], split: str, output_dir: Path
    ) -> None:
        for name, fig in plots.items():
            if fig is None:
                continue
            out_path = output_dir / f"{split}_{name}.png"
            try:
                fig.savefig(out_path, bbox_inches="tight", dpi=150)
                self.console.print(f"[dim]Saved plot: {out_path}[/dim]")
            except Exception as exc:
                self.logger.warning("Failed to save plot %s: %s", name, exc)


# ------------------------------------------------------------------
# MLflow run context wrapper
# ------------------------------------------------------------------

class _MlflowRunContext:
    """Wraps mlflow.start_run(), logs params on enter, returns run_id."""

    def __init__(self, ctx, meta: dict, checkpoint_path: Path, splits: list[str]):
        self._ctx = ctx
        self._meta = meta
        self._checkpoint_path = checkpoint_path
        self._splits = splits
        self._run_id: str | None = None

    def __enter__(self) -> str | None:
        import mlflow
        active = self._ctx.__enter__()
        self._run_id = active.info.run_id

        mlflow.log_param("eval_algorithm", self._meta.get("algorithm"))
        mlflow.log_param("eval_n_obs", self._meta.get("n_obs"))
        mlflow.log_param("eval_n_act", self._meta.get("n_act"))
        mlflow.log_param("eval_actor_hidden_dims", str(self._meta.get("actor_hidden_dims")))
        mlflow.log_param("eval_checkpoint", str(self._checkpoint_path))
        mlflow.log_param("eval_splits", ",".join(self._splits))

        return self._run_id

    def __exit__(self, *args):
        return self._ctx.__exit__(*args)


class _NoopContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args):
        pass


def _noop_context() -> _NoopContext:
    return _NoopContext()


# ------------------------------------------------------------------
# JSON serialization helper
# ------------------------------------------------------------------

def _json_default(obj: Any) -> Any:
    import numpy as np

    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)}")
