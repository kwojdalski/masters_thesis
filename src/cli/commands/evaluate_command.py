"""Evaluate a trained policy from a checkpoint without running training."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cli.services.evaluation_display_service import (
    PERF_ROWS,
    EvaluationDisplayService,
)
from trading_rl.callbacks.artifacts import (
    ArtifactPaths,
    save_observation_sample_artifact,
)
from trading_rl.evaluation.asset_meta import write_asset_meta
from trading_rl.evaluation.benchmark_table import save_benchmark_table_artifact

from .base_command import BaseCommand

if TYPE_CHECKING:
    import pandas as pd

_ALL_COMPONENTS = frozenset({"metrics", "benchmarks", "plots", "stats"})

_PERF_ROWS = PERF_ROWS


@dataclass
class EvaluateParams:
    """Parameters for the standalone evaluate command."""

    config_file: Path | None = None
    checkpoint: Path | None = None
    split: str = "all"
    only: list[str] | None = None
    output_dir: Path = field(default_factory=lambda: Path("./eval_results"))
    config_overrides: list[str] | None = None
    tracking_uri: str = "sqlite:///mlflow.db"
    no_mlflow: bool = False
    data_path: Path | None = None
    save_rollout: bool = False
    save_trades: bool = False
    per_symbol: bool = False


class EvaluateCommand(BaseCommand):
    """Evaluate a saved checkpoint without re-running training."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.display = EvaluationDisplayService(self.console)

    def execute(self, params: EvaluateParams) -> None:
        try:
            self._run(params)
        except Exception as e:
            self.handle_error(e, "Evaluation")

    # ------------------------------------------------------------------
    # Main flow
    # ------------------------------------------------------------------

    def _run(self, params: EvaluateParams) -> None:
        from trading_rl.constants import EnvBackend
        from trading_rl.evaluation import PolicyLoader, periods_per_year_from_timeframe

        components = frozenset(params.only) if params.only else _ALL_COMPONENTS

        config = self._load_config(params)
        self.console.print(f"[blue]Experiment: {config.experiment_name}[/blue]")

        algorithm = getattr(config.training, "algorithm", "").upper()
        is_random = algorithm == "RANDOM"

        if is_random:
            policy = None  # built per-split from the env action spec
            checkpoint_path = None
            meta = {}
            self.console.print("[dim]Algorithm: RANDOM (no checkpoint needed)[/dim]")
        else:
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

        params.output_dir.mkdir(parents=True, exist_ok=True)

        price_column = getattr(config.env, "price_column", None) or "close"
        backend = str(getattr(config.env, "backend", EnvBackend.TRADINGENV)).lower()
        reward_type = str(getattr(config.env, "reward_type", "log_return"))
        timeframe = getattr(config.data, "timeframe", "1d")
        timeframe_ppy = periods_per_year_from_timeframe(timeframe)

        splits_to_eval, split_dfs, train_symbols, val_symbols = self._resolve_splits(
            config, params, checkpoint_path
        )

        all_results: dict[str, Any] = {}

        mlflow_ctx = self._start_mlflow_run(
            config, meta, checkpoint_path, splits_to_eval, params
        )

        with mlflow_ctx as mlflow_run_id:
            for split in splits_to_eval:
                split_output = self._evaluate_one_split(
                    split,
                    split_dfs[split],
                    config=config,
                    params=params,
                    components=components,
                    policy=policy,
                    is_random=is_random,
                    meta=meta,
                    price_column=price_column,
                    backend=backend,
                    reward_type=reward_type,
                    timeframe_ppy=timeframe_ppy,
                    train_symbols=train_symbols,
                    val_symbols=val_symbols,
                    mlflow_run_id=mlflow_run_id,
                )
                if split_output is not None:
                    all_results[split] = split_output

            # Write and upload results.json
            out_json = params.output_dir / "results.json"
            with out_json.open("w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, default=_json_default)
            write_asset_meta(out_json, generator="cli/commands/evaluate_command.py")
            self.console.print(f"[green]Results written to {out_json}[/green]")

            if mlflow_run_id:
                import mlflow

                mlflow.log_artifact(str(out_json), "evaluation_summary")
                run_url = self._mlflow_run_url(params.tracking_uri, mlflow_run_id)
                self.console.print(f"[green]MLflow run: {run_url}[/green]")

    # ------------------------------------------------------------------
    # Split resolution
    # ------------------------------------------------------------------

    def _resolve_splits(
        self,
        config: Any,
        params: EvaluateParams,
        checkpoint_path: Path | None,
    ) -> tuple[list[str], dict[str, pd.DataFrame], list[str], list[str]]:
        """Resolve which splits to evaluate and their DataFrames.

        Handles four cases: an arbitrary ``--data-path`` file, per-symbol val/test
        files (``--per-symbol``), a multi-symbol scenario without ``--per-symbol``
        (train-only fallback or skip), and the standard prepared train/val/test
        dataset. Returns ``(splits_to_eval, split_dfs, train_symbols, val_symbols)``.
        """
        from trading_rl.constants import SplitName
        from trading_rl.data_utils import build_prepared_dataset

        if getattr(config.data, "data_paths", None):
            train_symbols = sorted(
                {Path(p).parent.name for p in config.data.data_paths}
            )
        elif getattr(config.data, "symbols", None):
            train_symbols = list(config.data.symbols)
        else:
            train_symbols = []
        if getattr(config.data, "val_data_paths", None):
            val_symbols = sorted(
                {Path(p).parent.name for p in config.data.val_data_paths}
            )
        else:
            val_symbols = train_symbols

        if params.data_path is not None:
            if not params.data_path.exists():
                raise FileNotFoundError(f"--data-path not found: {params.data_path}")
            self.console.print(
                f"[dim]Preparing arbitrary data: {params.data_path}[/dim]"
            )
            arbitrary_df = self._prepare_arbitrary_df(
                params.data_path, config, checkpoint_path
            )
            split_name = params.data_path.stem
            return [split_name], {split_name: arbitrary_df}, train_symbols, val_symbols

        val_data_paths = getattr(config.data, "val_data_paths", None)
        wants_val_or_test = params.split == "all" or params.split in {
            SplitName.VAL,
            SplitName.TEST,
        }
        if val_data_paths and wants_val_or_test and params.per_symbol:
            # Per-symbol eval: each symbol evaluated independently to avoid
            # cross-symbol price artefacts; results go to per_symbol/{SYMBOL}/.
            self.console.print(
                "[dim]--per-symbol: evaluating each symbol independently[/dim]"
            )
            splits_to_eval, split_dfs = self._resolve_per_symbol_splits(
                config, params, [str(p) for p in val_data_paths], checkpoint_path
            )
        elif val_data_paths and wants_val_or_test and not params.per_symbol:
            # Multi-symbol scenarios can only produce val/test through the
            # per-symbol path. Silently substituting the train split here made
            # in-sample metrics, benchmark tables and bootstrap significance
            # tests flow into the thesis snapshots as out-of-sample results.
            raise ValueError(
                f"Scenario '{config.experiment_name}' defines data.val_data_paths, so the "
                f"{params.split!r} split can only be evaluated per symbol. "
                "Re-run with --per-symbol, or pass --split train to evaluate the "
                "training split deliberately."
            )
        else:
            dataset = build_prepared_dataset(config, self.logger)
            splits_to_eval = (
                list(SplitName) if params.split == "all" else [SplitName(params.split)]
            )
            split_dfs = {
                SplitName.TRAIN: dataset.train_df,
                SplitName.VAL: dataset.val_df,
                SplitName.TEST: dataset.test_df,
            }

        return splits_to_eval, split_dfs, train_symbols, val_symbols

    # ------------------------------------------------------------------
    # Per-split evaluation
    # ------------------------------------------------------------------

    def _evaluate_one_split(
        self,
        split: str,
        split_df: pd.DataFrame,
        *,
        config: Any,
        params: EvaluateParams,
        components: frozenset[str],
        policy: Any,
        is_random: bool,
        meta: dict,
        price_column: str,
        backend: str,
        reward_type: str,
        timeframe_ppy: float,
        train_symbols: list[str],
        val_symbols: list[str],
        mlflow_run_id: str | None,
    ) -> dict[str, Any] | None:
        """Run rollout, metrics, benchmarks, and stats for one split.

        Returns the split's result dict, or None if the split was skipped
        for having fewer than 2 rows.
        """
        from trading_rl.evaluation import (
            StrategyEvaluator,
            StrategyEvaluatorConfig,
            run_all_statistical_tests,
        )
        from trading_rl.evaluation.benchmarks import (
            BenchmarkEngine,
            benchmarks_from_config,
        )
        from trading_rl.evaluation.metrics import build_metric_report
        from trading_rl.evaluation.report import _periods_per_year_from_index
        from trading_rl.pipeline.evaluation import build_evaluation_context_for_split

        if len(split_df) < 2:
            self.console.print(
                f"[yellow]Skipping {split} split: insufficient data[/yellow]"
            )
            return None

        self.console.print(
            f"[bold]Evaluating {split} split ({len(split_df):,} rows)...[/bold]"
        )

        # Per-symbol mode: route artifacts to per_symbol/{SYMBOL}/
        if params.per_symbol and isinstance(split, str) and "_" in split:
            sym = split.split("_", 1)[1]
            split_out_dir = params.output_dir / "per_symbol" / sym
        else:
            split_out_dir = params.output_dir
        split_out_dir.mkdir(parents=True, exist_ok=True)

        periods_py = _periods_per_year_from_index(split_df) or timeframe_ppy

        split_ctx = build_evaluation_context_for_split(
            split=split,
            df=split_df,
            config=config,
        )
        sample_path = save_observation_sample_artifact(
            split=split,
            df=split_ctx.df,
            output_dir=split_out_dir / ArtifactPaths.EVAL_DATA,
        )
        self.console.print(f"[dim]  Observation sample parquet → {sample_path}[/dim]")
        self.console.print(
            f"[dim]  Building environment ({split_ctx.max_steps:,} steps)...[/dim]"
        )
        eval_config = StrategyEvaluatorConfig(
            reward_type=reward_type,
            backend=backend,
            max_steps=split_ctx.max_steps,
            price_column=price_column,
            enable_plots="plots" in components,
            enable_metrics="metrics" in components,
            periods_per_year=periods_py,
            eval_plots=tuple(
                getattr(
                    config.evaluation,
                    "eval_plots",
                    ("rewards", "positions", "portfolio_value"),
                )
            ),
            training_steps=int(meta["total_count"])
            if meta.get("total_count") is not None
            else None,
            training_episodes=int(meta["total_episodes"])
            if meta.get("total_episodes") is not None
            else None,
            benchmarks=benchmarks_from_config(config.benchmarks)
            if getattr(config, "benchmarks", None)
            else None,
        )

        if is_random:
            from torchrl.envs.utils import RandomPolicy

            split_policy = RandomPolicy(split_ctx.env.action_spec)
        else:
            split_policy = policy
        evaluator = StrategyEvaluator(
            env_factory=lambda df, cfg, env=split_ctx.env: env,
            policy=split_policy,
            config=eval_config,
        )

        self.console.print("[dim]  Running policy rollout...[/dim]")
        result = evaluator.evaluate_split(split, split_df, env=split_ctx.env)
        self.console.print(
            f"[dim]  Rollout done: {len(result.simple_returns):,} steps, "
            f"final reward {result.final_reward:.4f}[/dim]"
        )

        split_output: dict[str, Any] = {
            "split": split,
            "final_reward": result.final_reward,
            "n_steps": len(result.simple_returns),
        }

        if "metrics" in components and result.metrics:
            self.console.print("[dim]  Computing metrics...[/dim]")
            split_output["metrics"] = result.metrics
            split_symbols = (
                train_symbols if str(split).startswith("train") else val_symbols
            )
            self._print_metrics_table(
                split, result.metrics, split_df=split_df, symbols=split_symbols
            )
            if mlflow_run_id:
                from trading_rl.callbacks.artifacts import log_evaluation_report

                log_evaluation_report(result.metrics, split_prefix=split)

        bench_returns_map: dict[str, Any] = {}
        if "benchmarks" in components or "stats" in components:
            self.console.print("[dim]  Building benchmarks...[/dim]")
            benchmarks, _ = BenchmarkEngine.build(
                split_df, config.benchmarks, price_column
            )

            if getattr(config.benchmarks, "is_random", False):
                try:
                    import numpy as np

                    from trading_rl.evaluation.statistical_benchmarks import (
                        compute_random_returns_from_prices,
                    )

                    self.console.print("[dim]  Computing random baseline...[/dim]")
                    _price_col = getattr(config.env, "price_column", None) or "close"
                    random_trials = compute_random_returns_from_prices(
                        split_df[_price_col],
                        split_ctx.max_steps,
                        n_trials=getattr(config.benchmarks, "n_random_trials", 10),
                        seed=getattr(config.benchmarks, "random_seed", None),
                    )
                    if random_trials:
                        min_len = min(len(t) for t in random_trials)
                        mean_returns = np.mean(
                            [t[:min_len] for t in random_trials], axis=0
                        )
                        benchmarks.append(BenchmarkEngine.random_actions(mean_returns))
                except Exception as _rand_err:
                    self.console.print(
                        f"[dim yellow]  Random baseline failed: {_rand_err}[/dim yellow]"
                    )

            self.console.print(f"[dim]  {len(benchmarks)} benchmark(s) ready[/dim]")

            if "benchmarks" in components and benchmarks:
                bench_out: dict[str, Any] = {}
                for spec in benchmarks:
                    self.console.print(
                        f"[dim]  Benchmark metrics: {spec.name}...[/dim]"
                    )
                    bench_returns = spec.compute_returns(split_ctx.max_steps)
                    bench_returns_map[spec.name] = bench_returns
                    n = min(len(result.simple_returns), len(bench_returns))
                    # Benchmark's own performance metrics
                    bench_own = build_metric_report(
                        strategy_simple_returns=bench_returns[:n],
                        benchmark_simple_returns=bench_returns[:n],
                        actions=None,
                        periods_per_year=periods_py,
                        risk_free_rate_annual=0.0,
                    )
                    # Relative metrics (alpha, beta, IR, TE) vs strategy
                    bench_rel = build_metric_report(
                        strategy_simple_returns=result.simple_returns[:n],
                        benchmark_simple_returns=bench_returns[:n],
                        actions=None,
                        periods_per_year=periods_py,
                        risk_free_rate_annual=0.0,
                    )
                    _enabled = getattr(
                        getattr(config, "metrics", None), "enabled_set", None
                    )
                    bench_out[spec.name] = {
                        "benchmark_metrics": bench_own.to_filtered_dict(_enabled),
                        "relative_metrics": {
                            k: getattr(bench_rel, k)
                            for k in (
                                "alpha",
                                "beta",
                                "information_ratio",
                                "tracking_error",
                            )
                        },
                    }
                split_output["benchmarks"] = bench_out
                _enabled = getattr(
                    getattr(config, "metrics", None), "enabled_set", None
                )
                strategy_dict = (
                    result.metrics.to_filtered_dict(_enabled)
                    if result.metrics
                    else None
                )
                self._print_benchmark_table(split, bench_out, strategy_dict)
                json_p, png_p = save_benchmark_table_artifact(
                    split, split_df, bench_out, strategy_dict, split_out_dir
                )
                self.console.print(f"[dim]  Benchmark table JSON → {json_p}[/dim]")
                self.console.print(f"[dim]  Benchmark table PNG  → {png_p}[/dim]")
                if mlflow_run_id:
                    self._log_benchmarks_to_mlflow(bench_out, split)

            if "stats" in components and getattr(config, "statistical_testing", None):
                stat_results = run_all_statistical_tests(
                    strategy_returns=result.simple_returns,
                    benchmarks=benchmarks,
                    max_steps=split_ctx.max_steps,
                    config=config.statistical_testing,
                    periods_per_year=periods_py,
                    status_fn=lambda msg: self.console.print(f"[dim]  {msg}[/dim]"),
                )
                split_output["statistical_tests"] = stat_results
                if mlflow_run_id:
                    from trading_rl.callbacks.artifacts import (
                        log_statistical_tests,
                    )

                    log_statistical_tests(stat_results, split_prefix=split)

        if params.save_rollout:
            self._save_rollout_data(result, split, split_df, split_out_dir)

        if params.save_trades:
            self._save_trade_log(split_ctx.env, split, split_out_dir)

        if "plots" in components and result.plots:
            self.console.print("[dim]  Saving plots...[/dim]")
            self._save_plots(result.plots, split, split_out_dir)
            if mlflow_run_id:
                from trading_rl.callbacks.artifacts import log_evaluation_plots

                _rollout = result.plots.get("_rollout_plot_data")
                _equity = result.plots.get("_equity_plot_data")
                _plot_data = (
                    {**(_rollout or {}), **(_equity or {})}
                    if (_rollout or _equity)
                    else None
                )
                log_evaluation_plots(
                    reward_plot=result.plots.get("reward_plot"),
                    action_plot=result.plots.get("action_plot"),
                    equity_curve_plot=result.plots.get("portfolio_value_plot"),
                    artifact_path_prefix=ArtifactPaths.eval_plots(split),
                    plot_data=_plot_data,
                )

        return split_output

    # ------------------------------------------------------------------
    # Arbitrary data preparation
    # ------------------------------------------------------------------

    def _prepare_arbitrary_df(
        self, data_path: Path, config: Any, checkpoint_path: Path | None = None
    ) -> pd.DataFrame:
        """Load a raw parquet file, apply the scenario's feature pipeline, and return a prepared DataFrame.

        Results are cached under ``config.data.feature_cache_dir`` keyed by the file's
        modification time and all preparation settings, so repeated calls on the
        same file are cheap after the first run.

        Args:
            data_path: Path to the eval data file
            config: Experiment config
            checkpoint_path: Path to checkpoint (used to restore training pipeline state)
        """
        import hashlib

        import pandas as pd

        from trading_rl.constants import EnvBackend, EnvMode
        from trading_rl.data.hft import (
            _deduplicate_hft_index_single,
            _derive_close_hft_single,
        )
        from trading_rl.data.loading import load_trading_data

        filter_lob_levels = getattr(config.data, "filter_lob_levels", None)
        feature_config = getattr(config.data, "feature_config", None)
        feature_cache_dir = getattr(config.data, "feature_cache_dir", None)
        mode = str(getattr(config.env, "mode", "mft")).lower().strip()
        backend = str(getattr(config.env, "backend", "")).lower().strip()
        stem = data_path.stem

        # --- Cache check ---
        cache_path: Path | None = None
        if feature_cache_dir and feature_config:
            ckpt_id = ""
            if checkpoint_path is not None and checkpoint_path.exists():
                ckpt_id = str(checkpoint_path.stat().st_mtime_ns)
            sig = hashlib.md5(
                f"{data_path}:{data_path.stat().st_mtime_ns}:{filter_lob_levels}:{feature_config}:{mode}:{backend}:{ckpt_id}".encode(),
                usedforsecurity=False,
            ).hexdigest()
            cache_path = Path(feature_cache_dir) / f"eval_{sig}.parquet"

        if cache_path is not None and cache_path.exists():
            self.console.print(f"[dim]  Loading from cache ({data_path.name})[/dim]")
            return pd.read_parquet(cache_path)

        # --- Compute from scratch ---
        df = load_trading_data(str(data_path)).dropna()
        self.console.print(
            f"[dim]  Loaded {len(df):,} rows from {data_path.name}[/dim]"
        )

        if filter_lob_levels is not None:
            from trading_rl.data.lob_filters import filter_unchanged_lob

            before = len(df)
            df = filter_unchanged_lob(df, levels=filter_lob_levels)
            self.console.print(
                f"[dim]  LOB filter: {before:,} → {len(df):,} rows[/dim]"
            )

        if feature_config:
            from trading_rl.data.loading import build_feature_pipeline_with_state

            restore_result = build_feature_pipeline_with_state(
                feature_config,
                checkpoint_path=checkpoint_path,
            )
            pipeline = restore_result.pipeline

            if restore_result.restored:
                self.console.print(
                    "[dim]  Restored training pipeline state from "
                    f"{restore_result.source} ({restore_result.state_size} features)[/dim]"
                )
            else:
                self.logger.warning(
                    "Pipeline state not available in checkpoint — normalizing eval data with eval statistics. "
                    "Metrics may not reflect true out-of-sample performance. Use --data-path mode only for "
                    "sanity checks; production eval should use prepared splits from training time."
                )
                pipeline.fit(df)

            features = pipeline.transform(df)
            df = pd.concat([df, features], axis=1)
            self.console.print(
                f"[dim]  Features computed: {len(features.columns)} columns[/dim]"
            )

        if mode == EnvMode.HFT:
            df = _derive_close_hft_single(df, stem, self.logger)
        if mode == EnvMode.HFT and backend == EnvBackend.TRADINGENV:
            df = _deduplicate_hft_index_single(df, stem, self.logger)

        if cache_path is not None:
            Path(feature_cache_dir).mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_path)
            self.console.print(f"[dim]  Cached → {cache_path.name}[/dim]")

        return df

    def _resolve_per_symbol_splits(
        self,
        config: Any,
        params: EvaluateParams,
        val_data_paths: list[str],
        checkpoint_path: Path | None = None,
    ) -> tuple[list[str], dict[str, pd.DataFrame]]:
        """Prepare each val file as an independent single-symbol DataFrame.

        Mirrors _build_per_day_splits: splits each file 50/50 so "val" is the
        first half and "test" is the second half. Returns a list of split-keys
        (e.g. ``["test_AAPL", "test_AMZN", ...]``) and the corresponding map.

        Fast path: if prepared_data_dir contains val_{sym}_prepared.parquet and
        test_{sym}_prepared.parquet (written by _build_per_day_splits during
        training), those are used directly — no feature recomputation.
        """
        from pathlib import Path

        import pandas as pd

        from trading_rl.constants import SplitName

        prepared_dir_str = getattr(
            getattr(config, "data", None), "prepared_data_dir", None
        )
        prepared_dir = Path(prepared_dir_str) if prepared_dir_str else None
        validation_size = getattr(
            getattr(config, "data", None), "validation_size", None
        )
        test_size = getattr(getattr(config, "data", None), "test_size", None)

        requested: set[str] = (
            {SplitName.VAL, SplitName.TEST}
            if params.split == "all"
            else {SplitName(params.split)}
        )
        splits_to_eval: list[str] = []
        split_dfs: dict[str, Any] = {}

        for val_path in val_data_paths:
            stem = Path(val_path).stem
            symbol = stem.split("_")[0]

            # Fast path: use already-prepared per-symbol parquets from training.
            val_prepared = (
                prepared_dir / f"val_{symbol}_prepared.parquet"
                if prepared_dir
                else None
            )
            test_prepared = (
                prepared_dir / f"test_{symbol}_prepared.parquet"
                if prepared_dir
                else None
            )

            if (
                val_prepared is not None
                and val_prepared.exists()
                and test_prepared is not None
                and test_prepared.exists()
            ):
                self.console.print(
                    f"[dim]  {symbol}: loading from prepared cache ({val_prepared.parent.name}/)[/dim]"
                )
                if SplitName.VAL in requested:
                    key = f"val_{symbol}"
                    df_val = pd.read_parquet(val_prepared)
                    if validation_size is not None:
                        df_val = df_val.iloc[:validation_size]
                    split_dfs[key] = df_val
                    splits_to_eval.append(key)
                if SplitName.TEST in requested:
                    key = f"test_{symbol}"
                    df_test = pd.read_parquet(test_prepared)
                    if test_size is not None:
                        df_test = df_test.iloc[:test_size]
                    split_dfs[key] = df_test
                    splits_to_eval.append(key)
            else:
                self.console.print(
                    f"[dim]  {symbol}: computing features ({Path(val_path).name})[/dim]"
                )
                df = self._prepare_arbitrary_df(Path(val_path), config, checkpoint_path)
                mid = len(df) // 2
                if SplitName.VAL in requested:
                    key = f"val_{symbol}"
                    split_dfs[key] = df.iloc[:mid].copy()
                    splits_to_eval.append(key)
                if SplitName.TEST in requested:
                    key = f"test_{symbol}"
                    split_dfs[key] = df.iloc[mid:].copy()
                    splits_to_eval.append(key)

        return splits_to_eval, split_dfs

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
            from datetime import UTC, datetime

            import mlflow

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
            self.logger.warning("mlflow unavailable skip run creation err={}", exc)
            return _noop_context()

    def _log_benchmarks_to_mlflow(self, bench_out: dict[str, Any], split: str) -> None:
        try:
            import mlflow
            import numpy as np

            for bench_name, entry in bench_out.items():
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False
                ) as f:
                    json.dump(entry, f, indent=2, default=_json_default)
                    f.flush()
                    mlflow.log_artifact(f.name, f"benchmarks/{split}/{bench_name}")
                    os.unlink(f.name)

                bench_metrics = entry.get("benchmark_metrics", entry)
                rel_metrics = entry.get("relative_metrics", {})
                for key, _, _ in _PERF_ROWS:
                    val = bench_metrics.get(key)
                    if val is not None and np.isfinite(float(val)):
                        mlflow.log_metric(
                            f"bench_{split}_{bench_name}_{key}", float(val)
                        )
                for key in ("alpha", "beta", "information_ratio", "tracking_error"):
                    val = rel_metrics.get(key)
                    if val is not None and np.isfinite(float(val)):
                        mlflow.log_metric(
                            f"bench_{split}_{bench_name}_{key}", float(val)
                        )
        except Exception as exc:
            self.logger.warning("log benchmarks to mlflow failed err={}", exc)

    @staticmethod
    def _mlflow_run_url(tracking_uri: str, run_id: str) -> str:
        if tracking_uri.startswith("sqlite:///"):
            db_path = tracking_uri[len("sqlite:///") :]
            abs_db = Path(db_path).resolve()
            return f"mlflow ui --backend-store-uri sqlite:///{abs_db}  →  run {run_id}"
        return f"{tracking_uri}  →  run {run_id}"

    # ------------------------------------------------------------------
    # Config and checkpoint helpers
    # ------------------------------------------------------------------

    def _load_config(self, params: EvaluateParams) -> Any:
        if params.config_file is None:
            raise ValueError("--config is required for the evaluate command.")

        config = self._load_experiment_config(
            params.config_file, command="evaluate", overrides=params.config_overrides
        )
        self.console.print(f"[dim]Config: {params.config_file}[/dim]")
        return config

    def _resolve_checkpoint(self, config: Any, params: EvaluateParams) -> Path:
        if params.checkpoint is not None:
            if not params.checkpoint.exists():
                raise FileNotFoundError(f"Checkpoint not found: {params.checkpoint}")
            return params.checkpoint

        log_dir = Path(config.logging.log_dir)
        matches = list(log_dir.rglob("*_checkpoint*.pt"))
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

    def _print_metrics_table(
        self,
        split: str,
        metrics: dict[str, float],
        split_df: pd.DataFrame | None = None,
        symbols: list[str] | None = None,
    ) -> None:
        self.display.print_metrics_table(split, metrics, split_df, symbols)

    def _print_benchmark_table(
        self,
        split: str,
        bench_out: dict[str, Any],
        strategy_metrics: dict[str, Any] | None = None,
    ) -> None:
        self.display.print_benchmark_table(split, bench_out, strategy_metrics)

    def _save_rollout_data(
        self,
        result: Any,
        split: str,
        split_df: pd.DataFrame,
        output_dir: Path,
    ) -> None:
        import numpy as np
        import pandas as pd

        n = min(len(result.last_positions), len(result.simple_returns), len(split_df))
        if n == 0:
            self.console.print(f"[yellow]No rollout data to save for {split}[/yellow]")
            return

        index = split_df.index[:n]
        data: dict[str, Any] = {
            "action": np.array(result.last_positions[:n], dtype=np.float32),
            "simple_return": result.simple_returns[:n].astype(np.float32),
        }
        if result.cumulative_returns is not None:
            cum = result.cumulative_returns
            # cumulative_returns is built with include_initial=True → length n+1
            if len(cum) == n + 1:
                cum = cum[1:]
            data["cumulative_log_return"] = cum[:n].astype(np.float32)

        out_df = pd.DataFrame(data, index=index)
        out_path = output_dir / f"{split}_rollout.parquet"
        out_df.to_parquet(out_path)
        write_asset_meta(out_path, generator="cli/commands/evaluate_command.py")
        self.console.print(f"[dim]Rollout data ({n:,} steps) → {out_path}[/dim]")

    def _save_trade_log(self, env: Any, split: str, output_dir: Path) -> None:
        from trading_rl.evaluation.returns import extract_trade_log

        trade_df = extract_trade_log(env)
        if trade_df is None:
            self.console.print(
                f"[yellow]Trade log unavailable for {split} (no broker found)[/yellow]"
            )
            return
        if trade_df.empty:
            self.console.print(f"[dim]  No trades recorded for {split}[/dim]")
            return

        out_path = output_dir / f"{split}_trades.csv"
        trade_df.to_csv(out_path, index=False)
        write_asset_meta(out_path, generator="cli/commands/evaluate_command.py")
        self.console.print(
            f"[dim]  Trade log ({len(trade_df):,} trades) → {out_path}[/dim]"
        )

    def _save_plots(self, plots: dict[str, Any], split: str, output_dir: Path) -> None:
        for name, fig in plots.items():
            if fig is None or name.startswith("_"):
                continue
            out_path = output_dir / f"{split}_{name}.png"
            try:
                if hasattr(fig, "save"):
                    fig.save(str(out_path), dpi=225, verbose=False)
                else:
                    fig.savefig(out_path, bbox_inches="tight", dpi=225)
                write_asset_meta(out_path, generator="cli/commands/evaluate_command.py")
                self.console.print(f"[dim]Saved plot: {out_path}[/dim]")
            except Exception as exc:
                self.logger.warning("save plot failed name={} err={}", name, exc)
                continue
            if hasattr(fig, "data") and fig.data is not None:
                data_path = output_dir / f"{split}_{name}.csv"
                try:
                    fig.data.to_csv(data_path, index=False)
                    write_asset_meta(
                        data_path, generator="cli/commands/evaluate_command.py"
                    )
                except Exception as exc:
                    self.logger.warning(
                        "save plot data failed name={} err={}", name, exc
                    )


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
        mlflow.log_param(
            "eval_actor_hidden_dims", str(self._meta.get("actor_hidden_dims"))
        )
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

    from trading_rl.evaluation.metrics import MetricReport

    if isinstance(obj, MetricReport):
        return obj.to_dict()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)}")
