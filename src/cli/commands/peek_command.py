"""Peek command — dataset and feature statistics summary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import typer
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .base_command import BaseCommand


@dataclass
class PeekParams:
    scenario: str | None = None
    config_file: Path | None = None
    config_override: list[str] | None = None
    n_features: int = 20
    skip_rows: int = 0


class PeekCommand(BaseCommand):
    """Show a prepared dataset summary: splits, feature stats, memmap inventory."""

    def execute(self, params: PeekParams) -> None:
        from logger import get_logger as _get_logger
        from trading_rl import ExperimentConfig
        from trading_rl.data_utils import build_prepared_dataset

        _log = _get_logger(__name__)

        if params.scenario and params.config_file:
            raise typer.BadParameter("Cannot specify both --config and --scenario.")
        if not params.scenario and not params.config_file:
            raise typer.BadParameter("Provide --scenario or --config.")

        if params.scenario:
            search = [
                Path(params.scenario),
                Path("src/configs/scenarios") / params.scenario,
                Path("src/configs/scenarios") / f"{params.scenario}.yaml",
            ]
            config_path = next((p for p in search if p.exists()), None)
            if config_path is None:
                raise typer.BadParameter(f"Scenario '{params.scenario}' not found.")
        else:
            config_path = params.config_file

        if config_path.is_dir():
            config = ExperimentConfig.from_scenario(config_path, overrides=params.config_override)
        else:
            config = ExperimentConfig.from_yaml(config_path, overrides=params.config_override)
        dataset = build_prepared_dataset(config, _log)

        detected_warmup = self._detect_warmup(config)
        effective_skip = params.skip_rows if params.skip_rows else detected_warmup

        self.console.print(Panel(Text(str(config_path), style="bold cyan"), title="scenario", expand=False))
        self._print_splits(dataset)
        self._print_feature_stats(dataset, config, params.n_features, effective_skip, detected_warmup, params.skip_rows)
        self._print_log_return_stats(dataset, config, effective_skip)
        self._print_memmaps(dataset)

    # ------------------------------------------------------------------

    def _detect_warmup(self, config) -> int:
        detected = 0
        feature_config_path = getattr(config.data, "feature_config", None)
        if not feature_config_path:
            return detected
        try:
            from trading_rl.features.pipeline import FeaturePipeline
            pipeline = FeaturePipeline.from_yaml(feature_config_path)
            for fc in pipeline.feature_configs:
                for key in ("window", "period", "slow_period", "long_period"):
                    val = fc.params.get(key)
                    if isinstance(val, int):
                        detected = max(detected, val)
                detected = max(detected, fc.rolling_window or 0)
        except Exception:
            pass
        return detected

    def _print_splits(self, dataset) -> None:
        tbl = Table(title="Splits", show_header=True, header_style="bold")
        tbl.add_column("split")
        tbl.add_column("rows", justify="right")
        tbl.add_column("columns", justify="right")
        tbl.add_column("first timestamp")
        tbl.add_column("last timestamp")
        for name, df in [("train", dataset.train_df), ("val", dataset.val_df), ("test", dataset.test_df)]:
            first = str(df.index[0]) if len(df) else "—"
            last = str(df.index[-1]) if len(df) else "—"
            tbl.add_row(name, f"{len(df):,}", str(df.shape[1]), first, last)
        self.console.print(tbl)

    def _print_feature_stats(self, dataset, config, n_features: int, effective_skip: int, detected_warmup: int, skip_rows: int) -> None:
        feat_cols = dataset.feature_columns
        env_selected = list(getattr(config.env, "feature_columns", None) or feat_cols)
        train = dataset.train_df[feat_cols].iloc[effective_skip:]

        warmup_note = f"detected warm-up={detected_warmup}"
        if skip_rows:
            warmup_note += f", overridden to {skip_rows}"
        skip_note = f", skip={effective_skip:,} ({warmup_note})" if effective_skip else ""

        tbl = Table(
            title=f"Feature statistics (train{skip_note}, top {min(n_features, len(feat_cols))} of {len(feat_cols)})",
            show_header=True,
            header_style="bold",
        )
        tbl.add_column("feature")
        tbl.add_column("selected", justify="center")
        tbl.add_column("mean", justify="right")
        tbl.add_column("std", justify="right")
        tbl.add_column("min", justify="right")
        tbl.add_column("max", justify="right")
        tbl.add_column("nulls", justify="right")

        selected_set = set(env_selected)
        desc = train.describe().T
        for col in feat_cols[:n_features]:
            row = desc.loc[col]
            null_count = int(train[col].isnull().sum())
            tick = "[green]yes[/green]" if col in selected_set else ""
            tbl.add_row(
                col, tick,
                f"{row['mean']:.4f}", f"{row['std']:.4f}",
                f"{row['min']:.4f}", f"{row['max']:.4f}",
                str(null_count) if null_count else "[green]0[/green]",
            )
        self.console.print(tbl)

        if len(feat_cols) > n_features:
            self.console.print(f"  [dim]… {len(feat_cols) - n_features} more features hidden (use --top {len(feat_cols)} to show all)[/dim]")

    def _print_log_return_stats(self, dataset, config, effective_skip: int) -> None:
        import numpy as np
        from trading_rl.data_utils import load_trading_data

        price_col = getattr(config.env, "price_column", "close")

        raw_df = load_trading_data(config.data.data_path).dropna()
        train_size = len(dataset.train_df) + effective_skip
        raw_train = raw_df.iloc[:train_size].iloc[effective_skip:]

        if price_col not in raw_train.columns:
            return

        prices = raw_train[price_col].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_rets = np.diff(np.log(prices))
        log_rets = log_rets[np.isfinite(log_rets)]

        if len(log_rets) == 0:
            return

        reward_type = getattr(config.env, "reward_type", "unknown")
        tbl = Table(
            title=f"Price log-returns on train (always-long proxy, price_column='{price_col}', reward_type='{reward_type}')",
            show_header=True,
            header_style="bold",
        )
        for col in ("n_steps", "mean", "std", "min", "p5", "p25", "p50", "p75", "p95", "max"):
            tbl.add_column(col, justify="right")

        p = np.percentile(log_rets, [5, 25, 50, 75, 95])
        tbl.add_row(
            f"{len(log_rets):,}",
            f"{log_rets.mean():.6f}",
            f"{log_rets.std():.6f}",
            f"{log_rets.min():.6f}",
            f"{p[0]:.6f}",
            f"{p[1]:.6f}",
            f"{p[2]:.6f}",
            f"{p[3]:.6f}",
            f"{p[4]:.6f}",
            f"{log_rets.max():.6f}",
        )
        self.console.print(tbl)

    def _print_memmaps(self, dataset) -> None:
        if not dataset.memmap_train_paths:
            return
        tbl = Table(title="Memmap files", show_header=True, header_style="bold")
        tbl.add_column("idx", justify="right")
        tbl.add_column("rows", justify="right")
        tbl.add_column("cols", justify="right")
        tbl.add_column("size")
        tbl.add_column("file")
        for idx, mp in enumerate(dataset.memmap_train_paths):
            size_mb = mp.data_path.stat().st_size / 1_048_576
            tbl.add_row(str(idx), f"{mp.n_rows:,}", str(len(mp.columns)), f"{size_mb:.1f} MB", mp.data_path.name)
        self.console.print(tbl)
