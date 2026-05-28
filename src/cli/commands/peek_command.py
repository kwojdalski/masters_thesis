"""Peek command — dataset and feature statistics summary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import typer
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from trading_rl.evaluation.asset_meta import write_asset_meta
from .base_command import BaseCommand


@dataclass
class PeekParams:
    scenario: str | None = None
    config_file: Path | None = None
    config_override: list[str] | None = None
    n_features: int = 20
    skip_rows: int = 0
    show_correlations: bool = False
    export: bool = False


class PeekCommand(BaseCommand):
    """Show a prepared dataset summary: splits, feature stats, memmap inventory."""

    def execute(self, params: PeekParams) -> None:
        import pandas as pd
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

        config = ExperimentConfig.load(config_path, overrides=params.config_override)
        dataset = build_prepared_dataset(config, _log)

        detected_warmup = self._detect_warmup(config)
        effective_skip = params.skip_rows if params.skip_rows else detected_warmup

        export_dir: Path | None = None
        if params.export:
            scenario_name = config_path.stem if config_path.is_file() else config_path.name
            export_dir = Path("reports/peek") / scenario_name
            export_dir.mkdir(parents=True, exist_ok=True)

        self.console.print(Panel(Text(str(config_path), style="bold cyan"), title="scenario", expand=False))
        splits_df = self._print_splits(dataset)
        feat_df = self._print_feature_stats(dataset, config, params.n_features, effective_skip, detected_warmup, params.skip_rows)
        ret_df = self._print_log_return_stats(dataset, config, effective_skip)
        corr_df: pd.DataFrame | None = None
        if params.show_correlations:
            corr_df = self._print_reward_correlations(dataset, config, effective_skip)
        inventory_df = self._print_raw_file_inventory(config)

        if export_dir is not None:
            import json
            splits_records = splits_df.to_dict(orient="records")
            _splits_path = export_dir / "splits.json"
            _splits_path.write_text(
                json.dumps(splits_records, indent=2, default=str), encoding="utf-8"
            )
            write_asset_meta(_splits_path, generator="cli/commands/peek_command.py")
            if inventory_df is not None:
                _inv_path = export_dir / "raw_file_inventory.json"
                _inv_path.write_text(
                    json.dumps(inventory_df.to_dict(orient="records"), indent=2), encoding="utf-8"
                )
                write_asset_meta(_inv_path, generator="cli/commands/peek_command.py")
            _feat_path = export_dir / "feature_stats.csv"
            feat_df.to_csv(_feat_path, index=False)
            write_asset_meta(_feat_path, generator="cli/commands/peek_command.py")
            if ret_df is not None:
                _ret_path = export_dir / "log_return_stats.csv"
                ret_df.to_csv(_ret_path, index=False)
                write_asset_meta(_ret_path, generator="cli/commands/peek_command.py")
            if corr_df is not None:
                _corr_path = export_dir / "correlations.csv"
                corr_df.to_csv(_corr_path, index=False)
                write_asset_meta(_corr_path, generator="cli/commands/peek_command.py")
            self.console.print(f"\n[green]Exported to {export_dir}/[/green]")

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

    def _print_splits(self, dataset) -> "pd.DataFrame":
        import numpy as np
        import pandas as pd

        def _delta_stats(df: "pd.DataFrame") -> tuple[float | None, float | None]:
            if not isinstance(df.index, pd.DatetimeIndex) or len(df) < 2:
                return None, None
            deltas = df.index.to_series().diff().dropna().dt.total_seconds().to_numpy()
            positive = deltas[deltas > 0]
            if positive.size == 0:
                return None, None
            return float(np.mean(positive)), float(np.median(positive))

        rows = []
        tbl = Table(title="Splits", show_header=True, header_style="bold")
        tbl.add_column("split")
        tbl.add_column("rows", justify="right")
        tbl.add_column("columns", justify="right")
        tbl.add_column("first timestamp")
        tbl.add_column("last timestamp")
        tbl.add_column("mean Δt", justify="right")
        tbl.add_column("median Δt", justify="right")
        for name, df in [("train", dataset.train_df), ("val", dataset.val_df), ("test", dataset.test_df)]:
            first = str(df.index[0]) if len(df) else ""
            last = str(df.index[-1]) if len(df) else ""
            mean_s, median_s = _delta_stats(df)

            def _fmt(s: float | None) -> str:
                if s is None:
                    return ""
                if s < 1e-3:
                    return f"{s * 1e6:.1f} µs"
                if s < 1.0:
                    return f"{s * 1e3:.2f} ms"
                return f"{s:.3f} s"

            tbl.add_row(name, f"{len(df):,}", str(df.shape[1]), first, last, _fmt(mean_s), _fmt(median_s))
            rows.append({
                "split": name,
                "rows": int(len(df)),
                "columns": int(df.shape[1]),
                "first_timestamp": first,
                "last_timestamp": last,
                "mean_delta_s": mean_s,
                "median_delta_s": median_s,
            })
        self.console.print(tbl)
        return pd.DataFrame(rows)

    def _print_feature_stats(self, dataset, config, n_features: int, effective_skip: int, detected_warmup: int, skip_rows: int) -> "pd.DataFrame":
        import pandas as pd

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
        tbl.add_column("skew", justify="right")
        tbl.add_column("kurt", justify="right")
        tbl.add_column("Q1", justify="right")
        tbl.add_column("Q2", justify="right")
        tbl.add_column("Q3", justify="right")
        tbl.add_column("min", justify="right")
        tbl.add_column("max", justify="right")
        tbl.add_column("nulls", justify="right")

        selected_set = set(env_selected)
        desc = train.describe(percentiles=[0.25, 0.50, 0.75]).T
        skewness = train.skew()
        kurtosis = train.kurt()
        export_rows = []
        for col in feat_cols[:n_features]:
            row = desc.loc[col]
            null_count = int(train[col].isnull().sum())
            skew_val = float(skewness[col])
            kurt_val = float(kurtosis[col])
            tick = "[green]yes[/green]" if col in selected_set else ""
            tbl.add_row(
                col, tick,
                f"{row['mean']:.4f}", f"{row['std']:.4f}",
                f"{skew_val:.3f}", f"{kurt_val:.3f}",
                f"{row['25%']:.4f}", f"{row['50%']:.4f}", f"{row['75%']:.4f}",
                f"{row['min']:.4f}", f"{row['max']:.4f}",
                str(null_count) if null_count else "[green]0[/green]",
            )
            export_rows.append({
                "feature": col,
                "selected": col in selected_set,
                "mean": float(row["mean"]),
                "std": float(row["std"]),
                "skew": skew_val,
                "kurt": kurt_val,
                "q1": float(row["25%"]),
                "q2": float(row["50%"]),
                "q3": float(row["75%"]),
                "min": float(row["min"]),
                "max": float(row["max"]),
                "nulls": null_count,
            })
        self.console.print(tbl)

        if len(feat_cols) > n_features:
            self.console.print(f"  [dim]… {len(feat_cols) - n_features} more features hidden (use --top {len(feat_cols)} to show all)[/dim]")

        return pd.DataFrame(export_rows)

    def _print_log_return_stats(self, dataset, config, effective_skip: int) -> "pd.DataFrame | None":
        import numpy as np
        import pandas as pd
        from trading_rl.data_utils import load_trading_data

        price_col = getattr(config.env, "price_column", "close")

        raw_df = load_trading_data(config.data.data_path).dropna()
        train_size = len(dataset.train_df) + effective_skip
        raw_train = raw_df.iloc[:train_size].iloc[effective_skip:]

        if price_col not in raw_train.columns:
            if {"ask_px_00", "bid_px_00"}.issubset(raw_train.columns):
                raw_train = raw_train.copy()
                raw_train[price_col] = ((raw_train["ask_px_00"] + raw_train["bid_px_00"]) / 2.0).ffill().bfill()
            else:
                return None

        prices = raw_train[price_col].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_rets = np.diff(np.log(prices))
        log_rets = log_rets[np.isfinite(log_rets)]

        if len(log_rets) == 0:
            return None

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
        return pd.DataFrame([{
            "price_column": price_col,
            "reward_type": reward_type,
            "n_steps": len(log_rets),
            "mean": float(log_rets.mean()),
            "std": float(log_rets.std()),
            "min": float(log_rets.min()),
            "p5": float(p[0]),
            "p25": float(p[1]),
            "p50": float(p[2]),
            "p75": float(p[3]),
            "p95": float(p[4]),
            "max": float(log_rets.max()),
        }])

    def _print_reward_correlations(self, dataset, config, effective_skip: int) -> "pd.DataFrame | None":
        import numpy as np
        import pandas as pd
        from scipy.stats import spearmanr
        from trading_rl.data_utils import load_trading_data

        price_col = getattr(config.env, "price_column", "close")
        feat_cols = list(getattr(config.env, "feature_columns", None) or dataset.feature_columns)

        raw_df = load_trading_data(config.data.data_path).dropna()
        train_size = len(dataset.train_df) + effective_skip
        raw_train = raw_df.iloc[:train_size].iloc[effective_skip:]

        if price_col not in raw_train.columns:
            if {"ask_px_00", "bid_px_00"}.issubset(raw_train.columns):
                raw_train = raw_train.copy()
                raw_train[price_col] = ((raw_train["ask_px_00"] + raw_train["bid_px_00"]) / 2.0).ffill().bfill()
            else:
                self.console.print("[yellow]Cannot compute correlations: price column not found.[/yellow]")
                return None

        prices = raw_train[price_col].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_rets = np.diff(np.log(prices))

        available = [c for c in feat_cols if c in dataset.train_df.columns]
        feat_df = dataset.train_df[available].iloc[effective_skip:]
        # Align: log_rets has one fewer row than prices
        n = min(len(log_rets), len(feat_df) - 1)
        log_rets = log_rets[:n]
        feat_aligned = feat_df.iloc[1:n + 1]

        rows = []
        for col in feat_cols:
            if col not in feat_aligned.columns:
                continue
            f = feat_aligned[col].to_numpy(dtype=float)
            mask = np.isfinite(f) & np.isfinite(log_rets)
            if mask.sum() < 10:
                pearson, spearman = float("nan"), float("nan")
            else:
                pearson = float(np.corrcoef(f[mask], log_rets[mask])[0, 1])
                spearman = float(spearmanr(f[mask], log_rets[mask]).statistic)
            rows.append((col, pearson, spearman))

        rows.sort(key=lambda r: abs(r[1]) if np.isfinite(r[1]) else 0, reverse=True)

        tbl = Table(
            title="Feature vs log-return correlations (train, sorted by |Pearson|)",
            show_header=True,
            header_style="bold",
        )
        tbl.add_column("feature")
        tbl.add_column("Pearson", justify="right")
        tbl.add_column("Spearman", justify="right")

        def _fmt(v: float) -> str:
            if not np.isfinite(v):
                return "n/a"
            color = "green" if abs(v) > 0.05 else ("yellow" if abs(v) > 0.01 else "red")
            return f"[{color}]{v:+.4f}[/{color}]"

        for col, p, s in rows:
            tbl.add_row(col, _fmt(p), _fmt(s))

        self.console.print(tbl)
        return pd.DataFrame([{"feature": col, "pearson": p, "spearman": s} for col, p, s in rows])

    def _print_raw_file_inventory(self, config) -> "pd.DataFrame | None":
        """Build and print a symbol × split table with event counts and delta stats."""
        import numpy as np
        import pandas as pd

        train_paths = list(getattr(config.data, "data_paths", None) or [])
        val_paths = list(getattr(config.data, "val_data_paths", None) or [])
        if not train_paths and not val_paths:
            return None

        def _symbol_of(p: str) -> str:
            return Path(p).name.split("_")[0]

        def _read_file(p: str) -> "pd.DataFrame":
            return pd.read_parquet(p, columns=["action"])

        def _delta_stats(deltas_s: np.ndarray) -> tuple[float | None, float | None]:
            pos = deltas_s[deltas_s > 0]
            if pos.size == 0:
                return None, None
            return float(np.mean(pos)), float(np.median(pos))

        rows = []

        # Train: pool within-file deltas per symbol (exclude overnight gaps)
        by_sym: dict[str, list[str]] = {}
        for p in train_paths:
            by_sym.setdefault(_symbol_of(p), []).append(p)

        for sym in sorted(by_sym):
            total_rows = 0
            total_trades = 0
            all_deltas: list[np.ndarray] = []
            for p in by_sym[sym]:
                try:
                    df = _read_file(p)
                    total_rows += len(df)
                    total_trades += int((df["action"] == "T").sum())
                    if len(df) >= 2:
                        d = np.diff(df.index.view(np.int64)) / 1e9
                        all_deltas.append(d)
                except Exception:
                    pass
            combined = np.concatenate(all_deltas) if all_deltas else np.array([])
            mean_s, median_s = _delta_stats(combined)
            rows.append({"symbol": sym, "split": "train", "rows": total_rows,
                         "trades": total_trades, "mean_delta_s": mean_s, "median_delta_s": median_s})

        # Val + test: each val file split 50/50 at midpoint
        for p in sorted(val_paths, key=_symbol_of):
            sym = _symbol_of(p)
            try:
                df = _read_file(p)
            except Exception:
                continue
            mid = len(df) // 2
            for split_name, split_df in [("val", df.iloc[:mid]), ("test", df.iloc[mid:])]:
                if len(split_df) >= 2:
                    d = np.diff(split_df.index.view(np.int64)) / 1e9
                    mean_s, median_s = _delta_stats(d)
                else:
                    mean_s, median_s = None, None
                rows.append({"symbol": sym, "split": split_name, "rows": len(split_df),
                             "trades": int((split_df["action"] == "T").sum()),
                             "mean_delta_s": mean_s, "median_delta_s": median_s})

        if not rows:
            return None

        def _fmt(s: float | None) -> str:
            if s is None:
                return "—"
            if s < 1e-3:
                return f"{s * 1e6:.1f} µs"
            if s < 1.0:
                return f"{s * 1e3:.2f} ms"
            return f"{s:.3f} s"

        flat_df = pd.DataFrame(rows)
        tbl = Table(title="Raw data inventory (events per symbol per split)", show_header=True, header_style="bold")
        tbl.add_column("Symbol")
        tbl.add_column("Split")
        tbl.add_column("Events", justify="right")
        tbl.add_column("Trades", justify="right")
        tbl.add_column("Mean Δt", justify="right")
        tbl.add_column("Median Δt", justify="right")
        for _, r in flat_df.iterrows():
            tbl.add_row(r["symbol"], r["split"], f"{int(r['rows']):,}",
                        f"{int(r['trades']):,}",
                        _fmt(r["mean_delta_s"]), _fmt(r["median_delta_s"]))
        self.console.print(tbl)
        return flat_df
