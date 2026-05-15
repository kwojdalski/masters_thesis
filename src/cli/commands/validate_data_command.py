"""validate-data command — run DataValidator checks against a prepared dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import typer
from rich.table import Table

from .base_command import BaseCommand

_CHECK_DESCRIPTIONS: dict[str, str] = {
    "empty splits":          "Each split (train/val/test) must contain at least one row.",
    "close column":          "A 'close' column must be present — used as the pricing signal by the environment.",
    "feature columns":       "At least one 'feature_*' column must exist in the prepared data.",
    "env feature prefix":    "All columns listed in env.feature_columns must start with 'feature_'.",
    "NaN values":            "No NaN (missing) values are allowed in any split — they break normalisation and cause silent gradient corruption.",
    "inf values":            "No infinite values are allowed — they cause loss NaN and kill training immediately.",
    "duplicate index":       "No two rows may share the same timestamp index — duplicates indicate a data pipeline bug.",
    "zero-variance features":"No feature column may be constant across a split — std=0 causes division-by-zero in z-score normalisation.",
    "LOB deltas":            "Every row must differ from the previous row in at least one price or size field across the tracked LOB levels — fully unchanged books are stale ticks that filter_unchanged_lob() should have removed.",
}


@dataclass
class ValidateDataParams:
    scenario: str | None = None
    config_file: Path | None = None
    config_override: list[str] | None = None
    check_nan: bool = True
    check_inf: bool = True
    check_duplicates: bool = True
    check_zero_variance: bool = True
    check_lob_deltas: bool = True
    lob_levels: int = 5
    verbose: bool = False


class ValidateDataCommand(BaseCommand):
    """Run DataValidator checks against the prepared dataset for a scenario."""

    def execute(self, params: ValidateDataParams) -> None:
        from logger import get_logger as _get_logger
        from trading_rl.data import build_prepared_dataset
        from trading_rl.data.validation import DataValidator

        _log = _get_logger(__name__)

        if params.scenario and params.config_file:
            raise typer.BadParameter("Cannot specify both --config and --scenario.")
        if not params.scenario and not params.config_file:
            raise typer.BadParameter("Provide --scenario or --config.")

        source = params.scenario or params.config_file
        config = self._load_experiment_config(source, overrides=params.config_override)

        self.console.print("[cyan]Building prepared dataset…[/cyan]")
        dataset = build_prepared_dataset(config, _log)

        validator = DataValidator(
            check_nan=params.check_nan,
            check_inf=params.check_inf,
            check_duplicates=params.check_duplicates,
            check_zero_variance=params.check_zero_variance,
            check_lob_deltas=params.check_lob_deltas,
            lob_levels=params.lob_levels,
        )

        if params.verbose:
            self._print_data_glimpse(dataset)

        self._print_checks_table(params)
        self.console.print("[cyan]Running checks…[/cyan]")

        lob_label = f"LOB deltas (L{params.lob_levels})"
        checks: list[tuple[str, str, object]] = [
            ("empty splits",       "empty splits",       lambda: validator.check_empty_splits(dataset.train_df, dataset.val_df, dataset.test_df)),
            ("close column",       "close column",       lambda: validator.check_close_column(dataset.train_df)),
            ("feature columns",    "feature columns",    lambda: validator.check_feature_columns(dataset.train_df)),
            ("env feature prefix", "env feature prefix", lambda: validator.check_env_feature_columns_prefix(config)),
        ]
        if params.check_nan:
            checks.append(("NaN values", "NaN values", lambda: validator.check_nan_values(dataset.train_df, dataset.val_df, dataset.test_df)))
        if params.check_inf:
            checks.append(("inf values", "inf values", lambda: validator.check_inf_values(dataset.train_df, dataset.val_df, dataset.test_df)))
        if params.check_duplicates:
            checks.append(("duplicate index", "duplicate index", lambda: validator.check_duplicate_index(dataset.train_df, dataset.val_df, dataset.test_df)))
        if params.check_zero_variance:
            checks.append(("zero-variance features", "zero-variance features", lambda: validator.check_zero_variance_features(dataset.train_df, dataset.val_df, dataset.test_df)))
        if params.check_lob_deltas:
            checks.append((lob_label, "LOB deltas", lambda: validator.check_lob_delta(dataset.train_df, dataset.val_df, dataset.test_df)))

        results: list[tuple[str, str, str, str]] = []
        failures: list[tuple[str, str]] = []
        for label, desc_key, fn in checks:
            try:
                fn()
                results.append((label, "[green]PASS[/green]", "", _CHECK_DESCRIPTIONS.get(desc_key, "")))
            except ValueError as exc:
                results.append((label, "[red]FAIL[/red]", str(exc), _CHECK_DESCRIPTIONS.get(desc_key, "")))
                failures.append((label, str(exc)))

        tbl = Table(title="Data Validation Results", show_header=True, header_style="bold")
        tbl.add_column("Check")
        tbl.add_column("Result", justify="center")
        tbl.add_column("Detail")
        if params.verbose:
            tbl.add_column("Description", style="dim")
        for label, status, detail, description in results:
            row = [label, status, detail]
            if params.verbose:
                row.append(description)
            tbl.add_row(*row)
        self.console.print(tbl)

        if failures:
            self.console.print(f"\n[red bold]{len(failures)} check(s) failed.[/red bold]")
            raise typer.Exit(1)
        else:
            self.console.print("\n[green bold]All checks passed.[/green bold]")

    def _print_data_glimpse(self, dataset, n_rows: int = 5, max_cols: int = 10) -> None:
        import numpy as np

        for split_name, df in [("train", dataset.train_df), ("val", dataset.val_df), ("test", dataset.test_df)]:
            head = df.head(n_rows)
            visible = list(head.columns[:max_cols])
            hidden = max(0, len(head.columns) - len(visible))

            tbl = Table(
                title=f"Data Glimpse — {split_name} ({len(df):,} rows × {df.shape[1]} cols)",
                show_header=True,
                header_style="bold",
            )
            tbl.add_column("index", style="dim")
            for col in visible:
                tbl.add_column(str(col), justify="right")

            for idx, row in head.iterrows():
                cells = [str(idx)]
                for col in visible:
                    val = row[col]
                    if isinstance(val, float):
                        cells.append("nan" if np.isnan(val) else f"{val:.4g}")
                    else:
                        cells.append(str(val))
                tbl.add_row(*cells)

            self.console.print(tbl)
            if hidden:
                self.console.print(f"  [dim]… {hidden} more columns hidden[/dim]")

    def _print_checks_table(self, params: ValidateDataParams) -> None:
        tbl = Table(title="Enabled Checks", show_header=True, header_style="bold dim")
        tbl.add_column("Check")
        tbl.add_column("Enabled", justify="center")
        if params.verbose:
            tbl.add_column("Description", style="dim")
        rows = [
            ("NaN values",                         params.check_nan,          "NaN values"),
            ("Inf values",                         params.check_inf,          "inf values"),
            ("Duplicate index",                    params.check_duplicates,   "duplicate index"),
            ("Zero-variance features",             params.check_zero_variance,"zero-variance features"),
            (f"LOB deltas (L{params.lob_levels})", params.check_lob_deltas,   "LOB deltas"),
        ]
        for name, enabled, desc_key in rows:
            cells = [name, "[green]yes[/green]" if enabled else "[dim]no[/dim]"]
            if params.verbose:
                cells.append(_CHECK_DESCRIPTIONS.get(desc_key, ""))
            tbl.add_row(*cells)
        self.console.print(tbl)
