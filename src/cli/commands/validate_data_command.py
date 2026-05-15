"""validate-data command — run DataValidator checks against a prepared dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import typer
from rich.table import Table

from .base_command import BaseCommand


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

        self._print_checks_table(params)
        self.console.print("[cyan]Running checks…[/cyan]")

        failures: list[tuple[str, str]] = []
        checks = [
            ("empty splits",        lambda: validator.check_empty_splits(dataset.train_df, dataset.val_df, dataset.test_df)),
            ("close column",        lambda: validator.check_close_column(dataset.train_df)),
            ("feature columns",     lambda: validator.check_feature_columns(dataset.train_df)),
            ("env feature prefix",  lambda: validator.check_env_feature_columns_prefix(config)),
        ]
        if params.check_nan:
            checks.append(("NaN values", lambda: validator.check_nan_values(dataset.train_df, dataset.val_df, dataset.test_df)))
        if params.check_inf:
            checks.append(("inf values", lambda: validator.check_inf_values(dataset.train_df, dataset.val_df, dataset.test_df)))
        if params.check_duplicates:
            checks.append(("duplicate index", lambda: validator.check_duplicate_index(dataset.train_df, dataset.val_df, dataset.test_df)))
        if params.check_zero_variance:
            checks.append(("zero-variance features", lambda: validator.check_zero_variance_features(dataset.train_df, dataset.val_df, dataset.test_df)))
        if params.check_lob_deltas:
            checks.append((f"LOB deltas (L{params.lob_levels})", lambda: validator.check_lob_delta(dataset.train_df, dataset.val_df, dataset.test_df)))

        results: list[tuple[str, str, str]] = []
        for name, fn in checks:
            try:
                fn()
                results.append((name, "[green]PASS[/green]", ""))
            except ValueError as exc:
                results.append((name, "[red]FAIL[/red]", str(exc)))
                failures.append((name, str(exc)))

        tbl = Table(title="Data Validation Results", show_header=True, header_style="bold")
        tbl.add_column("Check")
        tbl.add_column("Result", justify="center")
        tbl.add_column("Detail")
        for name, status, detail in results:
            tbl.add_row(name, status, detail)
        self.console.print(tbl)

        if failures:
            self.console.print(f"\n[red bold]{len(failures)} check(s) failed.[/red bold]")
            raise typer.Exit(1)
        else:
            self.console.print("\n[green bold]All checks passed.[/green bold]")

    def _print_checks_table(self, params: ValidateDataParams) -> None:
        tbl = Table(title="Enabled Checks", show_header=True, header_style="bold dim")
        tbl.add_column("Check")
        tbl.add_column("Enabled", justify="center")
        rows = [
            ("NaN values",              params.check_nan),
            ("Inf values",              params.check_inf),
            ("Duplicate index",         params.check_duplicates),
            ("Zero-variance features",  params.check_zero_variance),
            (f"LOB deltas (L{params.lob_levels})", params.check_lob_deltas),
        ]
        for name, enabled in rows:
            tbl.add_row(name, "[green]yes[/green]" if enabled else "[dim]no[/dim]")
        self.console.print(tbl)
