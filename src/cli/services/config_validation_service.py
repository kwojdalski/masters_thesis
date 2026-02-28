"""Validation service for experiment config and data dependencies."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from trading_rl import ExperimentConfig
from trading_rl.data_utils import load_trading_data
from trading_rl.features import FeaturePipeline, create_default_pipeline

_VALID_ENV_MODES = {"mft", "hft"}
_VALID_FEATURE_DOMAINS = {"shared", "mft", "hft"}


@dataclass
class ValidationIssue:
    """Single validation finding."""

    code: str
    severity: Literal["error", "warning"]
    check: str
    message: str


@dataclass
class ValidationReport:
    """Collection of validation findings."""

    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(issue.severity == "error" for issue in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(issue.severity == "warning" for issue in self.issues)

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "warning")

    def add(
        self,
        *,
        code: str,
        severity: Literal["error", "warning"],
        check: str,
        message: str,
    ) -> None:
        self.issues.append(
            ValidationIssue(code=code, severity=severity, check=check, message=message)
        )


def _validate_base_config(config: ExperimentConfig, report: ValidationReport) -> None:
    try:
        ExperimentConfig.from_dict(config.to_dict())
    except Exception as exc:  # pragma: no cover - defensive
        report.add(
            code="CONFIG_SCHEMA_INVALID",
            severity="error",
            check="base_config",
            message=str(exc),
        )


def _validate_paths(config: ExperimentConfig, report: ValidationReport) -> None:
    data_path = Path(config.data.data_path)
    if not data_path.exists():
        report.add(
            code="DATA_PATH_MISSING",
            severity="warning",
            check="paths",
            message=f"Dataset file does not exist: {data_path}",
        )

    feature_cfg = getattr(config.data, "feature_config", None)
    if feature_cfg:
        feature_path = Path(feature_cfg)
        if not feature_path.exists():
            report.add(
                code="FEATURE_CONFIG_MISSING",
                severity="error",
                check="paths",
                message=f"Feature config file does not exist: {feature_path}",
            )


def _build_feature_pipeline(config: ExperimentConfig, report: ValidationReport):
    feature_cfg = getattr(config.data, "feature_config", None)
    try:
        if feature_cfg:
            return FeaturePipeline.from_yaml(feature_cfg)
        return create_default_pipeline()
    except Exception as exc:
        report.add(
            code="FEATURE_PIPELINE_INVALID",
            severity="error",
            check="feature_pipeline",
            message=str(exc),
        )
        return None


def _validate_split_feasibility(
    config: ExperimentConfig, dataset_len: int, report: ValidationReport
) -> None:
    train_size = int(config.data.train_size)
    if train_size >= dataset_len:
        report.add(
            code="TRAIN_SPLIT_INVALID",
            severity="error",
            check="split",
            message=(
                f"train_size ({train_size}) must be smaller than dataset length "
                f"({dataset_len})"
            ),
        )
        return

    remaining = dataset_len - train_size
    validation_size = config.data.validation_size
    if validation_size is None:
        validation_size = remaining // 2
    if validation_size < 0:
        report.add(
            code="VALIDATION_SPLIT_NEGATIVE",
            severity="error",
            check="split",
            message=f"validation_size must be >= 0, got {validation_size}",
        )
        return
    if validation_size >= remaining:
        report.add(
            code="VALIDATION_SPLIT_INVALID",
            severity="error",
            check="split",
            message=(
                f"validation_size ({validation_size}) must be smaller than "
                f"remaining samples after train split ({remaining})"
            ),
        )


def _validate_env_columns(
    config: ExperimentConfig,
    available_columns: set[str],
    report: ValidationReport,
) -> None:
    price_column = getattr(config.env, "price_column", None)
    if price_column and price_column not in available_columns:
        mode = str(getattr(config.env, "mode", "mft")).lower().strip()
        can_derive_hft_close = (
            mode == "hft"
            and str(price_column) == "close"
            and {"ask_px_00", "bid_px_00"}.issubset(available_columns)
        )
        if can_derive_hft_close:
            # Align validation with training behavior where close is derived
            # from top-of-book midpoint for HFT datasets.
            return

        report.add(
            code="PRICE_COLUMN_MISSING",
            severity="error",
            check="env_columns",
            message=(
                "env.price_column contains unknown column: "
                f"{price_column}. Available: {sorted(available_columns)}"
            ),
        )

    feature_columns = getattr(config.env, "feature_columns", None)
    if feature_columns:
        missing_feat = sorted(set(feature_columns) - available_columns)
        if missing_feat:
            report.add(
                code="FEATURE_COLUMNS_MISSING",
                severity="error",
                check="env_columns",
                message=(
                    "env.feature_columns contains unknown columns: "
                    f"{missing_feat}. Available: {sorted(available_columns)}"
                ),
            )
        non_feature_prefixed = [
            col for col in feature_columns if not str(col).startswith("feature_")
        ]
        if non_feature_prefixed:
            report.add(
                code="FEATURE_COLUMNS_NOT_FEATURE_PREFIXED",
                severity="error",
                check="env_columns",
                message=(
                    "env.feature_columns must contain only feature pipeline outputs "
                    "(feature_*). Found non-feature columns: "
                    f"{non_feature_prefixed}"
                ),
            )


def _feature_output_name(feature_cfg) -> str:
    output_name = getattr(feature_cfg, "output_name", None)
    if output_name:
        return str(output_name)
    name = getattr(feature_cfg, "name", "unnamed")
    return f"feature_{name}"


def _feature_domain(feature_cfg) -> str:
    return str(getattr(feature_cfg, "domain", "shared")).lower().strip()


def _validate_feature_domains(
    config: ExperimentConfig,
    pipeline: FeaturePipeline,
    report: ValidationReport,
) -> None:
    mode = str(getattr(config.env, "mode", "mft")).lower().strip()
    if mode not in _VALID_ENV_MODES:
        report.add(
            code="ENV_MODE_INVALID",
            severity="error",
            check="feature_mode",
            message=(
                f"env.mode must be one of {sorted(_VALID_ENV_MODES)}, got '{mode}'."
            ),
        )
        return

    hft_features: list[str] = []
    mft_features: list[str] = []
    invalid_domain_features: list[str] = []

    for feature_cfg in pipeline.feature_configs:
        domain = _feature_domain(feature_cfg)
        output_name = _feature_output_name(feature_cfg)

        if domain not in _VALID_FEATURE_DOMAINS:
            invalid_domain_features.append(f"{output_name}({domain})")
            continue
        if domain == "hft":
            hft_features.append(output_name)
        elif domain == "mft":
            mft_features.append(output_name)

    if invalid_domain_features:
        report.add(
            code="FEATURE_DOMAIN_INVALID",
            severity="error",
            check="feature_mode",
            message=(
                "Feature config contains invalid domain values. "
                "Use one of ['shared', 'mft', 'hft']. Invalid entries: "
                f"{invalid_domain_features}"
            ),
        )

    if mode == "mft" and hft_features:
        report.add(
            code="MFT_HAS_HFT_FEATURES",
            severity="error",
            check="feature_mode",
            message=(
                "env.mode='mft' cannot include features tagged as domain='hft'. "
                f"Conflicting features: {hft_features}"
            ),
        )

    if mode == "hft":
        if not hft_features:
            report.add(
                code="HFT_FEATURES_MISSING",
                severity="error",
                check="feature_mode",
                message=(
                    "env.mode='hft' requires at least one feature tagged "
                    "domain='hft'."
                ),
            )
        if mft_features:
            report.add(
                code="HFT_HAS_MFT_FEATURES",
                severity="error",
                check="feature_mode",
                message=(
                    "env.mode='hft' cannot include features tagged as "
                    f"domain='mft'. Conflicting features: {mft_features}"
                ),
            )


def _validate_feature_input_columns(
    pipeline: FeaturePipeline,
    raw_columns: set[str],
    report: ValidationReport,
) -> None:
    required_columns: set[str] = set()
    for feature in pipeline.features:
        required_columns.update(feature.required_columns())

    missing_columns = sorted(required_columns - raw_columns)
    if missing_columns:
        report.add(
            code="FEATURE_INPUT_COLUMNS_MISSING",
            severity="error",
            check="feature_columns",
            message=(
                "Feature pipeline requires input columns missing from dataset: "
                f"{missing_columns}. Available columns: {sorted(raw_columns)}"
            ),
        )


def validate_experiment_config(config: ExperimentConfig) -> ValidationReport:
    """Validate experiment config with filesystem and data-aware checks."""
    report = ValidationReport()
    _validate_base_config(config, report)
    _validate_paths(config, report)

    pipeline = _build_feature_pipeline(config, report)

    data_path = Path(config.data.data_path)
    if not data_path.exists():
        return report

    try:
        file_signature = data_path.stat().st_mtime_ns
        df = load_trading_data(str(data_path), cache_bust=file_signature)
    except Exception as exc:
        report.add(
            code="DATA_LOAD_FAILED",
            severity="error",
            check="dataset_load",
            message=f"Failed to load dataset: {exc}",
        )
        return report

    dataset_len = len(df.dropna())
    if dataset_len == 0:
        report.add(
            code="DATASET_EMPTY",
            severity="error",
            check="dataset_load",
            message="Dataset is empty after dropping NaN rows.",
        )
        return report

    _validate_split_feasibility(config, dataset_len, report)

    raw_columns = set(df.columns)
    if pipeline is not None:
        _validate_feature_domains(config, pipeline, report)
        _validate_feature_input_columns(pipeline, raw_columns, report)
        available_columns = raw_columns.union(set(pipeline.get_feature_names()))
    else:
        available_columns = raw_columns

    _validate_env_columns(config, available_columns, report)
    return report
