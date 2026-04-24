"""Feature selector for research-based pre-selection.

Wraps the offline feature scoring and redundancy-aware selection
logic into a reusable component that operates on FeatureConfig lists
directly, without requiring a separate YAML config file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

from logger import get_logger
from trading_rl.features.base import FeatureConfig
from trading_rl.features.pipeline import FeaturePipeline

logger = get_logger(__name__)


@dataclass(frozen=True)
class FeatureSelectionResult:
    """Result of a feature selection run."""

    selected_configs: list[FeatureConfig]
    scores: pd.DataFrame
    correlation_matrix: pd.DataFrame
    selected_names: list[str]
    top_k: int
    corr_threshold: float


@dataclass
class FeatureSelectorConfig:
    """Configuration for feature selection.

    Attributes:
        top_k: Maximum number of features to select.
        corr_threshold: Maximum absolute correlation between selected features.
            Features correlated above this threshold with any already-selected
            feature are skipped during greedy selection.
        horizon: Forward return horizon for the proxy target.
        score_weights: Weights for the composite score components.
            Keys: spearman_val, pearson_val, mutual_information, stability_gap.
        train_size: Number of rows for the training split.
        validation_size: Number of rows for the validation split.
            If None, uses the remainder after train_size split in half.
    """

    top_k: int = 12
    corr_threshold: float = 0.85
    horizon: int = 1
    score_weights: dict[str, float] = field(
        default_factory=lambda: {
            "spearman_val": 0.45,
            "pearson_val": 0.35,
            "mutual_information": 0.20,
            "stability_gap": -0.20,
        }
    )
    train_size: int = 15000
    validation_size: int | None = None


def _resolve_price_series(df: pd.DataFrame) -> pd.Series:
    """Resolve a price series for proxy target computation."""
    if "close" in df.columns:
        return df["close"].astype(float)
    if {"bid_px_00", "ask_px_00"}.issubset(df.columns):
        return ((df["bid_px_00"] + df["ask_px_00"]) / 2.0).astype(float)
    raise ValueError(
        "Feature selection requires either 'close' or "
        "'bid_px_00'/'ask_px_00' columns to define proxy targets."
    )


def _build_proxy_target(df: pd.DataFrame, horizon: int) -> pd.Series:
    """Build a forward log-return proxy target."""
    price = _resolve_price_series(df)
    target = np.log(price.shift(-horizon) / price)
    return target.rename(f"target_log_return_h{horizon}")


def _safe_corr(series: pd.Series, target: pd.Series, method: str) -> float:
    """Compute a correlation safely for low-variance inputs."""
    if series.nunique(dropna=True) <= 1 or target.nunique(dropna=True) <= 1:
        return 0.0
    corr = series.corr(target, method=method)
    return float(corr) if pd.notna(corr) else 0.0


def _safe_mutual_information(
    frame: pd.DataFrame, target: pd.Series
) -> np.ndarray:
    """Compute mutual information with stable fallbacks."""
    if frame.empty:
        return np.zeros(0, dtype=float)
    valid_frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    valid_target = target.loc[valid_frame.index]
    if valid_frame.empty or valid_target.nunique(dropna=True) <= 1:
        return np.zeros(frame.shape[1], dtype=float)
    mi = mutual_info_regression(
        valid_frame.to_numpy(),
        valid_target.to_numpy(),
        random_state=0,
    )
    return np.asarray(mi, dtype=float)


def _build_score_table(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    weights: dict[str, float],
) -> pd.DataFrame:
    """Build per-feature offline screening metrics."""
    target_col = next(
        col for col in train_frame.columns if col.startswith("target_")
    )
    feature_cols = [col for col in train_frame.columns if col != target_col]

    train_target = train_frame[target_col]
    val_target = val_frame[target_col]

    mi_values = _safe_mutual_information(train_frame[feature_cols], train_target)
    mi_series = pd.Series(mi_values, index=feature_cols, dtype=float)

    rows: list[dict[str, float | str]] = []
    for feat in feature_cols:
        train_series = train_frame[feat]
        val_series = val_frame[feat]

        pearson_val = _safe_corr(val_series, val_target, "pearson")
        spearman_val = _safe_corr(val_series, val_target, "spearman")
        spearman_train = _safe_corr(train_series, train_target, "spearman")
        stability_gap = abs(spearman_train - spearman_val)
        mi_val = float(mi_series.get(feat, 0.0))

        score = (
            abs(spearman_val) * weights.get("spearman_val", 0.45)
            + abs(pearson_val) * weights.get("pearson_val", 0.35)
            + mi_val * weights.get("mutual_information", 0.20)
            - stability_gap * abs(weights.get("stability_gap", -0.20))
        )

        rows.append(
            {
                "feature": feat,
                "train_std": float(train_series.std(ddof=1) or 0.0),
                "val_std": float(val_series.std(ddof=1) or 0.0),
                "train_pearson": _safe_corr(train_series, train_target, "pearson"),
                "val_pearson": pearson_val,
                "train_spearman": spearman_train,
                "val_spearman": spearman_val,
                "mutual_information": mi_val,
                "stability_gap": stability_gap,
                "score": float(score),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )


def _select_features(
    scores: pd.DataFrame,
    correlation_matrix: pd.DataFrame,
    top_k: int,
    corr_threshold: float,
) -> list[str]:
    """Greedy redundancy-aware selection from ranked features."""
    selected: list[str] = []

    for feature in scores["feature"]:
        if len(selected) >= top_k:
            break
        if not selected:
            selected.append(feature)
            continue

        correlations = correlation_matrix.loc[feature, selected].abs()
        if bool((correlations >= corr_threshold).any()):
            continue
        selected.append(feature)

    return selected


class FeatureSelector:
    """Select features using research-based scoring and redundancy filtering.

    Operates directly on a list of FeatureConfig instances and a DataFrame,
    without requiring a YAML config file. This makes it composable with
    FeatureGroupResolver for group-based candidate pool construction.

    Usage::

        from trading_rl.features.groups import FeatureGroupResolver
        from trading_rl.features.selector import FeatureSelector, FeatureSelectorConfig

        resolver = FeatureGroupResolver.from_yaml("src/configs/features/feature_groups.yaml")
        candidates = resolver.resolve(["imbalance", "fair_value", "spread", "flow", "regime"])

        selector = FeatureSelector(FeatureSelectorConfig(top_k=12))
        result = selector.select(candidates, train_df, val_df)

        # Use selected configs to build a pipeline
        pipeline = FeaturePipeline(result.selected_configs)
    """

    def __init__(self, config: FeatureSelectorConfig | None = None) -> None:
        self.config = config or FeatureSelectorConfig()

    def select(
        self,
        feature_configs: list[FeatureConfig],
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
    ) -> FeatureSelectionResult:
        """Run feature selection on candidate features.

        Args:
            feature_configs: Candidate feature configurations.
            train_df: Training DataFrame with raw columns.
            val_df: Validation DataFrame with raw columns.

        Returns:
            FeatureSelectionResult with selected configs and scoring details.
        """
        cfg = self.config
        logger.info(
            "Starting feature selection: %d candidates, top_k=%d, "
            "corr_threshold=%.2f, horizon=%d",
            len(feature_configs),
            cfg.top_k,
            cfg.corr_threshold,
            cfg.horizon,
        )

        # Build and fit the feature pipeline on training data
        pipeline = FeaturePipeline(feature_configs)
        pipeline.fit(train_df)

        train_features = pipeline.transform(train_df)
        val_features = pipeline.transform(val_df)

        # Build proxy target
        train_target = _build_proxy_target(train_df, cfg.horizon)
        val_target = _build_proxy_target(val_df, cfg.horizon)

        # Align features and target, dropping NaN rows
        train_aligned = pd.concat([train_features, train_target], axis=1).dropna()
        val_aligned = pd.concat([val_features, val_target], axis=1).dropna()

        if train_aligned.empty or val_aligned.empty:
            raise ValueError(
                "Feature selection produced empty train/validation frames after "
                "target alignment. Reduce the horizon or provide more data."
            )

        logger.info(
            "Aligned frames: train=%d rows, val=%d rows",
            len(train_aligned),
            len(val_aligned),
        )

        # Score features
        scores = _build_score_table(train_aligned, val_aligned, cfg.score_weights)

        # Compute inter-feature correlation on training data
        feature_cols = scores["feature"].tolist()
        correlation_matrix = train_aligned[feature_cols].corr().fillna(0.0)

        # Select features with redundancy filter
        selected_names = _select_features(
            scores,
            correlation_matrix,
            top_k=cfg.top_k,
            corr_threshold=cfg.corr_threshold,
        )

        # Map selected names back to FeatureConfig instances
        output_name_map = {
            (fc.output_name or f"feature_{fc.name}"): fc
            for fc in feature_configs
        }
        selected_configs = [
            output_name_map[name]
            for name in selected_names
            if name in output_name_map
        ]

        logger.info(
            "Feature selection complete: %d of %d candidates selected",
            len(selected_configs),
            len(feature_configs),
        )

        return FeatureSelectionResult(
            selected_configs=selected_configs,
            scores=scores,
            correlation_matrix=correlation_matrix,
            selected_names=selected_names,
            top_k=cfg.top_k,
            corr_threshold=cfg.corr_threshold,
        )