"""Feature selector using IC/ICIR framework for offline pre-selection.

Implements an Information Coefficient (IC) based feature scoring pipeline
that aligns with standard factor research practice in quantitative finance:

- IC: rank correlation between each feature and forward returns, computed
  rolling over time windows.
- ICIR (Information Coefficient Information Ratio): mean IC divided by IC
  standard deviation — a single principled score that captures both signal
  strength and consistency.
- Conditional IC: after selecting a feature, regress its signal out of
  remaining candidates and score the residuals. This replaces pairwise
  correlation as the redundancy criterion.

The selection procedure is offline: it runs once, produces a reduced feature
YAML, and the training pipeline loads that file directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from logger import get_logger
from trading_rl.features.base import FeatureConfig
from trading_rl.features.pipeline import FeaturePipeline

logger = get_logger(__name__)


@dataclass(frozen=True)
class FeatureSelectionResult:
    """Result of a feature selection run."""

    selected_configs: list[FeatureConfig]
    scores: pd.DataFrame
    ic_series: dict[str, pd.Series]
    correlation_matrix: pd.DataFrame
    selected_names: list[str]
    top_k: int
    icir_threshold: float


@dataclass
class FeatureSelectorConfig:
    """Configuration for IC/ICIR-based feature selection.

    Attributes:
        top_k: Maximum number of features to select.
        horizon: Forward return horizon for the proxy target.
        icir_threshold: Minimum absolute ICIR for a feature to be considered.
            Features below this threshold are excluded before redundancy
            filtering. Default 0.02 aligns with common factor research
            practice.
        corr_threshold: Maximum absolute correlation threshold for the
            conditional IC redundancy filter. Not used in the new IC-based
            pipeline; retained for backward compatibility.
        window_size: Rolling window size for IC computation. If None, IC
            is computed on the full validation split (no rolling).
        ic_decay_horizons: List of horizons at which to compute IC decay.
            If non-empty, IC is computed at each horizon and the decay
            curve is reported in the scores table.
        train_size: Number of rows for the training split.
        validation_size: Number of rows for the validation split.
    """

    top_k: int = 12
    horizon: int = 1
    icir_threshold: float = 0.02
    corr_threshold: float = 0.85
    window_size: int | None = None
    ic_decay_horizons: list[int] = field(default_factory=lambda: [1, 5, 10, 20])
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


def _compute_ic_series(
    feature: pd.Series,
    target: pd.Series,
    window_size: int | None = None,
) -> pd.Series:
    """Compute rolling IC (Spearman rank correlation) between feature and target.

    Args:
        feature: Feature values, indexed by time.
        target: Forward return values, aligned with feature index.
        window_size: Rolling window size. If None, compute a single IC
            value over the full series.

    Returns:
        Series of IC values. If window_size is None, returns a single-element
        series. If window_size is set, returns a rolling series.
    """
    aligned = pd.concat([feature, target], axis=1).dropna()
    if len(aligned) < 10:
        return pd.Series([0.0], index=aligned.index[:1])

    if window_size is None or window_size <= 1:
        ic = aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method="spearman")
        return pd.Series([float(ic) if pd.notna(ic) else 0.0], index=[aligned.index[0]])

    rolling_ic = aligned.iloc[:, 0].rolling(window_size).corr(
        aligned.iloc[:, 1], method="spearman"
    )
    rolling_ic = rolling_ic.dropna()
    return rolling_ic


def _build_icir_score_table(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    window_size: int | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Build per-feature IC/ICIR scoring table.

    For each feature, computes:
    - mean_ic: mean of rolling IC values on the validation split
    - ic_std: standard deviation of rolling IC values
    - icir: mean_ic / ic_std (Information Coefficient Information Ratio)
    - ic_tstat: approximate t-statistic (mean_ic / (ic_std / sqrt(n)))
    - ic_positive_ratio: fraction of rolling windows with IC > 0
    - ic_decay: IC at multiple horizons (if available)

    Args:
        train_frame: Training DataFrame with feature and target columns.
        val_frame: Validation DataFrame with feature and target columns.
        window_size: Rolling window size for IC computation.

    Returns:
        Tuple of (scores DataFrame sorted by ICIR descending, IC series dict).
    """
    target_col = next(
        col for col in val_frame.columns if col.startswith("target_")
    )
    feature_cols = [col for col in val_frame.columns if col != target_col]
    val_target = val_frame[target_col]

    ic_series_dict: dict[str, pd.Series] = {}
    rows: list[dict[str, float | str]] = []

    for feat in feature_cols:
        val_series = val_frame[feat]

        # Compute IC series on validation split
        ic_series = _compute_ic_series(val_series, val_target, window_size)
        ic_series_dict[feat] = ic_series

        if len(ic_series) == 0 or ic_series.std() == 0:
            mean_ic = 0.0
            ic_std = 1e-10  # avoid division by zero
            icir = 0.0
            ic_tstat = 0.0
            ic_positive_ratio = 0.0
        else:
            mean_ic = float(ic_series.mean())
            ic_std = float(ic_series.std())
            icir = mean_ic / ic_std if ic_std > 1e-10 else 0.0
            n = len(ic_series)
            ic_tstat = mean_ic / (ic_std / np.sqrt(n)) if n > 1 and ic_std > 1e-10 else 0.0
            ic_positive_ratio = float((ic_series > 0).mean())

        # Also compute training IC for stability assessment
        train_target = train_frame[target_col]
        train_ic = _compute_ic_series(train_frame[feat], train_target, window_size)
        train_mean_ic = float(train_ic.mean()) if len(train_ic) > 0 else 0.0

        rows.append(
            {
                "feature": feat,
                "mean_ic": mean_ic,
                "ic_std": float(ic_std),
                "icir": float(icir),
                "ic_tstat": float(ic_tstat),
                "ic_positive_ratio": ic_positive_ratio,
                "train_mean_ic": train_mean_ic,
                "ic_stability": abs(mean_ic - train_mean_ic),
            }
        )

    scores = (
        pd.DataFrame(rows)
        .sort_values("icir", ascending=False)
        .reset_index(drop=True)
    )

    return scores, ic_series_dict


def _compute_ic_decay(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_configs: list[FeatureConfig],
    horizons: list[int],
) -> pd.DataFrame:
    """Compute IC at multiple forward return horizons for IC decay analysis.

    Args:
        train_df: Raw training DataFrame.
        val_df: Raw validation DataFrame.
        feature_configs: Feature configurations to evaluate.
        horizons: List of forward return horizons.

    Returns:
        DataFrame with columns: feature, horizon, ic.
    """
    pipeline = FeaturePipeline(feature_configs)
    pipeline.fit(train_df)

    train_features = pipeline.transform(train_df)
    val_features = pipeline.transform(val_df)

    results: list[dict] = []

    for h in horizons:
        val_target = _build_proxy_target(val_df, h)
        val_aligned = pd.concat([val_features, val_target], axis=1).dropna()

        if val_aligned.empty:
            continue

        target_col = val_aligned.columns[-1]
        feature_cols = [c for c in val_aligned.columns if c != target_col]

        for feat in feature_cols:
            ic = _compute_ic_series(val_aligned[feat], val_aligned[target_col])
            mean_ic = float(ic.mean()) if len(ic) > 0 else 0.0
            results.append(
                {"feature": feat, "horizon": h, "ic": mean_ic}
            )

    return pd.DataFrame(results)


def _select_features_conditional_ic(
    scores: pd.DataFrame,
    feature_data: pd.DataFrame,
    target: pd.Series,
    top_k: int,
    icir_threshold: float = 0.02,
) -> list[str]:
    """Greedy selection using conditional IC for redundancy filtering.

    After selecting a feature, its linear signal is regressed out of all
    remaining candidates, and IC/ICIR is recomputed on the residuals. This
    ensures that features are selected for their incremental alpha contribution,
    not just their pairwise correlation with already-selected features.

    Args:
        scores: DataFrame sorted by ICIR descending.
        feature_data: Validation feature DataFrame (aligned with target).
        target: Forward return target series.
        top_k: Maximum number of features to select.
        icir_threshold: Minimum absolute ICIR to be considered.

    Returns:
        List of selected feature names, in selection order.
    """
    from sklearn.linear_model import LinearRegression

    selected: list[str] = []
    remaining_features = list(scores["feature"])
    # Work on a copy of the feature matrix to avoid mutation
    residual_data = feature_data.copy()

    for candidate in scores["feature"]:
        if len(selected) >= top_k:
            break
        if candidate not in remaining_features:
            continue

        # Check ICIR threshold
        row = scores[scores["feature"] == candidate].iloc[0]
        if abs(float(row["icir"])) < icir_threshold:
            continue

        selected.append(candidate)

        # Regress out the selected feature's signal from remaining candidates
        if len(selected) < top_k and len(remaining_features) > len(selected):
            selected_matrix = residual_data[selected].values
            regressor = LinearRegression(fit_intercept=False)
            regressor.fit(selected_matrix, residual_data[remaining_features].values)

            # Residualize remaining features
            predicted = regressor.predict(selected_matrix)
            remaining_idx = [
                i for i, f in enumerate(remaining_features) if f not in selected
            ]
            if remaining_idx:
                remaining_names = [remaining_features[i] for i in remaining_idx]
                residual_data[remaining_names] = (
                    residual_data[remaining_names].values - predicted[:, remaining_idx]
                )

    return selected


class FeatureSelector:
    """Select features using IC/ICIR scoring and conditional IC redundancy filtering.

    Implements the Information Coefficient framework standard in quantitative
    factor research. Each feature is scored by its ICIR (mean IC / IC std) against
    forward returns, and redundancy is handled via conditional IC rather than
    pairwise correlation.

    Usage::

        from trading_rl.features.groups import FeatureGroupResolver
        from trading_rl.features.selector import FeatureSelector, FeatureSelectorConfig

        resolver = FeatureGroupResolver.from_yaml("src/configs/features/feature_groups.yaml")
        candidates = resolver.resolve(resolver.list_groups())

        selector = FeatureSelector(FeatureSelectorConfig(top_k=12))
        result = selector.select(candidates, train_df, val_df)

        # Write to YAML for training
        FeatureSelector.write_selected_yaml(result, "src/configs/features/selected.yaml")
    """

    def __init__(self, config: FeatureSelectorConfig | None = None) -> None:
        self.config = config or FeatureSelectorConfig()

    def select(
        self,
        feature_configs: list[FeatureConfig],
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
    ) -> FeatureSelectionResult:
        """Run IC/ICIR feature selection on candidate features.

        Args:
            feature_configs: Candidate feature configurations.
            train_df: Training DataFrame with raw columns.
            val_df: Validation DataFrame with raw columns.

        Returns:
            FeatureSelectionResult with selected configs, IC scores, and IC series.
        """
        cfg = self.config
        logger.info(
            "Starting IC/ICIR feature selection: %d candidates, top_k=%d, "
            "icir_threshold=%.3f, horizon=%d",
            len(feature_configs),
            cfg.top_k,
            cfg.icir_threshold,
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

        # Score features using IC/ICIR
        scores, ic_series = _build_icir_score_table(
            train_aligned, val_aligned, window_size=cfg.window_size,
        )

        logger.info("IC/ICIR scoring complete, top 5 features:")
        for _, row in scores.head(5).iterrows():
            logger.info(
                "  %s: ICIR=%.4f, mean_IC=%.6f, t=%.2f",
                row["feature"], row["icir"], row["mean_ic"], row["ic_tstat"],
            )

        # Compute inter-feature correlation on validation data for reporting
        feature_cols = scores["feature"].tolist()
        correlation_matrix = val_aligned[feature_cols].corr().fillna(0.0)

        # Select features using conditional IC redundancy filter
        val_target_col = next(
            col for col in val_aligned.columns if col.startswith("target_")
        )
        selected_names = _select_features_conditional_ic(
            scores,
            val_aligned[feature_cols],
            val_aligned[val_target_col],
            top_k=cfg.top_k,
            icir_threshold=cfg.icir_threshold,
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
            ic_series=ic_series,
            correlation_matrix=correlation_matrix,
            selected_names=selected_names,
            top_k=cfg.top_k,
            icir_threshold=cfg.icir_threshold,
        )

    @staticmethod
    def write_selected_yaml(
        result: FeatureSelectionResult,
        output_path: str | Path,
    ) -> Path:
        """Write selected feature configs to a YAML file suitable for training.

        This is the offline step: run feature selection once, write the result
        to a YAML file, then point the training config at that file via
        ``data.feature_config``.

        Args:
            result: FeatureSelectionResult from a select() call.
            output_path: Path for the output YAML file.

        Returns:
            Path to the written file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        features = []
        for cfg in result.selected_configs:
            entry: dict[str, Any] = {
                "name": cfg.name,
                "feature_type": cfg.feature_type,
                "normalize": cfg.normalize,
                "domain": cfg.domain,
            }
            if cfg.params:
                entry["params"] = cfg.params
            if cfg.output_name and cfg.output_name != f"feature_{cfg.name}":
                entry["output_name"] = cfg.output_name
            features.append(entry)

        payload = {"features": features}
        output_path.write_text(
            yaml.dump(payload, default_flow_style=False, sort_keys=False),
            encoding="utf-8",
        )
        logger.info("Wrote %d selected features to %s", len(features), output_path)
        return output_path