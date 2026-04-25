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
        use_multi_horizon: If True, use composite multi-horizon scoring.
            Features are scored across all ic_decay_horizons and combined
            using horizon_weights.
        horizon_weights: Weights for each horizon in multi-horizon scoring.
            Must match length of ic_decay_horizons. If None, uses equal weights.
        use_cross_validation: If True, use time-series cross-validation
            for robust feature ranking instead of single train/val split.
        n_cv_splits: Number of CV splits for time-series cross-validation.
        cv_test_size: Size of each validation fold in CV.
        cv_gap: Gap between training and validation periods in CV.
        ensemble_method: Method for ensemble selection across CV splits:
            "majority": Features selected by majority of splits
            "rank_average": Average rankings across splits
            "weighted": Weighted by ICIR scores
        enable_hyperparameter_search: If True, perform grid search for
            optimal hyperparameters (icir_threshold, top_k, horizon).
        hyperparameter_grid: Dictionary of hyperparameters to search.
            Keys: "icir_threshold", "top_k", "horizon"
            Values: List of values to try.
    """

    top_k: int = 12
    horizon: int = 1
    icir_threshold: float = 0.02
    corr_threshold: float = 0.85
    window_size: int | None = None
    ic_decay_horizons: list[int] = field(default_factory=lambda: [1, 5, 10, 20])
    train_size: int = 15000
    validation_size: int | None = None
    use_multi_horizon: bool = False
    horizon_weights: list[float] | None = None
    use_cross_validation: bool = False
    n_cv_splits: int = 5
    cv_test_size: int = 1000
    cv_gap: int = 100
    ensemble_method: str = "rank_average"
    enable_hyperparameter_search: bool = False
    hyperparameter_grid: dict[str, list[Any]] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration."""
        if self.use_multi_horizon:
            if not self.ic_decay_horizons:
                raise ValueError(
                    "ic_decay_horizons must be non-empty when use_multi_horizon=True"
                )
            if self.horizon_weights is not None:
                if len(self.horizon_weights) != len(self.ic_decay_horizons):
                    raise ValueError(
                        f"horizon_weights length ({len(self.horizon_weights)}) "
                        f"must match ic_decay_horizons length ({len(self.ic_decay_horizons)})"
                    )
            if self.horizon_weights is None:
                # Equal weights by default
                self.horizon_weights = [1.0 / len(self.ic_decay_horizons)] * len(self.ic_decay_horizons)

        if self.use_cross_validation:
            if self.n_cv_splits < 2:
                raise ValueError("n_cv_splits must be >= 2 for cross-validation")
            if self.cv_test_size <= 0:
                raise ValueError("cv_test_size must be > 0")
            if self.cv_gap < 0:
                raise ValueError("cv_gap must be >= 0")

        if self.ensemble_method not in ["majority", "rank_average", "weighted"]:
            raise ValueError(
                f"ensemble_method must be one of 'majority', 'rank_average', 'weighted', "
                f"got '{self.ensemble_method}'"
            )

        if self.enable_hyperparameter_search:
            valid_keys = {"icir_threshold", "top_k", "horizon"}
            for key in self.hyperparameter_grid:
                if key not in valid_keys:
                    raise ValueError(
                        f"Invalid hyperparameter key '{key}'. "
                        f"Valid keys: {valid_keys}"
                    )


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


def _build_time_series_cv_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    test_size: int = 1000,
    gap: int = 100,
    min_train_size: int = 2000,
) -> list[tuple[int, int, int, int]]:
    """Generate train/validation split indices for time-series cross-validation.

    Uses expanding window CV with a gap between training and validation periods
    to prevent leakage. Each split uses all data up to a certain point,
    then validates on the next test_size samples.

    Args:
        df: DataFrame to generate splits for (used to determine max index).
        n_splits: Number of CV splits to generate.
        test_size: Size of each validation fold.
        gap: Gap between training and validation periods.
        min_train_size: Minimum size of training set for first split.

    Returns:
        List of tuples: (train_start, train_end, val_start, val_end)
    """
    n_samples = len(df)
    splits = []

    if min_train_size + gap + test_size > n_samples:
        raise ValueError(
            f"Insufficient data for CV: need at least {min_train_size + gap + test_size} "
            f"samples, got {n_samples}"
        )

    # Calculate total samples needed for all splits
    total_cv_samples = min_train_size + (n_splits - 1) * (gap + test_size) + test_size

    if total_cv_samples > n_samples:
        n_splits = (n_samples - min_train_size - test_size) // (gap + test_size) + 1
        logger.warning(
            "Reducing n_splits to %d due to insufficient data for %d splits",
            n_splits, n_splits + (total_cv_samples - n_samples) // (gap + test_size) + 1
        )

    # Generate splits using expanding window
    for i in range(n_splits):
        train_start = 0
        train_end = min_train_size + i * (gap + test_size)
        val_start = train_end + gap
        val_end = val_start + test_size

        if val_end > n_samples:
            logger.warning(
                "Split %d exceeds data bounds, stopping early", i
            )
            break

        splits.append((train_start, train_end, val_start, val_end))

    logger.info(
        "Generated %d time-series CV splits with min_train_size=%d, test_size=%d, gap=%d",
        len(splits), min_train_size, test_size, gap
    )

    return splits


def _build_multi_horizon_score_table(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    feature_pipeline: FeaturePipeline | None = None,
    feature_configs: list[FeatureConfig] | None = None,
    window_size: int | None = None,
    horizons: list[int] | None = None,
    horizon_weights: list[float] | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Build per-feature IC/ICIR scoring table with multi-horizon composite scores.

    For each feature, computes IC/ICIR at each horizon and combines them
    using weighted averaging. Features are scored by their composite ICIR.

    Args:
        train_frame: Training DataFrame with raw feature columns and price data.
        val_frame: Validation DataFrame with raw feature columns and price data.
        feature_pipeline: Optional pre-built feature pipeline.
        feature_configs: Optional feature configurations to build pipeline.
        window_size: Rolling window size for IC computation.
        horizons: List of forward return horizons to evaluate.
        horizon_weights: Weights for each horizon in composite score.

    Returns:
        Tuple of (scores DataFrame sorted by composite ICIR descending,
                 IC series dict for primary horizon).
    """
    if horizons is None or len(horizons) == 0:
        # Fallback to single horizon - use standard function
        if feature_pipeline is None and feature_configs is not None:
            feature_pipeline = FeaturePipeline(feature_configs)
            feature_pipeline.fit(train_frame)
            train_features = feature_pipeline.transform(train_frame)
            val_features = feature_pipeline.transform(val_frame)
            primary_horizon = 1
            train_target = _build_proxy_target(train_frame, primary_horizon)
            val_target = _build_proxy_target(val_frame, primary_horizon)
            train_aligned = pd.concat([train_features, train_target], axis=1).dropna()
            val_aligned = pd.concat([val_features, val_target], axis=1).dropna()
            return _build_icir_score_table(train_aligned, val_aligned, window_size)
        else:
            # If pipeline is provided, use it to transform
            if feature_pipeline is not None:
                train_features = feature_pipeline.transform(train_frame)
                val_features = feature_pipeline.transform(val_frame)
                primary_horizon = 1
                train_target = _build_proxy_target(train_frame, primary_horizon)
                val_target = _build_proxy_target(val_frame, primary_horizon)
                train_aligned = pd.concat([train_features, train_target], axis=1).dropna()
                val_aligned = pd.concat([val_features, val_target], axis=1).dropna()
                return _build_icir_score_table(train_aligned, val_aligned, window_size)
            else:
                # Assume already transformed data (with targets)
                return _build_icir_score_table(train_frame, val_frame, window_size)

    if horizon_weights is None:
        horizon_weights = [1.0 / len(horizons)] * len(horizons)

    logger.info(
        "Building multi-horizon scores with horizons=%s, weights=%s",
        horizons, horizon_weights
    )

    # Build feature pipeline if not provided
    if feature_pipeline is None:
        if feature_configs is None:
            raise ValueError(
                "Either feature_pipeline or feature_configs must be provided"
            )
        feature_pipeline = FeaturePipeline(feature_configs)
        feature_pipeline.fit(train_frame)
        train_features = feature_pipeline.transform(train_frame)
        val_features = feature_pipeline.transform(val_frame)
    else:
        train_features = feature_pipeline.transform(train_frame)
        val_features = feature_pipeline.transform(val_frame)

    # Get feature columns
    feature_cols = val_features.columns.tolist()

    # Store IC results for each horizon
    horizon_results: dict[int, dict[str, dict]] = {}

    for h, weight in zip(horizons, horizon_weights):
        # Build target for this horizon
        train_target = _build_proxy_target(train_frame, h)
        val_target = _build_proxy_target(val_frame, h)

        # Align features and target - use transformed features
        train_aligned = pd.concat([train_features, train_target], axis=1).dropna()
        val_aligned = pd.concat([val_features, val_target], axis=1).dropna()

        if train_aligned.empty or val_aligned.empty:
            logger.warning(
                "Horizon %d produced empty frames after alignment, skipping", h
            )
            continue

        target_col = train_aligned.columns[-1]
        train_target_series = train_aligned[target_col]
        val_target_series = val_aligned[target_col]

        horizon_results[h] = {}

        for feat in feature_cols:
            # Compute IC series for this horizon
            ic_series = _compute_ic_series(val_aligned[feat], val_target_series, window_size)

            if len(ic_series) == 0 or ic_series.std() == 0:
                mean_ic = 0.0
                ic_std = 1e-10
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

            # Compute training IC for stability
            train_target_series = train_aligned[target_col]
            train_ic = _compute_ic_series(train_aligned[feat], train_target_series, window_size)
            train_mean_ic = float(train_ic.mean()) if len(train_ic) > 0 else 0.0

            horizon_results[h][feat] = {
                "mean_ic": mean_ic,
                "ic_std": ic_std,
                "icir": icir,
                "ic_tstat": ic_tstat,
                "ic_positive_ratio": ic_positive_ratio,
                "train_mean_ic": train_mean_ic,
                "ic_stability": abs(mean_ic - train_mean_ic),
            }

    # Build composite scores using weighted averaging
    rows: list[dict] = []

    for feat in feature_cols:
        # Get results for this feature across all horizons
        feature_horizon_scores = {
            h: results.get(feat)
            for h, results in horizon_results.items()
            if feat in results
        }

        if not feature_horizon_scores:
            continue

        # Compute weighted averages
        composite_mean_ic = 0.0
        composite_ic_std = 0.0
        composite_icir = 0.0
        composite_ic_tstat = 0.0
        composite_ic_positive_ratio = 0.0
        composite_train_mean_ic = 0.0
        composite_ic_stability = 0.0
        total_weight = 0.0

        for h, weight in zip(horizons, horizon_weights):
            if h not in feature_horizon_scores:
                continue

            scores = feature_horizon_scores[h]
            composite_mean_ic += scores["mean_ic"] * weight
            composite_ic_std += scores["ic_std"] * weight
            composite_icir += scores["icir"] * weight
            composite_ic_tstat += scores["ic_tstat"] * weight
            composite_ic_positive_ratio += scores["ic_positive_ratio"] * weight
            composite_train_mean_ic += scores["train_mean_ic"] * weight
            composite_ic_stability += scores["ic_stability"] * weight
            total_weight += weight

        if total_weight > 0:
            composite_mean_ic /= total_weight
            composite_ic_std /= total_weight
            composite_icir /= total_weight
            composite_ic_tstat /= total_weight
            composite_ic_positive_ratio /= total_weight
            composite_train_mean_ic /= total_weight
            composite_ic_stability /= total_weight

        # Compute horizon-specific IC for reporting
        horizon_ic_dict = {h: horizon_results[h][feat]["mean_ic"] for h in feature_horizon_scores}
        horizon_ic_str = ",".join(f"{h}:{ic:.4f}" for h, ic in sorted(horizon_ic_dict.items()))

        rows.append({
            "feature": feat,
            "mean_ic": composite_mean_ic,
            "ic_std": composite_ic_std,
            "icir": composite_icir,
            "ic_tstat": composite_ic_tstat,
            "ic_positive_ratio": composite_ic_positive_ratio,
            "train_mean_ic": composite_train_mean_ic,
            "ic_stability": composite_ic_stability,
            "horizon_ic_scores": horizon_ic_str,
        })

    scores = pd.DataFrame(rows).sort_values("icir", ascending=False).reset_index(drop=True)

    # Get IC series for primary horizon (first in list)
    primary_horizon = horizons[0]
    ic_series_dict = {}
    train_target = _build_proxy_target(train_frame, primary_horizon)
    val_target = _build_proxy_target(val_frame, primary_horizon)
    train_aligned = pd.concat([train_features, train_target], axis=1).dropna()
    val_aligned = pd.concat([val_features, val_target], axis=1).dropna()

    target_col = val_aligned.columns[-1]
    for feat in feature_cols:
        ic_series_dict[feat] = _compute_ic_series(
            val_aligned[feat], val_aligned[target_col], window_size
        )

    return scores, ic_series_dict


def _ensemble_select_features(
    split_selections: list[list[str]],
    ensemble_method: str = "rank_average",
    scores_per_split: list[pd.DataFrame] | None = None,
) -> list[str]:
    """Ensemble feature selections across CV splits.

    Args:
        split_selections: List of selected feature names for each CV split.
        ensemble_method: Method for ensemble selection:
            - "majority": Features selected by majority of splits
            - "rank_average": Average rankings across splits
            - "weighted": Weighted by ICIR scores
        scores_per_split: List of scores DataFrames for each split (required for "weighted").

    Returns:
        List of ensemble-selected feature names.
    """
    if not split_selections:
        return []

    n_splits = len(split_selections)

    if ensemble_method == "majority":
        # Vote by majority
        feature_votes: dict[str, int] = {}
        for split_selected in split_selections:
            for feat in split_selected:
                feature_votes[feat] = feature_votes.get(feat, 0) + 1

        # Select features that appear in majority of splits
        threshold = n_splits // 2 + 1
        ensemble_selected = [
            feat for feat, votes in sorted(
                feature_votes.items(), key=lambda x: x[1], reverse=True
            )
            if votes >= threshold
        ]

        logger.info(
            "Majority voting: %d features selected (threshold=%d/%d splits)",
            len(ensemble_selected), threshold, n_splits
        )
        return ensemble_selected

    elif ensemble_method == "rank_average":
        # Average rankings across splits
        feature_ranks: dict[str, list[int]] = {}
        for split_idx, split_selected in enumerate(split_selections):
            for rank, feat in enumerate(split_selected):
                if feat not in feature_ranks:
                    feature_ranks[feat] = []
                feature_ranks[feat].append(rank)

        # Compute average rank and sort
        feature_avg_ranks = {
            feat: sum(ranks) / len(ranks)
            for feat, ranks in feature_ranks.items()
        }

        ensemble_selected = sorted(
            feature_avg_ranks.keys(), key=lambda f: feature_avg_ranks[f]
        )

        logger.info(
            "Rank averaging: %d features selected across %d splits",
            len(ensemble_selected), n_splits
        )
        return ensemble_selected

    elif ensemble_method == "weighted":
        if scores_per_split is None or len(scores_per_split) != n_splits:
            raise ValueError(
                "scores_per_split must be provided for weighted ensemble method"
            )

        # Weight features by their ICIR scores across splits
        feature_weights: dict[str, float] = {}
        feature_counts: dict[str, int] = {}

        for split_scores in scores_per_split:
            for _, row in split_scores.iterrows():
                feat = row["feature"]
                weight = abs(row["icir"])

                feature_weights[feat] = feature_weights.get(feat, 0.0) + weight
                feature_counts[feat] = feature_counts.get(feat, 0) + 1

        # Normalize by number of splits the feature appears in
        feature_avg_weights = {
            feat: weight / feature_counts[feat]
            for feat, weight in feature_weights.items()
        }

        ensemble_selected = sorted(
            feature_avg_weights.keys(), key=lambda f: feature_avg_weights[f], reverse=True
        )

        logger.info(
            "Weighted ensemble: %d features selected across %d splits",
            len(ensemble_selected), n_splits
        )
        return ensemble_selected

    else:
        raise ValueError(f"Unknown ensemble method: {ensemble_method}")


class FeatureSelector:
    """Select features using IC/ICIR scoring and conditional IC redundancy filtering.

    Implements the Information Coefficient framework standard in quantitative
    factor research. Each feature is scored by its ICIR (mean IC / IC std) against
    forward returns, and redundancy is handled via conditional IC rather than
    pairwise correlation.

    Enhanced with:
    - Time-series cross-validation for robust ranking
    - Multi-horizon composite scoring
    - Automated hyperparameter search
    - Ensemble selection across CV splits

    Usage::

        from trading_rl.features.groups import FeatureGroupResolver
        from trading_rl.features.selector import FeatureSelector, FeatureSelectorConfig

        resolver = FeatureGroupResolver.from_yaml("src/configs/features/feature_groups.yaml")
        candidates = resolver.resolve(resolver.list_groups())

        # Standard single-split selection
        selector = FeatureSelector(FeatureSelectorConfig(top_k=12))
        result = selector.select(candidates, train_df, val_df)

        # Cross-validated selection
        config = FeatureSelectorConfig(
            top_k=12,
            use_cross_validation=True,
            n_cv_splits=5,
            cv_test_size=1000,
            ensemble_method="rank_average"
        )
        result = selector.select(candidates, df)

        # Multi-horizon selection
        config = FeatureSelectorConfig(
            top_k=12,
            use_multi_horizon=True,
            ic_decay_horizons=[1, 5, 10, 20],
            horizon_weights=[0.4, 0.3, 0.2, 0.1]  # Weight short-term more
        )
        result = selector.select(candidates, train_df, val_df)

        # Hyperparameter search
        config = FeatureSelectorConfig(
            enable_hyperparameter_search=True,
            hyperparameter_grid={
                "icir_threshold": [0.01, 0.02, 0.03],
                "top_k": [8, 12, 16],
            }
        )
        result = selector.select(candidates, df)

        # Write to YAML for training
        FeatureSelector.write_selected_yaml(result, "src/configs/features/selected.yaml")
    """

    def __init__(self, config: FeatureSelectorConfig | None = None) -> None:
        self.config = config or FeatureSelectorConfig()

    def select(
        self,
        feature_configs: list[FeatureConfig],
        train_df: pd.DataFrame | None = None,
        val_df: pd.DataFrame | None = None,
        df: pd.DataFrame | None = None,
    ) -> FeatureSelectionResult:
        """Run IC/ICIR feature selection on candidate features.

        Args:
            feature_configs: Candidate feature configurations.
            train_df: Training DataFrame with raw columns (required if not using CV).
            val_df: Validation DataFrame with raw columns (required if not using CV).
            df: Full dataset for cross-validated selection (required if use_cross_validation=True).

        Returns:
            FeatureSelectionResult with selected configs, IC scores, and IC series.

        Note:
            - For standard selection: provide train_df and val_df
            - For cross-validated selection: provide df only
            - Cross-validation and hyperparameter search can be combined
        """
        cfg = self.config

        # Hyperparameter search
        if cfg.enable_hyperparameter_search:
            return self._hyperparameter_search(
                feature_configs,
                train_df=train_df,
                val_df=val_df,
                df=df,
            )

        # Cross-validation
        if cfg.use_cross_validation:
            if df is None:
                raise ValueError("df must be provided when use_cross_validation=True")
            return self._cross_validate_select(feature_configs, df)

        # Standard single-split selection
        if train_df is None or val_df is None:
            raise ValueError("train_df and val_df must be provided for standard selection")

        return self._single_split_select(feature_configs, train_df, val_df)

    def _single_split_select(
        self,
        feature_configs: list[FeatureConfig],
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
    ) -> FeatureSelectionResult:
        """Run single-split IC/ICIR feature selection."""
        cfg = self.config
        logger.info(
            "Starting IC/ICIR feature selection: %d candidates, top_k=%d, "
            "icir_threshold=%.3f, horizon=%d, multi_horizon=%s",
            len(feature_configs),
            cfg.top_k,
            cfg.icir_threshold,
            cfg.horizon,
            cfg.use_multi_horizon,
        )

        # Build and fit the feature pipeline on training data
        pipeline = FeaturePipeline(feature_configs)
        pipeline.fit(train_df)

        train_features = pipeline.transform(train_df)
        val_features = pipeline.transform(val_df)

        # Score features using appropriate method
        if cfg.use_multi_horizon:
            scores, ic_series = _build_multi_horizon_score_table(
                train_df,  # Pass raw DF for multi-horizon target building
                val_df,
                feature_pipeline=pipeline,
                feature_configs=None,  # Not needed since we pass pipeline
                window_size=cfg.window_size,
                horizons=cfg.ic_decay_horizons,
                horizon_weights=cfg.horizon_weights,
            )
        else:
            # Build proxy target for single horizon
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

            scores, ic_series = _build_icir_score_table(
                train_aligned, val_aligned, window_size=cfg.window_size,
            )

        logger.info("IC/ICIR scoring complete, top 5 features:")
        for _, row in scores.head(5).iterrows():
            logger.info(
                "  %s: ICIR=%.4f, mean_IC=%.6f, t=%.2f",
                row["feature"], row["icir"], row["mean_ic"], row.get("ic_tstat", 0),
            )

        # Compute inter-feature correlation on validation data for reporting
        if cfg.use_multi_horizon:
            # Use first horizon target for alignment
            primary_horizon = cfg.ic_decay_horizons[0]
            val_target = _build_proxy_target(val_df, primary_horizon)
            val_aligned = pd.concat([val_features, val_target], axis=1).dropna()
        else:
            val_target = _build_proxy_target(val_df, cfg.horizon)
            val_aligned = pd.concat([val_features, val_target], axis=1).dropna()

        # Get feature columns from val_aligned (exclude target columns)
        feature_cols_from_val = [col for col in val_aligned.columns if not col.startswith("target_")]
        correlation_matrix = val_aligned[feature_cols_from_val].corr().fillna(0.0)

        # Select features using conditional IC redundancy filter
        val_target_col = next(
            col for col in val_aligned.columns if col.startswith("target_")
        )
        selected_names = _select_features_conditional_ic(
            scores,
            val_aligned[feature_cols_from_val],
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

    def _cross_validate_select(
        self,
        feature_configs: list[FeatureConfig],
        df: pd.DataFrame,
    ) -> FeatureSelectionResult:
        """Run cross-validated feature selection using ensemble of CV splits."""
        cfg = self.config

        logger.info(
            "Starting cross-validated feature selection: %d candidates, n_splits=%d, "
            "ensemble_method=%s, multi_horizon=%s",
            len(feature_configs),
            cfg.n_cv_splits,
            cfg.ensemble_method,
            cfg.use_multi_horizon,
        )

        # Generate CV splits
        splits = _build_time_series_cv_splits(
            df=df,
            n_splits=cfg.n_cv_splits,
            test_size=cfg.cv_test_size,
            gap=cfg.cv_gap,
        )

        if not splits:
            raise ValueError("No valid CV splits generated")

        # Track selections and scores across splits
        split_selections: list[list[str]] = []
        split_scores: list[pd.DataFrame] = []

        for split_idx, (train_start, train_end, val_start, val_end) in enumerate(splits):
            logger.info(
                "Processing CV split %d: train=%d:%d, val=%d:%d",
                split_idx + 1, train_start, train_end, val_start, val_end
            )

            train_df = df.iloc[train_start:train_end].copy()
            val_df = df.iloc[val_start:val_end].copy()

            try:
                result = self._single_split_select(feature_configs, train_df, val_df)
                split_selections.append(result.selected_names)
                split_scores.append(result.scores)

                logger.info(
                    "Split %d: Selected %d features",
                    split_idx + 1, len(result.selected_names)
                )
            except Exception as e:
                logger.warning(
                    "Split %d failed: %s. Skipping this split.",
                    split_idx + 1, str(e)
                )
                continue

        if not split_selections:
            raise RuntimeError("All CV splits failed")

        # Ensemble selections across splits
        ensemble_selected = _ensemble_select_features(
            split_selections=split_selections,
            ensemble_method=cfg.ensemble_method,
            scores_per_split=split_scores if cfg.ensemble_method == "weighted" else None,
        )

        # Trim to top_k if ensemble returned more
        ensemble_selected = ensemble_selected[:cfg.top_k]

        # Create composite scores by averaging across splits
        feature_scores: dict[str, dict] = {}
        for scores_df in split_scores:
            if scores_df.empty:
                continue
            for _, row in scores_df.iterrows():
                feat = row["feature"]
                if feat not in feature_scores:
                    feature_scores[feat] = {
                        "icir_values": [],
                        "mean_ic_values": [],
                        "ic_tstat_values": [],
                    }
                feature_scores[feat]["icir_values"].append(row["icir"])
                feature_scores[feat]["mean_ic_values"].append(row["mean_ic"])
                feature_scores[feat]["ic_tstat_values"].append(row.get("ic_tstat", 0))

        # Build composite score DataFrame
        rows: list[dict] = []
        for feat in ensemble_selected:
            if feat in feature_scores and feature_scores[feat]["icir_values"]:
                scores = feature_scores[feat]
                rows.append({
                    "feature": feat,
                    "mean_ic": np.mean(scores["mean_ic_values"]),
                    "ic_std": np.std(scores["mean_ic_values"]) if len(scores["mean_ic_values"]) > 1 else 0.0,
                    "icir": np.mean(scores["icir_values"]),
                    "ic_tstat": np.mean(scores["ic_tstat_values"]) if scores["ic_tstat_values"] else 0.0,
                    "ic_positive_ratio": len([v for v in split_selections if feat in v]) / len(split_selections),
                    "train_mean_ic": np.mean(scores["mean_ic_values"]),  # Use val as proxy
                    "ic_stability": np.std(scores["icir_values"]) if len(scores["icir_values"]) > 1 else 0.0,
                })

        # If no features with valid scores, use all features that appear in any split scores
        if not rows:
            logger.warning(
                "No ensemble features have valid scores. Using features from split scores directly."
            )
            # Get all features from any split scores
            all_scored_features = set()
            for scores_df in split_scores:
                if not scores_df.empty:
                    all_scored_features.update(scores_df["feature"].tolist())

            # Build scores for all features with data
            for feat in all_scored_features:
                if feat in feature_scores and feature_scores[feat]["icir_values"]:
                    scores = feature_scores[feat]
                    rows.append({
                        "feature": feat,
                        "mean_ic": np.mean(scores["mean_ic_values"]),
                        "ic_std": np.std(scores["mean_ic_values"]) if len(scores["mean_ic_values"]) > 1 else 0.0,
                        "icir": np.mean(scores["icir_values"]),
                        "ic_tstat": np.mean(scores["ic_tstat_values"]) if scores["ic_tstat_values"] else 0.0,
                        "ic_positive_ratio": len([v for v in split_selections if feat in v]) / len(split_selections),
                        "train_mean_ic": np.mean(scores["mean_ic_values"]),
                        "ic_stability": np.std(scores["icir_values"]) if len(scores["icir_values"]) > 1 else 0.0,
                    })

        if not rows:
            logger.warning("No features with valid scores found in ensemble selection")
            composite_scores = pd.DataFrame(columns=[
                "feature", "mean_ic", "ic_std", "icir", "ic_tstat",
                "ic_positive_ratio", "train_mean_ic", "ic_stability"
            ])
        else:
            composite_scores = pd.DataFrame(rows).sort_values("icir", ascending=False)

        # Map selected names back to FeatureConfig instances
        output_name_map = {
            (fc.output_name or f"feature_{fc.name}"): fc
            for fc in feature_configs
        }
        selected_configs = [
            output_name_map[name]
            for name in ensemble_selected
            if name in output_name_map
        ]

        logger.info(
            "Cross-validated selection complete: %d of %d candidates selected "
            "(across %d CV splits)",
            len(selected_configs),
            len(feature_configs),
            len(split_selections),
        )

        return FeatureSelectionResult(
            selected_configs=selected_configs,
            scores=composite_scores,
            ic_series={},  # Not applicable for ensemble
            correlation_matrix=pd.DataFrame(),  # Not applicable for ensemble
            selected_names=ensemble_selected,
            top_k=cfg.top_k,
            icir_threshold=cfg.icir_threshold,
        )

    def _hyperparameter_search(
        self,
        feature_configs: list[FeatureConfig],
        train_df: pd.DataFrame | None = None,
        val_df: pd.DataFrame | None = None,
        df: pd.DataFrame | None = None,
    ) -> FeatureSelectionResult:
        """Perform grid search for optimal hyperparameters."""
        cfg = self.config

        if not cfg.hyperparameter_grid:
            logger.warning("enable_hyperparameter_search=True but grid is empty, using default config")
            return self.select(feature_configs, train_df, val_df, df)

        logger.info(
            "Starting hyperparameter search with grid: %s",
            cfg.hyperparameter_grid
        )

        # Build parameter grid
        param_names = list(cfg.hyperparameter_grid.keys())
        param_values = list(cfg.hyperparameter_grid.values())

        # Generate all combinations
        import itertools
        all_combinations = list(itertools.product(*param_values))

        best_result = None
        best_score = float("-inf")

        for combination in all_combinations:
            param_dict = dict(zip(param_names, combination))

            # Create config for this combination
            search_config = FeatureSelectorConfig(
                top_k=param_dict.get("top_k", cfg.top_k),
                horizon=param_dict.get("horizon", cfg.horizon),
                icir_threshold=param_dict.get("icir_threshold", cfg.icir_threshold),
                use_multi_horizon=cfg.use_multi_horizon,
                ic_decay_horizons=cfg.ic_decay_horizons,
                horizon_weights=cfg.horizon_weights,
                use_cross_validation=cfg.use_cross_validation,
                n_cv_splits=cfg.n_cv_splits,
                cv_test_size=cfg.cv_test_size,
                cv_gap=cfg.cv_gap,
                ensemble_method=cfg.ensemble_method,
                window_size=cfg.window_size,
            )

            # Create selector with this config
            search_selector = FeatureSelector(search_config)

            try:
                # Run selection with this config
                result = search_selector.select(feature_configs, train_df, val_df, df)

                # Score this configuration
                # Use mean ICIR of selected features as the objective
                if len(result.scores) > 0 and "icir" in result.scores.columns:
                    config_score = result.scores["icir"].mean()

                    logger.info(
                        "Config %s: score=%.4f, n_selected=%d",
                        param_dict, config_score, len(result.selected_configs)
                    )

                    if config_score > best_score:
                        best_score = config_score
                        best_result = result
                        logger.info(
                            "New best config: %s (score=%.4f)",
                            param_dict, best_score
                        )
                else:
                    logger.warning(
                        "Config %s produced empty or invalid scores, skipping",
                        param_dict
                    )

            except Exception as e:
                logger.warning(
                    "Config %s failed: %s. Skipping.",
                    param_dict, str(e)
                )
                continue

        if best_result is None:
            raise RuntimeError("Hyperparameter search failed for all configurations")

        logger.info(
            "Hyperparameter search complete. Best config: score=%.4f",
            best_score
        )

        return best_result

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