"""Offline feature research workflow.

This module is intentionally separate from the main RL training pipeline.
It scores engineered features against cheap proxy targets and emits a reduced
feature configuration that can be used for later RL experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_selection import mutual_info_regression

from logger import get_logger
from trading_rl.data_utils import load_trading_data
from trading_rl.feature_research.config import FeatureResearchConfig
from trading_rl.features import FeaturePipeline

logger = get_logger(__name__)


@dataclass(frozen=True)
class FeatureResearchArtifacts:
    """Paths to generated offline feature research outputs."""

    output_dir: Path
    scores_csv: Path
    correlation_csv: Path
    selected_yaml: Path
    summary_md: Path


def _resolve_price_series(df: pd.DataFrame) -> pd.Series:
    """Resolve a price series for offline proxy targets."""
    if "close" in df.columns:
        return df["close"].astype(float)
    if {"bid_px_00", "ask_px_00"}.issubset(df.columns):
        return ((df["bid_px_00"] + df["ask_px_00"]) / 2.0).astype(float)
    raise ValueError(
        "Offline feature research requires either 'close' or "
        "'bid_px_00'/'ask_px_00' columns to define proxy targets."
    )


def _build_proxy_target(df: pd.DataFrame, horizon: int) -> pd.Series:
    """Build a forward return proxy target."""
    price = _resolve_price_series(df)
    target = np.log(price.shift(-horizon) / price)
    return target.rename(f"target_log_return_h{horizon}")


def _align_feature_frames(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_config_path: str,
    horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Create train/validation frames with aligned feature and target columns."""
    pipeline = FeaturePipeline.from_yaml(feature_config_path)
    pipeline.fit(train_df)

    train_features = pipeline.transform(train_df)
    val_features = pipeline.transform(val_df)

    train_target = _build_proxy_target(train_df, horizon)
    val_target = _build_proxy_target(val_df, horizon)

    train_aligned = pd.concat([train_features, train_target], axis=1).dropna()
    val_aligned = pd.concat([val_features, val_target], axis=1).dropna()

    if train_aligned.empty or val_aligned.empty:
        raise ValueError(
            "Feature research produced empty train/validation frames after "
            "target alignment. Reduce the horizon or provide more data."
        )

    return (
        train_aligned,
        val_aligned,
        [cfg.__dict__.copy() for cfg in pipeline.feature_configs],
    )


def _safe_corr(series: pd.Series, target: pd.Series, method: str) -> float:
    """Compute a correlation safely for low-variance inputs."""
    if series.nunique(dropna=True) <= 1 or target.nunique(dropna=True) <= 1:
        return 0.0
    corr = series.corr(target, method=method)
    return float(corr) if pd.notna(corr) else 0.0


def _safe_mutual_information(frame: pd.DataFrame, target: pd.Series) -> np.ndarray:
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
) -> pd.DataFrame:
    """Build per-feature offline screening metrics."""
    target_col = next(col for col in train_frame.columns if col.startswith("target_"))
    feature_cols = [col for col in train_frame.columns if col != target_col]

    train_target = train_frame[target_col]
    val_target = val_frame[target_col]

    mi_values = _safe_mutual_information(train_frame[feature_cols], train_target)
    mi_series = pd.Series(mi_values, index=feature_cols, dtype=float)

    rows: list[dict[str, float | str]] = []
    for feature in feature_cols:
        train_series = train_frame[feature]
        val_series = val_frame[feature]

        pearson_train = _safe_corr(train_series, train_target, "pearson")
        pearson_val = _safe_corr(val_series, val_target, "pearson")
        spearman_train = _safe_corr(train_series, train_target, "spearman")
        spearman_val = _safe_corr(val_series, val_target, "spearman")
        stability_gap = abs(spearman_train - spearman_val)
        score = (
            abs(spearman_val) * 0.45
            + abs(pearson_val) * 0.35
            + float(mi_series.get(feature, 0.0)) * 0.20
            - stability_gap * 0.20
        )

        rows.append(
            {
                "feature": feature,
                "train_std": float(train_series.std(ddof=1) or 0.0),
                "val_std": float(val_series.std(ddof=1) or 0.0),
                "train_pearson": pearson_train,
                "val_pearson": pearson_val,
                "train_spearman": spearman_train,
                "val_spearman": spearman_val,
                "mutual_information": float(mi_series.get(feature, 0.0)),
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


def _write_selected_feature_config(
    feature_configs: list[dict],
    selected_features: list[str],
    output_path: Path,
) -> None:
    """Write a reduced feature YAML preserving original config blocks."""
    selected_set = set(selected_features)
    reduced = [
        cfg
        for cfg in feature_configs
        if f"feature_{cfg.get('name')}" in selected_set
        or cfg.get("output_name") in selected_set
    ]
    payload = {"features": reduced}
    output_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )


def _write_summary(
    summary_path: Path,
    *,
    config: FeatureResearchConfig,
    selected_features: list[str],
    scores: pd.DataFrame,
) -> None:
    """Write a compact Markdown summary."""
    top_preview = scores.head(min(10, len(scores)))
    lines = [
        "# Offline Feature Research Summary",
        "",
        f"- Experiment: `{config.experiment_name}`",
        f"- Feature config: `{config.data.feature_config}`",
        f"- Proxy target horizon: `{config.research.horizon}`",
        f"- Requested top_k: `{config.research.top_k}`",
        f"- Correlation threshold: `{config.research.corr_threshold}`",
        "",
        "## Selected Features",
        "",
    ]
    lines.extend(f"- `{name}`" for name in selected_features)
    lines.extend(
        [
            "",
            "## Top Ranked Features",
            "",
            "| feature | score | val_spearman | val_pearson | mutual_information |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in top_preview.itertuples(index=False):
        lines.append(
            f"| `{row.feature}` | {row.score:.6f} | {row.val_spearman:.6f} | "
            f"{row.val_pearson:.6f} | {row.mutual_information:.6f} |"
        )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_feature_research(
    *,
    config: FeatureResearchConfig,
) -> FeatureResearchArtifacts:
    """Run offline feature scoring and redundancy-aware selection."""
    output_dir = config.research.output_dir or (
        Path("logs") / config.experiment_name / "feature_research"
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    file_signature = Path(config.data.data_path).stat().st_mtime_ns
    raw_df = load_trading_data(
        config.data.data_path,
        cache_bust=file_signature,
    ).dropna()

    train_size = config.data.train_size
    remaining = len(raw_df) - train_size
    validation_size = (
        config.data.validation_size
        if config.data.validation_size is not None
        else remaining // 2
    )

    if train_size >= len(raw_df):
        raise ValueError("train_size must be smaller than dataset length.")
    if validation_size <= 0 or validation_size >= remaining:
        raise ValueError(
            "validation_size must be > 0 and smaller than the post-train remainder."
        )

    train_df = raw_df.iloc[:train_size].copy()
    val_df = raw_df.iloc[train_size : train_size + validation_size].copy()

    train_frame, val_frame, feature_configs = _align_feature_frames(
        train_df,
        val_df,
        config.data.feature_config,
        config.research.horizon,
    )
    scores = _build_score_table(train_frame, val_frame)

    feature_cols = scores["feature"].tolist()
    correlation_matrix = train_frame[feature_cols].corr().fillna(0.0)
    selected_features = _select_features(
        scores,
        correlation_matrix,
        top_k=config.research.top_k,
        corr_threshold=config.research.corr_threshold,
    )

    scores_csv = output_path / "feature_scores.csv"
    correlation_csv = output_path / "feature_correlations.csv"
    selected_yaml = output_path / "selected_features.yaml"
    summary_md = output_path / "summary.md"

    scores.to_csv(scores_csv, index=False)
    correlation_matrix.to_csv(correlation_csv, index=True)
    _write_selected_feature_config(feature_configs, selected_features, selected_yaml)
    _write_summary(
        summary_md,
        config=config,
        selected_features=selected_features,
        scores=scores,
    )

    logger.info(
        "Offline feature research complete: %d features ranked, %d selected",
        len(scores),
        len(selected_features),
    )

    return FeatureResearchArtifacts(
        output_dir=output_path,
        scores_csv=scores_csv,
        correlation_csv=correlation_csv,
        selected_yaml=selected_yaml,
        summary_md=summary_md,
    )
