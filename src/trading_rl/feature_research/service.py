"""Offline feature research workflow.

This module is intentionally separate from the main RL training pipeline.
It scores engineered features against cheap proxy targets using the IC/ICIR
framework and emits a reduced feature configuration for later RL experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LinearRegression

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


def _compute_ic_series(
    feature: pd.Series,
    target: pd.Series,
    window_size: int | None = None,
) -> pd.Series:
    """Compute rolling IC (Spearman rank correlation) between feature and target."""
    aligned = pd.concat([feature, target], axis=1).dropna()
    if len(aligned) < 10:
        return pd.Series([0.0], index=aligned.index[:1] if len(aligned) > 0 else [0])

    if window_size is None or window_size <= 1:
        ic = aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method="spearman")
        return pd.Series([float(ic) if pd.notna(ic) else 0.0], index=[aligned.index[0]])

    rolling_ic = aligned.iloc[:, 0].rolling(window_size).corr(
        aligned.iloc[:, 1], method="spearman"
    )
    return rolling_ic.dropna()


def _build_score_table(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    window_size: int | None = None,
) -> pd.DataFrame:
    """Build per-feature IC/ICIR scoring table.

    Primary scoring uses the training split so the validation split stays
    clean for model comparison. Validation IC is reported for stability only.
    """
    target_col = next(col for col in train_frame.columns if col.startswith("target_"))
    feature_cols = [col for col in train_frame.columns if col != target_col]

    train_target = train_frame[target_col]
    val_target = val_frame[target_col]

    rows: list[dict[str, float | str]] = []
    for feat in feature_cols:
        # Primary IC/ICIR scored on training split
        ic_series = _compute_ic_series(train_frame[feat], train_target, window_size)

        if len(ic_series) == 0 or ic_series.std() == 0:
            mean_ic, ic_std, icir, ic_tstat, ic_positive_ratio = 0.0, 1e-10, 0.0, 0.0, 0.0
        else:
            mean_ic = float(ic_series.mean())
            ic_std = float(ic_series.std())
            icir = mean_ic / ic_std if ic_std > 1e-10 else 0.0
            n = len(ic_series)
            ic_tstat = mean_ic / (ic_std / np.sqrt(n)) if n > 1 and ic_std > 1e-10 else 0.0
            ic_positive_ratio = float((ic_series > 0).mean())

        # Validation IC for out-of-sample stability reporting only
        val_ic = _compute_ic_series(val_frame[feat], val_target, window_size)
        val_mean_ic = float(val_ic.mean()) if len(val_ic) > 0 else 0.0

        rows.append(
            {
                "feature": feat,
                "mean_ic": mean_ic,
                "ic_std": float(ic_std),
                "icir": float(icir),
                "ic_tstat": float(ic_tstat),
                "ic_positive_ratio": ic_positive_ratio,
                "val_mean_ic": val_mean_ic,
                "ic_stability": abs(mean_ic - val_mean_ic),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("icir", ascending=False)
        .reset_index(drop=True)
    )


def _select_features(
    scores: pd.DataFrame,
    feature_data: pd.DataFrame,
    target: pd.Series,
    top_k: int,
    icir_threshold: float = 0.02,
) -> list[str]:
    """Greedy selection using conditional IC for redundancy filtering."""
    selected: list[str] = []
    remaining_features = list(scores["feature"])
    residual_data = feature_data.copy()

    for candidate in scores["feature"]:
        if len(selected) >= top_k:
            break
        if candidate not in remaining_features:
            continue

        row = scores[scores["feature"] == candidate].iloc[0]
        if abs(float(row["icir"])) < icir_threshold:
            continue

        selected.append(candidate)

        if len(selected) < top_k and len(remaining_features) > len(selected):
            selected_matrix = residual_data[selected].values
            regressor = LinearRegression(fit_intercept=False)
            regressor.fit(selected_matrix, residual_data[remaining_features].values)

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
        f"- ICIR threshold: `{config.research.icir_threshold}`",
        "",
        "## Selected Features",
        "",
    ]
    lines.extend(f"- `{name}`" for name in selected_features)
    lines.extend(
        [
            "",
            "## Top Ranked Features (by ICIR)",
            "",
            "| feature | icir | mean_ic | ic_tstat | ic_positive_ratio |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in top_preview.itertuples(index=False):
        lines.append(
            f"| `{row.feature}` | {row.icir:.4f} | {row.mean_ic:.6f} | "
            f"{row.ic_tstat:.2f} | {row.ic_positive_ratio:.3f} |"
        )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_feature_research(
    *,
    config: FeatureResearchConfig,
) -> FeatureResearchArtifacts:
    """Run offline IC/ICIR feature scoring and conditional IC selection."""
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
    scores = _build_score_table(train_frame, val_frame, window_size=config.research.window_size)

    val_target_col = next(col for col in val_frame.columns if col.startswith("target_"))
    feature_cols = scores["feature"].tolist()
    val_aligned = pd.concat([val_frame[feature_cols], val_frame[val_target_col]], axis=1).dropna()

    selected_features = _select_features(
        scores,
        val_aligned[feature_cols],
        val_aligned[val_target_col],
        top_k=config.research.top_k,
        icir_threshold=config.research.icir_threshold,
    )

    correlation_matrix = val_frame[feature_cols].corr().fillna(0.0)

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
