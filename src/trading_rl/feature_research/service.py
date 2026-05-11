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
from trading_rl.data_utils import load_trading_data, _feature_cache_key
from trading_rl.feature_research.config import FeatureResearchConfig, TargetType
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
    scores: pd.DataFrame
    selected_names: list[str]


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


def _build_return_target(df: pd.DataFrame, horizon: int) -> pd.Series:
    """Raw forward log-return proxy target."""
    price = _resolve_price_series(df)
    return np.log(price.shift(-horizon) / price).rename(f"target_return_h{horizon}")


def _build_target(
    df: pd.DataFrame, horizon: int, target_type: TargetType, vol_window: int
) -> pd.Series:
    """Dispatch to the right target builder based on target_type."""
    if target_type == TargetType.RETURN:
        return _build_return_target(df, horizon)
    if target_type == TargetType.SHARPE:
        return _build_sharpe_proxy_target(df, horizon, vol_window)
    raise ValueError(f"Unknown target_type '{target_type}'.")


def _build_sharpe_proxy_target(df: pd.DataFrame, horizon: int, vol_window: int) -> pd.Series:
    """Build a Sharpe-proxy target: forward return scaled by recent realised vol.

    Dividing by rolling vol rewards features that predict good risk-adjusted
    returns rather than raw price movement, which is closer to what DSR
    optimises over an episode.
    """
    price = _resolve_price_series(df)
    tick_log_ret = np.log(price / price.shift(1))
    rolling_vol = tick_log_ret.rolling(vol_window).std()
    forward_ret = np.log(price.shift(-horizon) / price)
    sharpe_proxy = forward_ret / (rolling_vol + 1e-10)
    return sharpe_proxy.rename(f"target_sharpe_h{horizon}_v{vol_window}")


def _align_feature_frames(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_config_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Fit feature pipeline on train, transform both splits; return feature-only frames."""
    pipeline = FeaturePipeline.from_yaml(feature_config_path)
    pipeline.fit(train_df)

    train_features = pipeline.transform(train_df)
    val_features = pipeline.transform(val_df)

    if train_features.empty or val_features.empty:
        raise ValueError(
            "Feature pipeline produced empty frames. "
            "Check the feature config and data."
        )

    return (
        train_features,
        val_features,
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

    feat_ranked = aligned.iloc[:, 0].rolling(window_size).rank()
    tgt_ranked = aligned.iloc[:, 1].rolling(window_size).rank()
    rolling_ic = feat_ranked.rolling(window_size).corr(tgt_ranked)
    return rolling_ic.dropna()


def _score_feature_at_horizon(
    feat_train: pd.Series,
    feat_val: pd.Series,
    train_target: pd.Series,
    val_target: pd.Series,
    window_size: int,
) -> dict[str, float]:
    """Return IC/ICIR stats for one feature against one horizon target."""
    ic_series = _compute_ic_series(feat_train, train_target, window_size)
    ic_std_raw = ic_series.std()
    if len(ic_series) == 0 or not (ic_std_raw > 0):
        mean_ic = ic_std = icir = ic_tstat = ic_positive_ratio = 0.0
    else:
        mean_ic = float(ic_series.mean())
        ic_std = float(ic_std_raw)
        icir = mean_ic / ic_std if ic_std > 1e-10 else 0.0
        n = len(ic_series)
        ic_tstat = mean_ic / (ic_std / np.sqrt(n)) if n > 1 and ic_std > 1e-10 else 0.0
        ic_positive_ratio = float((ic_series > 0).mean())

    val_ic = _compute_ic_series(feat_val, val_target, window_size)
    val_mean_ic = float(val_ic.mean()) if len(val_ic) > 0 else 0.0

    return {
        "mean_ic": mean_ic,
        "ic_std": ic_std,
        "icir": icir,
        "ic_tstat": ic_tstat,
        "ic_positive_ratio": ic_positive_ratio,
        "val_mean_ic": val_mean_ic,
    }


def _build_score_table(
    train_features: pd.DataFrame,
    val_features: pd.DataFrame,
    train_raw: pd.DataFrame,
    val_raw: pd.DataFrame,
    horizons: list[int],
    vol_window: int,
    window_size: int,
    target_type: TargetType = TargetType.SHARPE,
) -> pd.DataFrame:
    """Build per-feature IC/ICIR scoring table across multiple Sharpe-proxy horizons.

    For each feature, scores are computed against every horizon target. The row
    with the highest |ICIR| is kept as the representative score, and
    ``best_horizon`` records which horizon produced it.
    """
    feature_cols = list(train_features.columns)
    rows: list[dict] = []

    for feat in feature_cols:
        best: dict[str, float] = {}
        best_icir_abs = -1.0
        best_h = horizons[0]

        for h in horizons:
            train_target = _build_target(train_raw, h, target_type, vol_window)
            val_target = _build_target(val_raw, h, target_type, vol_window)

            # Align feature with target (drop NaNs from vol warm-up + forward shift)
            train_aligned = pd.concat([train_features[feat], train_target], axis=1).dropna()
            val_aligned = pd.concat([val_features[feat], val_target], axis=1).dropna()

            if len(train_aligned) < 10:
                continue

            stats = _score_feature_at_horizon(
                train_aligned.iloc[:, 0],
                val_aligned.iloc[:, 0] if len(val_aligned) >= 10 else train_aligned.iloc[:10, 0],
                train_aligned.iloc[:, 1],
                val_aligned.iloc[:, 1] if len(val_aligned) >= 10 else train_aligned.iloc[:10, 1],
                window_size,
            )

            if abs(stats["icir"]) > best_icir_abs:
                best_icir_abs = abs(stats["icir"])
                best = stats
                best_h = h

        rows.append(
            {
                "feature": feat,
                "best_horizon": best_h,
                "mean_ic": best.get("mean_ic", 0.0),
                "ic_std": best.get("ic_std", 1e-10),
                "icir": best.get("icir", 0.0),
                "ic_tstat": best.get("ic_tstat", 0.0),
                "ic_positive_ratio": best.get("ic_positive_ratio", 0.0),
                "val_mean_ic": best.get("val_mean_ic", 0.0),
                "ic_stability": abs(best.get("mean_ic", 0.0) - best.get("val_mean_ic", 0.0)),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("icir", ascending=False, key=abs)
        .reset_index(drop=True)
    )


def _select_features(
    scores: pd.DataFrame,
    feature_data: pd.DataFrame,
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
        f"- Proxy target horizons: `{config.research.horizons}` (Sharpe-proxy, vol window={config.research.vol_window})",
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


# Max val rows per symbol kept for the pooled correlation / redundancy-pruning
# step. IC scoring uses the full train split; only this downstream pool is capped.
_VAL_POOL_CAP = 20_000


def _score_single_symbol(
    data_path: str,
    config: FeatureResearchConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Load one symbol, split, compute features, score IC/ICIR.

    Returns (scores, val_features_sample, feature_configs).

    All large intermediate frames (raw data, train split, train features) are
    explicitly deleted before returning so peak RSS stays proportional to one
    symbol at a time. val_features_sample is capped at _VAL_POOL_CAP rows —
    sufficient for pooled correlation and redundancy pruning.
    """
    symbol = Path(data_path).stem

    # --- feature cache lookup -------------------------------------------
    # Reuses the same cache directory as the training pipeline so a symbol
    # that was already prepared for training gets a free hit here.
    # Cache stores: train_features, val_features, train_price, val_price.
    cache_entry: Path | None = None
    validation_size_resolved: int | None = None

    if config.data.feature_cache_dir and Path(data_path).exists():
        # Resolve validation_size now so the cache key is stable.
        raw_len = len(load_trading_data(data_path).dropna())
        train_size_cfg = config.data.train_size
        # Apply the same proportional fallback used below so the cache key
        # is consistent with the actual split that will be computed.
        if train_size_cfg >= raw_len:
            train_size_cfg = max(10, int(raw_len * 0.8))
        remaining_cfg = raw_len - train_size_cfg
        validation_size_resolved = (
            config.data.validation_size
            if config.data.validation_size is not None
            else remaining_cfg // 2
        )
        cache_key = _feature_cache_key(
            data_path,
            train_size_cfg,
            validation_size_resolved,
            test_size=None,
            feature_config_path=config.data.feature_config,
            feature_pipeline=None,
        )
        cache_entry = Path(config.data.feature_cache_dir) / f"fr_{cache_key}"
        _fr_files = ("train_features", "val_features", "train_price", "val_price")
        if all((cache_entry / f"{f}.parquet").exists() for f in _fr_files):
            logger.info("feature research cache hit symbol=%s key=%s", symbol, cache_key[:8])
            train_features = pd.read_parquet(cache_entry / "train_features.parquet")
            val_features = pd.read_parquet(cache_entry / "val_features.parquet")
            train_price_df = pd.read_parquet(cache_entry / "train_price.parquet")
            val_price_df = pd.read_parquet(cache_entry / "val_price.parquet")
            feature_configs = [
                cfg.__dict__.copy()
                for cfg in FeaturePipeline.from_yaml(config.data.feature_config).feature_configs
            ]
            scores = _build_score_table(
                train_features, val_features, train_price_df, val_price_df,
                horizons=config.research.horizons,
                vol_window=config.research.vol_window,
                window_size=config.research.window_size,
                target_type=config.research.target_type,
            )
            del train_features, train_price_df, val_price_df
            if len(val_features) > _VAL_POOL_CAP:
                val_features = val_features.sample(
                    n=_VAL_POOL_CAP, random_state=0
                ).reset_index(drop=True)
            return scores, val_features, feature_configs
        logger.info("feature research cache miss symbol=%s key=%s", symbol, cache_key[:8])
    # --- end cache lookup ------------------------------------------------

    raw_df = load_trading_data(data_path).dropna()

    train_size = config.data.train_size
    # When train_size is larger than the file (per-day mode where we want the
    # full file as training), use 80 % of the file as train and 10 % as val.
    if train_size >= len(raw_df):
        train_size = max(10, int(len(raw_df) * 0.8))
        logger.info(
            "train_size exceeds file length for '%s' — using proportional split "
            "train=%d (80 %%) val=%d (10 %%)",
            symbol, train_size, int(len(raw_df) * 0.1),
        )

    remaining = len(raw_df) - train_size
    validation_size = (
        validation_size_resolved
        if validation_size_resolved is not None
        else (
            config.data.validation_size
            if config.data.validation_size is not None
            else remaining // 2
        )
    )

    if validation_size <= 0 or validation_size >= remaining:
        raise ValueError(
            f"validation_size={validation_size} invalid for symbol '{symbol}' "
            f"(post-train remainder is {remaining})."
        )

    train_df = raw_df.iloc[:train_size].copy()
    val_df = raw_df.iloc[train_size : train_size + validation_size].copy()
    del raw_df

    train_price_df = pd.DataFrame(
        {"close": _resolve_price_series(train_df).values}, index=train_df.index
    )
    val_price_df = pd.DataFrame(
        {"close": _resolve_price_series(val_df).values}, index=val_df.index
    )

    train_features, val_features, feature_configs = _align_feature_frames(
        train_df, val_df, config.data.feature_config
    )
    del train_df, val_df

    # Persist to cache before scoring so repeated runs with different research
    # settings (horizons, target_type) reuse the expensive feature computation.
    if cache_entry is not None:
        cache_entry.mkdir(parents=True, exist_ok=True)
        train_features.to_parquet(cache_entry / "train_features.parquet")
        val_features.to_parquet(cache_entry / "val_features.parquet")
        train_price_df.to_parquet(cache_entry / "train_price.parquet")
        val_price_df.to_parquet(cache_entry / "val_price.parquet")
        logger.info("feature research cache write symbol=%s key=%s", symbol, cache_key[:8])

    scores = _build_score_table(
        train_features, val_features, train_price_df, val_price_df,
        horizons=config.research.horizons,
        vol_window=config.research.vol_window,
        window_size=config.research.window_size,
        target_type=config.research.target_type,
    )
    del train_features, train_price_df, val_price_df

    if len(val_features) > _VAL_POOL_CAP:
        val_features = val_features.sample(
            n=_VAL_POOL_CAP, random_state=0
        ).reset_index(drop=True)

    return scores, val_features, feature_configs


def _aggregate_symbol_scores(per_symbol_scores: list[pd.DataFrame]) -> pd.DataFrame:
    """Average per-feature IC statistics across symbols.

    best_horizon is set to the most frequently dominant horizon across symbols.
    """
    combined = pd.concat(per_symbol_scores, ignore_index=True)
    agg = (
        combined.groupby("feature")
        .agg(
            best_horizon=("best_horizon", lambda s: int(s.mode().iloc[0])),
            mean_ic=("mean_ic", "mean"),
            ic_std=("ic_std", "mean"),
            icir=("icir", "mean"),
            ic_tstat=("ic_tstat", "mean"),
            ic_positive_ratio=("ic_positive_ratio", "mean"),
            val_mean_ic=("val_mean_ic", "mean"),
            ic_stability=("ic_stability", "mean"),
        )
        .reset_index()
    )
    return (
        agg.sort_values("icir", ascending=False, key=abs)
        .reset_index(drop=True)
    )


def run_feature_research(
    *,
    config: FeatureResearchConfig,
) -> FeatureResearchArtifacts:
    """Run offline IC/ICIR feature scoring and conditional IC selection.

    When ``config.data.data_paths`` contains more than one path, IC statistics
    are computed independently per symbol and then averaged.  Features that
    score consistently across all symbols are ranked highest, giving a more
    generalizable shortlist than single-symbol scoring.
    """
    output_dir = config.research.output_dir or (
        Path("logs") / config.experiment_name / "feature_research"
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    data_paths = config.data.resolve_paths()
    multi_symbol = len(data_paths) > 1

    logger.info(
        "feature scoring target_type=%s horizons=%s vol_window=%d n_symbols=%d",
        config.research.target_type,
        config.research.horizons,
        config.research.vol_window,
        len(data_paths),
    )

    per_symbol_scores: list[pd.DataFrame] = []
    all_val_features: list[pd.DataFrame] = []
    feature_configs: list[dict] = []

    for idx, data_path in enumerate(data_paths):
        symbol = Path(data_path).stem
        logger.info("scoring symbol %d/%d: %s", idx + 1, len(data_paths), symbol)
        sym_scores, sym_val_features, sym_feature_configs = _score_single_symbol(
            data_path, config
        )
        per_symbol_scores.append(sym_scores)
        all_val_features.append(sym_val_features)
        if not feature_configs:
            feature_configs = sym_feature_configs

        if multi_symbol:
            sym_scores.to_csv(
                output_path / f"feature_scores_{symbol}.csv", index=False
            )

    scores = (
        _aggregate_symbol_scores(per_symbol_scores)
        if multi_symbol
        else per_symbol_scores[0]
    )

    feature_cols = scores["feature"].tolist()
    pooled_val_features = pd.concat(all_val_features, ignore_index=True)
    del all_val_features

    # _select_features uses only the feature values for linear residualization;
    # drop NaN rows so the regression is clean.
    val_features_clean = pooled_val_features[feature_cols].dropna()

    selected_features = _select_features(
        scores,
        val_features_clean,
        top_k=config.research.top_k,
        icir_threshold=config.research.icir_threshold,
    )

    correlation_matrix = pooled_val_features[feature_cols].corr().fillna(0.0)
    del pooled_val_features

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
        scores=scores,
        selected_names=list(selected_features),
    )
