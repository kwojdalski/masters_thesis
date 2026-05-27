"""Raw and transformed data overview logging helpers for MLflow."""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any

import mlflow
import numpy as np
import pandas as pd

from logger import get_logger as get_project_logger


def _log_overview_impl(
    df: pd.DataFrame,
    config: Any,
    artifact_dir: str,
    data_label: str,
    n_plot_samples: int = 200,
    max_features: int | None = 5,
    plots_subdir: str = "plots",
    obs_clip: float | None = None,
) -> None:
    """Shared implementation for raw and transformed data overview logging."""
    logger = get_project_logger(__name__)

    if not mlflow.active_run():
        logger.warning("no active mlflow run skipping %s logging", artifact_dir)
        return

    try:
        from plotnine import aes, geom_hline, geom_step, ggplot, labs
        from trading_rl.evaluation.thesis_theme import thesis_theme

        param_prefix = artifact_dir.replace("/", "_")
        mlflow.log_param(f"{param_prefix}_shape", f"{df.shape[0]}x{df.shape[1]}")
        mlflow.log_param(f"{param_prefix}_columns", list(df.columns))
        mlflow.log_param(f"{param_prefix}_date_range", f"{df.index.min()} to {df.index.max()}")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.head(50).to_csv(f.name)
            mlflow.log_artifact(f.name, artifact_dir)
            os.unlink(f.name)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            separator = "=" * (len(data_label) + 9)
            f.write(f"{data_label} Overview\n{separator}\n\n")
            f.write(f"Shape: {df.shape}\n")
            f.write(f"Columns: {list(df.columns)}\n")
            f.write(f"Date Range: {df.index.min()} to {df.index.max()}\n\n")
            f.write("Data Types:\n")
            f.write(str(df.dtypes))
            f.write("\n\nStatistical Summary:\n")
            f.write(str(df.describe()))
            f.flush()
            mlflow.log_artifact(f.name, artifact_dir)
            os.unlink(f.name)

        plot_df = df.head(n_plot_samples).reset_index()
        plot_df["time_index"] = range(len(plot_df))
        index_col = plot_df.columns[0]
        all_cols = [c for c in df.columns if c in plot_df.columns and c != index_col]
        columns_to_plot = all_cols if max_features is None else all_cols[:max_features]

        for column in columns_to_plot:
            try:
                p = (
                    ggplot(plot_df, aes(x="time_index", y=column))
                    + geom_step(color="steelblue", size=0.8)
                    + labs(
                        title=f"{column.title()} Over Time",
                        x="Time Index",
                        y=column.title(),
                    )
                    + thesis_theme()
                )
                if obs_clip is not None:
                    p = (
                        p
                        + geom_hline(yintercept=obs_clip, linetype="dashed", color="red", size=0.6)
                        + geom_hline(yintercept=-obs_clip, linetype="dashed", color="red", size=0.6)
                    )
                temp_path = os.path.join(tempfile.gettempdir(), f"{column}.png")
                p.save(temp_path, width=16, height=10, dpi=225)
                mlflow.log_artifact(temp_path, f"{artifact_dir}/{plots_subdir}")
                os.unlink(temp_path)
            except Exception as plot_error:  # pragma: no cover
                logger.warning("create plot failed column=%s err=%s", column, plot_error)

        ohlc_cols = ["open", "high", "low", "close"]
        if all(col in plot_df.columns for col in ohlc_cols):
            try:
                ohlc_melted = pd.melt(
                    plot_df[["time_index", *ohlc_cols]].dropna(),
                    id_vars=["time_index"],
                    value_vars=ohlc_cols,
                    var_name="price_type",
                    value_name="price",
                )
                p_combined = (
                    ggplot(ohlc_melted, aes(x="time_index", y="price", color="price_type"))
                    + geom_step(size=0.8)
                    + labs(
                        title="OHLC Prices Over Time",
                        x="Time Index",
                        y="Price",
                        color="Price Type",
                    )
                    + thesis_theme()
                )
                temp_path = os.path.join(tempfile.gettempdir(), "ohlc_combined.png")
                p_combined.save(temp_path, width=20, height=10, dpi=225)
                mlflow.log_artifact(temp_path, f"{artifact_dir}/{plots_subdir}")
                os.unlink(temp_path)
            except Exception as combined_error:  # pragma: no cover
                logger.warning("create combined ohlc plot failed err=%s", combined_error)

    except Exception as e:  # pragma: no cover
        logger.warning("log %s failed err=%s", artifact_dir, e)


def log_raw_data_overview(df: pd.DataFrame, config: Any) -> None:
    """Log raw (pre-transformation) dataset overview, sample, and visuals to MLflow."""
    raw_cols = [c for c in df.columns if not str(c).startswith("feature_")]
    raw_df = df[raw_cols] if raw_cols else df
    mlflow.log_param("data_source", config.data.data_path)
    _log_overview_impl(raw_df, config, "raw_data_overview", "Raw Data")


def log_transformed_data_overview(df: pd.DataFrame, config: Any) -> None:
    """Log transformed (feature-engineered) dataset overview, sample, and visuals to MLflow."""
    logger = get_project_logger(__name__)
    all_feat_cols = [c for c in df.columns if str(c).startswith("feature_")]
    feat_df = df[all_feat_cols] if all_feat_cols else df

    obs_clip: float | None = getattr(getattr(config, "env", None), "obs_clip", None)

    # All computed features
    _log_overview_impl(
        feat_df,
        config,
        "transformed_data_overview/all_features",
        "Transformed Data (all features)",
        n_plot_samples=1000,
        max_features=None,
        plots_subdir="plots",
        obs_clip=obs_clip,
    )

    # Only the features actually selected for the observation space
    selected_cols = getattr(getattr(config, "env", None), "feature_columns", None) or []
    selected_feat_cols = [c for c in selected_cols if c in df.columns and c != "feature_position"]
    logger.debug(
        "log_transformed_data_overview: selected_cols=%d in_df=%d",
        len(selected_cols),
        len(selected_feat_cols),
    )
    if selected_feat_cols:
        sel_df = df[selected_feat_cols]
        _log_overview_impl(
            sel_df,
            config,
            "transformed_data_overview/selected_features",
            "Transformed Data (selected features)",
            n_plot_samples=1000,
            max_features=None,
            plots_subdir="plots",
            obs_clip=obs_clip,
        )
    else:
        logger.warning(
            "log_transformed_data_overview: no selected features found — "
            "env.feature_columns=%s; selected_features folder will not be created.",
            selected_cols or "not set",
        )

    log_feature_descriptive_stats(df, config)
    _log_feature_vs_return_scatter(df, config)
    if getattr(getattr(config, "logging", None), "log_oracle_alignment_plot", False):
        _log_oracle_vs_reward_alignment(df, config)


def log_feature_descriptive_stats(df: pd.DataFrame, config: Any) -> None:
    """Log per-feature descriptive statistics for all training variables.

    Computes min, max, mean, median, std, percentiles, skewness, kurtosis, and
    null counts for every feature column in *df* and logs the result as
    ``transformed_data_overview/stats/feature_stats.json`` and ``.csv``.
    """
    logger = get_project_logger(__name__)

    if not mlflow.active_run():
        logger.warning("no active mlflow run skipping feature descriptive stats")
        return

    feat_cols = [c for c in df.columns if str(c).startswith("feature_")]
    if not feat_cols:
        logger.warning("log_feature_descriptive_stats: no feature_ columns found, skipping")
        return

    feat_df = df[feat_cols]
    n_total = len(feat_df)

    rows: list[dict] = []
    percentiles = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]

    for col in feat_cols:
        s = feat_df[col].dropna().astype(float)
        non_null = len(s)
        null_count = n_total - non_null

        row: dict = {
            "feature": col,
            "count_total": n_total,
            "count_non_null": non_null,
            "count_null": null_count,
            "null_pct": round(null_count / n_total * 100, 4) if n_total > 0 else None,
        }

        if non_null == 0:
            for stat in ("mean", "std", "min", "p01", "p05", "p25", "p50", "p75", "p95", "p99", "max",
                         "skewness", "kurtosis", "range", "iqr", "cv"):
                row[stat] = None
        else:
            q = s.quantile(percentiles).to_dict()
            mean_val = float(s.mean())
            std_val = float(s.std(ddof=1)) if non_null > 1 else 0.0
            min_val = float(s.min())
            max_val = float(s.max())
            p25 = float(q[0.25])
            p75 = float(q[0.75])

            row.update({
                "mean": round(mean_val, 8),
                "std": round(std_val, 8),
                "min": round(min_val, 8),
                "p01": round(float(q[0.01]), 8),
                "p05": round(float(q[0.05]), 8),
                "p25": round(p25, 8),
                "p50": round(float(q[0.50]), 8),
                "p75": round(p75, 8),
                "p95": round(float(q[0.95]), 8),
                "p99": round(float(q[0.99]), 8),
                "max": round(max_val, 8),
                "skewness": round(float(s.skew()), 6) if non_null > 2 else None,
                "kurtosis": round(float(s.kurt()), 6) if non_null > 3 else None,
                "range": round(max_val - min_val, 8),
                "iqr": round(p75 - p25, 8),
                "cv": round(std_val / abs(mean_val), 6) if abs(mean_val) > 1e-12 else None,
            })

        rows.append(row)

    stats_df = pd.DataFrame(rows)
    artifact_dir = "transformed_data_overview/stats"

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "feature_stats.csv")
            json_path = os.path.join(tmpdir, "feature_stats.json")

            stats_df.to_csv(csv_path, index=False)
            mlflow.log_artifact(csv_path, artifact_dir)

            # Serialize NaN as null in JSON
            serializable = []
            for row in rows:
                clean_row = {}
                for k, v in row.items():
                    if v is None:
                        clean_row[k] = None
                    elif isinstance(v, float) and not np.isfinite(v):
                        clean_row[k] = None
                    else:
                        clean_row[k] = v
                serializable.append(clean_row)

            with open(json_path, "w") as fh:
                json.dump(serializable, fh, indent=2)
            mlflow.log_artifact(json_path, artifact_dir)

        logger.info(
            "log_feature_descriptive_stats: logged stats for %d features to %s",
            len(feat_cols),
            artifact_dir,
        )
    except Exception as e:  # pragma: no cover
        logger.warning("log_feature_descriptive_stats failed err=%s", e)


def _log_feature_vs_return_scatter(df: pd.DataFrame, config: Any) -> None:
    """Log scatter plots of feature_bid_px_00 and feature_ask_px_00 vs next-step log return."""
    logger = get_project_logger(__name__)

    if not mlflow.active_run():
        return

    price_col = getattr(getattr(config, "env", None), "price_column", "close")
    target_features = [f for f in ("feature_bid_px_00", "feature_ask_px_00") if f in df.columns]

    if not target_features or price_col not in df.columns:
        return

    try:
        from plotnine import (
            aes,
            geom_point,
            geom_smooth,
            ggplot,
            labs,
        )
        from trading_rl.evaluation.thesis_theme import thesis_theme

        prices = df[price_col].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_ret = np.concatenate([np.diff(np.log(prices)), [np.nan]])
        log_ret = np.where(np.isfinite(log_ret), log_ret, np.nan)

        scatter_df = df[target_features].copy()
        scatter_df["log_return"] = log_ret
        scatter_df = scatter_df.dropna()

        # Subsample for plotting performance
        if len(scatter_df) > 2000:
            scatter_df = scatter_df.sample(2000, random_state=42)

        for feat in target_features:
            try:
                plot_data = scatter_df[[feat, "log_return"]].rename(columns={feat: "feature_value"})
                p = (
                    ggplot(plot_data, aes(x="feature_value", y="log_return"))
                    + geom_point(alpha=0.2, size=0.8, color="steelblue")
                    + geom_smooth(method="lm", color="red", size=0.8)
                    + labs(
                        title=f"{feat} vs Next-Step Log Return",
                        x=feat,
                        y="Log Return (t+1)",
                    )
                    + thesis_theme()
                )
                temp_path = os.path.join(tempfile.gettempdir(), f"{feat}_vs_log_return.png")
                p.save(temp_path, width=12, height=8, dpi=225)
                mlflow.log_artifact(temp_path, "transformed_data_overview/plots")
                os.unlink(temp_path)
            except Exception as e:  # pragma: no cover
                logger.warning("feature vs return scatter failed feat=%s err=%s", feat, e)

    except Exception as e:  # pragma: no cover
        logger.warning("_log_feature_vs_return_scatter failed err=%s", e)


def _log_oracle_vs_reward_alignment(
    df: pd.DataFrame,
    config: Any,
    n_points: int = 10_000,
) -> None:
    """Scatter plot of feature_future_close_vel[t] vs next-step log return[t+1].

    Checks whether the oracle feature and reward signal are temporally aligned.
    A correlation near 1.0 confirms they measure the same price movement.
    """
    logger = get_project_logger(__name__)

    if not mlflow.active_run():
        return

    oracle_col = "feature_future_close_vel"
    if oracle_col not in df.columns:
        logger.debug("oracle alignment plot skipped: %s not in df", oracle_col)
        return

    price_col = getattr(getattr(config, "env", None), "price_column", "close")
    if price_col not in df.columns:
        logger.debug("oracle alignment plot skipped: price column %s not in df", price_col)
        return

    try:
        from plotnine import (
            aes,
            annotate,
            geom_point,
            geom_smooth,
            ggplot,
            labs,
        )
        from trading_rl.evaluation.thesis_theme import thesis_theme

        prices = df[price_col].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_ret = np.concatenate([[np.nan], np.diff(np.log(prices))])
        log_ret = np.where(np.isfinite(log_ret), log_ret, np.nan)
        log_return_next = np.roll(log_ret, -1)

        def _make_plot_df(oracle_values: np.ndarray) -> pd.DataFrame:
            d = pd.DataFrame({"oracle": oracle_values, "log_return_next": log_return_next})
            d.iloc[-1, d.columns.get_loc("log_return_next")] = np.nan
            d = d.dropna()
            if len(d) > n_points:
                d = d.sample(n_points, random_state=42)
            return d

        def _scatter(plot_df: pd.DataFrame, x_label: str, filename: str, metric_key: str) -> None:
            corr = float(plot_df["oracle"].corr(plot_df["log_return_next"]))
            mlflow.log_metric(metric_key, corr)
            p = (
                ggplot(plot_df, aes(x="oracle", y="log_return_next"))
                + geom_point(alpha=0.15, size=0.6, color="steelblue")
                + geom_smooth(method="lm", color="red", size=1.0)
                + annotate("text", x=plot_df["oracle"].quantile(0.05),
                           y=plot_df["log_return_next"].max() * 0.9,
                           label=f"r = {corr:.4f}", size=11, color="darkred")
                + labs(
                    title="Oracle Feature vs Next-Step Log Return (alignment check)",
                    x=x_label,
                    y="Log Return [t+1]",
                )
                + thesis_theme()
            )
            temp_path = os.path.join(tempfile.gettempdir(), filename)
            p.save(temp_path, width=12, height=8, dpi=225)
            mlflow.log_artifact(temp_path, "transformed_data_overview/plots")
            os.unlink(temp_path)
            logger.info("log oracle alignment plot filename=%s corr=%.4f n=%d", filename, corr, len(plot_df))

        # Normalised feature column
        plot_df_norm = _make_plot_df(df[oracle_col].to_numpy(dtype=float))
        _scatter(plot_df_norm, "feature_future_close_vel (normalised)",
                 "oracle_vs_reward_alignment_normalised.png",
                 "oracle_reward_alignment_corr")

        # Raw (unnormalised) mid-price velocity, computed directly from bid/ask
        if {"bid_px_00", "ask_px_00"}.issubset(df.columns):
            mid = (df["bid_px_00"] + df["ask_px_00"]) / 2.0
            raw_oracle = mid.diff().shift(-1).fillna(0.0).to_numpy(dtype=float)
            plot_df_raw = _make_plot_df(raw_oracle)
            _scatter(plot_df_raw, "mid-price velocity raw (bid+ask)/2 diff [t+1]",
                     "oracle_vs_reward_alignment_raw.png",
                     "oracle_reward_alignment_corr_raw")
        else:
            logger.debug("oracle alignment raw plot skipped: bid_px_00/ask_px_00 not in df")

    except Exception as e:  # pragma: no cover
        logger.warning("oracle alignment plot failed err=%s", e)
