"""Evaluation result logging helpers for MLflow."""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlflow
import numpy as np
import pandas as pd

from logger import get_logger as get_project_logger
from trading_rl.evaluation.asset_meta import write_asset_meta
from trading_rl.evaluation.metrics import MetricReport

if TYPE_CHECKING:
    from trading_rl.callbacks.mlflow_callback import MLflowTrainingCallback


class ArtifactPaths:
    """Central registry of MLflow artifact folder names.

    Root constants are used as default argument values in artifact functions.
    Builder staticmethods produce the full parameterised path for each use case.
    """

    EVAL_PLOTS = "evaluation_plots"
    EVAL_DATA = "evaluation_data"
    EVAL_PLOTS_TEMP = "evaluation_plots_temp"
    EVAL_DATA_TEMP = "evaluation_data_temp"
    EXPLAINABILITY = "explainability"

    @classmethod
    def eval_plots(cls, split: str) -> str:
        return f"{cls.EVAL_PLOTS}/{split}"

    @classmethod
    def eval_plots_temp(cls, split: str, step: int) -> str:
        return f"{cls.EVAL_PLOTS_TEMP}/{split}/step_{step:08d}"

    @classmethod
    def eval_data(cls, split: str) -> str:
        return f"{cls.EVAL_DATA}/{split}"

    @classmethod
    def eval_data_temp(cls, split: str, step: int) -> str:
        return f"{cls.EVAL_DATA_TEMP}/{split}/step_{step:08d}"

    @classmethod
    def explainability(cls, split: str) -> str:
        return f"{cls.EXPLAINABILITY}/{split}"


def save_observation_sample_artifact(
    *,
    split: str,
    df: pd.DataFrame,
    output_dir: str | Path,
    max_rows: int = 5000,
    artifact_path_prefix: str = ArtifactPaths.EVAL_DATA,
) -> Path:
    """Save and optionally log the first rows used for split evaluation."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_split = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in split)
    n_rows = min(max_rows, len(df))
    out_path = output_dir / f"{safe_split}_observations_head_{n_rows}.parquet"
    df.head(max_rows).to_parquet(out_path)
    write_asset_meta(out_path, generator="callbacks/artifacts_evaluation.py")

    if mlflow.active_run() is not None:
        mlflow.log_artifact(str(out_path), artifact_path_prefix)

    return out_path


def save_eval_rollout_artifact(
    *,
    split: str,
    last_positions: list[Any],
    simple_returns: np.ndarray,
    cumulative_returns: np.ndarray | None,
    df_index: pd.Index,
    output_dir: str | Path,
    artifact_path_prefix: str = ArtifactPaths.EVAL_DATA,
) -> Path:
    """Serialize per-step rollout data to CSV and log as an MLflow artifact.

    Columns: action, simple_return, cumulative_log_return (when available).
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n = min(len(last_positions), len(simple_returns), len(df_index))
    data: dict = {
        "action": np.array(last_positions[:n], dtype=np.float32),
        "simple_return": np.asarray(simple_returns[:n], dtype=np.float32),
    }
    if cumulative_returns is not None:
        cum = np.asarray(cumulative_returns)
        if len(cum) == n + 1:
            cum = cum[1:]
        data["cumulative_log_return"] = cum[:n].astype(np.float32)

    safe_split = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in split)
    out_path = output_dir / f"{safe_split}_rollout.parquet"
    pd.DataFrame(data, index=df_index[:n]).to_parquet(out_path)
    write_asset_meta(out_path, generator="callbacks/artifacts_evaluation.py")

    if mlflow.active_run() is not None:
        mlflow.log_artifact(str(out_path), artifact_path_prefix)

    return out_path


def log_final_metrics(
    logs: dict,
    final_metrics: dict,
    training_callback: MLflowTrainingCallback | None = None,
) -> None:
    """Log final training metrics to MLflow."""
    logger = get_project_logger(__name__)
    mlflow.log_metric("final_reward", final_metrics["final_reward"])
    mlflow.log_metric(
        "optimizer_steps",
        final_metrics.get("optimizer_steps", final_metrics.get("training_steps", 0)),
    )
    mlflow.log_metric("total_env_steps", final_metrics.get("total_env_steps", 0))
    mlflow.log_metric("total_episodes", final_metrics.get("total_episodes", 0))

    if "last_position_per_episode" in final_metrics:
        positions = final_metrics["last_position_per_episode"]
        if positions:
            mlflow.log_metric("last_position_sequence_length", len(positions))
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                f.write(json.dumps(positions[:100]))
                f.flush()
                mlflow.log_artifact(f.name, "position_data")
                os.unlink(f.name)
    elif "portfolio_weights" in final_metrics:
        weights = final_metrics["portfolio_weights"]
        if weights:
            timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S")
            temp_path = os.path.join(tempfile.gettempdir(), f"{timestamp}_weights.json")
            with open(temp_path, "w") as f:
                f.write(json.dumps(weights[:100]))
            mlflow.log_artifact(temp_path, "portfolio_weights_data")
            os.unlink(temp_path)

    if logs.get("loss_actor"):
        mlflow.log_metric("avg_actor_loss", np.mean(logs["loss_actor"]))
    else:
        logger.warning(
            "No actor loss data available for logging - training may have been skipped "
            "due to tensor shape issues"
        )

    if training_callback:
        training_curves = training_callback.get_training_curves()

        if training_curves["episode_rewards"]:
            mlflow.log_metric(
                "episode_avg_reward", np.mean(training_curves["episode_rewards"])
            )

        if training_curves["portfolio_valuations"]:
            mlflow.log_metric(
                "episode_portfolio_valuation",
                training_curves["portfolio_valuations"][-1],
            )

        if training_curves["position_change_counts"]:
            position_changes = training_curves["position_change_counts"]
            mlflow.log_metric(
                "episode_avg_position_change", float(np.mean(position_changes))
            )
            mlflow.log_metric("total_position_changes", int(np.sum(position_changes)))

            total_episodes = len(training_curves["episode_rewards"])
            total_actions = len(training_callback.training_stats["actions_taken"])
            avg_transitions = (
                (total_actions - total_episodes) / total_episodes
                if total_episodes > 0 and total_actions > total_episodes
                else 1.0
            )
            avg_position_change_ratio = np.mean(position_changes) / avg_transitions
            mlflow.log_metric(
                "episode_avg_position_change_ratio", avg_position_change_ratio
            )


def log_evaluation_report(
    report: MetricReport | dict[str, float],
    split_prefix: str | None = None,
) -> None:
    """Log evaluation report metrics and JSON artifact to MLflow.

    Args:
        report: Dictionary of evaluation metrics.
        split_prefix: Optional split name (e.g. "train", "val", "test").
    """
    logger = get_project_logger(__name__)
    if not mlflow.active_run():
        logger.warning("no active mlflow run skipping evaluation report logging")
        return

    artifact_dir = (
        f"evaluation_data/{split_prefix}" if split_prefix else "evaluation_data"
    )

    report_dict = report.to_dict() if isinstance(report, MetricReport) else report

    # Convert NaN/inf → null so every field always appears in the JSON.
    serializable: dict[str, float | None] = {}
    for key, value in report_dict.items():
        try:
            fv = float(value)
            serializable[key] = None if not np.isfinite(fv) else fv
        except (TypeError, ValueError):
            serializable[key] = None

    if not any(v is not None for v in serializable.values()):
        logger.warning("no finite evaluation metrics to log")
        return

    metric_prefix = f"{split_prefix}_" if split_prefix else ""
    for key, value in serializable.items():
        if value is not None:
            mlflow.log_metric(f"{metric_prefix}{key}", value)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "metrics.json")
        with open(path, "w") as fh:
            json.dump(serializable, fh, indent=2, sort_keys=True)
        mlflow.log_artifact(path, artifact_dir)


def log_statistical_tests(
    test_results: dict[str, Any],
    *,
    split_prefix: str | None = None,
    log_to_research_artifacts: bool = False,
    research_artifact_subdir: str = "research_artifacts/statistical_tests",
) -> None:
    """Log statistical significance test results to MLflow.

    Args:
        test_results: Dictionary with all statistical test results.
        split_prefix: Optional split name (e.g. "train", "val", "test").
        log_to_research_artifacts: If True, also log a compact summary bundle.
        research_artifact_subdir: MLflow artifact subdirectory for the summary.
    """
    logger = get_project_logger(__name__)
    if not mlflow.active_run():
        logger.warning("no active mlflow run skipping statistical test logging")
        return

    if not test_results.get("enabled", False):
        logger.debug("statistical testing disabled skipping logging")
        return

    stat_artifact_dir = (
        f"statistical_tests/{split_prefix}" if split_prefix else "statistical_tests"
    )
    split_infix = f"{split_prefix}_" if split_prefix else ""

    mlflow.log_param("stat_tests_enabled", True)
    mlflow.log_param(
        "stat_tests_configured", ",".join(test_results.get("tests_configured", []))
    )

    for baseline_result in test_results.get("baselines", []):
        baseline_name = baseline_result.get("baseline", "unknown")
        if "error" in baseline_result:
            logger.warning(
                "skip baseline baseline={} err={}",
                baseline_name,
                baseline_result["error"],
            )
            continue

        if "n_strategy_samples" in baseline_result:
            mlflow.log_metric(
                f"stat_{split_infix}{baseline_name}_n_strategy",
                baseline_result["n_strategy_samples"],
            )
        if "n_baseline_samples" in baseline_result:
            mlflow.log_metric(
                f"stat_{split_infix}{baseline_name}_n_baseline",
                baseline_result["n_baseline_samples"],
            )

        for test_name, test_data in baseline_result.items():
            if not isinstance(test_data, dict):
                continue
            prefix = f"stat_{split_infix}{baseline_name}_{test_name}"
            for key, value in test_data.items():
                if key in ["test_name", "error"]:
                    continue
                try:
                    if isinstance(value, bool):
                        mlflow.log_metric(f"{prefix}_{key}", float(value))
                    elif isinstance(value, int | float):
                        if np.isfinite(value):
                            mlflow.log_metric(f"{prefix}_{key}", float(value))
                except (TypeError, ValueError):
                    continue

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as handle:
        json.dump(test_results, handle, indent=2, sort_keys=True, default=str)
        handle.flush()
        mlflow.log_artifact(handle.name, stat_artifact_dir)
        os.unlink(handle.name)

    vwap_volume_source = test_results.get("vwap_volume_source")
    if isinstance(vwap_volume_source, str) and vwap_volume_source:
        mlflow.log_param("stat_vwap_volume_source", vwap_volume_source)

    benchmark_table = test_results.get("benchmark_comparison_table")
    if isinstance(benchmark_table, list) and benchmark_table:
        benchmark_df = pd.DataFrame(benchmark_table)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as handle:
            benchmark_df.to_csv(handle.name, index=False)
            mlflow.log_artifact(handle.name, stat_artifact_dir)
            os.unlink(handle.name)

    if log_to_research_artifacts:
        significant_findings: list[dict[str, Any]] = []
        for baseline_result in test_results.get("baselines", []):
            baseline_name = baseline_result.get("baseline", "unknown")
            if not isinstance(baseline_result, dict):
                continue
            for test_name, test_data in baseline_result.items():
                if not isinstance(test_data, dict) or "p_value" not in test_data:
                    continue
                finding: dict[str, Any] = {
                    "baseline": baseline_name,
                    "test": test_name,
                    "p_value": test_data.get("p_value"),
                    "significant": bool(test_data.get("significant", False)),
                }
                if "effect_size" in test_data:
                    finding["effect_size"] = test_data.get("effect_size")
                significant_findings.append(finding)

        summary_payload = {
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "tests_configured": test_results.get("tests_configured", []),
            "n_baselines": len(test_results.get("baselines", [])),
            "n_findings": len(significant_findings),
            "findings": significant_findings,
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as handle:
            json.dump(summary_payload, handle, indent=2, sort_keys=True, default=str)
            handle.flush()
            mlflow.log_artifact(handle.name, research_artifact_subdir)
            os.unlink(handle.name)

    logger.info("log statistical test results to mlflow")


def log_explainability_results(
    importance_df: pd.DataFrame | None,
    importance_plot: Any,
    method: str = "permutation",
    metrics: dict[str, float] | None = None,
    artifact_path_prefix: str | None = None,
) -> None:
    """Log explainability plots and importance data to MLflow.

    Args:
        importance_df: DataFrame with importance scores (None for merged plots).
        importance_plot: Plot object to save.
        method: Method name for artifact naming.
        metrics: Optional metrics dictionary to log.
        artifact_path_prefix: Optional path prefix for MLflow artifacts.
    """
    if not mlflow.active_run():
        return

    artifact_dir = (
        artifact_path_prefix if artifact_path_prefix else ArtifactPaths.EXPLAINABILITY
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        plot_path = tmp_path / f"explainability_{method}.png"
        importance_plot.save(str(plot_path))
        mlflow.log_artifact(str(plot_path), artifact_dir)

        if importance_df is not None:
            csv_path = tmp_path / f"importance_{method}.csv"
            importance_df.to_csv(csv_path, index=False)
            mlflow.log_artifact(str(csv_path), artifact_dir)

        if metrics:
            mlflow.log_metrics({f"{method}_{k}": v for k, v in metrics.items()})


def log_evaluation_plots(
    reward_plot,
    action_plot,
    action_probs_plot=None,
    equity_curve_plot=None,
    logs=None,
    merged_plot=None,
    artifact_path_prefix=None,
    debug: bool = False,
    plot_data: dict | None = None,
) -> None:
    """Save evaluation/training plots as MLflow artifacts.

    Args:
        reward_plot: Cumulative rewards comparison plot.
        action_plot: Actions/portfolio weights plot.
        action_probs_plot: Optional action probabilities plot.
        equity_curve_plot: Portfolio equity curve plot.
        logs: Optional training logs for loss plots.
        merged_plot: Optional merged comparison plot (rewards + actions).
        artifact_path_prefix: Optional path prefix for MLflow artifacts.
        debug: Enable debug mode for plot rendering.
        plot_data: Optional dict from build_rollout_plot_data / build_equity_plot_data.
            When provided, DataFrames are saved as parquet so QMD can re-render plots
            at any figure size without re-running the rollout.
    """
    import contextlib
    import io
    import warnings

    from plotnine.exceptions import PlotnineWarning

    logger = get_project_logger(__name__)

    if not mlflow.active_run():
        logger.warning("no active mlflow run skipping plot artifact logging")
        return

    @contextlib.contextmanager
    def _suppress_plotnine():
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            yield

    artifact_dir = (
        artifact_path_prefix if artifact_path_prefix else ArtifactPaths.EVAL_PLOTS
    )
    saved_paths: dict[str, str] = {}
    batch_temp_dir = tempfile.mkdtemp()
    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S")

    try:
        from trading_rl.evaluation.thesis_theme import save_plot as _save_plot

        def _save(plot_obj, filename, key, dir_, width=8, height=5):
            tmp_path = os.path.join(batch_temp_dir, filename)
            try:
                t_render = time.monotonic()
                logger.trace("render plot filename={} debug={}", filename, debug)
                with warnings.catch_warnings(), _suppress_plotnine():
                    warnings.simplefilter("ignore", PlotnineWarning)
                    _save_plot(
                        plot_obj,
                        tmp_path,
                        width=width,
                        height=height,
                        dpi=225,
                        debug=debug,
                    )
                logger.trace(
                    "render done filename={} elapsed_s={:.2f}",
                    filename,
                    time.monotonic() - t_render,
                )

                pil_logger = logging.getLogger("PIL.PngImagePlugin")
                prev_level = pil_logger.level
                pil_logger.setLevel(logging.INFO)
                t_upload = time.monotonic()
                logger.trace("mlflow log_artifact filename={} dir={}", filename, dir_)
                try:
                    mlflow.log_artifact(tmp_path, dir_)
                finally:
                    pil_logger.setLevel(prev_level)
                logger.trace(
                    "mlflow log_artifact done filename={} elapsed_s={:.2f}",
                    filename,
                    time.monotonic() - t_upload,
                )

                if key:
                    saved_paths[key] = tmp_path
            except Exception:
                logger.exception("save plot failed filename={}", filename)

        _save(reward_plot, f"{timestamp}_rewards.png", "rewards", artifact_dir, 16, 10)
        _save(
            action_plot, f"{timestamp}_positions.png", "positions", artifact_dir, 16, 10
        )

        if action_probs_plot is not None:
            _save(
                action_probs_plot,
                f"{timestamp}_action_probabilities.png",
                "action_probabilities",
                artifact_dir,
                16,
                10,
            )
        else:
            logger.info("action probability plot missing skipping artifact")

        if equity_curve_plot is not None:
            _save(
                equity_curve_plot,
                f"{timestamp}_equity_curve.png",
                "equity_curve",
                artifact_dir,
                16,
                10,
            )
        else:
            logger.warning("equity curve plot missing skipping artifact")

        if merged_plot is not None:
            _save(
                merged_plot,
                f"{timestamp}_merged_comparison.png",
                "merged_comparison",
                artifact_dir,
                16,
                27,
            )
        else:
            logger.info("merged comparison plot missing skipping artifact")

        if logs and (logs.get("loss_value") or logs.get("loss_actor")):
            from plotnine import aes, facet_wrap, geom_line, ggplot, labs

            from trading_rl.evaluation.thesis_theme import thesis_theme

            loss_data = []
            if logs.get("loss_value"):
                loss_data.extend(
                    {"step": i, "loss": loss, "type": "Value Loss"}
                    for i, loss in enumerate(logs["loss_value"])
                )
            if logs.get("loss_actor"):
                loss_data.extend(
                    {"step": i, "loss": loss, "type": "Actor Loss"}
                    for i, loss in enumerate(logs["loss_actor"])
                )
            if loss_data:
                loss_df = pd.DataFrame(loss_data)
                loss_plot = (
                    ggplot(loss_df, aes(x="step", y="loss", color="type"))
                    + geom_line(size=0.72)
                    + facet_wrap("type", ncol=1, scales="free")
                    + labs(
                        title="Training Losses",
                        x="Training Step",
                        y="Loss Value",
                        color="Loss Type",
                    )
                    + thesis_theme()
                )
                _save(
                    loss_plot,
                    f"{timestamp}_training_losses.png",
                    None,
                    "training_plots",
                    16,
                    10,
                )

        # Attempt patchwork combination, fall back to Pillow
        try:
            import importlib.util

            if importlib.util.find_spec("plotnine.patchwork") is None:
                raise ImportError("plotnine.patchwork is unavailable")

            combined_plot = None
            if reward_plot is not None and action_plot is not None:
                combined_plot = reward_plot | action_plot
                if action_probs_plot is not None:
                    combined_plot = combined_plot / action_probs_plot
            elif reward_plot is not None:
                combined_plot = reward_plot
            elif action_plot is not None:
                combined_plot = action_plot
            elif action_probs_plot is not None:
                combined_plot = action_probs_plot

            if combined_plot is not None:
                _save(
                    combined_plot,
                    f"{timestamp}_combined_evaluation.png",
                    None,
                    artifact_dir,
                    20,
                    12,
                )

        except ImportError:
            if {"rewards", "positions", "action_probabilities"} <= set(
                saved_paths.keys()
            ):
                try:
                    from PIL import Image

                    with (
                        Image.open(saved_paths["rewards"]) as reward_img,
                        Image.open(saved_paths["positions"]) as action_img,
                        Image.open(saved_paths["action_probabilities"]) as probs_img,
                    ):
                        top_width = reward_img.width + action_img.width
                        top_height = max(reward_img.height, action_img.height)
                        combined = Image.new(
                            "RGB",
                            (
                                max(top_width, probs_img.width),
                                top_height + probs_img.height,
                            ),
                            "white",
                        )
                        combined.paste(reward_img, (0, 0))
                        combined.paste(action_img, (reward_img.width, 0))
                        combined.paste(probs_img, (0, top_height))

                    tmp_combined = os.path.join(
                        batch_temp_dir, f"{timestamp}_combined_evaluation.png"
                    )
                    combined.save(tmp_combined, format="PNG")
                    mlflow.log_artifact(tmp_combined, artifact_dir)
                except Exception as combine_error:  # pragma: no cover
                    logger.warning(
                        "create combined evaluation plot failed err={}", combine_error
                    )

        # Save plot DataFrames as parquet so QMD can re-render at any figure size
        if plot_data:
            import json

            frames: dict[str, Any] = {}
            if "rewards" in plot_data:
                frames["rewards"] = plot_data["rewards"]
            if "actions" in plot_data:
                frames["actions"] = plot_data["actions"]
            if plot_data.get("actions_ma") is not None:
                frames["actions_ma"] = plot_data["actions_ma"]
            if "returns" in plot_data:
                frames["equity"] = plot_data["returns"]

            meta_keys = {
                "stride",
                "date_str",
                "reward_type",
                "is_portfolio",
                "training_steps",
                "training_episodes",
                "n_obs",
                "allocation_ma_window",
                "initial_portfolio_value",
                "policy_mode",
                "symbols",
                "n_total_symbols",
            }
            meta = {k: plot_data[k] for k in meta_keys if k in plot_data}

            from trading_rl.evaluation.asset_meta import write_asset_meta

            for frame_name, df_frame in frames.items():
                pq_path = os.path.join(
                    batch_temp_dir, f"{timestamp}_{frame_name}_data.parquet"
                )
                df_frame.assign(Run=df_frame["Run"].astype(str)).to_parquet(
                    pq_path, index=False
                )
                write_asset_meta(pq_path, generator="callbacks/artifacts_evaluation.py")
                mlflow.log_artifact(pq_path, artifact_dir)
                sidecar = pq_path + ".meta.json"
                if os.path.exists(sidecar):
                    mlflow.log_artifact(sidecar, artifact_dir)

            meta_path = os.path.join(batch_temp_dir, f"{timestamp}_plot_meta.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, default=str)
            mlflow.log_artifact(meta_path, artifact_dir)
            logger.info("saved plot data parquets frames={}", list(frames))

        logger.info("save evaluation and training plots as mlflow artifacts")
    except Exception as e:  # pragma: no cover
        logger.warning("save plots as artifacts failed err={}", e)
    finally:
        if os.path.exists(batch_temp_dir):
            shutil.rmtree(batch_temp_dir)
