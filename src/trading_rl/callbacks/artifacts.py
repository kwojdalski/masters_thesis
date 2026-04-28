"""Standalone MLflow artifact and metric logging helpers.

These are pure functions — no state, no class required.  They were
previously housed as static methods on ``MLflowTrainingCallback``, which
kept them coupled to the callback lifecycle for no good reason.

All functions are re-exported on ``MLflowTrainingCallback`` as class-level
``staticmethod`` attributes so existing callers (``MLflowTrainingCallback.log_*(…)``)
continue to work without modification.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlflow
import numpy as np
import pandas as pd

from logger import get_logger as get_project_logger

if TYPE_CHECKING:
    from trading_rl.callbacks.mlflow_callback import MLflowTrainingCallback


def log_config_artifact(config) -> None:
    """Log YAML config file as an MLflow artifact."""
    config_dir = Path("src/configs/scenarios")
    config_file = None
    for candidate in config_dir.rglob("*.yaml"):
        cparts = candidate.parts
        try:
            idx = list(cparts).index("scenarios")
            rel_parts = cparts[idx + 1:]
            candidate_name = (
                "_".join(Path(*rel_parts).with_suffix("").parts)
                if rel_parts
                else candidate.stem
            )
        except ValueError:
            candidate_name = candidate.stem
        if candidate_name == config.experiment_name:
            config_file = candidate
            break

    if config_file and config_file.exists():
        mlflow.log_artifact(str(config_file), "config")
        return

    # Serialize the in-memory config
    config_dict = {
        "experiment_name": config.experiment_name,
        "seed": config.seed,
        "data": {
            "data_path": config.data.data_path,
            "download_data": config.data.download_data,
            "exchange_names": config.data.exchange_names,
            "symbols": config.data.symbols,
            "timeframe": config.data.timeframe,
            "data_dir": config.data.data_dir,
            "download_since": config.data.download_since,
            "train_size": config.data.train_size,
        },
        "env": {
            "name": config.env.name,
            "positions": config.env.positions,
            "trading_fees": config.env.trading_fees,
            "borrow_interest_rate": config.env.borrow_interest_rate,
        },
        "network": {
            "actor_hidden_dims": config.network.actor_hidden_dims,
            "value_hidden_dims": config.network.value_hidden_dims,
        },
        "training": {
            "algorithm": config.training.algorithm,
            "actor_lr": config.training.actor_lr,
            "value_lr": config.training.value_lr,
            "value_weight_decay": config.training.value_weight_decay,
            "max_steps": config.training.max_steps,
            "init_rand_steps": config.training.init_rand_steps,
            "frames_per_batch": config.training.frames_per_batch,
            "optim_steps_per_batch": config.training.optim_steps_per_batch,
            "sample_size": config.training.sample_size,
            "buffer_size": config.training.buffer_size,
            "loss_function": config.training.loss_function,
            "eval_steps": config.training.eval_steps,
            "eval_interval": config.training.eval_interval,
            "log_interval": config.training.log_interval,
        },
        "logging": {
            "log_dir": config.logging.log_dir,
            "log_level": config.logging.log_level,
        },
    }

    for attr in ("tau", "clip_epsilon", "entropy_bonus", "vf_coef", "ppo_epochs"):
        if hasattr(config.training, attr):
            config_dict["training"][attr] = getattr(config.training, attr)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        import yaml

        yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)
        mlflow.log_artifact(f.name, "config")
        os.unlink(f.name)


def log_training_parameters(config) -> None:
    """Log core training parameters to MLflow."""
    from trading_rl.config import DEFAULT_INITIAL_PORTFOLIO_VALUE

    try:
        mlflow.log_param("experiment_name", str(config.experiment_name))
        mlflow.log_param("seed", int(config.seed))

        mlflow.log_param("data_train_size", int(config.data.train_size))
        mlflow.log_param("data_timeframe", str(config.data.timeframe))
        mlflow.log_param("data_exchange_names", json.dumps(config.data.exchange_names))
        mlflow.log_param("data_symbols", json.dumps(config.data.symbols))
        mlflow.log_param("data_download_data", bool(config.data.download_data))

        mlflow.log_param("env_name", str(config.env.name))
        mlflow.log_param("env_positions", json.dumps(config.env.positions))
        mlflow.log_param("env_trading_fees", float(config.env.trading_fees))
        mlflow.log_param("env_borrow_interest_rate", float(config.env.borrow_interest_rate))
        mlflow.log_param(
            "env_initial_portfolio_value",
            float(getattr(config.env, "initial_portfolio_value", DEFAULT_INITIAL_PORTFOLIO_VALUE)),
        )

        mlflow.log_param("network_actor_hidden_dims", json.dumps(config.network.actor_hidden_dims))
        mlflow.log_param("network_value_hidden_dims", json.dumps(config.network.value_hidden_dims))

        mlflow.log_param("training_algorithm", str(config.training.algorithm))
        mlflow.log_param("training_actor_lr", float(config.training.actor_lr))
        mlflow.log_param("training_value_lr", float(config.training.value_lr))
        mlflow.log_param("training_value_weight_decay", float(config.training.value_weight_decay))
        mlflow.log_param("training_max_steps", int(config.training.max_steps))
        mlflow.log_param("training_init_rand_steps", int(config.training.init_rand_steps))
        mlflow.log_param("training_frames_per_batch", int(config.training.frames_per_batch))
        mlflow.log_param("training_optim_steps_per_batch", int(config.training.optim_steps_per_batch))
        mlflow.log_param("training_sample_size", int(config.training.sample_size))
        mlflow.log_param("training_buffer_size", int(config.training.buffer_size))
        mlflow.log_param(
            "training_checkpoint_interval",
            int(getattr(config.training, "checkpoint_interval", 0)),
        )
        mlflow.log_param("training_loss_function", str(config.training.loss_function))
        mlflow.log_param("training_eval_steps", int(config.training.eval_steps))
        mlflow.log_param("training_eval_interval", int(config.training.eval_interval))
        mlflow.log_param("training_log_interval", int(config.training.log_interval))

        for attr, cast in (
            ("tau", float),
            ("clip_epsilon", float),
            ("entropy_bonus", float),
            ("vf_coef", float),
            ("ppo_epochs", int),
        ):
            if hasattr(config.training, attr):
                mlflow.log_param(f"training_{attr}", cast(getattr(config.training, attr)))

        mlflow.log_param("logging_log_dir", str(config.logging.log_dir))
        mlflow.log_param("logging_log_level", str(config.logging.log_level))

    except Exception as e:  # pragma: no cover - defensive
        get_project_logger(__name__).warning(f"Failed to log some training parameters: {e}")


def log_parameter_faq_artifact() -> None:
    """Log parameter FAQ as both markdown and HTML artifacts."""
    logger = get_project_logger(__name__)

    try:
        if not mlflow.active_run():
            logger.warning("no active mlflow run skipping faq artifacts")
            return

        faq_path = Path(__file__).resolve().parent.parent / "docs" / "parameter_faq.md"
        if not faq_path.exists():
            logger.warning("faq file not found path=%s", faq_path)
            return

        try:
            mlflow.log_artifact(str(faq_path), "documentation")
            logger.info("log faq markdown artifact")
        except Exception as md_error:
            logger.error("log faq markdown failed err=%s", md_error)
            return

        try:
            import markdown

            with open(faq_path, encoding="utf-8") as f:
                md_content = f.read()

            try:
                html_content = markdown.markdown(
                    md_content, extensions=["tables", "fenced_code", "toc"]
                )
            except Exception as ext_error:
                logger.warning("markdown extensions failed trying basic conversion err=%s", ext_error)
                html_content = markdown.markdown(md_content)

            styled_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Parameter FAQ - Trading RL Experiments</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; line-height: 1.6; }}
        h1, h2, h3 {{ color: #333; }}
        h1 {{ border-bottom: 2px solid #eee; padding-bottom: 10px; }}
        h2 {{ border-bottom: 1px solid #eee; padding-bottom: 5px; margin-top: 30px; }}
        code {{ background: #f5f5f5; padding: 2px 4px; border-radius: 3px; font-family: 'Monaco', 'Consolas', monospace; }}
        pre {{ background: #f5f5f5; padding: 10px; border-radius: 5px; overflow-x: auto; }}
        ul, ol {{ padding-left: 20px; }}
        li {{ margin: 5px 0; }}
        strong {{ color: #2c3e50; }}
        blockquote {{ border-left: 4px solid #ddd; margin-left: 0; padding-left: 20px; color: #666; }}
        table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f5f5f5; }}
    </style>
</head>
<body>
{html_content}
</body>
</html>"""

            temp_dir = tempfile.gettempdir()
            html_temp_path = os.path.join(temp_dir, "parameter_faq.html")

            with open(html_temp_path, "w", encoding="utf-8") as f:
                f.write(styled_html)

            if os.path.exists(html_temp_path):
                mlflow.log_artifact(html_temp_path, "documentation")
                logger.info("log faq html artifact")
                os.unlink(html_temp_path)
            else:
                logger.error("html file was not created")

        except ImportError:
            logger.warning("markdown library not available skipping html conversion")
        except Exception as html_error:
            logger.error("log faq html failed err=%s", html_error)

    except Exception as e:
        logger.error("faq artifact logging failed err=%s", e)


def log_data_overview(df, config) -> None:
    """Log dataset overview, sample, and quick visuals to MLflow."""
    logger = get_project_logger(__name__)

    if not mlflow.active_run():
        logger.warning("no active mlflow run skipping data overview logging")
        return

    try:
        from plotnine import aes, element_text, geom_step, ggplot, labs, theme

        mlflow.log_param("dataset_shape", f"{df.shape[0]}x{df.shape[1]}")
        mlflow.log_param("dataset_columns", list(df.columns))
        mlflow.log_param("date_range", f"{df.index.min()} to {df.index.max()}")
        mlflow.log_param("data_source", config.data.data_path)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.head(50).to_csv(f.name)
            mlflow.log_artifact(f.name, "data_overview")
            os.unlink(f.name)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Dataset Overview\n")
            f.write("================\n\n")
            f.write(f"Shape: {df.shape}\n")
            f.write(f"Columns: {list(df.columns)}\n")
            f.write(f"Date Range: {df.index.min()} to {df.index.max()}\n\n")
            f.write("Data Types:\n")
            f.write(str(df.dtypes))
            f.write("\n\nStatistical Summary:\n")
            f.write(str(df.describe()))
            f.flush()
            mlflow.log_artifact(f.name, "data_overview")
            os.unlink(f.name)

        plot_df = df.head(200).reset_index()
        plot_df["time_index"] = range(len(plot_df))

        ohlcv_columns = ["open", "high", "low", "close", "volume"]
        available_columns = [col for col in ohlcv_columns if col in plot_df.columns]
        feature_columns = [
            col
            for col in plot_df.columns
            if col not in [*ohlcv_columns, plot_df.columns[0], "time_index"]
        ]

        for column in available_columns + feature_columns[:5]:
            try:
                p = (
                    ggplot(plot_df, aes(x="time_index", y=column))
                    + geom_step(color="steelblue", size=0.8)
                    + labs(
                        title=f"{column.title()} Over Time",
                        x="Time Index",
                        y=column.title(),
                    )
                    + theme(plot_title=element_text(size=14, face="bold"))
                )
                temp_path = os.path.join(tempfile.gettempdir(), f"{column}.png")
                p.save(temp_path, width=16, height=10, dpi=150)
                mlflow.log_artifact(temp_path, "data_overview/plots")
                os.unlink(temp_path)
            except Exception as plot_error:  # pragma: no cover
                logger.warning("create plot failed column=%s err=%s", column, plot_error)

        if all(col in plot_df.columns for col in ["open", "high", "low", "close"]):
            try:
                ohlc_melted = pd.melt(
                    plot_df[["time_index", "open", "high", "low", "close"]].dropna(),
                    id_vars=["time_index"],
                    value_vars=["open", "high", "low", "close"],
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
                    + theme(plot_title=element_text(size=14, face="bold"))
                )
                temp_path = os.path.join(tempfile.gettempdir(), "ohlc_combined.png")
                p_combined.save(temp_path, width=20, height=10, dpi=150)
                mlflow.log_artifact(temp_path, "data_overview/plots")
                os.unlink(temp_path)
            except Exception as combined_error:  # pragma: no cover
                logger.warning("create combined ohlc plot failed err=%s", combined_error)

    except Exception as e:  # pragma: no cover
        logger.warning("log data overview failed err=%s", e)


def log_final_metrics(
    logs: dict,
    final_metrics: dict,
    training_callback: MLflowTrainingCallback | None = None,
) -> None:
    """Log final training metrics to MLflow."""
    logger = get_project_logger(__name__)
    mlflow.log_metric("final_reward", final_metrics["final_reward"])
    mlflow.log_metric("training_steps", final_metrics["training_steps"])

    if "last_position_per_episode" in final_metrics:
        positions = final_metrics["last_position_per_episode"]
        if positions:
            mlflow.log_metric("last_position_sequence_length", len(positions))
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
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
            mlflow.log_metric("episode_avg_reward", np.mean(training_curves["episode_rewards"]))

        if training_curves["portfolio_valuations"]:
            mlflow.log_metric(
                "episode_portfolio_valuation",
                training_curves["portfolio_valuations"][-1],
            )

        if training_curves["position_change_counts"]:
            position_changes = training_curves["position_change_counts"]
            mlflow.log_metric("episode_avg_position_change", float(np.mean(position_changes)))
            mlflow.log_metric("total_position_changes", int(np.sum(position_changes)))

            total_episodes = len(training_curves["episode_rewards"])
            total_actions = len(training_callback.training_stats["actions_taken"])
            avg_episode_length = (
                total_actions / total_episodes
                if total_episodes > 0 and total_actions > 0
                else 1.0
            )
            avg_position_change_ratio = np.mean(position_changes) / avg_episode_length
            mlflow.log_metric("episode_avg_position_change_ratio", avg_position_change_ratio)


def log_evaluation_report(
    report: dict[str, float],
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

    metric_prefix = f"eval_{split_prefix}_" if split_prefix else "eval_"
    artifact_dir = f"evaluation_metrics/{split_prefix}" if split_prefix else "evaluation_metrics"

    clean_report: dict[str, float] = {}
    for key, value in report.items():
        try:
            metric_value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(metric_value):
            clean_report[key] = metric_value
            mlflow.log_metric(f"{metric_prefix}{key}", metric_value)

    if not clean_report:
        logger.warning("no finite evaluation metrics to log")
        return

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as handle:
        json.dump(clean_report, handle, indent=2, sort_keys=True)
        handle.flush()
        mlflow.log_artifact(handle.name, artifact_dir)
        os.unlink(handle.name)


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

    stat_artifact_dir = f"statistical_tests/{split_prefix}" if split_prefix else "statistical_tests"
    split_infix = f"{split_prefix}_" if split_prefix else ""

    mlflow.log_param("stat_tests_enabled", True)
    mlflow.log_param(
        "stat_tests_configured", ",".join(test_results.get("tests_configured", []))
    )

    for baseline_result in test_results.get("baselines", []):
        baseline_name = baseline_result.get("baseline", "unknown")
        if "error" in baseline_result:
            logger.warning("skip baseline baseline=%s err=%s", baseline_name, baseline_result['error'])
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
                    elif isinstance(value, (int, float)):
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
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as handle:
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
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as handle:
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

    artifact_dir = artifact_path_prefix if artifact_path_prefix else "explainability"

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
    actual_returns_plot=None,
    logs=None,
    merged_plot=None,
    artifact_path_prefix=None,
) -> None:
    """Save evaluation/training plots as MLflow artifacts.

    Args:
        reward_plot: Cumulative rewards comparison plot.
        action_plot: Actions/portfolio weights plot.
        action_probs_plot: Optional action probabilities plot.
        actual_returns_plot: Actual portfolio returns plot.
        logs: Optional training logs for loss plots.
        merged_plot: Optional merged comparison plot (rewards + actions).
        artifact_path_prefix: Optional path prefix for MLflow artifacts.
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

    artifact_dir = artifact_path_prefix if artifact_path_prefix else "evaluation_plots"
    saved_paths: dict[str, str] = {}
    batch_temp_dir = tempfile.mkdtemp()
    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S")

    try:
        def _save(plot_obj, filename, key, dir_, width=8, height=5):
            tmp_path = os.path.join(batch_temp_dir, filename)
            try:
                with warnings.catch_warnings(), _suppress_plotnine():
                    warnings.simplefilter("ignore", PlotnineWarning)
                    if hasattr(plot_obj, "save"):
                        plot_obj.save(tmp_path, width=width, height=height, dpi=150)
                    elif hasattr(plot_obj, "savefig"):
                        plot_obj.savefig(tmp_path, dpi=150)
                    else:
                        raise RuntimeError("Unsupported plot object for saving")
                pil_logger = logging.getLogger("PIL.PngImagePlugin")
                prev_level = pil_logger.level
                pil_logger.setLevel(logging.INFO)
                try:
                    mlflow.log_artifact(tmp_path, dir_)
                finally:
                    pil_logger.setLevel(prev_level)
                if key:
                    saved_paths[key] = tmp_path
            except Exception:
                logger.exception("save plot failed filename=%s", filename)

        _save(reward_plot, f"{timestamp}_rewards.png", "rewards", artifact_dir, 16, 10)
        _save(action_plot, f"{timestamp}_positions.png", "positions", artifact_dir, 16, 10)

        if action_probs_plot is not None:
            _save(action_probs_plot, f"{timestamp}_action_probabilities.png", "action_probabilities", artifact_dir, 16, 10)
        else:
            logger.info("action probability plot missing skipping artifact")

        if actual_returns_plot is not None:
            _save(actual_returns_plot, f"{timestamp}_actual_returns.png", "actual_returns", artifact_dir, 16, 10)
        else:
            logger.warning("actual returns plot missing skipping artifact")

        if merged_plot is not None:
            _save(merged_plot, f"{timestamp}_merged_comparison.png", "merged_comparison", artifact_dir, 16, 16)
        else:
            logger.info("merged comparison plot missing skipping artifact")

        if logs and (logs.get("loss_value") or logs.get("loss_actor")):
            from plotnine import aes, facet_wrap, geom_line, ggplot, labs

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
                    + geom_line(size=1.2)
                    + facet_wrap("type", ncol=1, scales="free")
                    + labs(title="Training Losses", x="Training Step", y="Loss Value", color="Loss Type")
                )
                _save(loss_plot, f"{timestamp}_training_losses.png", None, "training_plots", 16, 10)

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
                _save(combined_plot, f"{timestamp}_combined_evaluation.png", None, artifact_dir, 20, 12)

        except ImportError:
            if {"rewards", "positions", "action_probabilities"} <= set(saved_paths.keys()):
                try:
                    from PIL import Image

                    reward_img = Image.open(saved_paths["rewards"])
                    action_img = Image.open(saved_paths["positions"])
                    probs_img = Image.open(saved_paths["action_probabilities"])

                    top_width = reward_img.width + action_img.width
                    top_height = max(reward_img.height, action_img.height)
                    combined = Image.new("RGB", (max(top_width, probs_img.width), top_height + probs_img.height), "white")
                    combined.paste(reward_img, (0, 0))
                    combined.paste(action_img, (reward_img.width, 0))
                    combined.paste(probs_img, (0, top_height))

                    tmp_combined = os.path.join(batch_temp_dir, f"{timestamp}_combined_evaluation.png")
                    combined.save(tmp_combined, format="PNG")
                    mlflow.log_artifact(tmp_combined, artifact_dir)
                except Exception as combine_error:  # pragma: no cover
                    logger.warning("create combined evaluation plot failed err=%s", combine_error)

        logger.info("save evaluation and training plots as mlflow artifacts")
    except Exception as e:  # pragma: no cover
        logger.warning("save plots as artifacts failed err=%s", e)
    finally:
        if os.path.exists(batch_temp_dir):
            shutil.rmtree(batch_temp_dir)
