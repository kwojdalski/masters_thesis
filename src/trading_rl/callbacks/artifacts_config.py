"""Config and parameter logging helpers for MLflow."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import mlflow

from logger import get_logger as get_project_logger


def _to_yaml_serializable(obj):
    """Recursively convert enum values and other non-YAML-safe types to primitives."""
    import enum

    if isinstance(obj, dict):
        return {k: _to_yaml_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_yaml_serializable(v) for v in obj]
    if isinstance(obj, enum.Enum):
        return obj.value
    return obj


def log_config_artifact(config) -> None:
    """Log effective runtime config as an MLflow artifact.

    Always serializes the in-memory config (including any CLI overrides) to
    ``config/effective_config.yaml``.  If the originating scenario YAML can be
    identified it is also logged to ``config/source_scenario.yaml`` for
    reference, but it never replaces the effective config.
    """
    import yaml

    # Log source YAML as reference only — do not return early.
    config_dir = Path("src/configs/scenarios")
    for candidate in config_dir.rglob("*.yaml"):
        cparts = candidate.parts
        try:
            idx = list(cparts).index("scenarios")
            rel_parts = cparts[idx + 1 :]
            candidate_name = (
                "_".join(Path(*rel_parts).with_suffix("").parts)
                if rel_parts
                else candidate.stem
            )
        except ValueError:
            candidate_name = candidate.stem
        if candidate_name == config.experiment_name and candidate.exists():
            mlflow.log_artifact(str(candidate), "config/source")
            break

    # Always log the effective in-memory config (captures overrides).
    config_dict = _to_yaml_serializable(config.to_dict())

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix="effective_config_"
    ) as f:
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
        mlflow.log_param(
            "env_borrow_interest_rate", float(config.env.borrow_interest_rate)
        )
        mlflow.log_param(
            "env_initial_portfolio_value",
            float(
                getattr(
                    config.env,
                    "initial_portfolio_value",
                    DEFAULT_INITIAL_PORTFOLIO_VALUE,
                )
            ),
        )

        mlflow.log_param(
            "network_actor_hidden_dims", json.dumps(config.network.actor_hidden_dims)
        )
        mlflow.log_param(
            "network_value_hidden_dims", json.dumps(config.network.value_hidden_dims)
        )

        mlflow.log_param("training_algorithm", str(config.training.algorithm))
        mlflow.log_param("training_actor_lr", float(config.training.actor_lr))
        mlflow.log_param("training_value_lr", float(config.training.value_lr))
        mlflow.log_param(
            "training_value_weight_decay", float(config.training.value_weight_decay)
        )
        mlflow.log_param("training_max_steps", int(config.training.max_steps))
        mlflow.log_param(
            "training_init_rand_steps", int(config.training.init_rand_steps)
        )
        mlflow.log_param(
            "training_frames_per_batch", int(config.training.frames_per_batch)
        )
        mlflow.log_param(
            "training_optim_steps_per_batch", int(config.training.optim_steps_per_batch)
        )
        mlflow.log_param("training_sample_size", int(config.training.sample_size))
        mlflow.log_param("training_buffer_size", int(config.training.buffer_size))
        mlflow.log_param(
            "training_checkpoint_interval",
            int(getattr(config.training, "checkpoint_interval", 0)),
        )
        mlflow.log_param("training_loss_function", str(config.training.loss_function))
        mlflow.log_param("eval_steps", int(config.evaluation.eval_steps))
        if config.evaluation.eval_fraction is not None:
            mlflow.log_param("eval_fraction", float(config.evaluation.eval_fraction))
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
                mlflow.log_param(
                    f"training_{attr}", cast(getattr(config.training, attr))
                )

        mlflow.log_param("logging_log_dir", str(config.logging.log_dir))
        mlflow.log_param("logging_log_level", str(config.logging.log_level))

    except Exception as e:  # pragma: no cover - defensive
        get_project_logger(__name__).warning(
            f"Failed to log some training parameters: {e}"
        )


def log_parameter_faq_artifact() -> None:
    """Log parameter FAQ as both markdown and HTML artifacts."""
    logger = get_project_logger(__name__)

    try:
        if not mlflow.active_run():
            logger.warning("no active mlflow run skipping faq artifacts")
            return

        faq_path = Path(__file__).resolve().parent.parent / "docs" / "parameter_faq.md"
        if not faq_path.exists():
            logger.warning("faq file not found path={}", faq_path)
            return

        with open(faq_path, encoding="utf-8") as f:
            md_content = f.read()

        try:
            import shutil
            import subprocess

            git_bin = shutil.which("git") or "git"
            commit = (
                subprocess.check_output(  # noqa: S603 -- resolved git binary, fixed args
                    [git_bin, "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
                )
                .decode()
                .strip()
            )
            short = commit[:8]
        except Exception:
            commit, short = "unknown", "unknown"

        run_header = (
            f"<!-- generated -->\n"
            f"## Run Metadata\n\n"
            f"| Field | Value |\n"
            f"|---|---|\n"
            f"| Git commit | `{short}` (`{commit}`) |\n\n"
            f"---\n\n"
        )
        md_content = run_header + md_content

        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".md", delete=False, encoding="utf-8"
            ) as tmp_md:
                tmp_md.write(md_content)
                tmp_md_path = tmp_md.name
            mlflow.log_artifact(tmp_md_path, "documentation")
            os.unlink(tmp_md_path)
            logger.info("log faq markdown artifact commit={}", short)
        except Exception as md_error:
            logger.error("log faq markdown failed err={}", md_error)
            return

        try:
            import markdown

            try:
                html_content = markdown.markdown(
                    md_content, extensions=["tables", "fenced_code", "toc"]
                )
            except Exception as ext_error:
                logger.warning(
                    "markdown extensions failed trying basic conversion err={}",
                    ext_error,
                )
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
            logger.error("log faq html failed err={}", html_error)

    except Exception as e:
        logger.error("faq artifact logging failed err={}", e)
