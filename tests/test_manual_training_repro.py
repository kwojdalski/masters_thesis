import os
from pathlib import Path

import pytest
from rich.console import Console

from cli.commands import TrainingCommand, TrainingParams
from logger import configure_logging


@pytest.fixture(autouse=True)
def _end_mlflow_run():
    """Ensure any active MLflow run from a previous test is closed before each test."""
    try:
        import mlflow

        if mlflow.active_run():
            mlflow.end_run()
    except ImportError:
        pass
    yield
    try:
        import mlflow

        if mlflow.active_run():
            mlflow.end_run()
    except ImportError:
        pass


@pytest.fixture(autouse=True)
def _restore_logging():
    """Restore logging state and LOG_LEVEL env var after each test."""
    import logging
    import os

    root = logging.getLogger()
    saved_root_level = root.level
    saved_levels = {
        name: lgr.level
        for name, lgr in logging.Logger.manager.loggerDict.items()
        if isinstance(lgr, logging.Logger)
    }
    saved_log_level_env = os.environ.get("LOG_LEVEL")
    yield
    root.setLevel(saved_root_level)
    for name, level in saved_levels.items():
        logging.getLogger(name).setLevel(level)
    if saved_log_level_env is None:
        os.environ.pop("LOG_LEVEL", None)
    else:
        os.environ["LOG_LEVEL"] = saved_log_level_env


# @pytest.mark.skip(reason="Long running training test for manual debugging")
def test_sine_wave_ppo_training_debug():
    """
    Reproduces the CLI command:
    LOG_LEVEL=DEBUG python src/cli.py train \
      --config src/configs/scenarios/sine_wave/ppo_no_trend.yaml \
      # Edit config for max_steps/actor_lr as needed
    """
    # 1. Configure Logging (equivalent to LOG_LEVEL=DEBUG)
    # The CLI main callback does this:
    os.environ["LOG_LEVEL"] = "DEBUG"
    configure_logging(component="test_repro", level="DEBUG")

    # 2. Setup Parameters (equivalent to CLI args)
    # Note: paths should be relative to project root where pytest is run
    config_path = Path("src/configs/scenarios/sine_wave/ppo_no_trend_continuous")

    params = TrainingParams(
        config_file=config_path,
        max_steps=600,  # Divisible by frames_per_batch (200)
        experiment_name="test_repro_ppo_sine",
    )

    # 3. Initialize Command
    # We use a Console that doesn't capture output so we can see it in real-time
    # when running with `pytest -s`
    console = Console(force_terminal=True)
    cmd = TrainingCommand(console)

    # 4. Execute
    # This calls run_single_experiment internally
    try:
        cmd.execute(params)
    except Exception as e:
        pytest.fail(f"Training failed with error: {e}")


def test_upward_trend_td3_training_debug():
    """
    Reproduces the CLI command (shortened for test):
    LOG_LEVEL=DEBUG python src/cli.py train \
      --config src/configs/scenarios/synthetic/upward_trend_td3_tradingenv.yaml \
      # Edit config for max_steps/actor_lr/init_rand_steps as needed

    In test form we lower max_steps to keep runtime reasonable.
    """
    os.environ["LOG_LEVEL"] = "DEBUG"
    configure_logging(component="test_repro", level="DEBUG")

    config_path = Path("src/configs/scenarios/synthetic/upward_trend_td3_tradingenv")

    params = TrainingParams(
        config_file=config_path,
        max_steps=2000,  # shorter than full run for test
        config_overrides=["data.train_size=400"],
        experiment_name="test_repro_td3_uptrend",
    )

    console = Console(force_terminal=True)
    cmd = TrainingCommand(console)

    try:
        cmd.execute(params)
    except Exception as e:
        pytest.fail(f"TD3 training failed with error: {e}")


def test_upward_trend_ddpg_training_debug():
    """
    Quick DDPG tradingenv uptrend repro (shortened for test runtime).
    Mirrors:
    LOG_LEVEL=DEBUG python src/cli.py train \
      --config src/configs/scenarios/synthetic/upward_trend_ddpg_tradingenv.yaml \
      # Edit config for max_steps/init_rand_steps as needed
    """
    os.environ["LOG_LEVEL"] = "DEBUG"
    configure_logging(component="test_repro", level="DEBUG")

    config_path = Path("src/configs/scenarios/synthetic/upward_trend_ddpg_tradingenv")

    params = TrainingParams(
        config_file=config_path,
        max_steps=2000,  # shorten for test
        experiment_name="test_repro_ddpg_uptrend",
    )

    console = Console(force_terminal=True)
    cmd = TrainingCommand(console)

    try:
        cmd.execute(params)
    except Exception as e:
        pytest.fail(f"DDPG training failed with error: {e}")


if __name__ == "__main__":
    # Allows running directly with python tests/test_manual_training_repro.py
    test_sine_wave_ppo_training_debug()
