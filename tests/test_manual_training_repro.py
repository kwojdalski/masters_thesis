import os
from pathlib import Path

import pytest
from rich.console import Console

from cli.commands import TrainingCommand, TrainingParams
from logger import configure_logging


# @pytest.mark.skip(reason="Long running training test for manual debugging")
def test_sine_wave_ppo_training_debug():
    """
    Reproduces the CLI command:
    LOG_LEVEL=DEBUG python src/cli.py train \
      --config src/configs/scenarios/sine_wave_ppo_no_trend.yaml \
      --max-steps 50000 \
      --actor-lr 0.0001
    """
    # 1. Configure Logging (equivalent to LOG_LEVEL=DEBUG)
    # The CLI main callback does this:
    os.environ["LOG_LEVEL"] = "DEBUG"
    configure_logging(component="test_repro", level="DEBUG")

    # 2. Setup Parameters (equivalent to CLI args)
    # Note: paths should be relative to project root where pytest is run
    config_path = Path("src/configs/scenarios/sine_wave_ppo_no_trend.yaml")

    params = TrainingParams(
        config_file=config_path,
        max_steps=600,  # Divisible by frames_per_batch (200)
        actor_lr=0.0001,
        # experiment_name="debug_repro_test", # Optional: set a name if desired
        # log_dir=Path("logs/debug_test"),    # Optional: override log dir
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
      --config src/configs/scenarios/upward_trend_td3_tradingenv.yaml \
      --max-steps 40000 \
      --actor-lr 0.0003 \
      --init-rand-steps 900

    In test form we lower max_steps to keep runtime reasonable.
    """
    os.environ["LOG_LEVEL"] = "DEBUG"
    configure_logging(component="test_repro", level="DEBUG")

    config_path = Path("src/configs/scenarios/upward_trend_td3_tradingenv.yaml")

    params = TrainingParams(
        config_file=config_path,
        max_steps=2000,  # shorter than full run for test
        actor_lr=0.0003,
        init_rand_steps=900,
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
      --config src/configs/scenarios/upward_trend_ddpg_tradingenv.yaml \
      --max-steps 40000 \
      --init-rand-steps 900
    """
    os.environ["LOG_LEVEL"] = "DEBUG"
    configure_logging(component="test_repro", level="DEBUG")

    config_path = Path("src/configs/scenarios/upward_trend_ddpg_tradingenv.yaml")

    params = TrainingParams(
        config_file=config_path,
        max_steps=2000,  # shorten for test
        init_rand_steps=900,
        actor_lr=0.0003,
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
