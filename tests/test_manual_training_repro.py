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
      --config src/configs/sine_wave_ppo_no_trend.yaml \
      --max-steps 50000 \
      --actor-lr 0.0001
    """
    # 1. Configure Logging (equivalent to LOG_LEVEL=DEBUG)
    # The CLI main callback does this:
    os.environ["LOG_LEVEL"] = "DEBUG"
    configure_logging(component="test_repro", level="DEBUG")

    # 2. Setup Parameters (equivalent to CLI args)
    # Note: paths should be relative to project root where pytest is run
    config_path = Path("src/configs/sine_wave_ppo_no_trend.yaml")

    params = TrainingParams(
        config_file=config_path,
        max_steps=500,
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


if __name__ == "__main__":
    # Allows running directly with python tests/test_manual_training_repro.py
    test_sine_wave_ppo_training_debug()
