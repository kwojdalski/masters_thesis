"""
Plotting utilities for trading RL experiments.

This module contains plotting functions for visualizing training progress
using plotnine.
"""

import pandas as pd
from plotnine import (
    aes,
    facet_wrap,
    geom_line,
    ggplot,
)

from logger import get_logger
from trading_rl.evaluation.thesis_theme import thesis_theme

logger = get_logger(__name__)


def visualize_training(logs: dict, save_path: str | None = None):
    """Visualize training progress.

    Args:
        logs: Dictionary of training logs
        save_path: Optional path to save plot
    """
    # Create loss dataframe with safety checks for empty logs
    value_loss = logs.get("loss_value", [])
    actor_loss = logs.get("loss_actor", [])

    # Use the longer of the two for step range, or empty if both are empty
    max_steps = max(len(value_loss), len(actor_loss))
    if max_steps == 0:
        # Return empty plot if no training occurred
        logger.warning(
            "No training loss data available for plotting - all optimization steps may have been skipped due to tensor shape issues"
        )
        return None

    # Pad shorter lists with None values and warn about mismatched lengths
    if len(value_loss) < max_steps:
        logger.warning(
            f"Value loss data incomplete ({len(value_loss)} vs {max_steps} steps) - some optimization steps may have been skipped"
        )
        value_loss = value_loss + [None] * (max_steps - len(value_loss))
    if len(actor_loss) < max_steps:
        logger.warning(
            f"Actor loss data incomplete ({len(actor_loss)} vs {max_steps} steps) - some optimization steps may have been skipped"
        )
        actor_loss = actor_loss + [None] * (max_steps - len(actor_loss))

    loss_df = pd.DataFrame(
        {
            "step": range(max_steps),
            "Value Loss": value_loss,
            "Actor Loss": actor_loss,
        }
    )

    # Create plot
    plot = (
        ggplot(loss_df.melt(id_vars=["step"], var_name="Loss Type", value_name="Loss"))
        + geom_line(aes(x="step", y="Loss", color="Loss Type"), size=0.3)
        + facet_wrap("Loss Type", ncol=1, scales="free")
        + thesis_theme()
    )

    if save_path:
        import contextlib
        import io

        @contextlib.contextmanager
        def suppress_plotnine_output():
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                yield

        with suppress_plotnine_output():
            plot.save(save_path)

    return plot
