"""
Plotting utilities for trading RL experiments.

This module contains plotting functions for visualizing training progress
and experiment comparisons using plotnine.
"""

import mlflow
import pandas as pd
from plotnine import (
    aes,
    element_text,
    facet_wrap,
    geom_col,
    geom_line,
    ggplot,
    ggsave,
    labs,
    theme,
)

from logger import get_logger

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
        print("Warning: No training loss data available for plotting - all optimization steps may have been skipped due to tensor shape issues")
        return None
        
    # Pad shorter lists with None values and warn about mismatched lengths
    if len(value_loss) < max_steps:
        print(f"Warning: Value loss data incomplete ({len(value_loss)} vs {max_steps} steps) - some optimization steps may have been skipped")
        value_loss = value_loss + [None] * (max_steps - len(value_loss))
    if len(actor_loss) < max_steps:
        print(f"Warning: Actor loss data incomplete ({len(actor_loss)} vs {max_steps} steps) - some optimization steps may have been skipped")
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
        + geom_line(aes(x="step", y="Loss", color="Loss Type"))
        + facet_wrap("Loss Type", ncol=1, scales="free")
    )

    if save_path:
        import contextlib
        import io
        
        @contextlib.contextmanager
        def suppress_plotnine_output():
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                yield
        
        with suppress_plotnine_output():
            plot.save(save_path)

    return plot


def create_mlflow_comparison_plots(experiment_name: str, results: list):
    """Create comparison plots for MLflow experiments using plotnine."""
    # Extract data from results
    trial_numbers = list(range(len(results)))
    final_rewards = [r["final_metrics"]["final_reward"] for r in results]

    # Get position data from training stats if available
    total_positions = []
    avg_positions = []

    for result in results:
        # Try to get position data from final metrics
        total_pos = result["final_metrics"].get("total_position_changes", 0)
        avg_pos = result["final_metrics"].get("avg_position_change_per_episode", 0)
        total_positions.append(total_pos)
        avg_positions.append(avg_pos)

    # Create combined dataframe for plotting
    plot_data = pd.DataFrame(
        {
            "trial": trial_numbers * 3,
            "metric_type": (
                ["Final Reward"] * len(trial_numbers)
                + ["Total Position Changes"] * len(trial_numbers)
                + ["Avg Position Changes/Episode"] * len(trial_numbers)
            ),
            "value": final_rewards + total_positions + avg_positions,
        }
    )

    # Create the combined plot
    plot = (
        ggplot(plot_data, aes(x="trial", y="value", fill="metric_type"))
        + geom_col(alpha=0.7, show_legend=False)
        + facet_wrap("metric_type", scales="free_y", ncol=1)
        + labs(
            title=f"{experiment_name}: Experiment Comparison",
            x="Trial Number",
            y="Value",
        )
        + theme(
            plot_title=element_text(size=16, face="bold"),
            strip_text=element_text(size=12, face="bold"),
            axis_title=element_text(size=11),
            figure_size=(10, 8),
        )
    )

    # Save plot with output suppression
    import contextlib
    import io
    
    @contextlib.contextmanager
    def suppress_plotnine_output():
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            yield
    
    plot_path = f"{experiment_name}_comparison.png"
    with suppress_plotnine_output():
        ggsave(plot, plot_path, width=10, height=8, dpi=150)

    # Log the comparison plot as an artifact to MLflow
    with mlflow.start_run(run_name="experiment_summary"):
        mlflow.log_artifact(plot_path)

    logger.info(f"Comparison plots saved to: {plot_path}")
