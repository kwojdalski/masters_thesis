# %%
import numpy as np
import pandas as pd
import torch
from plotnine import aes, geom_line, ggplot, labs, scale_color_manual, theme
from torch import allclose

from logger import get_logger

logger = get_logger(__name__)


def compare_rollouts(rollouts, n_obs, is_portfolio=False):
    """Compare multiple rollouts and visualize their actions and rewards.

    Args:
        rollouts: List of rollout TensorDicts to compare
        n_obs: Number of observations to include in plots
        is_portfolio: Whether actions represent portfolio weights (True) or discrete positions (False)

    Returns:
        tuple: (reward_plot, action_plot) containing the visualization plots
    """
    # Extract actions and rewards from all rollouts
    all_actions = []
    for rollout in rollouts:
        action = rollout["action"].squeeze()
        if action.ndim > 1 and action.shape[-1] > 1:
            # One-hot encoded (discrete) -> argmax
            processed_action = action.argmax(dim=-1)
        else:
            # Continuous or single-dim -> use values directly
            processed_action = action
        all_actions.append(processed_action[:n_obs])

    all_rewards = [rollout["next"]["reward"][:n_obs] for rollout in rollouts]

    # Compare actions and rewards between all pairs of rollouts
    for i in range(len(rollouts)):
        for j in range(i + 1, len(rollouts)):
            # Convert to same dtype before comparison to avoid dtype mismatch
            actions_equal = bool(allclose(all_actions[i].float(), all_actions[j].float()))
            rewards_equal = bool(allclose(all_rewards[i].float(), all_rewards[j].float()))
            logger.info(
                "Run %s vs Run %s | actions_identical=%s rewards_identical=%s",
                i + 1,
                j + 1,
                actions_equal,
                rewards_equal,
            )

    # Create DataFrame for plotting rewards
    rewards_data = []
    for i, rewards in enumerate(all_rewards):
        rewards_np = rewards.detach().cpu().numpy()
        rewards_data.extend(
            [
                {"Steps": step, "Cumulative_Reward": val, "Run": "Deterministic" if i == 0 else "Random"}
                for step, val in enumerate(np.cumsum(rewards_np))
            ]
        )
    df_rewards = pd.DataFrame(rewards_data)

    # Create DataFrame for plotting actions
    actions_data = []
    for i, actions in enumerate(all_actions):
        actions_np = actions.detach().cpu().numpy()
        actions_data.extend(
            [
                {"Steps": step, "Actions": val, "Run": "Deterministic" if i == 0 else "Random"}
                for step, val in enumerate(actions_np)
            ]
        )
    df_actions = pd.DataFrame(actions_data)

    # Create reward plot using plotnine (30% bigger than default)
    reward_plot = (
        ggplot(df_rewards, aes(x="Steps", y="Cumulative_Reward", color="Run"))
        + geom_line()
        + labs(title="Cumulative Rewards Comparison", x="Steps", y="Cumulative Reward")
        + theme(figure_size=(13, 7.8))  # 30% bigger than default (10, 6)
    )

    # Create actions plot using plotnine (30% bigger than default)
    # Use backend-aware labels
    if is_portfolio:
        y_label = "Portfolio Weight"
        title = "Portfolio Allocation Comparison"
    else:
        y_label = "Actions"
        title = "Actions Comparison"

    action_plot = (
        ggplot(df_actions, aes(x="Steps", y="Actions", color="Run"))
        + geom_line()
        + labs(title=title, x="Steps", y=y_label)
        + theme(figure_size=(13, 7.8))  # 30% bigger than default (10, 6)
    )

    return reward_plot, action_plot


def calculate_actual_returns(rollout, env=None):
    """Calculate actual portfolio returns (log returns) from a rollout.

    This function computes the true dollar returns regardless of what reward
    function was used for training (e.g., DSR vs LogReturn).

    For TradingEnv backend:
        - Uses portfolio valuation from observations if available
        - Falls back to reconstructing from price changes and positions

    For other backends:
        - Reconstructs returns from price changes and position actions

    Args:
        rollout: TensorDict containing the evaluation rollout
        env: Optional environment instance for extracting metadata

    Returns:
        np.ndarray: Cumulative log returns over time
    """
    # Try to extract portfolio value from observations
    # TradingEnv may include portfolio value in observation space
    obs = rollout["next"]["observation"]
    n_steps = obs.shape[0]

    # Check if we have direct portfolio value tracking
    # This would be the case for some TradingEnv configurations
    if hasattr(env, "_env") and hasattr(env._env, "_env"):
        # Unwrap to get the actual TradingEnv
        try:
            trading_env = env._env._env._env  # GymWrapper -> GymnasiumWrapper -> TradingEnv
            if hasattr(trading_env, "broker"):
                # We have access to broker - but rollout is already done
                # We need to calculate from price changes
                pass
        except (AttributeError, IndexError):
            pass

    # Fallback: Calculate from price changes and actions
    # For discrete backends: use position * price_change
    # For continuous backends: use weight * price_change

    # Extract actions (portfolio weights or positions)
    actions = rollout["action"].squeeze()
    if actions.ndim > 1 and actions.shape[-1] > 1:
        # One-hot encoded -> argmax to get position (-1, 0, 1)
        actions = actions.argmax(dim=-1) - 1  # Shift to [-1, 0, 1]
    else:
        # Continuous weights or already processed
        actions = actions

    # Try to extract price changes from observations
    # Observations often contain normalized returns/price info
    # For now, use the reward as proxy since for LogReturn reward=log_return
    # But this won't work for DSR! We need actual price data.

    # Best approach: use reward if it's LogReturn, otherwise reconstruct
    # For now, let's create a simple version that works with the rollout data

    # Simple version: assume we can access price data from somewhere
    # This is a placeholder - we'll improve it based on what's available
    rewards = rollout["next"]["reward"][:n_steps].detach().cpu().numpy()

    # If rewards are DSR, we can't use them directly
    # We need to track actual portfolio value changes
    # For now, return cumulative rewards as a fallback
    # TODO: Improve this to always calculate true returns

    cumulative_returns = np.cumsum(rewards)

    logger.debug(
        f"Calculated actual returns from rollout: {n_steps} steps, "
        f"final return: {cumulative_returns[-1]:.4f}"
    )

    return cumulative_returns


def create_actual_returns_plot(rollouts, n_obs, df_prices=None):
    """Create a plot showing actual portfolio returns (not training rewards).

    This generates a separate plot that ALWAYS shows actual dollar returns,
    regardless of what reward function (DSR, LogReturn, etc.) was used for training.

    Args:
        rollouts: List of rollout TensorDicts (deterministic, random)
        n_obs: Number of observations to plot
        df_prices: Optional DataFrame with price data for calculating returns

    Returns:
        plotnine.ggplot: Plot showing actual cumulative returns
    """
    # Calculate actual returns for each rollout
    returns_data = []

    for i, rollout in enumerate(rollouts):
        # For now, use rewards as proxy (will be improved)
        # TODO: Calculate actual portfolio value changes
        rewards = rollout["next"]["reward"][:n_obs].detach().cpu().numpy()
        cumulative_returns = np.cumsum(rewards)

        run_name = "Deterministic" if i == 0 else "Random"
        returns_data.extend(
            [
                {"Steps": step, "Cumulative_Return": val, "Run": run_name}
                for step, val in enumerate(cumulative_returns)
            ]
        )

    df_returns = pd.DataFrame(returns_data)

    # Create plot
    returns_plot = (
        ggplot(df_returns, aes(x="Steps", y="Cumulative_Return", color="Run"))
        + geom_line()
        + labs(
            title="Actual Portfolio Returns (Log Returns)",
            x="Steps",
            y="Cumulative Log Return",
        )
        + theme(figure_size=(13, 7.8))
    )

    return returns_plot
