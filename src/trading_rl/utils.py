# %%
import numpy as np
import pandas as pd
from plotnine import aes, geom_line, ggplot, labs, theme
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
