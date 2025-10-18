# %%
import numpy as np
import pandas as pd
from plotnine import aes, geom_line, ggplot, labs
from torch import allclose

from logger import get_logger

logger = get_logger(__name__)


def compare_rollouts(rollouts, n_obs):
    """Compare multiple rollouts and visualize their actions and rewards.

    Args:
        rollouts: List of rollout TensorDicts to compare
    Returns:
        tuple: (reward_plot, action_plot) containing the visualization plots
    """
    # Extract actions and rewards from all rollouts
    all_actions = [
        rollout["action"].squeeze().argmax(dim=-1)[:n_obs] for rollout in rollouts
    ]
    all_rewards = [rollout["next"]["reward"][:n_obs] for rollout in rollouts]

    # Compare actions and rewards between all pairs of rollouts
    for i in range(len(rollouts)):
        for j in range(i + 1, len(rollouts)):
            actions_equal = bool(allclose(all_actions[i], all_actions[j]))
            rewards_equal = bool(allclose(all_rewards[i], all_rewards[j]))
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
        rewards_data.extend(
            [
                {"Steps": step, "Cumulative_Reward": val, "Run": f"Run {i + 1}"}
                for step, val in enumerate(np.cumsum(rewards.numpy()))
            ]
        )
    df_rewards = pd.DataFrame(rewards_data)

    # Create DataFrame for plotting actions
    actions_data = []
    for i, actions in enumerate(all_actions):
        actions_data.extend(
            [
                {"Steps": step, "Actions": val, "Run": f"Run {i + 1}"}
                for step, val in enumerate(actions.numpy())
            ]
        )
    df_actions = pd.DataFrame(actions_data)

    # Create reward plot using plotnine
    reward_plot = (
        ggplot(df_rewards, aes(x="Steps", y="Cumulative_Reward", color="Run"))
        + geom_line()
        + labs(title="Cumulative Rewards Comparison", x="Steps", y="Cumulative Reward")
    )

    # Create actions plot using plotnine
    action_plot = (
        ggplot(df_actions, aes(x="Steps", y="Actions", color="Run"))
        + geom_line()
        + labs(title="Actions Comparison", x="Steps", y="Actions")
    )

    return reward_plot, action_plot
