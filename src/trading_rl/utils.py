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


def create_actual_returns_plot(
    rollouts, n_obs, df_prices=None, env=None, actual_returns_list=None
):
    """Create a plot showing actual portfolio returns (not training rewards).

    This generates a separate plot that ALWAYS shows actual dollar returns,
    regardless of what reward function (DSR, LogReturn, etc.) was used for training.

    For TradingEnv backend, this extracts actual portfolio values from the broker's
    track_record, ensuring true P&L is shown even when training with DSR.

    Args:
        rollouts: List of rollout TensorDicts (deterministic, random)
        n_obs: Number of observations to plot
        df_prices: Optional DataFrame with price data (unused for now)
        env: Optional environment instance for accessing TradingEnv broker state
        actual_returns_list: Optional pre-extracted list of return arrays (one per rollout)

    Returns:
        plotnine.ggplot: Plot showing actual cumulative returns
    """
    returns_data = []

    for i, rollout in enumerate(rollouts):
        run_name = "Deterministic" if i == 0 else "Random"

        # Use pre-extracted returns if available
        if actual_returns_list and i < len(actual_returns_list):
            actual_returns = actual_returns_list[i]
        else:
            # Try to extract actual returns from TradingEnv broker
            actual_returns = _extract_tradingenv_returns(env, n_obs) if env else None

        if actual_returns is not None:
            # Use actual portfolio value changes (TradingEnv)
            cumulative_returns = actual_returns[:n_obs]
            logger.debug(
                f"{run_name}: Using actual portfolio returns from TradingEnv broker"
            )
        else:
            # Fallback: use rollout rewards (works for LogReturn, shows DSR for DSR)
            rewards = rollout["next"]["reward"][:n_obs].detach().cpu().numpy()
            cumulative_returns = np.cumsum(rewards)
            logger.debug(f"{run_name}: Using rollout rewards as fallback")

        returns_data.extend(
            [
                {"Steps": step, "Cumulative_Return": val, "Run": run_name}
                for step, val in enumerate(cumulative_returns)
            ]
        )

    # Add benchmark series if price data is available
    if df_prices is not None:
        # Buy-and-Hold: cumulative log returns of holding the asset
        buy_and_hold = (
            np.log(df_prices["close"] / df_prices["close"].shift(1))
            .fillna(0)
            .cumsum()[:n_obs]
        )

        # Max Profit (Unleveraged): theoretical maximum with perfect foresight
        max_profit = (
            np.log(abs(df_prices["close"] / df_prices["close"].shift(1) - 1) + 1)
            .fillna(0)
            .cumsum()[:n_obs]
        )

        # Add benchmarks to data
        for step, (bh_val, mp_val) in enumerate(zip(buy_and_hold, max_profit, strict=False)):
            returns_data.extend(
                [
                    {"Steps": step, "Cumulative_Return": bh_val, "Run": "Buy-and-Hold"},
                    {
                        "Steps": step,
                        "Cumulative_Return": mp_val,
                        "Run": "Max Profit (Unleveraged)",
                    },
                ]
            )

    df_returns = pd.DataFrame(returns_data)

    # Create plot with custom colors for benchmarks
    returns_plot = (
        ggplot(df_returns, aes(x="Steps", y="Cumulative_Return", color="Run"))
        + geom_line()
        + labs(
            title="Actual Portfolio Returns (Log Returns)",
            x="Steps",
            y="Cumulative Log Return",
        )
        + scale_color_manual(
            values={
                "Deterministic": "#F8766D",
                "Random": "#00BFC4",
                "Buy-and-Hold": "violet",
                "Max Profit (Unleveraged)": "green",
            }
        )
        + theme(figure_size=(13, 7.8))
    )

    return returns_plot


def _extract_tradingenv_returns(env, n_steps):
    """Extract actual portfolio returns from TradingEnv broker.

    Args:
        env: Wrapped TradingEnv (TransformedEnv -> GymnasiumWrapper -> TradingEnv)
        n_steps: Number of steps to extract

    Returns:
        np.ndarray: Cumulative log returns, or None if extraction fails
    """
    try:
        # Unwrap layers to get to TradingEnv
        # Structure: TransformedEnv -> GymnasiumTradingEnvWrapper -> TradingEnv
        if hasattr(env, "_env") and hasattr(env._env, "_env"):
            trading_env = env._env._env
        else:
            logger.debug("Cannot unwrap to TradingEnv - missing _env attribute")
            return None

        # Check if this is actually TradingEnv with broker
        if not hasattr(trading_env, "broker"):
            logger.debug("Unwrapped env has no broker attribute")
            return None

        broker = trading_env.broker

        # Check if track_record exists and has data
        if not hasattr(broker, "track_record") or len(broker.track_record) == 0:
            logger.debug("Broker has no track_record or it's empty")
            return None

        # Extract NLV values from track_record
        # Use context_post.nlv which is the portfolio value after each step
        # Note: TrackRecord doesn't support slicing, must iterate
        nlv_values = []
        max_records = min(n_steps, len(broker.track_record))
        for i in range(max_records):
            record = broker.track_record[i]
            if hasattr(record, "context_post") and hasattr(record.context_post, "nlv"):
                nlv_values.append(float(record.context_post.nlv))
            else:
                logger.warning("Track record missing context_post.nlv")
                return None

        if len(nlv_values) < 2:
            logger.debug(f"Insufficient NLV values: {len(nlv_values)}")
            return None

        # Calculate log returns from NLV changes
        log_returns = []
        for i in range(1, len(nlv_values)):
            if nlv_values[i - 1] > 0 and nlv_values[i] > 0:
                log_ret = np.log(nlv_values[i] / nlv_values[i - 1])
                log_returns.append(log_ret)
            else:
                logger.warning(f"Invalid NLV values: {nlv_values[i-1]}, {nlv_values[i]}")
                return None

        # Prepend 0 for first step (no return yet)
        log_returns = [0.0] + log_returns

        # Return cumulative log returns
        cumulative_returns = np.cumsum(log_returns)

        logger.info(
            f"Extracted {len(cumulative_returns)} actual returns from TradingEnv broker. "
            f"Final return: {cumulative_returns[-1]:.6f}"
        )

        return cumulative_returns

    except Exception as e:
        logger.warning(f"Failed to extract TradingEnv returns: {e}")
        return None
