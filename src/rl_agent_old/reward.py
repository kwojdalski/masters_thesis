"""
Reward functions for reinforcement learning trading agent.
"""

import numpy as np


def reward(
    pair: np.ndarray,
    coef: np.ndarray,
    action_track: np.ndarray,
    cost: float,
    reward_function: str = "raw_return",
    eta_dsr: float = 0.5,
) -> np.ndarray:
    """
    Returns the Return for each (s,a) pair in the episode.
    The Return is part of the RL framework and is defined
    as the sum of the rewards in an episode.

    Args:
        pair: DataFrame containing price series of both assets
              of length equal to the length of the episode
        coef: Cointegrating coefficient for the assets in the pair
        action_track: Vector of actions carried out in the episode
        cost: Cost of engaging in a trade (changing action)
        reward_function: Type of reward function to use
        eta_dsr: Parameter for differential Sharpe ratio

    Returns:
        Array of rewards for each state-action pair
    """
    # Get the number of observations
    n = len(pair)

    # Initialize vector of returns. The reward is defined as
    # percentage return
    additive = np.zeros(len(pair[:, 0]))

    # Reward in the last visited state interval is 0
    # Therefore up to n-1
    for i in range(n - 1):
        # Calculate return of the pair following each (s,a) pair in the episode.
        # An episode consists only of a=1 or a=-1 and a final different action which
        # terminates the episode.

        if action_track[i] == -1:
            additive[i] = -(
                coef[0] * (pair[n - 1, 0] - pair[i, 0])
                + coef[1] * (pair[n - 1, 1] - pair[i, 1])
            ) / (pair[i, 0] + abs(coef[1]) * pair[i, 1])

        elif action_track[i] == 1:  # Action taken (long)
            additive[i] = (
                coef[0] * (pair[n - 1, 0] - pair[i, 0])
                + coef[1] * (pair[n - 1, 1] - pair[i, 1])
            ) / (pair[i, 0] + abs(coef[1]) * pair[i, 1])

    if reward_function == "dsr":
        additive = np.concatenate(([additive[-1]], additive[:-1]))
        dsr = diff_sharpe_ratio(additive, eta=eta_dsr)
        reward_values = np.concatenate((dsr[1:], [dsr[0]]))
    else:
        reward_values = additive

    return reward_values


def diff_sharpe_ratio(returns: np.ndarray, eta: float = 0.5) -> np.ndarray:
    """
    Calculate the differential Sharpe ratio.

    Args:
        returns: Array of returns
        eta: Time scale parameter

    Returns:
        Array of differential Sharpe ratios
    """
    # Initialize A and B
    A = np.zeros_like(returns)
    B = np.zeros_like(returns)

    # Calculate K_eta
    K_eta = (1 - eta / 2) / (1 - eta)

    # Calculate A and B recursively
    for t in range(1, len(returns)):
        A[t] = eta * returns[t] + (1 - eta) * A[t - 1]
        B[t] = eta * returns[t] ** 2 + (1 - eta) * B[t - 1]

    # Calculate Sharpe ratio
    S = A / (K_eta * np.sqrt(B - A**2))

    # Calculate differential Sharpe ratio
    dsr = np.zeros_like(returns)
    dsr[1:] = S[1:] - S[:-1]

    return dsr


def q_reward(
    pair: np.ndarray,
    coef: np.ndarray,
    action: int,
    cost: float,
    reward_func: str = "raw_return",
    eta_dsr: float = 0.5,
) -> float:
    """
    Calculate reward for Q-learning agent.

    Args:
        pair: Price data for the trading pair
        coef: Cointegration coefficients
        action: Action taken (-1: short, 0: neutral, 1: long)
        cost: Transaction cost
        reward_func: Reward function to use ('dsr' or 'raw_return')
        eta_dsr: Time scale parameter for differential Sharpe ratio

    Returns:
        Reward value
    """
    # Calculate raw return
    if action == -1:  # Short position
        raw_return = -(
            coef[0] * (pair[1, 0] - pair[0, 0]) + coef[1] * (pair[1, 1] - pair[0, 1])
        ) / (pair[0, 0] + abs(coef[1]) * pair[0, 1])
    elif action == 1:  # Long position
        raw_return = (
            coef[0] * (pair[1, 0] - pair[0, 0]) + coef[1] * (pair[1, 1] - pair[0, 1])
        ) / (pair[0, 0] + abs(coef[1]) * pair[0, 1])
    else:  # Neutral position
        raw_return = 0

    # Apply reward function
    if reward_func == "dsr":
        # For DSR, we need at least 2 returns
        returns = np.array([0, raw_return])  # Add 0 for t=0
        reward = diff_sharpe_ratio(returns, eta=eta_dsr)[-1]
    else:
        reward = raw_return

    return reward


def episode_backup(
    action_track: list[int],
    state_track: np.ndarray,
    reward: np.ndarray,
    diff: int,
    coef: np.ndarray,
    cost: float,
    qsa: np.ndarray,
    alpha: float,
    pre_ns: np.ndarray,
    ns: np.ndarray,
    nsa: np.ndarray,
) -> tuple:
    """
    Perform episode backup for Monte Carlo Control.

    Args:
        action_track: List of actions taken
        state_track: Array of states visited
        reward: Array of rewards received
        diff: Difference type (1 or 2)
        coef: Cointegration coefficients
        cost: Transaction cost
        qsa: Q-value matrix
        alpha: Learning rate
        pre_ns: Previous state visit counts
        ns: Current state visit counts
        nsa: State-action visit counts

    Returns:
        Tuple of updated parameters
    """
    actions = [-1, 0, 1]  # short, neutral, long

    if diff == 1:  # Episode terminated with a 0
        for l in range(len(state_track)):
            action = action_track[l]
            state = state_track[l]

            action_dim = actions.index(action)
            state_action = np.array([*state, action_dim])

            # Update visit counts
            counts = update_n(state, ns, nsa, actions, action)
            ns, nsa = counts

            # Update learning rate
            alpha = 1 / (np.log(nsa[tuple(state_action)] + 1))

            # MCC backup
            qsa[tuple(state_action)] += alpha * (reward[l] - qsa[tuple(state_action)])

    elif diff == 2:
        for j in range(len(state_track)):
            action = action_track[j]
            state = state_track[j]

            action_dim = actions.index(action)
            state_action = np.array([*state, action_dim])

            # Update visit counts
            counts = update_n(state, ns, nsa, actions, action)
            ns, nsa = counts

            # Update learning rate
            alpha = 1 / (np.log(nsa[tuple(state_action)] + 1))

            # Update Q-values
            qsa[tuple(state_action)] += alpha * (reward[j] - qsa[tuple(state_action)])

        # Update state and action tracks
        state_track = state_track[-1:]
        action_track = action_track[-1:]
        pre_ns = ns

    return action_track, state_track, qsa, alpha, ns, nsa, pre_ns
