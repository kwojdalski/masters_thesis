"""
Monte Carlo Control module for reinforcement learning.

This module implements the Monte Carlo Control algorithm for
reinforcement learning in statistical arbitrage.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from src.rl_agent_python.reward import episode_backup, reward
from src.rl_agent_python.rl_utils import choose_action, initialize_rl_framework
from src.rl_agent_python.state_space import state_indexes, update_n


@dataclass
class MCCResult:
    """Class to store MCC algorithm results."""

    Qsa: np.ndarray
    Position_track: np.ndarray
    Nsa: np.ndarray
    Ns: np.ndarray
    epsilon: float
    alpha: float
    cut_points: np.ndarray


def mcc_bootstrap(
    features: pd.DataFrame,
    asset: pd.DataFrame,
    coef: Optional[np.ndarray],
    cost: float,
    n_episodes: int,
    pretrained_agent: Optional[MCCResult] = None,
    algorithm: str = "mcc",
) -> MCCResult:
    """
    Monte Carlo Control bootstrap algorithm.

    Args:
        features: DataFrame of state features
        asset: DataFrame of asset prices
        coef: Cointegration coefficients
        cost: Transaction cost
        n_episodes: Number of episodes to run
        pretrained_agent: Optional pretrained agent
        algorithm: RL algorithm to use

    Returns:
        MCCResult containing algorithm results
    """
    # Initialize RL framework
    (
        cut_points,
        actions,
        alpha,
        epsilon,
        N_0,
        episode,
        Ns,
        Nsa,
        Qsa,
        Position_track,
        diff,
        terminate,
    ) = initialize_rl_framework(features, algorithm)

    # Use pretrained agent if available
    if pretrained_agent is not None:
        Qsa = pretrained_agent.Qsa
        Position_track = pretrained_agent.Position_track
        Nsa = pretrained_agent.Nsa
        Ns = pretrained_agent.Ns
        epsilon = pretrained_agent.epsilon
        alpha = pretrained_agent.alpha
        cut_points = pretrained_agent.cut_points

    # Initialize tracking variables
    i = 0
    state_track = pd.DataFrame()
    action_track = []

    while episode < n_episodes:
        state_track = pd.DataFrame()
        action_track = []

        # Get initial state and action
        state = state_indexes(features.iloc[i], cut_points)
        action = choose_action(Qsa, state, epsilon, actions)
        Position_track = np.append(Position_track, action)

        # Handle end of data
        if i == len(asset) - 1:
            action = 0
            episode += 1
            print(f"Episode {episode}")

        # Update visit counts
        Ns, Nsa = update_n(state, Ns, Nsa, actions, action, cut_points)

        if action == 0:
            epsilon = N_0 / (N_0 + Ns[tuple(state)])
            i = i + 1 if i < len(asset) - 1 else 0
        else:
            terminate = False
            i += 1
            state_track = pd.concat([state_track, pd.DataFrame([state])])
            action_track.append(action)
            pre_ns = Ns.copy()

        # Main episode loop
        while not terminate:
            state = state_indexes(features.iloc[i], cut_points)
            state_track = pd.concat([state_track, pd.DataFrame([state])])
            epsilon = N_0 / (N_0 + pre_ns[tuple(state)])
            action = choose_action(Qsa, state, epsilon, actions)
            action_track.append(action)
            Position_track = np.append(Position_track, action)

            # Check for position change
            if len(action_track) > 1:
                diff = abs(action_track[-1] - action_track[-2])
            else:
                diff = 0

            # Force position close at end of data
            if i == len(asset) - 1 and diff == 0:
                diff = 1

            # Handle episode end
            if diff != 0:
                terminate = True if diff == 1 else False
                epi_pair = asset.iloc[i - len(action_track) + 1 : i + 1]

                # Calculate reward
                rwrd = reward(
                    epi_pair,
                    coef,
                    np.array(action_track),
                    cost,
                    reward_function="dsr",
                    eta_dsr=0.8,
                )

                # Backup episode
                (
                    action_track,
                    state_track,
                    Qsa,
                    alpha,
                    Ns,
                    Nsa,
                    pre_ns,
                ) = episode_backup(
                    action_track,
                    state_track,
                    rwrd,
                    diff,
                    coef,
                    cost,
                    Qsa,
                    alpha,
                    pre_ns,
                    Ns,
                    Nsa,
                )

            # Move to next time step
            if i < len(asset) - 1:
                i += 1
            else:
                i = 0
                terminate = True
                action = 0
                episode += 1
                print(f"Episode {episode}")

    return MCCResult(
        Qsa=Qsa,
        Position_track=Position_track,
        Nsa=Nsa,
        Ns=Ns,
        epsilon=epsilon,
        alpha=alpha,
        cut_points=cut_points,
    )
