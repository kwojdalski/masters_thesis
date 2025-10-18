"""
Q-learning implementation for statistical arbitrage.
"""

import numpy as np

from src.rl_agent_python.reward import q_reward
from src.rl_agent_python.utils import (
    choose_action,
    initialize_rl_framework,
    state_indexes,
    update_n,
)


class QLearning:
    def __init__(self, features: np.ndarray, actions: list[int] = [-1, 0, 1]):
        """
        Initialize Q-learning agent.

        Args:
            features: Feature matrix for state representation
            actions: List of possible actions (default: [-1, 0, 1] for short, neutral, long)
        """
        self.actions = actions
        self.features = features

        # Initialize RL framework
        (
            self.cut_points,
            self.actions,
            self.alpha,
            self.epsilon,
            self.n_0,
            self.episode,
            self.ns,
            self.nsa,
            self.qsa,
            self.action_track,
        ) = initialize_rl_framework(features, "qlearning")

    def train(
        self,
        pair: np.ndarray,
        coef: np.ndarray,
        n_episodes: int,
        cost: float,
        pretrained_agent: dict | None = None,
        verbose: bool = False,
        reward_func: str = "dsr",
        **kwargs,
    ) -> dict:
        """
        Train the Q-learning agent.

        Args:
            pair: Price data for the trading pair
            coef: Cointegration coefficients
            n_episodes: Number of training episodes
            cost: Transaction cost
            pretrained_agent: Optional pretrained agent to continue training from
            verbose: Whether to print training progress
            reward_func: Reward function to use ('dsr' or 'raw_return')
            **kwargs: Additional parameters

        Returns:
            Dictionary containing trained agent parameters
        """
        # Load pretrained agent if provided
        if pretrained_agent is not None:
            self.qsa = pretrained_agent["qsa"]
            self.action_track = pretrained_agent["action_track"]
            self.nsa = pretrained_agent["nsa"]
            self.ns = pretrained_agent["ns"]
            self.epsilon = pretrained_agent["epsilon"]
            self.alpha = pretrained_agent["alpha"]
            self.cut_points = pretrained_agent["cut_points"]

        # Training loop
        while self.episode <= n_episodes:
            i = 0
            while i < len(pair) - 1:
                # Get current state
                state = state_indexes(self.features[i], self.cut_points)

                # Choose action
                action = choose_action(self.qsa, state, self.epsilon, self.actions)
                action_dim = self.actions.index(action)
                state_action = np.array([*state, action_dim])

                # Track action
                self.action_track.append(action)

                # Update visit counts
                counts = update_n(
                    state, self.ns, self.nsa, self.actions, action, self.cut_points
                )
                self.ns, self.nsa = counts

                # Update epsilon and alpha
                self.epsilon = 1 / (np.log(self.nsa[tuple(state_action)] + 1))
                self.alpha = 1 / (np.log(self.nsa[tuple(state_action)] + 1))

                # Get next state and best action
                state_next = state_indexes(self.features[i + 1], self.cut_points)
                action_target = choose_action(self.qsa, state_next, 0, self.actions)
                action_target_dim = self.actions.index(action_target)
                state_action_target = np.array([*state_next, action_target_dim])

                # Calculate reward
                reward = q_reward(
                    pair[i : i + 2],
                    coef,
                    action,
                    cost,
                    reward_func=reward_func,
                    eta_dsr=0.2,
                )

                # Q-learning update
                self.qsa[tuple(state_action)] += self.alpha * (
                    reward
                    + self.qsa[tuple(state_action_target)]
                    - self.qsa[tuple(state_action)]
                )

                i += 1

            self.episode += 1

        return {
            "qsa": self.qsa,
            "action_track": self.action_track,
            "nsa": self.nsa,
            "ns": self.ns,
            "epsilon": self.epsilon,
            "alpha": self.alpha,
            "cut_points": self.cut_points,
        }
