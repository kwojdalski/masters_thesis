"""
Object-oriented classes for the trading system.

This module defines the core classes for managing currency portfolios
and cointegrated pairs in the reinforcement learning trading system.
"""

from typing import Any, Optional

import numpy as np
import pandas as pd
from src.rl_agent_python.mcc import mcc_bootstrap
from src.rl_agent_python.q_learning import q_control
from src.rl_agent_python.rl_utils import RLFramework, initialize_rl_framework


class CurrencyPortfolio:
    """Base class for managing currency portfolios."""

    def __init__(
        self,
        assets: pd.DataFrame,
        features: pd.DataFrame,
        n_episodes: int,
        cost: float,
        pretrained_agent: Optional[RLFramework] = None,
        algorithm: Optional[str] = None,
    ):
        """
        Initialize currency portfolio.

        Args:
            assets: DataFrame of asset prices
            features: DataFrame of features
            n_episodes: Number of episodes for training
            cost: Transaction cost
            pretrained_agent: Optional pretrained agent
            algorithm: RL algorithm to use
        """
        self.assets = assets
        self.features = features
        self.n_episodes = n_episodes
        self.cost = cost
        self.algorithm = algorithm

        # Validate pretrained agent if provided
        if pretrained_agent is not None:
            if self.features.shape[1] != pretrained_agent.cut_points.shape[1]:
                raise ValueError(
                    f"Pretrained agent has {pretrained_agent.cut_points.shape[1]} features "
                    f"but state space has {self.features.shape[1]} features"
                )
        self.pretrained_agent = pretrained_agent

    def initialize_rl_framework(self) -> RLFramework:
        """
        Initialize reinforcement learning framework.

        Returns:
            Initialized RL framework
        """
        return initialize_rl_framework(self.features, self.algorithm)

    def train(self, algorithm: Optional[str] = None) -> Any:
        """
        Train the portfolio using specified algorithm.

        Args:
            algorithm: RL algorithm to use (overrides instance algorithm)

        Returns:
            Trained agent
        """
        if algorithm is None:
            algorithm = self.algorithm

        if algorithm not in ["tdva", "rlalgo", "mcc"]:
            raise ValueError("Only tdva, rlalgo and mcc supported")

        print(f"Training with {algorithm} algorithm")

        if algorithm == "mcc":
            print("Using MCC bootstrap")
            return mcc_bootstrap(
                self.features,
                self.assets,
                None,  # coef will be calculated internally
                self.cost,
                self.n_episodes,
                pretrained_agent=self.pretrained_agent,
            )
        else:
            return q_control(
                self.assets,
                None,  # coef will be calculated internally
                self.features,
                self.n_episodes,
                self.cost,
                pretrained_agent=self.pretrained_agent,
            )


class CointegratedPairs(CurrencyPortfolio):
    """Class for managing cointegrated currency pairs."""

    def __init__(
        self,
        asset: pd.Series,
        pair: pd.DataFrame,
        coef: np.ndarray,
        features: pd.DataFrame,
        n_episodes: int,
        cost: float,
        pretrained_agent: Optional[RLFramework] = None,
        algorithm: Optional[str] = None,
    ):
        """
        Initialize cointegrated pairs.

        Args:
            asset: Series of asset prices
            pair: DataFrame of pair prices
            coef: Cointegration coefficients
            features: DataFrame of features
            n_episodes: Number of episodes for training
            cost: Transaction cost
            pretrained_agent: Optional pretrained agent
            algorithm: RL algorithm to use
        """
        super().__init__(
            assets=pair,
            features=features,
            n_episodes=n_episodes,
            cost=cost,
            pretrained_agent=pretrained_agent,
            algorithm=algorithm,
        )

        self.asset = asset
        self.pair = pair
        self.coef = coef

        # Validate algorithm
        if algorithm is None:
            raise ValueError("Algorithm can't be None")
        if algorithm not in ["tdva", "rlalgo", "mcc"]:
            raise ValueError("Only tdva, rlalgo and mcc supported")
