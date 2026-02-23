"""Explainability and interpretability for trading RL using Captum."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from captum.attr import FeaturePermutation, IntegratedGradients
from plotnine import (
    aes,
    coord_flip,
    geom_col,
    ggplot,
    labs,
    theme_minimal,
    theme,
    element_text,
)
from tensordict import TensorDict

from logger import get_logger

logger = get_logger(__name__)


class ActorCaptumProxy(nn.Module):
    """
    A proxy wrapper that converts TensorDict-based TorchRL actors 
    to standard tensor-in/tensor-out modules for Captum.
    """

    def __init__(self, actor: Any, observation_key: str = "observation"):
        super().__init__()
        self.actor = actor
        self.observation_key = observation_key

    def forward(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        """
        Captum calls this with a raw tensor. 
        We wrap it in a TensorDict, call the actor, and return the action/mean.
        """
        # Create a dummy TensorDict
        td = TensorDict({self.observation_key: obs_tensor}, batch_size=obs_tensor.shape[:-1])
        
        # Run the actor (TensorDictModule)
        result_td = self.actor(td)
        
        # We want to attribute the 'action' or the distribution 'loc' (mean)
        if "loc" in result_td.keys():
            return result_td["loc"]
        elif "action" in result_td.keys():
            # For discrete actions, attribution is tricky; 
            # for continuous, 'action' is what we want.
            return result_td["action"].float()
        
        raise ValueError("Actor output TensorDict must contain 'loc' or 'action' for attribution.")


class RLInterpretabilityAnalyzer:
    """
    Main class for post-mortem feature importance analysis.
    """

    def __init__(
        self, 
        trainer: Any, 
        feature_names: list[str], 
        observation_key: str = "observation"
    ):
        self.trainer = trainer
        self.feature_names = feature_names
        self.proxy = ActorCaptumProxy(trainer.actor, observation_key)
        self.proxy.eval()

    def compute_global_importance(self, obs_batch: torch.Tensor) -> pd.DataFrame:
        """
        Use Feature Permutation to calculate global importance over a batch of data.
        This is the 'Lucarelli-style' analysis.
        """
        logger.info("Computing Feature Permutation Importance...")
        fp = FeaturePermutation(self.proxy)
        
        # Attribution shape will match obs_batch shape
        attributions = fp.attribute(obs_batch, target=None)
        
        # Take the absolute mean across the batch (and time steps if present)
        # Assuming obs_batch is [batch, features] or [batch, time, features]
        importance = torch.abs(attributions).mean(dim=tuple(range(attributions.ndim - 1)))
        
        df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance.detach().cpu().numpy()
        })
        return df.sort_values("importance", ascending=False)

    def compute_global_ig(self, obs_batch: torch.Tensor) -> pd.DataFrame:
        """
        Use Integrated Gradients to calculate global importance over a batch of data.
        """
        logger.info("Computing Global Integrated Gradients Importance...")
        ig = IntegratedGradients(self.proxy)
        
        # Baselines are typically zero
        baseline = torch.zeros_like(obs_batch)
        
        attributions = ig.attribute(obs_batch, baselines=baseline, target=None)
        
        # Average absolute attributions
        importance = torch.abs(attributions).mean(dim=tuple(range(attributions.ndim - 1)))
        
        df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance.detach().cpu().numpy()
        })
        return df.sort_values("importance", ascending=False)

    def compute_local_attribution(self, obs_sample: torch.Tensor) -> pd.DataFrame:
        """
        Use Integrated Gradients to see why a specific trade was made.
        """
        logger.info("Computing Integrated Gradients Attribution for sample...")
        ig = IntegratedGradients(self.proxy)
        
        # Baselines are typically zero (no information)
        baseline = torch.zeros_like(obs_sample)
        
        attributions, delta = ig.attribute(
            obs_sample, 
            baselines=baseline, 
            target=None, 
            return_convergence_delta=True
        )
        
        df = pd.DataFrame({
            "feature": self.feature_names,
            "attribution": attributions.detach().cpu().numpy().flatten()
        })
        return df

    def plot_importance(self, df: pd.DataFrame, title: str = "Feature Importance", color: str = "steelblue"):
        """Visualize the importance using plotnine."""
        # Clean up column names for plotting
        if "importance" in df.columns:
            val_col = "importance"
        else:
            val_col = "attribution"

        plot = (
            ggplot(df, aes(x=f"reorder(feature, {val_col})", y=val_col))
            + geom_col(fill=color)
            + coord_flip()
            + labs(title=title, x="Feature", y="Importance Score")
            + theme_minimal()
            + theme(
                figure_size=(24, 20),  # 2x taller: was (24, 10), originally (12, 5)
                plot_title=element_text(size=16, weight="bold"),
                axis_title=element_text(size=14),
                axis_text=element_text(size=12)
            )
        )
        return plot

    def plot_importance_merged(
        self,
        df_permutation: pd.DataFrame,
        df_ig: pd.DataFrame,
    ):
        """Create a merged plot with both permutation and integrated gradients importance stacked vertically."""
        p1 = self.plot_importance(
            df_permutation,
            title="Global Feature Importance (Permutation)",
            color="steelblue"
        )

        p2 = self.plot_importance(
            df_ig,
            title="Global Feature Importance (Integrated Gradients)",
            color="coral"
        )

        # Stack vertically using / operator
        merged_plot = p1 / p2
        return merged_plot

    def quantify_interpretability(self, importance_df: pd.DataFrame) -> dict[str, float]:
        """Quantify interpretability metrics from importance data."""
        importance = importance_df["importance"].values
        normalized = importance / (importance.sum() + 1e-9)
        
        # Sparsity: Percentage of features with very low importance (< 1% of mean)
        threshold = normalized.mean() * 0.01
        sparsity = (normalized < threshold).mean()
        
        # Concentration (Gini coefficient-like): Lorenz curve top features
        # How many features account for 80% of the information?
        sorted_norm = np.sort(normalized)[::-1]
        cumulative = np.cumsum(sorted_norm)
        top_80_count = np.where(cumulative >= 0.8)[0][0] + 1 if any(cumulative >= 0.8) else len(normalized)
        concentration = top_80_count / len(normalized)
        
        return {
            "interpretability_sparsity": float(sparsity),
            "interpretability_top_80_feature_ratio": float(concentration),
            "interpretability_top_feature_count": float(top_80_count)
        }


def analyze_trained_agent(trainer: Any, env: Any, feature_names: list[str], n_steps: int = 100):
    """
    Helper function to run a rollout and analyze feature importance.
    """
    # 1. Collect data
    rollout = env.rollout(max_steps=n_steps)
    obs_batch = rollout["observation"] # [T, D]
    
    # 2. Analyze
    analyzer = RLInterpretabilityAnalyzer(trainer, feature_names)
    importance_df = analyzer.compute_global_importance(obs_batch)
    
    # 3. Plot
    plot = analyzer.plot_importance(importance_df, title=f"Global Importance (n={n_steps} steps)")
    
    return importance_df, plot
