"""Random policy baseline trainer — takes uniformly random actions, does no learning."""

from __future__ import annotations

from typing import Any

from torchrl.envs.utils import RandomPolicy

from trading_rl.trainers.base import BaseTrainer


class RandomTrainer(BaseTrainer):
    """Baseline trainer that selects uniformly random actions and skips optimization.

    Useful as a lower-bound reference when comparing against trained agents.
    Fully compatible with the training pipeline: it runs the same data loading,
    environment setup, and evaluation as real trainers — it just does nothing in
    the training loop.
    """

    @staticmethod
    def build_models(n_obs: int, n_act: int, config: Any, env: Any):
        actor = RandomPolicy(env.action_spec)
        return actor, None

    def _optimization_step(self, batch_idx: int, max_length: int, buffer_len: int) -> None:
        pass

    def _evaluate(self) -> None:
        pass

    def train(self, callback: Any = None) -> dict[str, list]:
        """Skip training — random policy requires no learning."""
        return dict(self.logs)

    def load_checkpoint(self, path: str) -> None:
        """No-op — random policy has no learnable state to restore."""
        from logger import get_logger
        get_logger(__name__).warning(
            "load_checkpoint called on RandomTrainer; no state to restore path=%s", path
        )

    def save_checkpoint(self, path: str, feature_pipeline_state: dict[str, dict[str, float]] | None = None) -> None:
        import torch
        torch.save(
            {
                "algorithm": "RANDOM",
                "n_obs": self.n_obs,
                "n_act": self.n_act,
                "feature_pipeline_state": feature_pipeline_state,
            },
            path,
        )

    def setup_periodic_explainability(
        self,
        df: Any,
        max_steps: int,
        config: Any,
        eval_env: Any = None,
    ) -> None:
        pass

    def setup_periodic_evaluation(
        self,
        splits: Any,
        config: Any,
        algorithm: str,
    ) -> None:
        pass
