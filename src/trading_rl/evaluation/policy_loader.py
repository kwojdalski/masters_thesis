"""Load a trained policy from a checkpoint for standalone evaluation.

Usage::

    from trading_rl.evaluation.policy_loader import PolicyLoader

    policy = PolicyLoader.from_checkpoint("path/to/checkpoint.pt")
    # policy is a ready-to-use actor (no trainer needed)

    from trading_rl.evaluation import EvaluationConfig, StrategyEvaluator

    evaluator = StrategyEvaluator(
        env_factory=my_env_factory,
        policy=policy,
        config=EvaluationConfig(reward_type="log_return", max_steps=500),
    )
    result = evaluator.evaluate_split("test", test_df)
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


class PolicyLoader:
    """Reconstruct a trained actor from a checkpoint file.

    Supports PPO, TD3, and DDPG checkpoints produced by the trainers in
    ``trading_rl.trainers``.  The checkpoint must contain the architecture
    metadata saved since the decoupling refactor (``algorithm``, ``n_obs``,
    ``n_act``, ``actor_hidden_dims``).
    """

    @staticmethod
    def from_checkpoint(path: str, device: str = "cpu") -> Any:
        """Load and return an actor network ready for inference.

        Args:
            path: Path to a ``.pt`` checkpoint file.
            device: Torch device string (``"cpu"``, ``"cuda"``, etc.).

        Returns:
            Actor module with weights loaded, set to ``eval()`` mode.

        Raises:
            ValueError: If the checkpoint is missing required architecture keys
                or specifies an unsupported algorithm.
        """
        import torch

        checkpoint = torch.load(path, map_location=device, weights_only=False)
        return PolicyLoader._build_actor(checkpoint, device)

    @staticmethod
    def _build_actor(checkpoint: dict, device: str) -> Any:
        import torch

        for key in ("algorithm", "n_obs", "n_act"):
            if checkpoint.get(key) is None:
                raise ValueError(
                    f"Checkpoint is missing '{key}'. Re-train with the current codebase "
                    "to generate portable checkpoints."
                )

        algorithm: str = checkpoint["algorithm"].lower()
        n_obs: int = int(checkpoint["n_obs"])
        n_act: int = int(checkpoint["n_act"])
        hidden_dims: list[int] | None = checkpoint.get("actor_hidden_dims")
        state_dict: dict = checkpoint["actor_state_dict"]

        if algorithm == "ppo":
            actor = PolicyLoader._build_ppo_actor(n_obs, n_act, hidden_dims)
        elif algorithm in ("td3", "ddpg"):
            action_low = checkpoint.get("action_low")
            action_high = checkpoint.get("action_high")
            spec = None
            if action_low is not None and action_high is not None:
                spec = SimpleNamespace(
                    low=torch.tensor(action_low, dtype=torch.float32),
                    high=torch.tensor(action_high, dtype=torch.float32),
                )
            actor = PolicyLoader._build_continuous_actor(n_obs, n_act, hidden_dims, spec)
        else:
            raise ValueError(
                f"Unsupported algorithm '{algorithm}'. "
                "Expected one of: ppo, td3, ddpg."
            )

        actor.load_state_dict(state_dict)
        actor.to(device)
        actor.eval()
        return actor

    @staticmethod
    def _build_ppo_actor(n_obs: int, n_act: int, hidden_dims: list[int] | None) -> Any:
        from trading_rl.models import create_ppo_actor
        return create_ppo_actor(n_obs, n_act, hidden_dims=hidden_dims, spec=None)

    @staticmethod
    def _build_continuous_actor(
        n_obs: int,
        n_act: int,
        hidden_dims: list[int] | None,
        spec: Any | None,
    ) -> Any:
        from trading_rl.models import create_ddpg_actor
        return create_ddpg_actor(n_obs, n_act, hidden_dims=hidden_dims, spec=spec)

    @staticmethod
    def inspect(path: str) -> dict:
        """Return the architecture metadata stored in a checkpoint.

        Useful for verifying what was saved without loading the full weights.

        Args:
            path: Path to a ``.pt`` checkpoint file.

        Returns:
            Dict with keys: ``algorithm``, ``n_obs``, ``n_act``,
            ``actor_hidden_dims``, ``value_hidden_dims``, ``action_low``,
            ``action_high``, ``total_count``, ``total_episodes``,
            ``mlflow_run_id``, ``mlflow_experiment_name``.
        """
        import torch

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        keys = (
            "algorithm",
            "n_obs",
            "n_act",
            "actor_hidden_dims",
            "value_hidden_dims",
            "action_low",
            "action_high",
            "total_count",
            "total_episodes",
            "mlflow_run_id",
            "mlflow_experiment_name",
        )
        return {k: checkpoint.get(k) for k in keys}
