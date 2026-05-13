"""Wrapper to convert continuous actions to discrete trading actions."""

import torch
from tensordict import TensorDict
from torchrl.data import Bounded
from torchrl.envs.transforms import Transform

from logger import get_logger
from trading_rl.constants import TradePosition

logger = get_logger(__name__)


class ContinuousToDiscreteAction(Transform):
    """Transform that converts continuous actions [-1, 1] to discrete trading actions.

    This allows TD3/DDPG (continuous action algorithms) to work with discrete trading
    environments by mapping continuous action values to discrete position choices.

    Action Mapping:
    - action < -0.33  → discrete action 0 (Short position)
    - -0.33 ≤ action ≤ 0.33 → discrete action 1 (Hold/neutral)
    - action > 0.33   → discrete action 2 (Long position)
    """

    def __init__(self, discrete_actions=None, thresholds=None, device="cpu"):
        """Initialize the continuous to discrete action transform.

        Args:
            discrete_actions: List of discrete actions (default: [-1, 0, 1])
            thresholds: Thresholds for mapping continuous to discrete
                       (default: [-0.33, 0.33])
            device: Device for tensors
        """
        super().__init__()

        if discrete_actions is None:
            discrete_actions = list(TradePosition)
        if thresholds is None:
            thresholds = [-0.33, 0.33]

        self.discrete_actions = discrete_actions
        self.thresholds = thresholds
        self.n_actions = len(discrete_actions)
        self.device = device

        # Create continuous action spec for TD3/DDPG
        self.action_spec = Bounded(
            low=-1.0, high=1.0, shape=(1,), dtype=torch.float32, device=device
        )

        logger.info(
            "init continuous_to_discrete actions=%s thresholds=%s action_spec=%s",
            discrete_actions, thresholds, self.action_spec,
        )

    def _apply_transform(self, obs: TensorDict) -> TensorDict:
        """Apply transform to observations (no-op for this transform)."""
        return obs

    def _step(self, tensordict: TensorDict, next_tensordict: TensorDict) -> TensorDict:
        """Convert continuous action to discrete before stepping environment."""
        return next_tensordict

    def _call(self, tensordict: TensorDict) -> TensorDict:
        """Transform continuous actions to discrete before environment step."""
        if "action" in tensordict.keys():
            continuous_action = tensordict["action"]
            discrete_action = self._continuous_to_discrete(continuous_action)
            tensordict = tensordict.clone()
            tensordict["action"] = discrete_action
        return tensordict

    def _inv_call(self, tensordict: TensorDict) -> TensorDict:
        """Inverse transform - not applicable for action conversion."""
        return tensordict

    def _continuous_to_discrete(self, continuous_action: torch.Tensor) -> torch.Tensor:
        """Convert continuous action [-1, 1] to discrete action indices.

        Args:
            continuous_action: Tensor with continuous action values in [-1, 1]

        Returns:
            Tensor with discrete action indices, shape = continuous_action.shape[:-1]
        """
        if not isinstance(continuous_action, torch.Tensor):
            continuous_action = torch.tensor(continuous_action, device=self.device)

        values = continuous_action.reshape(-1).float()

        if len(self.thresholds) == 2 and len(self.discrete_actions) == 3:
            indices = torch.ones(values.shape, dtype=torch.long, device=self.device)
            indices[values < self.thresholds[0]] = 0
            indices[values > self.thresholds[1]] = 2
        else:
            indices = torch.zeros(values.shape, dtype=torch.long, device=self.device)
            for i, threshold in enumerate(self.thresholds):
                indices[values > threshold] = i + 1
            indices.clamp_(max=len(self.discrete_actions) - 1)

        return indices.reshape(continuous_action.shape[:-1])

    def transform_action_spec(self, action_spec):
        """Transform action spec to continuous space for TD3/DDPG.

        Args:
            action_spec: The parent environment's action spec (typically CompositeSpec)

        Returns:
            Updated CompositeSpec with continuous action entry
        """
        # Clone to avoid modifying original in place if that's a concern
        # typically action_spec is a CompositeSpec containing "action"
        if isinstance(action_spec, TensorDict) or hasattr(action_spec, "set"):
            # It's a CompositeSpec (which behaves like TensorDict for setting)
            full_spec = action_spec.clone()
            # Replace the "action" key with our continuous spec
            # We assume the key is "action" as is standard in TorchRL
            # If strictness is required, we could check for other keys
            full_spec["action"] = self.action_spec
            return full_spec
        else:
            # Fallback if it's not a CompositeSpec (unlikely for EnvBase)
            return self.action_spec

    def transform_reward_spec(self, reward_spec):
        """Pass through reward spec unchanged."""
        return reward_spec

    def transform_observation_spec(self, observation_spec):
        """Pass through observation spec unchanged."""
        return observation_spec
