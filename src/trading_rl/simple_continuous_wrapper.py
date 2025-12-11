"""Simple wrapper to handle continuous actions for TD3/DDPG in discrete environments."""

import torch
from torchrl.data import Bounded
from torchrl.envs import EnvBase
from tensordict import TensorDict

from logger import get_logger

logger = get_logger(__name__)


class ContinuousActionWrapper(EnvBase):
    """Wrapper that presents continuous action space but converts to discrete internally."""
    
    def __init__(self, base_env, discrete_actions=None, thresholds=None, device="cpu"):
        """Initialize the wrapper.
        
        Args:
            base_env: The underlying discrete trading environment
            discrete_actions: List of discrete actions (default: [-1, 0, 1])  
            thresholds: Thresholds for continuous->discrete mapping
            device: Device for tensors
        """
        if discrete_actions is None:
            discrete_actions = [-1, 0, 1]
        if thresholds is None:
            thresholds = [-0.33, 0.33]
            
        self.base_env = base_env
        self.discrete_actions = discrete_actions
        self.thresholds = thresholds
        self.device = device
        
        # Initialize parent class
        super().__init__(device=device, batch_size=base_env.batch_size)
        
        logger.info(f"ContinuousActionWrapper initialized:")
        logger.info(f"  Discrete actions: {discrete_actions}")
        logger.info(f"  Thresholds: {thresholds}")
    
    @property
    def action_spec(self):
        """Return continuous action spec."""
        return Bounded(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=torch.float32,
            device=self.device
        )
    
    @property
    def reward_spec(self):
        """Pass through reward spec from base environment."""
        return self.base_env.reward_spec
    
    @property
    def observation_spec(self):
        """Pass through observation spec from base environment.""" 
        return self.base_env.observation_spec
    
    def _reset(self, tensordict=None, **kwargs):
        """Reset the base environment."""
        return self.base_env.reset(tensordict, **kwargs)
    
    def _step(self, tensordict):
        """Convert continuous action to discrete and step base environment."""
        # Get continuous action
        continuous_action = tensordict["action"]
        
        # Convert to discrete 
        discrete_action = self._continuous_to_discrete(continuous_action)
        
        # Create new tensordict with discrete action
        discrete_tensordict = tensordict.clone()
        discrete_tensordict["action"] = discrete_action
        
        # Step base environment
        result = self.base_env.step(discrete_tensordict)
        
        return result
        
    def _continuous_to_discrete(self, continuous_action: torch.Tensor) -> torch.Tensor:
        """Convert continuous action [-1, 1] to discrete action index."""
        # Handle different tensor shapes
        if continuous_action.numel() == 1:
            action_val = continuous_action.item()
        else:
            action_val = continuous_action.flatten()[0].item()
        
        # Map to discrete action index
        if action_val < self.thresholds[0]:
            discrete_idx = 0  # Short
        elif action_val > self.thresholds[1]:  
            discrete_idx = 2  # Long
        else:
            discrete_idx = 1  # Hold
            
        return torch.tensor(discrete_idx, dtype=torch.long, device=self.device)

    def set_seed(self, seed):
        """Set random seed."""
        return self.base_env.set_seed(seed)
        
    def close(self):
        """Close the environment."""
        return self.base_env.close()