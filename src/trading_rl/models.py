"""Neural network models for trading RL."""

from typing import Any

import torch
import torch.nn as nn
from joblib import Memory
from tensordict import TensorDict
from tensordict.nn import InteractionType, TensorDictModule
from torch import distributions as d
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator

from logger import get_logger

logger = get_logger(__name__)

# Setup joblib memory for caching model creation
memory = Memory(location=".cache/joblib", verbose=1)


def clear_model_cache():
    """Clear model creation cache."""
    memory.clear(warn=True)


class DiscreteNet(nn.Module):
    """Neural network for discrete action selection with probability outputs.

    This network takes observations as input and outputs action probabilities
    suitable for categorical distributions.
    """

    def __init__(
        self, input_dim: int, n_actions: int, hidden_dims: list[int] | None = None
    ):
        """Initialize the discrete action network.

        Args:
            input_dim: Dimension of input observations
            n_actions: Number of discrete actions
            hidden_dims: List of hidden layer dimensions. Defaults to [64, 32].
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 32]

        # Build network layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                ]
            )
            prev_dim = hidden_dim

        # Output layer with softmax
        layers.extend(
            [
                nn.Linear(prev_dim, n_actions),
                nn.Softmax(dim=-1),
            ]
        )

        self.network = nn.Sequential(*layers)

        logger.info(
            f"Created DiscreteNet: input_dim={input_dim}, "
            f"n_actions={n_actions}, hidden_dims={hidden_dims}"
        )

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input tensor

        Returns:
            Dictionary with "probs" key containing action probabilities
        """
        probs = self.network(x)
        return {"probs": probs}


def create_actor(
    n_obs: int,
    n_act: int,
    hidden_dims: list[int] | None = None,
    spec: Any | None = None,
) -> ProbabilisticActor:
    """Create a probabilistic actor for discrete action spaces.

    Args:
        n_obs: Number of observations
        n_act: Number of actions
        hidden_dims: Hidden layer dimensions
        spec: Action spec from environment

    Returns:
        ProbabilisticActor module
    """
    logger.info("Creating actor network")

    # Create base network
    net = DiscreteNet(n_obs, n_act, hidden_dims)

    # Wrap in TensorDictModule
    module = TensorDictModule(
        net,
        in_keys=["observation"],
        out_keys=["probs"],
    )

    # Create probabilistic actor
    actor = ProbabilisticActor(
        module=module,
        distribution_class=d.OneHotCategorical,
        in_keys=["probs"],
        spec=spec,
        safe=True,
        default_interaction_type=InteractionType.RANDOM,
    )

    logger.info("Actor network created")
    return actor


def create_value_network(
    n_obs: int,
    n_act: int,
    hidden_dims: list[int] | None = None,
) -> ValueOperator:
    """Create a value network for state-action value estimation.

    Args:
        n_obs: Number of observations
        n_act: Number of actions
        hidden_dims: Hidden layer dimensions for MLP

    Returns:
        ValueOperator module
    """
    logger.info("Creating value network")

    if hidden_dims is None:
        hidden_dims = [64, 32, 16]

    value_net = ValueOperator(
        MLP(
            in_features=n_obs + n_act,
            out_features=1,
            num_cells=hidden_dims,
        ),
        in_keys=["observation", "probs"],
        out_keys=["state_action_value"],
    )

    logger.info(f"Value network created with hidden_dims={hidden_dims}")
    return value_net


def create_ppo_actor(
    n_obs: int,
    n_act: int,
    hidden_dims: list[int] | None = None,
    spec: Any | None = None,
) -> ProbabilisticActor:
    """Create a probabilistic actor specifically for PPO.

    Args:
        n_obs: Number of observations
        n_act: Number of actions
        hidden_dims: Hidden layer dimensions
        spec: Action spec from environment

    Returns:
        ProbabilisticActor module configured for PPO
    """
    logger.info("Creating PPO actor network")

    # Create base network
    net = DiscreteNet(n_obs, n_act, hidden_dims)

    # Wrap in TensorDictModule
    module = TensorDictModule(
        net,
        in_keys=["observation"],
        out_keys=["probs"],
    )

    # Create probabilistic actor with proper PPO configuration
    # Don't specify both out_keys and return_log_prob to avoid conflicts
    actor = ProbabilisticActor(
        module=module,
        distribution_class=d.OneHotCategorical,
        in_keys=["probs"],
        spec=None,  # Remove spec to avoid conflicts with multiple out_keys
        safe=False,  # Must be False when spec=None
        default_interaction_type=InteractionType.RANDOM,
        return_log_prob=True,  # This automatically creates action_log_prob
    )

    logger.info("PPO actor network created")
    return actor


def create_ppo_value_network(
    n_obs: int,
    hidden_dims: list[int] | None = None,
) -> ValueOperator:
    """Create a value network for PPO (state value estimation).

    Args:
        n_obs: Number of observations
        hidden_dims: Hidden layer dimensions for MLP

    Returns:
        ValueOperator module for V(s) estimation
    """
    logger.info("Creating PPO value network")

    if hidden_dims is None:
        hidden_dims = [64, 32, 16]

    value_net = ValueOperator(
        MLP(
            in_features=n_obs,  # PPO value network only takes state
            out_features=1,
            num_cells=hidden_dims,
        ),
        in_keys=["observation"],  # Only state input for V(s)
        out_keys=["state_value"],  # V(s) not Q(s,a)
    )

    logger.info(f"PPO value network created with hidden_dims={hidden_dims}")
    return value_net


def create_ddpg_actor(
    n_obs: int,
    n_act: int,
    hidden_dims: list[int] | None = None,
    spec: Any | None = None,
) -> TensorDictModule:
    """Create a deterministic actor for DDPG (continuous actions).

    Args:
        n_obs: Number of observations
        n_act: Number of actions
        hidden_dims: Hidden layer dimensions
        spec: Action spec from environment

    Returns:
        Deterministic actor module for DDPG
    """
    logger.info("Creating DDPG deterministic actor")

    if hidden_dims is None:
        hidden_dims = [64, 32]

    # Create deterministic network (no softmax)
    actor_net = MLP(
        in_features=n_obs,
        out_features=n_act,
        num_cells=hidden_dims,
        activation_class=nn.ReLU,
        activate_last_layer=False,  # No activation on output
    )

    # Apply tanh for bounded continuous actions
    # Always apply Tanh for DDPG/TD3 to ensure actions are in [-1, 1]
    # If spec is provided, we could scale, but standard DDPG assumes [-1, 1]
    actor_net = nn.Sequential(
        actor_net,
        nn.Tanh(),  # Output in [-1, 1]
    )

    # Wrap in TensorDictModule
    actor = TensorDictModule(
        actor_net,
        in_keys=["observation"],
        out_keys=["action"],
    )

    logger.info("DDPG deterministic actor created")
    return actor


def create_td3_actor(
    n_obs: int,
    n_act: int,
    hidden_dims: list[int] | None = None,
    spec: Any | None = None,
) -> TensorDictModule:
    """Create deterministic actor for TD3 (continuous actions)."""
    return create_ddpg_actor(n_obs, n_act, hidden_dims=hidden_dims, spec=spec)


def create_td3_qvalue_network(
    n_obs: int,
    n_act: int,
    hidden_dims: list[int] | None = None,
) -> ValueOperator:
    """Create Q-value network for TD3 taking observation and action."""
    if hidden_dims is None:
        hidden_dims = [64, 32, 16]

    value_net = ValueOperator(
        MLP(
            in_features=n_obs + n_act,
            out_features=1,
            num_cells=hidden_dims,
        ),
        in_keys=["observation", "action"],
        out_keys=["state_action_value"],
    )

    logger.info(f"TD3 Q-value network created with hidden_dims={hidden_dims}")
    return value_net


class StackedQValueNetwork(TensorDictModule):


    """


    A TensorDictModule that combines two Q-value networks (ValueOperator) and stacks their outputs.


    The combined module takes 'observation' and 'action' and outputs 'state_action_value'


    where the last dimension of 'state_action_value' contains the Q-values from each critic.


    """


    def __init__(self, qvalue_net1: ValueOperator, qvalue_net2: ValueOperator):


        # Pass a dummy nn.Identity() module as the 'module' argument


        # The actual logic is in the forward method of StackedQValueNetwork itself.


        super().__init__(


            module=nn.Identity(), # Dummy module to satisfy TensorDictModule's __init__


            in_keys=qvalue_net1.in_keys, # Assume in_keys are consistent


            out_keys=qvalue_net1.out_keys # Assume out_keys are consistent


        )


        self.qvalue_nets = nn.ModuleList([qvalue_net1, qvalue_net2])





    def forward(self, tensordict: TensorDict) -> TensorDict:


        # Call each Q-network sequentially, passing a clone to avoid in-place modification issues


        q_outputs = []


        for q_net in self.qvalue_nets:


            q_output_td = q_net(tensordict.clone()) # Clone to prevent in-place modification of tensordict for next net


            q_outputs.append(q_output_td.get(self.out_keys[0])) # Assuming single output key for ValueOperator





        # Stack the Q-values along a new dimension


        stacked_q_values = torch.stack(q_outputs, dim=-1) # Output shape will be [..., 1, 2]





        # Set the stacked Q-values in the original tensordict


        tensordict.set(self.out_keys[0], stacked_q_values)


        return tensordict


def create_td3_twin_qvalue_network(
    n_obs: int,
    n_act: int,
    hidden_dims: list[int] | None = None,
    num_qvalue_nets: int = 2,
) -> ValueOperator:
    """
    Create a twin Q-value network for TD3 that outputs multiple Q-values.

    This creates a single ValueOperator that outputs num_qvalue_nets Q-values,
    which is what TorchRL's TD3Loss expects.
    """
    logger.info(f"Creating TD3 twin Q-value network with {num_qvalue_nets} outputs")
    if hidden_dims is None:
        hidden_dims = [64, 32, 16]

    # Create MLP that outputs num_qvalue_nets Q-values
    value_net = ValueOperator(
        MLP(
            in_features=n_obs + n_act,
            out_features=num_qvalue_nets,  # Output multiple Q-values
            num_cells=hidden_dims,
        ),
        in_keys=["observation", "action"],
        out_keys=["state_action_value"],
    )

    logger.info(f"TD3 twin Q-value network created with hidden_dims={hidden_dims}, outputs={num_qvalue_nets}")
    return value_net


def create_td3_stacked_qvalue_network(
    n_obs: int,
    n_act: int,
    hidden_dims: list[int] | None = None,
) -> TensorDictModule:
    """
    Create a combined Q-value network for TD3 that outputs stacked Q-values.

    Deprecated: Use create_td3_twin_qvalue_network instead for better TD3Loss compatibility.
    """
    logger.info("Creating TD3 stacked Q-value networks")
    # Create two individual Q-networks using the existing factory
    qvalue_net1 = create_td3_qvalue_network(n_obs, n_act, hidden_dims)
    qvalue_net2 = create_td3_qvalue_network(n_obs, n_act, hidden_dims)

    # Combine them into a single TensorDictModule
    stacked_q_net = StackedQValueNetwork(qvalue_net1, qvalue_net2)
    logger.info("TD3 stacked Q-value network created")
    return stacked_q_net


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
