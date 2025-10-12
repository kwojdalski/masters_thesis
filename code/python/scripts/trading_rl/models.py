"""Neural network models for trading RL."""

import logging

import torch.nn as nn
from tensordict.nn import InteractionType, TensorDictModule
from torch import distributions as d
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator

logger = logging.getLogger(__name__)


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
    spec: any | None = None,
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


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
