"""Neural network models for trading RL."""

from typing import Any

import torch
import torch.nn as nn
from joblib import Memory
from tensordict import TensorDict
from tensordict.nn import InteractionType, TensorDictModule, NormalParamExtractor
from torch import distributions as d
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator
from torchrl.modules.distributions import TanhNormal

from logger import get_logger

logger = get_logger(__name__)

# Setup joblib memory for caching model creation
memory = Memory(location=".cache/joblib", verbose=1)


class _FlattenObs(nn.Module):
    """Flatten a multi-dimensional observation to a 1-D feature vector.

    gym_anytrading returns per-step observations shaped [window, n_features].
    Batched, this gives [T, window, n_features]; unbatched, [window, n_features].

    By recording obs_ndim (number of dims per single observation), we can always
    flatten exactly the obs dims regardless of whether a batch dim is present:
        x.flatten(start_dim=-obs_ndim)  →  [..., window*n_features]
    For 1-D obs (obs_ndim=1) this is a no-op because flattening a single dim
    leaves the tensor unchanged.
    """

    def __init__(self, obs_ndim: int = 1):
        super().__init__()
        self.obs_ndim = obs_ndim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.obs_ndim > 1:
            return x.flatten(start_dim=-self.obs_ndim)
        return x


class ScaleFromUnitRange(nn.Module):
    """Map actions from normalized [-1, 1] range to environment action bounds."""

    def __init__(self, low: torch.Tensor, high: torch.Tensor):
        super().__init__()
        self.register_buffer("low", torch.as_tensor(low))
        self.register_buffer("high", torch.as_tensor(high))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low = self.low.to(device=x.device, dtype=x.dtype)
        high = self.high.to(device=x.device, dtype=x.dtype)
        return low + (x + 1.0) * (high - low) / 2.0


def _extract_action_bounds_from_spec(spec: Any) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Extract low/high action bounds from a TorchRL spec if available."""
    if spec is None:
        return None

    low = None
    high = None

    # TorchRL Bounded specs commonly expose bounds via spec.space.low/high.
    space = getattr(spec, "space", None)
    if space is not None:
        low = getattr(space, "low", None)
        high = getattr(space, "high", None)

    # Fallback for specs that expose low/high directly.
    if low is None:
        low = getattr(spec, "low", None)
    if high is None:
        high = getattr(spec, "high", None)

    if low is None or high is None:
        return None

    return torch.as_tensor(low), torch.as_tensor(high)


def clear_model_cache():
    """Clear model creation cache."""
    memory.clear(warn=True)


class DiscreteNet(nn.Module):
    """Neural network for discrete action selection with probability outputs.

    This network takes observations as input and outputs action probabilities
    suitable for categorical distributions.
    """

    def __init__(
        self,
        input_dim: int,
        n_actions: int,
        hidden_dims: list[int] | None = None,
        obs_ndim: int = 1,
    ):
        """Initialize the discrete action network.

        Args:
            input_dim: Total flattened dimension of one observation.
            n_actions: Number of discrete actions.
            hidden_dims: List of hidden layer dimensions. Defaults to [64, 32].
            obs_ndim: Number of dims per single observation (1 for flat, 2 for
                      gym_anytrading's [window, features] obs, etc.).
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 32]

        # Build network layers; _FlattenObs handles multi-dim obs (e.g. anytrading)
        layers: list[nn.Module] = [_FlattenObs(obs_ndim=obs_ndim)]
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

        logger.info("build discrete_net input_dim=%d n_actions=%d hidden_dims=%s", input_dim, n_actions, hidden_dims)

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
    logger.info("build actor network")

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

    logger.info("build actor network complete")
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
    logger.info("build value network")

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

    logger.info("build value network hidden_dims=%s", hidden_dims)
    return value_net


def create_ppo_actor(
    n_obs: int,
    n_act: int,
    hidden_dims: list[int] | None = None,
    spec: Any | None = None,
    obs_ndim: int = 1,
) -> ProbabilisticActor:
    """Create a probabilistic actor specifically for PPO.

    Args:
        n_obs: Total flattened observation size.
        n_act: Number of actions.
        hidden_dims: Hidden layer dimensions.
        spec: Action spec from environment.
        obs_ndim: Number of dims per single observation (1 for flat obs,
                  2 for gym_anytrading's [window, features] obs).

    Returns:
        ProbabilisticActor module configured for PPO
    """
    logger.info("build ppo actor network")

    # Create base network
    net = DiscreteNet(n_obs, n_act, hidden_dims, obs_ndim=obs_ndim)

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

    logger.info("build ppo actor network complete")
    return actor


def create_continuous_ppo_actor(
    n_obs: int,
    n_act: int,
    hidden_dims: list[int] | None = None,
    spec: Any | None = None,
) -> ProbabilisticActor:
    """Create a probabilistic actor for continuous action spaces (PPO).

    Uses a TanhNormal distribution where the network outputs mean (loc)
    and scale (standard deviation).

    Args:
        n_obs: Number of observations
        n_act: Number of actions
        hidden_dims: Hidden layer dimensions
        spec: Action spec from environment

    Returns:
        ProbabilisticActor module configured for continuous PPO
    """
    logger.info("build continuous ppo actor network")

    if hidden_dims is None:
        hidden_dims = [64, 32]

    # Create base MLP that outputs 2 * n_act (loc and scale for each action dim)
    net = MLP(
        in_features=n_obs,
        out_features=n_act * 2,  # Output loc and scale
        num_cells=hidden_dims,
        activation_class=nn.Tanh,
    )

    # Add NormalParamExtractor to split output into loc and scale
    extractor = NormalParamExtractor()
    
    # Combined network
    net = nn.Sequential(net, extractor)

    # Wrap in TensorDictModule
    module = TensorDictModule(
        net,
        in_keys=["observation"],
        out_keys=["loc", "scale"],
    )

    # Create probabilistic actor using TanhNormal
    # TanhNormal ensures actions are bounded in [-1, 1] (or spec bounds)
    actor = ProbabilisticActor(
        module=module,
        distribution_class=TanhNormal,
        distribution_kwargs={},
        in_keys=["loc", "scale"],
        out_keys=["action"],
        spec=spec,
        safe=False,
        default_interaction_type=InteractionType.RANDOM,
        return_log_prob=True,
    )

    logger.info("build continuous ppo actor network complete")
    return actor


def create_ppo_value_network(
    n_obs: int,
    hidden_dims: list[int] | None = None,
    obs_ndim: int = 1,
) -> ValueOperator:
    """Create a value network for PPO (state value estimation).

    Args:
        n_obs: Total flattened observation size.
        hidden_dims: Hidden layer dimensions for MLP.
        obs_ndim: Number of dims per single observation (1 for flat, 2 for
                  gym_anytrading's [window, features] obs).

    Returns:
        ValueOperator module for V(s) estimation
    """
    logger.info("build ppo value network")

    if hidden_dims is None:
        hidden_dims = [64, 32, 16]

    value_net = ValueOperator(
        nn.Sequential(
            _FlattenObs(obs_ndim=obs_ndim),
            MLP(
                in_features=n_obs,
                out_features=1,
                num_cells=hidden_dims,
            ),
        ),
        in_keys=["observation"],  # Only state input for V(s)
        out_keys=["state_value"],  # V(s) not Q(s,a)
    )

    logger.info("build ppo value network hidden_dims=%s", hidden_dims)
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
    logger.info("build ddpg actor")

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

    # Produce normalized actions in [-1, 1], then map to env bounds if a bounded
    # spec is provided. This keeps the actor stable while supporting non-unit
    # action domains (e.g. [0, 1] long-only allocations).
    actor_layers: list[nn.Module] = [actor_net, nn.Tanh()]
    bounds = _extract_action_bounds_from_spec(spec)
    if bounds is not None:
        low, high = bounds
        actor_layers.append(ScaleFromUnitRange(low=low, high=high))

    actor_net = nn.Sequential(*actor_layers)

    # Wrap in TensorDictModule
    actor = TensorDictModule(
        actor_net,
        in_keys=["observation"],
        out_keys=["action"],
    )

    logger.info("build ddpg actor complete")
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

    logger.info("build td3 qvalue network hidden_dims=%s", hidden_dims)
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
            module=nn.Identity(),  # Dummy module to satisfy TensorDictModule's __init__
            in_keys=qvalue_net1.in_keys,  # Assume in_keys are consistent
            out_keys=qvalue_net1.out_keys,  # Assume out_keys are consistent
        )

        self.qvalue_nets = nn.ModuleList([qvalue_net1, qvalue_net2])

        def forward(self, tensordict: TensorDict) -> TensorDict:
            # Call each Q-network sequentially, passing a clone to avoid in-place modification issues

            q_outputs = []

            for q_net in self.qvalue_nets:
                q_output_td = q_net(
                    tensordict.clone()
                )  # Clone to prevent in-place modification of tensordict for next net

                q_outputs.append(
                    q_output_td.get(self.out_keys[0])
                )  # Assuming single output key for ValueOperator

            # Concatenate the Q-values along the last dimension

            # Each q_output is [..., 1]. We want [..., 2].

            stacked_q_values = torch.cat(
                q_outputs, dim=-1
            )  # Output shape will be [..., 2]

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
    logger.info("build td3 twin qvalue network n_outputs=%d", num_qvalue_nets)
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

    logger.info("build td3 twin qvalue network hidden_dims=%s n_outputs=%d", hidden_dims, num_qvalue_nets)
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
    logger.info("build td3 stacked qvalue networks")
    # Create two individual Q-networks using the existing factory
    qvalue_net1 = create_td3_qvalue_network(n_obs, n_act, hidden_dims)
    qvalue_net2 = create_td3_qvalue_network(n_obs, n_act, hidden_dims)

    # Combine them into a single TensorDictModule
    stacked_q_net = StackedQValueNetwork(qvalue_net1, qvalue_net2)
    logger.info("build td3 stacked qvalue networks complete")
    return stacked_q_net


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
