"""Neural network models for trading RL."""

from typing import Any

import torch
import torch.nn as nn
from tensordict.nn import InteractionType, TensorDictModule, NormalParamExtractor
from torch import distributions as d
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator
from torchrl.modules.distributions import TanhNormal

from logger import get_logger

logger = get_logger(__name__)


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

        logger.info("build discrete_net input_dim={} n_actions={} hidden_dims={}", input_dim, n_actions, hidden_dims)

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

    logger.info("build value network hidden_dims={}", hidden_dims)
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

    logger.info("build ppo value network hidden_dims={}", hidden_dims)
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

    # Small output-layer init: tanh saturates at ±1 when the last layer has
    # large weights, killing gradients before training begins. [-3e-3, 3e-3]
    # keeps the initial policy near-zero (flat), consistent with the TD3 paper.
    for m in reversed(list(actor_net.modules())):
        if isinstance(m, nn.Linear):
            nn.init.uniform_(m.weight, -3e-3, 3e-3)
            nn.init.zeros_(m.bias)
            break

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

    logger.info("build td3 qvalue network hidden_dims={}", hidden_dims)
    return value_net


def create_sac_actor(
    n_obs: int,
    n_act: int,
    hidden_dims: list[int] | None = None,
    spec: Any | None = None,
) -> ProbabilisticActor:
    """Create a stochastic TanhNormal actor for SAC (continuous actions).

    Identical in structure to the continuous PPO actor — a TanhNormal
    policy outputs a sampled action together with its log-probability, which
    SACLoss requires for the entropy-regularised objective.
    """
    return create_continuous_ppo_actor(n_obs, n_act, hidden_dims=hidden_dims, spec=spec)


def create_sac_qvalue_network(
    n_obs: int,
    n_act: int,
    hidden_dims: list[int] | None = None,
) -> ValueOperator:
    """Create a Q-value network for SAC taking (observation, action) as input.

    Identical in structure to the TD3 Q-value network — SAC also uses double
    Q-learning to reduce overestimation bias.
    """
    return create_td3_qvalue_network(n_obs, n_act, hidden_dims=hidden_dims)


class _GRUActorNet(nn.Module):
    """GRU-based network backbone for recurrent stochastic actor.

    Processes observation sequences in temporal order and outputs the (loc, scale)
    parameters of a TanhNormal distribution for each time step.
    """

    def __init__(
        self,
        n_obs: int,
        n_act: int,
        gru_hidden_dim: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(n_obs, gru_hidden_dim, num_layers, batch_first=True)
        self.head = nn.Linear(gru_hidden_dim, n_act * 2)
        self._extractor = NormalParamExtractor()

        # Small output-layer init — keeps the initial policy near-neutral.
        nn.init.uniform_(self.head.weight, -3e-3, 3e-3)
        nn.init.zeros_(self.head.bias)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # obs arrives as (n_obs,), (T, n_obs), or (B, T, n_obs) from TorchRL.
        was_1d = obs.dim() == 1
        was_2d = obs.dim() == 2
        if was_1d:
            obs = obs.unsqueeze(0).unsqueeze(0)   # → (1, 1, n_obs)
        elif was_2d:
            obs = obs.unsqueeze(0)                 # → (1, T, n_obs)
        gru_out, _ = self.gru(obs)                 # (B, T, gru_hidden_dim)
        if was_1d:
            gru_out = gru_out.squeeze(0).squeeze(0)  # → (gru_hidden_dim,)
        elif was_2d:
            gru_out = gru_out.squeeze(0)             # → (T, gru_hidden_dim)
        return self._extractor(self.head(gru_out))   # (loc, scale) pair


class _GRUValueNet(nn.Module):
    """GRU-based network backbone for recurrent state-value critic.

    Processes observation sequences in temporal order and outputs a scalar
    V(s) estimate for each time step.
    """

    def __init__(
        self,
        n_obs: int,
        gru_hidden_dim: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(n_obs, gru_hidden_dim, num_layers, batch_first=True)
        self.head = nn.Linear(gru_hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        was_1d = obs.dim() == 1
        was_2d = obs.dim() == 2
        if was_1d:
            obs = obs.unsqueeze(0).unsqueeze(0)
        elif was_2d:
            obs = obs.unsqueeze(0)
        gru_out, _ = self.gru(obs)
        if was_1d:
            gru_out = gru_out.squeeze(0).squeeze(0)
        elif was_2d:
            gru_out = gru_out.squeeze(0)
        return self.head(gru_out)


def create_recurrent_ppo_actor(
    n_obs: int,
    n_act: int,
    hidden_dims: list[int] | None = None,
    spec: Any | None = None,
    gru_num_layers: int = 1,
) -> ProbabilisticActor:
    """Create a GRU-backed stochastic actor for RecurrentPPO.

    Args:
        n_obs: Flattened observation dimension.
        n_act: Action dimension.
        hidden_dims: Network width per layer; hidden_dims[0] is used as the GRU
            hidden size. Defaults to [64, 32] (GRU hidden = 64).
        spec: Action spec from the environment.
        gru_num_layers: Number of stacked GRU layers.

    Returns:
        ProbabilisticActor with TanhNormal distribution and return_log_prob=True.
    """
    logger.info("build recurrent_ppo actor")

    if hidden_dims is None:
        hidden_dims = [64, 32]
    gru_hidden_dim = hidden_dims[0]

    net = _GRUActorNet(n_obs, n_act, gru_hidden_dim, gru_num_layers)
    module = TensorDictModule(
        net,
        in_keys=["observation"],
        out_keys=["loc", "scale"],
    )
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
    logger.info(
        "build recurrent_ppo actor complete gru_hidden_dim={} gru_num_layers={}",
        gru_hidden_dim, gru_num_layers,
    )
    return actor


def create_recurrent_ppo_value_network(
    n_obs: int,
    hidden_dims: list[int] | None = None,
    gru_num_layers: int = 1,
) -> ValueOperator:
    """Create a GRU-backed state-value critic for RecurrentPPO.

    Args:
        n_obs: Flattened observation dimension.
        hidden_dims: hidden_dims[0] is the GRU hidden size. Defaults to [64, 32, 16].
        gru_num_layers: Number of stacked GRU layers.

    Returns:
        ValueOperator estimating V(s) for each step in the episode rollout.
    """
    logger.info("build recurrent_ppo value network")

    if hidden_dims is None:
        hidden_dims = [64, 32, 16]
    gru_hidden_dim = hidden_dims[0]

    value_net = ValueOperator(
        _GRUValueNet(n_obs, gru_hidden_dim, gru_num_layers),
        in_keys=["observation"],
        out_keys=["state_value"],
    )
    logger.info(
        "build recurrent_ppo value network complete gru_hidden_dim={} gru_num_layers={}",
        gru_hidden_dim, gru_num_layers,
    )
    return value_net


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
