# %%

from torchrl.envs import GymEnv

env = GymEnv("Pendulum-v1")

from torchrl.modules import MLP, Actor, ValueOperator
from torchrl.objectives import DDPGLoss

n_obs = env.observation_spec["observation"].shape[-1]
n_act = env.action_spec.shape[-1]
actor = Actor(MLP(in_features=n_obs, out_features=n_act, num_cells=[32, 32]))
value_net = ValueOperator(
    MLP(in_features=n_obs + n_act, out_features=1, num_cells=[32, 32]),
    in_keys=["observation", "action"],
)

ddpg_loss = DDPGLoss(actor_network=actor, value_network=value_net)

# %%
ddpg_loss
rollout = env.rollout(max_steps=100, policy=actor)
loss_vals = ddpg_loss(rollout)

# %%
# %%
