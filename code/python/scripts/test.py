# %%
import torch
from torch import nn
from torchrl.data import Bounded
from torchrl.modules.tensordict_module.actors import Actor, ValueOperator
from torchrl.objectives.ddpg import DDPGLoss

_ = torch.manual_seed(42)
n_act, n_obs = 4, 3
spec = Bounded(-torch.ones(n_act), torch.ones(n_act), (n_act,))
actor = Actor(spec=spec, module=nn.Linear(n_obs, n_act))


# %%
class ValueClass(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(n_obs + n_act, 1)

    def forward(self, obs, act):
        return self.linear(torch.cat([obs, act], -1))


module = ValueClass()
value = ValueOperator(module=module, in_keys=["observation", "action"])
# %%
loss = DDPGLoss(actor, value)
# %%
(
    loss_actor,
    loss_value,
    pred_value,
    target_value,
    pred_value_max,
    target_value_max,
) = loss(
    observation=torch.randn(n_obs),
    action=spec.rand(),
    next_done=torch.zeros(1, dtype=torch.bool),
    next_terminated=torch.zeros(1, dtype=torch.bool),
    next_observation=torch.randn(n_obs),
    next_reward=torch.randn(1),
)

loss_actor.backward()

# The output keys can also be filtered using the DDPGLoss.select_out_keys() method.
# Examples

loss.select_out_keys("loss_actor", "loss_value")
# %%
loss_actor, loss_value = loss(
    observation=torch.randn(n_obs),
    action=spec.rand(),
    next_done=torch.zeros(1, dtype=torch.bool),
    next_terminated=torch.zeros(1, dtype=torch.bool),
    next_observation=torch.randn(n_obs),
    next_reward=torch.randn(1),
)
loss_actor.backward()
