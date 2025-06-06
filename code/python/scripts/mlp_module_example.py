# %%
import torch
import torch.distributions as D
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.envs import GymEnv
from torchrl.modules import MLP
from torchrl.modules.tensordict_module.actors import ProbabilisticActor

# 1. Create the environment
env = GymEnv("CartPole-v1")
obs_dim = env.observation_spec["observation"].shape[-1]
n_actions = env.action_spec.n

# 2. Define the base MLP
mlp = MLP(
    in_features=obs_dim,
    out_features=n_actions,
    depth=2,
    num_cells=64,
)


# 3. Wrap MLP so it modifies the TensorDict in-place
class LogitsModule(nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp

    def forward(self, obs):  # <--- receives just the observation
        logits = self.mlp(obs)
        return {"logits": logits}


# %%

wrapped = TensorDictModule(
    module=LogitsModule(mlp), in_keys=["observation"], out_keys=["logits"]
)

# 4. Define the ProbabilisticActor (without dist_in_keys)
actor = ProbabilisticActor(
    module=wrapped,
    in_keys=["observation"],
    out_keys=["action"],
    distribution_class=D.Categorical,
    distribution_kwargs={"logits": True},
)
# %%
# 5. Run a test
sample_obs = torch.randn(1, obs_dim)
td = TensorDict({"observation": sample_obs}, batch_size=[1])
out_td = actor(td)

print("Sampled action:", out_td["action"])

# %%
import torch
import torch.distributions as D
import torch.nn as nn
from tensordict import TensorDict
from torchrl.envs import GymEnv
from torchrl.modules import MLP
from torchrl.modules.tensordict_module.actors import ProbabilisticActor

# %%
# 1. Create the environment
env = GymEnv("CartPole-v1")
n_actions = env.action_spec.n

# 2. MLP that outputs logits
mlp = MLP(in_features=obs_dim, out_features=n_actions, depth=2, num_cells=64)


# %%
# 3. Wrap MLP to match TorchRL expectations
class LogitsModule(nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp

    def forward(self, obs):
        return {"logits": self.mlp(obs)}


# 4. Wrap in TensorDictModule
wrapped = TensorDictModule(
    module=LogitsModule(mlp), in_keys=["observation"], out_keys=["logits"]
)

# 5. Create the ProbabilisticActor with correct dist keys
actor = ProbabilisticActor(
    module=wrapped,
    in_keys=["observation"],
    out_keys=["action"],
    distribution_class=D.Categorical,
    distribution_kwargs={"logits": True},
    return_log_prob=False,
)

# Set distribution input keys explicitly to avoid using 'observation'
actor.dist_keys = ["logits"]  # 👈 IMPORTANT FIX

# 6. Test
sample_obs = torch.randn(1, obs_dim)
td = TensorDict({"observation": sample_obs}, batch_size=[1])
out_td = actor(td)

print("Sampled action:", out_td["action"])

# %%
import torch
import torch.distributions as D
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictSequential
from torchrl.envs import GymEnv
from torchrl.modules import MLP

# %%
# 1. Environment
env = GymEnv("CartPole-v1")
n_actions = env.action_spec.n

# 2. MLP
mlp = MLP(in_features=obs_dim, out_features=n_actions, depth=2, num_cells=64)


# 3. LogitsModule to generate logits from observation
class LogitsModule(nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp

    def forward(self, obs):
        return {"logits": self.mlp(obs)}


logits_module = TensorDictModule(
    module=LogitsModule(mlp), in_keys=["observation"], out_keys=["logits"]
)


# 4. Distribution module that builds a Categorical from logits
class CategoricalModule(nn.Module):
    def forward(self, logits):
        return {"_dist": D.Categorical(logits=logits)}


dist_module = TensorDictModule(
    module=CategoricalModule(), in_keys=["logits"], out_keys=["_dist"]
)


# 5. Sample from the distribution
class SamplerModule(nn.Module):
    def forward(self, dist):
        return {"action": dist.sample()}


sample_module = TensorDictModule(
    module=SamplerModule(), in_keys=["_dist"], out_keys=["action"]
)

# 6. Assemble full actor pipeline
actor = TensorDictSequential(logits_module, dist_module, sample_module)
# 7. Test

obs = torch.randn(1, obs_dim)
td = TensorDict({"observation": obs}, batch_size=[1])
td = actor(td)
print("Sampled action:", td["action"])

# %%

# %%
