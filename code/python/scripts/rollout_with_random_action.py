# %%
import gym_trading_env  # noqa: F401
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.envs import EnvBase, GymWrapper, TransformedEnv
from torchrl.envs.transforms import StepCounter

# %%
from torchrl.envs.utils import check_env_specs, step_mdp


def simple_rollout(steps=100):
    # preallocate:
    data = TensorDict(batch_size=[steps])
    # reset
    _data = env.reset()
    for i in range(steps):
        _data["action"] = env.action_spec.rand()
        _data = env.step(_data)
        data[i] = _data
        _data = step_mdp(_data, keep_other=True)
    return data


# %%
# Getting data
df = pd.read_pickle("./data/raw/binance/binance-BTCUSDT-1h.pkl")


# %%
def reward_function(history):
    return np.log(
        history["portfolio_valuation", -1] / history["portfolio_valuation", -2]
    )


# %%
# Features must have features in their name
df["feature"] = df["close"] / df["close"].shift(1)
df["feature_pct_change"] = df["close"].pct_change()
df["feature_high"] = df["high"] / df["close"] - 1
df["feature_close"] = df["close"]
df = df.dropna()

# Make the environment
base_env = gym.make(
    "TradingEnv",
    name="BTCUSD",
    df=df,
    positions=[-1, 0, 1],  # -1 (=SHORT), 0(=OUT), +1 (=LONG)
    trading_fees=0,
    borrow_interest_rate=0,
    reward_function=reward_function,
)

# %%
# Wrap environment for torchrl
env = GymWrapper(base_env)
env = TransformedEnv(env, StepCounter())


# %%
class MyCustomEnv(EnvBase):
    def __init__(self, base_env):
        super().__init__(batch_size=[])
        self.base_env = base_env
        self.observation_space = base_env.observation_space
        self.action_space = base_env.action_space
        # self._unwrapped = base_env
        self.batch_size = base_env.batch_size

    def _set_seed(self, seed):
        self.base_env.set_seed(seed)

    def _reset(self, tensordict):
        initial_obs = self.base_env.reset()
        return initial_obs

    def _step(self, tensordict):
        # Get step results from base environment
        return self.base_env._step(tensordict)

    # def _step(self, tensordict):
    #     # Get step results from base environment
    #     if tensordict.is_empty():
    #         tensordict = self.base_env.step(tensordict)

    #     step_result = self.base_env.step(tensordict)

    #     # Extract observation and generate random action
    #     obs = step_result["observation"]
    #     action = np.random.choice([-1, 0, 1])

    #     reward = step_result["next", "observation"][3] - step_result["observation"][3]
    #     # * action
    #     # Get termination flags
    #     terminated = step_result["next", "terminated"]
    #     truncated = step_result["next", "truncated"]

    #     # Return results in TensorDict format with next_ prefix
    #     return TensorDict(
    #         {
    #             "next": {
    #                 "observation": obs,
    #                 "reward": reward,
    #                 "action": action,
    #                 "done": terminated or truncated,
    #                 "terminated": terminated,
    #                 "truncated": truncated,
    #             }
    #         },
    #         [],
    #     )


# %%
rollout = env.rollout(max_steps=10)
for action, reward in zip(rollout["action"], rollout["next", "reward"]):
    print(f"Action: {action}, Reward: {reward}")
# %%
# Create custom environment wrapper
MyCustomEnv(env).rollout(max_steps=10)
custom_env = MyCustomEnv(env)
custom_env.action_space

# %%

print("data from rollout:", simple_rollout(100))
# Verify environment specifications
check_env_specs(env)
dir(MyCustomEnv(env))
dir((env))
print("observation_spec:", env.observation_spec)
print("state_spec:", env.state_spec)
print("reward_spec:", env.reward_spec)
# %%
# rollout = env.rollout(max_steps=300)
custom_rollout = custom_env.rollout(max_steps=300)
# %%
rollout["next", "observation"]
custom_rollout["next", "observation"]
rollout["next", "action"]
custom_rollout["next", "action"]
rollout["next", "reward"]
custom_rollout["next", "reward"]
# %%
# Run multiple rollouts with random actions
n_rollouts = 2
max_steps = 3000

rollouts = []
# %%
# Create a policy that always returns action 1
module = torch.nn.LazyLinear(out_features=env.action_spec.shape[-1])
policy = TensorDictModule(
    module,
    in_keys=["observation"],
    out_keys=["action"],
)
# %%
rollout = custom_env.rollout(max_steps=max_steps, policy=policy)
len(rollout["observation"])
len(rollout["next", "observation"])

df["close"][0:3000].plot()
df["close"][0:3000].diff()

plt.figure(figsize=(10, 6))
plt.plot(rollout["next"]["reward"].detach().numpy().cumsum(), label="Reward per step")
plt.title("Rewards per Step")
plt.xlabel("Steps")
plt.ylabel("Reward")
plt.legend()
plt.show()
# %%
# Plot cumulative rewards
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
for i, rollout in enumerate(rollouts):
    plt.plot(np.cumsum(rollout["next"]["reward"].numpy()), label=f"Run {i+1}")
plt.title("Cumulative Rewards Comparison")
plt.xlabel("Steps")
plt.ylabel("Cumulative Reward")
plt.show()

# Check environment specs
check_env_specs(env)
# Check environment specs
check_env_specs(env)
