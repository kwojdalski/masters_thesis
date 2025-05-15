# %%
# Using gym-anytrading with TorchRL


import gym
import gym_anytrading  # noqa: F401
import gymnasium as gym
import torch
from gym_anytrading.datasets import FOREX_EURUSD_1H_ASK
from torchrl.data import BoundedTensorSpec
from torchrl.envs import GymEnv

# %%
env = gym.make(
    "forex-v0",
    df=FOREX_EURUSD_1H_ASK,
    window_size=10,
    frame_bound=(10, 300),
    unit_side="right",
)


# env.set_seed(0)

# %%
# Load sample data or your own data
# For this example, we'll use the default data from gym-anytrading
print(f"Action Space: {env.action_space}")
print(f"Observation Space: {env.observation_space}")

# Create a TorchRL compatible environment
# We need to wrap the gym-anytrading environment
trading_env = GymEnv("forex-v0")

# Define the observation and action specs

# %%
action_spec = BoundedTensorSpec(
    shape=(1,),
    dtype=torch.int64,
    low=torch.zeros(1, dtype=torch.int64),
    high=torch.ones(1, dtype=torch.int64) * (trading_env.action_spec.shape[0] - 1),
    device="cpu",
)
# %%
# Create a new policy network for trading

print("env information:")
print("> shape:", env.unwrapped.shape)
print("> df.shape:", env.unwrapped.df.shape)
print("> prices.shape:", env.unwrapped.prices.shape)
print("> signal_features.shape:", env.unwrapped.signal_features.shape)
print("> max_possible_profit:", env.unwrapped.max_possible_profit())

print()
print("custom_env information:")
print("> shape:", env.unwrapped.shape)
print("> df.shape:", env.unwrapped.df.shape)
print("> prices.shape:", env.unwrapped.prices.shape)
print("> signal_features.shape:", env.unwrapped.signal_features.shape)
print("> max_possible_profit:", env.unwrapped.max_possible_profit())
# %%
env.reset()
env.render()

# %%
# ##
# Implementing a DQN algorithm for the trading environment
# Get attributes from the environment
td = env.rand_step()
env.action_space.sample()
env.step(1)
print("Observation shape:", td["observation"].shape)
print("Reward:", td["reward"].item())
print("Done:", td["done"].item())
print("Next observation shape:", td["next_observation"].shape)

# %%
# Set seeds for reproducibility
observation = env.reset()
while True:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    # env.render()
    if done:
        print("info:", info)
        break

# %%
# %%
