# %%
import datetime
import logging
import os
import sys
import time

import gym_trading_env  # noqa: F401
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from gym_trading_env.downloader import download
from gym_trading_env.renderer import Renderer
from tensordict.nn import TensorDictModule
from torch.optim import Adam
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.envs import GymWrapper, TransformedEnv
from torchrl.envs.transforms import StepCounter
from torchrl.envs.utils import check_env_specs
from torchrl.modules import MLP, Actor, ValueOperator
from torchrl.objectives import DDPGLoss, SoftUpdate

# %%
# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/trading_env_debug.log"),
        logging.StreamHandler(sys.stdout),
    ],
)

# Create a logger for this module
logger = logging.getLogger("gym_trading_env")
logger.info("Initializing trading environment debugging")

# %%
# Getting data
# Download BTC/USDT historical data from Binance and stores it to directory ./data/binance-BTCUSDT-1h.pkl
download_data = False
if download_data:
    download(
        exchange_names=["binance"],
        symbols=["BTC/USDT"],
        timeframe="1s",
        dir="data",
        since=datetime.datetime(year=2025, month=4, day=27),
    )
df = pd.read_pickle("./data/raw/binance/binance-BTCUSDT-1h.pkl")


# %%
def reward_function(history):
    return np.log(
        history["portfolio_valuation", -1] / history["portfolio_valuation", -2]
    )


# %%
# Features must have features in their name, crazy ikr
# Normalize features using z-score normalization
# Variables must be scaled otherwise the model will not learn properly
df["feature"] = (df["close"] / df["close"].shift(1) - 1).fillna(0)
df["feature"] = (df["feature"] - df["feature"].mean()) / df["feature"].std()

df["feature_pct_chng"] = df["close"].pct_change().fillna(0)
df["feature_pct_chng"] = (df["feature_pct_chng"] - df["feature_pct_chng"].mean()) / df[
    "feature_pct_chng"
].std()

df["feature_high"] = (df["high"] / df["close"] - 1).fillna(0)
df["feature_high"] = (df["feature_high"] - df["feature_high"].mean()) / df[
    "feature_high"
].std()

df["feature_low"] = (df["low"] / df["close"] - 1).fillna(0)
df["feature_low"] = (df["feature_low"] - df["feature_low"].mean()) / df[
    "feature_low"
].std()


# Make the environment out of the data
base_env = gym.make(
    "TradingEnv",
    name="BTCUSD",
    df=df,  # Your dataset with your custom features
    positions=[-1, 0, 1],  # -1 (=SHORT), 0(=OUT), +1 (=LONG)
    trading_fees=0.01 / 100,  # 0.01% per stock buy / sell (Binance fees)
    borrow_interest_rate=0.0003 / 100,  # 0.0003% per timestep (one timestep = 1h here)
    reward_function=reward_function,
)
base_env.device = "cuda" if torch.cuda.is_available() else "cpu"

# %%
# GymWrapper is a wrapper that allows the environment to be used with torchrl
env = GymWrapper(base_env)
env = TransformedEnv(env, StepCounter())
n_obs = env.observation_spec["observation"].shape[-1]
n_act = env.action_spec.shape[-1]


# %%
# Create a policy network
class ActionSelector(torch.nn.Module):
    """
    Module that selects actions from logits and formats them appropriately.
    Converts softmax outputs to one-hot encoded actions for discrete action space.
    """

    def forward(self, x):
        # Convert logits to one-hot encoding
        # Shape: [batch_size, 3] -> [batch_size, 3]
        one_hot = torch.nn.functional.one_hot(
            torch.argmax(x, dim=-1), num_classes=3
        ).to(x.dtype)
        return one_hot


# %%
module = torch.nn.Sequential(
    torch.nn.LazyLinear(out_features=3), torch.nn.Softmax(dim=-1), ActionSelector()
)
# %%
# policy is a function that takes an observation (state) and returns an action
policy = TensorDictModule(
    module,
    in_keys=["observation"],
    out_keys=["action"],
)
# %%
# The rollout method simulates the environment for a specified number of steps
# using the provided policy. It collects and returns a TensorDict containing:
#   - observations from each state
#   - actions taken by the policy
#   - rewards received
#   - next states
#   - termination signals
#   - other environment-specific information

rollout = env.rollout(max_steps=50, policy=policy)
for action, reward in zip(rollout["action"], rollout["next", "reward"]):
    print(f"Action: {action}, Reward: {reward}")

# %%
render = False
if render:
    env.save_for_render(dir="render_logs")
    renderer = Renderer(render_logs_dir="render_logs")


# %%
# Example with DDPG
# DDPG is a policy gradient method that uses a policy network and a value network
# %%
# For the previous example, we didn't have a policy network or a value network
# Now we want to eventually OPTIMIZE something, not just run the environment

# n_obs was defined earlier
# out_features is the number of actions, so in our case it's 3 (-1, 0, 1)
actor = Actor(MLP(in_features=n_obs, out_features=n_act))
# %%
# Value network is used to estimate the value of the current state-action pair
# It takes the state and action as input and outputs a single value
value_net = ValueOperator(
    MLP(
        in_features=n_obs + n_act,
        out_features=1,
        num_cells=[4, 4],
    ),
    in_keys=["observation", "action"],
)
# %%
# Create target network updater to fix the warning
ddpg_loss = DDPGLoss(
    actor_network=actor,
    value_network=value_net,
)

# %%
# Wrap the loss with the target network updater
updater = SoftUpdate(ddpg_loss, tau=0.05)

# %%
# Rollout the policy network
rollout = env.rollout(max_steps=1000, policy=actor)

loss_vals = ddpg_loss(rollout)
optim = Adam(ddpg_loss.parameters())

# %%
collector = SyncDataCollector(env, actor, frames_per_batch=200, total_frames=-1)
# %%
# Buffer size is 100k which is the max number of samples that can be stored
rb = ReplayBuffer(storage=LazyTensorStorage(100_000))

# %%
total_count = 0
total_episodes = 0
t0 = time.time()

init_rand_steps = 100
frames_per_batch = 100
optim_steps = 10000

# %%
for i, data in enumerate(collector):
    # Write data in replay buffer
    rb.extend(data)

    max_length = rb[:]["next", "step_count"].max()
    if len(rb) > init_rand_steps:
        # Optim loop (we do several optim steps
        # per batch collected for efficiency)
        for j, _ in enumerate(range(optim_steps)):
            if j % 10000 == 0:
                print(f"Optim step {j}")
            sample = rb.sample(128)
            loss_vals = ddpg_loss(sample)
            loss_vals["loss_value"].backward()
            optim.step()
            optim.zero_grad()
            # Update exploration factor
            # exploration_module.step(data.numel())
            # Update target params
            # Store previous parameters to check for significant changes
            prev_params = {
                name: param.clone().detach()
                for name, param in policy.named_parameters()
            }

            # Check if parameters changed significantly
            for name, param in policy.named_parameters():
                if torch.any(torch.abs(param - prev_params[name]) > 0.0001):
                    print(f"{name} changed")
                    for i in policy.parameters():
                        print(f"{i}")

            # time.sleep(0.5)  # Delay execution for 1 second
            updater.step()

            # Track parameter changes
            if j % 1000 == 0:
                logger.info(f"Max num steps: {max_length}, rb length {len(rb)}")
            total_count += data.numel()
            total_episodes += data["next", "done"].sum()
    if max_length > 200:
        break

t1 = time.time()

logger.info(
    f"solved after {total_count} steps, {total_episodes} episodes and in {t1 - t0}s."
)
# Print parameters of value network


# %%
env_to_render = env.rollout(max_steps=10000, policy=policy)
env_to_render["action"]
# %%


plt.figure(figsize=(10, 6))
plt.plot(
    env_to_render["next", "reward"].detach().numpy().cumsum(), label="Cumulative Reward"
)
plt.title("Cumulative Rewards Over Time")
plt.xlabel("Steps")
plt.ylabel("Cumulative Reward")
plt.legend()
plt.grid(True)
plt.show()

check_env_specs(env)

print("Shape of the rollout TensorDict:", rollout.batch_size)
rollout

# %%
render = False
if render:
    env_to_render.save_for_render(dir="render_logs")
    renderer = Renderer(render_logs_dir="render_logs")
    renderer.run()
    renderer.run()
