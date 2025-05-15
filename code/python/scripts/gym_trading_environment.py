# %%
import datetime
import logging
import os
import sys

import gym_trading_env  # noqa: F401
import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from gym_trading_env.downloader import download
from gym_trading_env.renderer import Renderer
from tensordict.nn import TensorDictModule
from torch.optim import Adam
from torchrl.envs import GymWrapper
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
df = pd.read_pickle("./data/raw/binance/binance-BTCUSDT-1h.pkl")

# Features must have features in their name, crazy ikr
df["feature"] = df["close"] / df["close"].shift(1)

df["feature_pct_change"] = df["close"].pct_change()
df["feature_high"] = df["high"] / df["close"] - 1
df["feature_low"] = df["low"]


def dynamic_feature_last_position_taken(history):
    return history["position", -1]


# Make the environment out of the data
base_env = gym.make(
    "TradingEnv",
    name="BTCUSD",
    df=df,  # Your dataset with your custom features
    positions=[-1, 0, 1],  # -1 (=SHORT), 0(=OUT), +1 (=LONG)
    trading_fees=0.01 / 100,  # 0.01% per stock buy / sell (Binance fees)
    borrow_interest_rate=0.0003 / 100,  # 0.0003% per timestep (one timestep = 1h here)
    reward_function=reward_function,
    dynamic_feature_functions=[dynamic_feature_last_position_taken],
)
base_env.device = "cuda" if torch.cuda.is_available() else "cpu"

# Adding metrics
base_env.add_metric(
    "Position Changes", lambda history: np.sum(np.diff(history["position"]) != 0)
)
base_env.add_metric("Episode Length", lambda history: len(history["position"]))
# base_env.add_metric("Profit", lambda history: history["profit"][-1])

observation, info = base_env.reset()

# %%
# GymWrapper is a wrapper that allows the environment to be used with torchrl
env = GymWrapper(base_env)
env.observation_spec["observation"].shape[-1]
observation, info = base_env.reset()

n_obs = env.observation_spec["observation"].shape[-1]
n_act = env.action_spec.shape[-1]


# %%
# Create a policy network
class ActionSelector(torch.nn.Module):
    def forward(self, x):
        return torch.argmax(x, dim=-1) - 1


module = torch.nn.Sequential(
    torch.nn.LazyLinear(out_features=3), torch.nn.Softmax(dim=-1), ActionSelector()
)
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

rollout = env.rollout(max_steps=5000, policy=policy)
rollout["next"]["observation"]
rollout["next"]["reward"]
# Print the entire rollout to see its structure
print("Rollout structure:", rollout)

# %%
# Check specific values in the rollout
print("\nKeys in rollout:", rollout.keys())
print("\nShape of rollout:", rollout.shape)

# %%
# Access specific tensors
# Display observation sample if available
print("\nObservation sample:", rollout.get("observation", [None])[0])
print("\nAction sample:", rollout.get("action", [None])[0])
print("\nNext state keys:", rollout.get("next", {}).keys())

# it's always -1 bc there's nothing we optimized
# %%
print("\nReward statistics:")
print("  Mean:", rollout["next"]["reward"].mean().item())

# %%
# Looks like the action consists of 0, i.e.
# %%
env.save_for_render(dir="render_logs")

renderer = Renderer(render_logs_dir="render_logs")


# %%
# Example with DDPG
# %%
n_obs = env.observation_spec["observation"].shape[-1]
actor = Actor(MLP(in_features=n_obs, out_features=env.action_spec.shape[-1]))
# %%
value_net = ValueOperator(
    MLP(
        in_features=n_obs + env.action_spec.shape[-1],
        out_features=1,
        num_cells=[32, 32],
    ),
    in_keys=["observation", "action"],
)

# Create target network updater to fix the warning
ddpg_loss = DDPGLoss(
    actor_network=actor,
    value_network=value_net,
)
#

SoftUpdate(ddpg_loss, tau=0.005)
# Initialize DDPG loss with the target network updater

rollout = env.rollout(max_steps=100, policy=actor)

loss_vals = ddpg_loss(rollout)
print(loss_vals)

# %%
total_loss = 0
for key, val in loss_vals.items():
    if key.startswith("loss_"):
        total_loss += val
# %%
optim = Adam(ddpg_loss.parameters())
total_loss.backward()
optim.step()
optim.zero_grad()
updater.step()
# %%
from collections import defaultdict

from tqdm import tqdm

logs = defaultdict(list)
pbar = tqdm(total=total_frames)
eval_str = ""

# We iterate over the collector until it reaches the total number of frames it was
# designed to collect:
for i, tensordict_data in enumerate(collector):
    # we now have a batch of data to work with. Let's learn something from it.
    for _ in range(num_epochs):
        # We'll need an "advantage" signal to make PPO work.
        # We re-compute it at each epoch as its value depends on the value
        # network which is updated in the inner loop.
        advantage_module(tensordict_data)
        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view.cpu())
        for _ in range(frames_per_batch // sub_batch_size):
            subdata = replay_buffer.sample(sub_batch_size)
            loss_vals = loss_module(subdata.to(device))
            loss_value = (
                loss_vals["loss_objective"]
                + loss_vals["loss_critic"]
                + loss_vals["loss_entropy"]
            )

            # Optimization: backward, grad clipping and optimization step
            loss_value.backward()
            # this is not strictly mandatory but it's good practice to keep
            # your gradient norm bounded
            torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
            optim.step()
            optim.zero_grad()

    logs["reward"].append(tensordict_data["next", "reward"].mean().item())
    pbar.update(tensordict_data.numel())
    cum_reward_str = (
        f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
    )
    logs["step_count"].append(tensordict_data["step_count"].max().item())
    stepcount_str = f"step count (max): {logs['step_count'][-1]}"
    logs["lr"].append(optim.param_groups[0]["lr"])
    lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"
    if i % 10 == 0:
        # We evaluate the policy once every 10 batches of data.
        # Evaluation is rather simple: execute the policy without exploration
        # (take the expected value of the action distribution) for a given
        # number of steps (1000, which is our ``env`` horizon).
        # The ``rollout`` method of the ``env`` can take a policy as argument:
        # it will then execute this policy at each step.
        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
            # execute a rollout with the trained policy
            eval_rollout = env.rollout(1000, policy_module)
            logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
            logs["eval reward (sum)"].append(
                eval_rollout["next", "reward"].sum().item()
            )
            logs["eval step_count"].append(eval_rollout["step_count"].max().item())
            eval_str = (
                f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
                f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
                f"eval step-count: {logs['eval step_count'][-1]}"
            )
            del eval_rollout
    pbar.set_description(", ".join([eval_str, cum_reward_str, stepcount_str, lr_str]))

    # We're also using a learning rate scheduler. Like the gradient clipping,
    # this is a nice-to-have but nothing necessary for PPO to work.
    scheduler.step()
    scheduler.step()
    scheduler.step()
