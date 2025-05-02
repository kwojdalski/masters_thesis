# %%
import datetime

# Set up logging for debugging
import logging
import os
import sys

import gymnasium as gym
import pandas as pd
import torch
from gym_trading_env.downloader import download
from tensordict.nn import TensorDictModule
from torchrl.envs import Compose, GymEnv, ObservationNorm, StepCounter, TransformedEnv
from torchrl.envs.libs.gym import GymEnv

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
# %%
# Import your fresh data
df = pd.read_pickle("./data/raw/binance/binance-BTCUSDT-1h.pkl")


# %%
base_env = gym.make(
    "TradingEnv",
    name="BTCUSD",
    df=df,  # Your dataset with your custom features
    positions=[-1, 0, 1],  # -1 (=SHORT), 0(=OUT), +1 (=LONG)
    trading_fees=0.01 / 100,  # 0.01% per stock buy / sell (Binance fees)
    borrow_interest_rate=0.0003 / 100,  # 0.0003% per timestep (one timestep = 1h here)
)


gym_env = GymEnv(env_name="TradingEnv", df=df)


# %%

# Run an episode until it ends :
done, truncated = False, False
observation, info = env.reset()
while not done and not truncated:
    # Pick a position by its index in your position list (=[-1, 0, 1])....usually something like : position_index = your_policy(observation)
    position_index = (
        env.action_space.sample()
    )  # At every timestep, pick a random position index from your position list (=[-1, 0, 1])

    observation, reward, done, truncated, info = env.step(position_index)
# %%
base_env.add_metric(
    "Position Changes", lambda history: np.sum(np.diff(history["position"]) != 0)
)
base_env.add_metric("Episode Length", lambda history: len(history["position"]))
base_env
# %%
base_env.device = "cuda" if torch.cuda.is_available() else "cpu"
env = TransformedEnv(
    base_env,
    Compose(
        # normalize observations
        ObservationNorm(in_keys=["observation"]),
        StepCounter(),
    ),
)
# To add a new attribute to base_env, you should use dot notation instead of dictionary-style access
# You can access it later using base_env.device
env

# %%
frames_per_batch = 1000
# For a complete training, bring the number of frames up to 1M
total_frames = 10_000
sub_batch_size = 64  # cardinality of the sub-samples gathered from the current data in the inner loop
num_epochs = 10  # optimization steps per batch of data collected
clip_epsilon = (
    0.2  # clip value for PPO loss: see the equation in the intro for more context.
)
gamma = 0.99
lmbda = 0.95
entropy_eps = 1e-4
# %%
# %%


import gymnasium as gym
from torchrl.envs.libs.gym import GymWrapper

env = GymWrapper(base_env)


# %%
module = torch.nn.LazyLinear(out_features=env.action_spec.shape[-1])
policy = TensorDictModule(
    module,
    in_keys=["observation"],
    out_keys=["action"],
)
rollout = env.rollout(max_steps=10, policy=policy)
# Print the entire rollout to see its structure
print("Rollout structure:", rollout)

# Check specific values in the rollout
print("\nKeys in rollout:", rollout.keys())
print("\nShape of rollout:", rollout.shape)

# Access specific tensors
if "observation" in rollout:
    print("\nObservation sample:", rollout["observation"][0])
if "action" in rollout:
    print("\nAction sample:", rollout["action"][0])
if "next" in rollout:
    print("\nNext state keys:", rollout["next"].keys())

# Check statistics
if "reward" in rollout:
    print("\nReward statistics:")
    print("  Mean:", rollout["reward"].mean().item())
    print("  Sum:", rollout["reward"].sum().item())
    print("  Min:", rollout["reward"].min().item())
    print("  Max:", rollout["reward"].max().item())

# %%
