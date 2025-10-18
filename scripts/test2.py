# %%
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
# sys.path.insert(0, PROJECT_ROOT)
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
import torch.nn as nn
from gym_trading_env.downloader import download
from tensordict import TensorDict
from tensordict.nn import InteractionType, TensorDictModule, set_composite_lp_aggregate
from torch import distributions as d
from torch.optim import Adam
from torchrl.envs import GymWrapper, TransformedEnv
from torchrl.envs.transforms import StepCounter
from torchrl.modules import ProbabilisticActor

# %%
# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)
set_composite_lp_aggregate(True).set()
# Configure logging
# Disable matplotlib logging
logging.getLogger("matplotlib").setLevel(logging.WARNING)

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
# Download BTC/USDT historical data from Binance and stores it to directory ./data/binance-BTCUSDT-1h.parquet
download_data = False
if download_data:
    download(
        exchange_names=["binance"],
        symbols=["BTC/USDT"],
        timeframe="1s",
        dir="data",
        since=datetime.datetime(year=2025, month=4, day=27, tzinfo=datetime.UTC),
    )
# %%
df = pd.read_parquet("./data/raw/binance/binance-BTCUSDT-1h.parquet")


# %%
def reward_function(history):
    # Calculate returns
    returns = np.log(
        history["portfolio_valuation", -1] / history["portfolio_valuation", -2]
    )

    return returns


# %%
# Features must have features in their name, crazy ikr
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
# volatility realized
# Count NaN values in each column
df = df.dropna()
# %%
# Make the environment out of the data
size = 500
base_env = gym.make(
    "TradingEnv",
    name="BTCUSD",
    df=df[:size],  # Your dataset with your custom features
    positions=[-1, 0, 1],  # -1 (=SHORT), 0(=OUT), +1 (=LONG)
    trading_fees=0.15 / 100,  # 0.01 / 100,  # 0.01% per stock buy / sell (Binance fees)
    borrow_interest_rate=0,  # 0.0003 / 100,  # 0.0003% per timestep (one timestep = 1h here)
    reward_function=reward_function,
)

# %%
# GymWrapper is a wrapper that allows the environment to be used with torchrl
env = GymWrapper(base_env)
env = TransformedEnv(env, StepCounter())

n_obs = env.observation_spec["observation"].shape[-1]
n_act = env.action_spec.shape[-1]


# %%
# Step 1: Define a neural network for discrete action selection
# - Takes input observations and outputs action probabilities
# - Uses multiple layers with non-linear activations
class DiscreteNet(nn.Module):
    def __init__(self, input_dim, n_actions):
        super().__init__()
        # Create network layers
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions),
            nn.Softmax(dim=-1),
        )

        # Initialize weights using Xavier/Glorot initialization
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        # Ensure input is float32 for better performance
        x = x.to(torch.float32)
        probs = self.network(x)
        return {"probs": probs}

    def check_gradients(self):
        """Helper method to check if network parameters have gradients"""
        for name, param in self.named_parameters():
            requires_grad = param.requires_grad
            has_grad = param.grad is not None
            logger.info(f"Layer {name}:")
            logger.info(f"  Requires grad: {requires_grad}")
            logger.info(f"  Has gradient: {has_grad}")
            if has_grad:
                logger.info(f"  Gradient norm: {param.grad.norm()}")


# %%
# Step 2: Create TensorDictModule wrapper
# - Wraps the neural network for use with TensorDict
# - Specifies input/output key mappings
module = TensorDictModule(
    module=DiscreteNet(n_obs, n_act),
    in_keys=["observation"],  # Takes observation as input
    out_keys=["probs"],  # Outputs action probabilities
)

# Verify module is on correct device and in training mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
module.to(device)
module.train()

# %%
actor = ProbabilisticActor(
    module=module,
    distribution_class=d.OneHotCategorical,
    in_keys=["probs"],  # logits for Categorical
    out_keys=["action"],
    spec=env.action_spec,
    safe=True,  # Enable safety checks
    default_interaction_type=InteractionType.RANDOM,  # Return one-hot encoded actions
)

# Create optimizer for actor network
actor_optimizer = Adam(actor.parameters(), lr=1e-3)


# Sample batch of observations for testing gradients
test_batch = TensorDict(
    {"observation": torch.randn(32, n_obs)},  # Batch of 32 random observations
    batch_size=[32],
)
test_batch = test_batch.to(device)

# Forward pass through actor
with torch.enable_grad():
    output = actor(test_batch)

    # Calculate "loss" (using mean of probabilities as dummy objective)
    # In real training this would be your actual loss function
    loss = output["probs"].mean()

    # Backward pass
    actor_optimizer.zero_grad()
    loss.backward()

    # Log gradient information
    logger.info("Checking actor gradients:")
    actor.module.module.check_gradients()
