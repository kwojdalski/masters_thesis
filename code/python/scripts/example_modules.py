# %%
import logging
import os
import sys

import gym_trading_env  # noqa: F401
import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch.distributions import Normal
from torchrl.envs import GymWrapper
from torchrl.modules import MLP, Actor, ProbabilisticActor

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
df = pd.read_pickle("./data/raw/binance/binance-BTCUSDT-1h.pkl")


# %%
def reward_function(history):
    return np.log(
        history["portfolio_valuation", -1] / history["portfolio_valuation", -2]
    )


# %%
# Features must have features in their name, crazy ikr
df["feature"] = df["close"] / df["close"].shift(1)
df["feature_pct_change"] = df["close"].pct_change()
df["feature_high"] = df["high"] / df["close"] - 1
df["feature_low"] = df["low"]


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

# %%
# GymWrapper is a wrapper that allows the environment to be used with torchrl
env = GymWrapper(base_env)

# %%
n_obs = env.observation_spec["observation"].shape[-1]
n_act = env.action_spec.shape[-1]

module = torch.nn.LazyLinear(out_features=n_act)
policy = TensorDictModule(
    module,
    in_keys=["observation"],  # Market data observations
    out_keys=["action"],  # Trading actions
)
# %%
###################################
# Execute trading policy with lazy module that automatically determines observation space shape
rollout = env.rollout(max_steps=10, policy=policy)
print(rollout)

###################################
# Specialized wrappers for trading
# --------------------

policy = Actor(module)
rollout = env.rollout(max_steps=10, policy=policy)
print(rollout)

###################################
# Trading-specific networks
# --------

module = MLP(
    out_features=n_act,
    num_cells=[64, 128],  # Larger network for complex market patterns
    activation_class=torch.nn.Tanh,
)
policy = Actor(module)
rollout = env.rollout(max_steps=10, policy=policy)

###################################
# Probabilistic trading policies
# ----------------------


# MLP for market data processing
backbone = MLP(in_features=n_obs, out_features=2)
extractor = NormalParamExtractor()
module = torch.nn.Sequential(backbone, extractor)
td_module = TensorDictModule(module, in_keys=["observation"], out_keys=["loc", "scale"])
policy = ProbabilisticActor(
    td_module,
    in_keys=["loc", "scale"],
    out_keys=["action"],
    distribution_class=Normal,
    return_log_prob=True,
)

rollout = env.rollout(max_steps=10, policy=policy)
print(rollout)

###################################
# Exploration control for trading
from torchrl.envs.utils import ExplorationType, set_exploration_type

with set_exploration_type(ExplorationType.DETERMINISTIC):
    # Conservative trading using mean values
    rollout = env.rollout(max_steps=10, policy=policy)
with set_exploration_type(ExplorationType.RANDOM):
    # More exploratory trading
    rollout = env.rollout(max_steps=10, policy=policy)

###################################
# Trading exploration strategies
from tensordict.nn import TensorDictSequential
from torchrl.modules import EGreedyModule

policy = Actor(
    MLP(env.observation_spec.shape[-1], env.action_spec.shape[-1], num_cells=[64, 128])
)

# Epsilon-greedy exploration for trading
exploration_module = EGreedyModule(
    spec=env.action_spec,
    annealing_num_steps=1000,  # Gradual reduction in exploration
    eps_init=0.5,
)

exploration_policy = TensorDictSequential(policy, exploration_module)

with set_exploration_type(ExplorationType.DETERMINISTIC):
    # Conservative trading
    rollout = env.rollout(max_steps=10, policy=exploration_policy)
with set_exploration_type(ExplorationType.RANDOM):
    # More aggressive trading
    rollout = env.rollout(max_steps=10, policy=exploration_policy)

###################################
# Q-Value based trading
env = GymEnv("Trading-v0")
print(env.action_spec)

# Value network for trading decisions
num_actions = env.action_spec.shape[-1]
value_net = TensorDictModule(
    MLP(out_features=num_actions, num_cells=[64, 128]),
    in_keys=["observation"],
    out_keys=["action_value"],
)

from torchrl.modules import QValueModule

policy = TensorDictSequential(
    value_net,
    QValueModule(spec=env.action_spec),
)

rollout = env.rollout(max_steps=3, policy=policy)
print(rollout)

# Exploration policy for Q-learning
policy_explore = TensorDictSequential(policy, EGreedyModule(env.action_spec))

with set_exploration_type(ExplorationType.RANDOM):
    rollout_explore = env.rollout(max_steps=3, policy=policy_explore)

###################################
# Next steps for trading systems:
#
# - Implement compound distributions for multi-asset trading
# - Use RNNs for time series analysis
# - Explore transformer-based architectures for market prediction

# %%
