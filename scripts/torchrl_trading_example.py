# %%
import datetime
import logging
import time
from collections import defaultdict

import gym_trading_env  # noqa: F401
import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from gym_trading_env.downloader import download
from plotnine import aes, facet_wrap, geom_line, ggplot
from tensordict.nn import InteractionType, TensorDictModule, set_composite_lp_aggregate
from torch import distributions as d
from torch.optim import Adam
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.envs import GymWrapper, TransformedEnv
from torchrl.envs.transforms import StepCounter
from torchrl.envs.utils import set_exploration_type
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator
from torchrl.objectives import DDPGLoss, SoftUpdate
from utils import compare_rollouts

from logger import (
    LogContext,
    get_logger,
    log_dataframe_info,
    log_performance_metrics,
    log_processing_step,
    setup_logging,
)

# %%
# Setup logging using custom logger
setup_logging(
    log_level="INFO",
    log_dir="logs/torchrl_trading",
    console_output=True,
    file_output=True,
)

# Disable matplotlib logging
logging.getLogger("matplotlib").setLevel(logging.WARNING)

# Create a logger for this module
logger = get_logger("torchrl_trading_example")
logger.info("Initializing TorchRL trading environment example")

# Set TensorDict configuration
set_composite_lp_aggregate(True).set()

# %%
# Getting data
log_processing_step(logger, "Data Loading", "Loading BTC/USDT historical data")

download_data = False
if download_data:
    logger.info("Downloading fresh data from Binance")
    download(
        exchange_names=["binance"],
        symbols=["BTC/USDT"],
        timeframe="1s",
        dir="data",
        since=datetime.datetime(year=2025, month=4, day=27, tzinfo=datetime.UTC),
    )

df = pd.read_parquet("./data/raw/binance/binance-BTCUSDT-1h.parquet")
log_dataframe_info(logger, df, "BTC/USDT Market Data", level="INFO")


# %%
def reward_function(history):
    # Calculate returns
    returns = np.log(
        history["portfolio_valuation", -1] / history["portfolio_valuation", -2]
    )

    return returns


# %%
# Feature Engineering
log_processing_step(
    logger,
    "Feature Engineering",
    "Creating normalized features from OHLC data",
    extra_data={"features": ["return", "pct_change", "high_ratio", "low_ratio"]},
)

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

# Drop NaN values
rows_before = len(df)
df = df.dropna()
rows_after = len(df)
logger.info(f"Dropped {rows_before - rows_after} rows with NaN values")
log_dataframe_info(logger, df, "Processed Features DataFrame", level="INFO")
# %%
# Environment Setup
log_processing_step(
    logger,
    "Environment Setup",
    "Creating trading environment",
    extra_data={
        "size": 1000,
        "positions": [-1, 0, 1],
        "trading_fees": 0.0,
        "borrow_interest_rate": 0.0,
    },
)

size = 1000
base_env = gym.make(
    "TradingEnv",
    name="BTCUSD",
    df=df[:size],  # Your dataset with your custom features
    positions=[-1, 0, 1],  # -1 (=SHORT), 0(=OUT), +1 (=LONG)
    trading_fees=0 / 100,  # 0.01 / 100,  # 0.01% per stock buy / sell (Binance fees)
    borrow_interest_rate=0,  # 0.0003 / 100,  # 0.0003% per timestep (one timestep = 1h here)
    reward_function=reward_function,
)

# %%
# GymWrapper is a wrapper that allows the environment to be used with torchrl
logger.info("Wrapping environment with TorchRL GymWrapper")
env = GymWrapper(base_env)
env = TransformedEnv(env, StepCounter())

n_obs = env.observation_spec["observation"].shape[-1]
n_act = env.action_spec.shape[-1]
logger.info(f"Environment specs - Observations: {n_obs}, Actions: {n_act}")


# %%
# Step 1: Define a neural network for discrete action selection
# - Takes input observations and outputs action probabilities
# - Uses a single linear layer followed by softmax activation
class DiscreteNet(nn.Module):
    def __init__(self, input_dim, n_actions):
        super().__init__()
        # Add multiple layers with non-linear activations
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions),
            nn.Softmax(dim=-1),
        )
        self.linear = nn.Linear(input_dim, n_actions)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        probs = self.network(x)
        return {"probs": probs}


# %%
# Model Setup
log_processing_step(
    logger,
    "Model Initialization",
    "Creating actor and value networks",
    extra_data={
        "actor_layers": [64, 32],
        "value_layers": [64, 32, 16],
        "n_observations": n_obs,
        "n_actions": n_act,
    },
)

# Step 2: Create TensorDictModule wrapper
# - Wraps the neural network for use with TensorDict
# - Specifies input/output key mappings
module = TensorDictModule(
    DiscreteNet(n_obs, n_act),
    in_keys=["observation"],  # Takes observation as input
    out_keys=["probs"],  # Outputs action probabilities
)

# %%
actor = ProbabilisticActor(
    module=module,
    distribution_class=d.OneHotCategorical,
    in_keys=["probs"],  # logits for Categorical
    # out_keys=["action"],
    spec=env.action_spec,
    safe=True,  # Enable safety checks
    # either InteractionType.MODE, InteractionType.RANDOM,
    # InteractionType.MEDIAN, InteractionType.DETERMINISTIC
    default_interaction_type=InteractionType.RANDOM,  # Return one-hot encoded actions
)
logger.info("Actor network initialized")

# %%
# Value network
value_net = ValueOperator(
    MLP(
        in_features=n_obs + n_act,
        out_features=1,
        num_cells=[64, 32, 16],  # Larger network
    ),
    in_keys=["observation", "probs"],
    out_keys=["state_action_value"],
)
logger.info("Value network initialized")

# %%
# Loss and target network updater
# Initialize DDPG loss with actor and value networks
# Uses L2 loss for value function approximation

ddpg_loss = DDPGLoss(
    actor_network=actor,  # Policy network for action selection
    value_network=value_net,  # Value network for state-value estimation
    loss_function="l2",  # L2 loss for stable value function learning
)
updater = SoftUpdate(ddpg_loss, tau=0.001)  # Slower target network updates


# %%
# Rollout the policy network
# Rollout simulates the environment for a specified number of steps using the policy
# It collects and returns a TensorDict containing:
#   - observations from each state
#   - actions taken by the policy
#   - rewards received
#   - next states
#   - termination signals
#   - other environment-specific information
rollout = env.rollout(max_steps=3000, policy=actor)
reward_plot, action_plot = compare_rollouts([rollout], n_obs=3000)
_ = reward_plot  # Display plot in notebook
_ = action_plot  # Display plot in notebook
# %%
loss_vals = ddpg_loss(rollout)
loss_vals["loss_value"].backward()
loss_vals["loss_actor"].backward()


# Adam optimizer with learning rate 1e-4 for gradient-based optimization
# of the DDPG loss parameters, combining benefits of RMSprop and momentum


optimizer_actor = Adam(ddpg_loss.actor_network_params.values(True, True), lr=1e-4)
optimizer_value = Adam(
    ddpg_loss.value_network_params.values(True, True), lr=1e-3, weight_decay=1e-2
)
optim = Adam(ddpg_loss.parameters(), lr=1e-4)

# %%
# Creates a synchronous data collector that:
# - Uses the trading environment (env)
# - Collects experience using the actor policy (a batch is a group of state-action-reward
# samples collected during training)
# - Collects 200 state-action-reward samples per batch of experience collection
# - Each frame represents one timestep of environment interaction
# - These frames contain the state, action taken, reward received, and next state
# - Runs indefinitely (total_frames=-1)
# %%
# Buffer size is 100k which is the max number of samples that can be stored
rb = ReplayBuffer(storage=LazyTensorStorage(100_000))

# %%
# Initialize counters for tracking training progress
total_count = 0  # Total number of environment steps taken
total_episodes = 0  # Total number of completed episodes
# Stopping conditions
max_training_steps = 5000000  # Maximum number of training steps

# Training hyperparameters
init_rand_steps = 50  # More exploration
frames_per_batch = 200  # Keep same
optim_steps = 50  # Much fewer optimization steps per batch
sample_size = 50


collector = SyncDataCollector(
    create_env_fn=lambda: env,
    policy=actor,
    frames_per_batch=frames_per_batch,
    total_frames=max_training_steps,  # Set maximum total frames
)

# %%
# Training Loop
log_processing_step(
    logger,
    "Training Start",
    "Beginning DDPG training loop",
    extra_data={
        "max_training_steps": max_training_steps,
        "frames_per_batch": frames_per_batch,
        "optim_steps": optim_steps,
        "sample_size": sample_size,
        "init_rand_steps": init_rand_steps,
    },
)

t0 = time.time()
# Initialize lists to store loss values
loss_value_history = []
loss_actor_history = []

logs = defaultdict(list)

with LogContext(logger, "Training Loop"):
    for i, data in enumerate(collector):
        # Add collected experience data to replay buffer for later training
        rb.extend(data)
        # Get the maximum number of steps taken in any episode in the replay buffer
        max_length = rb[:]["next", "step_count"].max()
        if len(rb) > init_rand_steps:
            # Optim loop
            for j, _ in enumerate(range(optim_steps)):
                if j % 10000 == 0:
                    logger.info(f"Optim step {j}")
                    for i in ddpg_loss.named_parameters():
                        logger.debug(f"{i}")

                sample = rb.sample(sample_size)
                loss_vals = ddpg_loss(sample)
                # optimize
                loss_vals["loss_actor"].backward()
                optimizer_actor.step()
                optimizer_actor.zero_grad()

                loss_vals["loss_value"].backward()
                optimizer_value.step()
                optimizer_value.zero_grad()

                # optim.step()
                # optim.zero_grad()
                updater.step()

                # Store loss values

                logs["loss_value"].append(loss_vals["loss_value"].item())
                logs["loss_actor"].append(loss_vals["loss_actor"].item())

                # Track parameter changes
                if j % 1000 == 0:
                    logger.info(f"Max num steps: {max_length}, rb length {len(rb)}")
                    # Track changes in loss values since last logging
                    if not hasattr(ddpg_loss, "prev_loss_value"):
                        ddpg_loss.prev_loss_value = loss_vals["loss_value"].item()
                        ddpg_loss.prev_loss_actor = loss_vals["loss_actor"].item()

                    curr_loss_value = loss_vals["loss_value"].item()
                    curr_loss_actor = loss_vals["loss_actor"].item()

                    logger.info(
                        f"Loss value: {curr_loss_value} "
                        f"(change: {curr_loss_value - ddpg_loss.prev_loss_value:+.4f})"
                    )
                    logger.info(
                        f"Loss actor: {curr_loss_actor} "
                        f"(change: {curr_loss_actor - ddpg_loss.prev_loss_actor:+.4f})"
                    )

                    ddpg_loss.prev_loss_value = curr_loss_value
                    ddpg_loss.prev_loss_actor = curr_loss_actor
                    # We evaluate the policy once every 10 batches of data.
                    # Evaluation is rather simple: execute the policy without exploration
                    # (take the expected value of the action distribution) for a given
                    # number of steps (1000, which is our ``env`` horizon).
                    # The ``rollout`` method of the ``env`` can take a policy as argument:
                    # it will then execute this policy at each step.

                    with (
                        set_exploration_type(InteractionType.DETERMINISTIC),
                        torch.no_grad(),
                    ):
                        # execute a rollout with the trained policy
                        eval_rollout = env.rollout(500, actor)
                        logs["eval reward"].append(
                            eval_rollout["next", "reward"].mean().item()
                        )
                        logs["eval reward (sum)"].append(
                            eval_rollout["next", "reward"].sum().item()
                        )
                        logs["eval step_count"].append(
                            eval_rollout["step_count"].max().item()
                        )
                        eval_str = (
                            f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
                            f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
                            f"eval step-count: {logs['eval step_count'][-1]}"
                        )
                        del eval_rollout

                # Update training progress counters:
                # - data.numel() returns total number of elements in the batch
                # - data["next", "done"].sum() counts completed episodes in batch
                total_count += data.numel()  # Increment total environment steps
                total_episodes += data[
                    "next", "done"
                ].sum()  # Increment completed episodes

                # Check if we've exceeded maximum training steps
                if total_count >= max_training_steps:
                    logger.info(
                        f"Training stopped after reaching maximum steps: "
                        f"{max_training_steps}"
                    )
                    break

        # Break outer loop if inner loop was broken
        if total_count >= max_training_steps:
            break

t1 = time.time()

# Log training performance metrics
training_duration = t1 - t0
log_performance_metrics(
    logger,
    "DDPG Training",
    training_duration,
    extra_metrics={
        "total_steps": int(total_count),
        "total_episodes": int(total_episodes),
        "steps_per_second": int(total_count / training_duration),
        "final_loss_value": logs["loss_value"][-1] if logs["loss_value"] else 0,
        "final_loss_actor": logs["loss_actor"][-1] if logs["loss_actor"] else 0,
    },
)

logger.info(
    f"Training completed: {total_count} steps, {total_episodes} episodes in {training_duration:.2f}s"
)

# %%


loss_df = pd.DataFrame(
    {
        "step": range(len(logs["loss_value"])),
        "Value Loss": logs["loss_value"],
        "Actor Loss": logs["loss_actor"],
    }
)
# %%
(
    ggplot(loss_df.melt(id_vars=["step"], var_name="Loss Type", value_name="Loss"))
    + geom_line(aes(x="step", y="Loss", color="Loss Type"))
    + facet_wrap("Loss Type", ncol=1, scales="free")
)
# %%
logger.info(
    f"solved after {total_count} steps, {total_episodes} episodes and in {t1 - t0}s."
)

# %%
# Evaluation Phase
log_processing_step(
    logger,
    "Model Evaluation",
    "Running rollouts with trained policy",
    extra_data={"max_steps": 1000, "strategies": ["MODE", "RANDOM"]},
)

# Run first rollout
# Use the mode of the distribution (argmax for categorical, mean for normal, etc).
max_steps = 1000
logger.info("Running deterministic (MODE) rollout")
with set_exploration_type(InteractionType.MODE):
    env_to_render_1 = env.rollout(max_steps=max_steps, policy=actor)

logger.info("Running stochastic (RANDOM) rollout")
with set_exploration_type(InteractionType.RANDOM):
    env_to_render_2 = env.rollout(max_steps=max_steps, policy=actor)

# Log evaluation metrics
eval_reward_mode = env_to_render_1["next", "reward"].sum().item()
eval_reward_random = env_to_render_2["next", "reward"].sum().item()
logger.info(f"Evaluation reward (MODE): {eval_reward_mode:.4f}")
logger.info(f"Evaluation reward (RANDOM): {eval_reward_random:.4f}")


# %%
# Call the function with the rollouts
# %%
benchmark_df = pd.DataFrame(
    {
        "x": range(max_steps),
        "buy_and_hold": np.log(df["close"] / df["close"].shift(1))
        .fillna(0)
        .cumsum()[:max_steps],
        "max_profit": np.log(abs(df["close"] / df["close"].shift(1) - 1) + 1)
        .fillna(0)
        .cumsum()[:max_steps],
    }
)
# benchmark_df
# %%
reward_plot, action_plot = compare_rollouts(
    [env_to_render_1, env_to_render_2], n_obs=max_steps
)
(
    reward_plot
    + geom_line(data=benchmark_df, mapping=aes(x="x", y="buy_and_hold"), color="violet")
    + geom_line(data=benchmark_df, mapping=aes(x="x", y="buy_and_hold"), color="violet")
)
# %%
_ = action_plot  # Display plot in notebook

# %%
