# %%

import gym_trading_env  # noqa: F401
import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule
from torch import distributions as d
from torch.optim import Adam
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.envs import GymWrapper, TransformedEnv
from torchrl.envs.transforms import StepCounter
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator
from torchrl.objectives import DDPGLoss, SoftUpdate


# %%
# Set seeds for reproducibility
def set_seeds(seed=42):
    """Set seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# Create synthetic data with clear patterns
def create_synthetic_data(n_steps=200):
    """Create synthetic price data with clear patterns."""
    t = np.linspace(0, 4 * np.pi, n_steps)
    price = 100 + 10 * np.sin(t) + np.random.normal(0, 1, n_steps)
    volume = np.abs(10 + 5 * np.sin(t) + np.random.normal(0, 1, n_steps))

    df = pd.DataFrame({"close": price, "volume": volume})

    # Add simple features
    df["returns"] = df["close"].pct_change().fillna(0)
    df["vol_change"] = df["volume"].pct_change().fillna(0)

    # Normalize features
    for col in ["returns", "vol_change"]:
        df[f"feature_{col}"] = (df[col] - df[col].mean()) / df[col].std()

    return df


# Simple reward function
def reward_function(history):
    """Calculate simple returns-based reward."""
    return history["portfolio_valuation", -1] / history["portfolio_valuation", -2] - 1


# Network architecture
class SimpleNet(nn.Module):
    def __init__(self, input_dim, n_actions):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(), nn.Linear(32, n_actions)
        )

    def forward(self, x):
        logits = self.network(x)
        probs = nn.functional.softmax(logits, dim=-1)
        return {"probs": probs}


# Set random seeds
set_seeds(42)

# Create synthetic data
df = create_synthetic_data(n_steps=200)

# Create environment
base_env = gym.make(
    "TradingEnv",
    name="TEST",
    df=df,
    positions=[-1, 0, 1],  # Short, Neutral, Long
    trading_fees=0.001,  # 0.1% trading fee
    reward_function=reward_function,
)

# Wrap environment
env = GymWrapper(base_env)
env = TransformedEnv(env, StepCounter())

# Get dimensions
n_obs = env.observation_spec["observation"].shape[-1]
n_act = env.action_spec.shape[-1]

# Create network and actor
module = TensorDictModule(
    module=SimpleNet(n_obs, n_act),
    in_keys=["observation"],
    out_keys=["probs"],
)

actor = ProbabilisticActor(
    module=module,
    distribution_class=d.OneHotCategorical,
    in_keys=["probs"],
    out_keys=["action"],
    spec=env.action_spec,
)

# Training parameters
frames_per_batch = 50  # Small batch size
total_frames = 1000  # Short training run

# Create collector
collector = SyncDataCollector(
    create_env_fn=lambda: env,
    policy=actor,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
)

# Create replay buffer
rb = ReplayBuffer(storage=LazyTensorStorage(1000))

# Create a value network for better stability
value_net = ValueOperator(
    MLP(
        in_features=n_obs,
        out_features=1,
        num_cells=[32, 32],  # Smaller network for the simple task
    ),
    in_keys=["observation"],
    out_keys=["state_action_value"],
)

# Initialize DDPG loss and updater
ddpg_loss = DDPGLoss(
    actor_network=actor,
    value_network=value_net,
    loss_function="l2",
)

# Use soft updates for stability
updater = SoftUpdate(ddpg_loss, tau=0.005)  # Slower target network updates

# Initialize optimizer with lower learning rate and gradient clipping
optimizer = Adam(ddpg_loss.parameters(), lr=3e-4)  # Lower learning rate

# Lists to track metrics
actor_losses = []
value_losses = []
returns_history = []

# Training loop with improved stability
for i, data in enumerate(collector):
    rb.extend(data)
    if len(rb) >= frames_per_batch:
        # Sample from buffer
        sample = rb.sample(frames_per_batch)

        # Calculate losses
        loss_vals = ddpg_loss(sample)
        loss = loss_vals["loss_actor"] + loss_vals["loss_value"]

        # Optimize
        optimizer.zero_grad()
        loss.backward()
        # Clip gradients for stability
        torch.nn.utils.clip_grad_norm_(ddpg_loss.parameters(), max_norm=1.0)
        optimizer.step()

        # Soft update target networks
        updater.step()

        # Track metrics
        actor_losses.append(loss_vals["loss_actor"].item())
        value_losses.append(loss_vals["loss_value"].item())

        # Calculate episode return
        episode_return = sample["next", "reward"].sum().item()
        returns_history.append(episode_return)

        # Print progress
        if i % 10 == 0:  # Print every 10 steps
            avg_actor_loss = np.mean(actor_losses[-10:])
            # avg_value_loss = np.mean(value_losses[-10:])
            avg_return = np.mean(returns_history[-10:])
            print(f"Step {i}")
            print(f"  Actor Loss: {avg_actor_loss:.4f}")
            # print(f"  Value Loss: {avg_value_loss:.4f}")
            print(f"  Avg Return: {avg_return:.4f}")

    if i * frames_per_batch >= total_frames:
        break

# Plot training curves
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.plot(actor_losses)
plt.title("Actor Loss")
plt.xlabel("Step")
plt.ylabel("Loss")

plt.subplot(132)
plt.plot(value_losses)
plt.title("Value Loss")
plt.xlabel("Step")
plt.ylabel("Loss")

plt.subplot(133)
plt.plot(returns_history)
plt.title("Episode Returns")
plt.xlabel("Step")
plt.ylabel("Return")

plt.tight_layout()
plt.show()

# Test the trained agent
with torch.no_grad():
    test_rollout = env.rollout(max_steps=100, policy=actor)
    returns = test_rollout["next", "reward"].sum().item()
    print(f"Test Returns: {returns:.4f}")

# %%
