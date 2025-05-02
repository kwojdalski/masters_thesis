# %%
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule as Mod
from tensordict.nn import TensorDictSequential as Seq
from torch.optim import Adam
from torchrl._utils import logger as torchrl_logger
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.data.tensor_specs import BoundedTensorSpec
from torchrl.modules import MLP, EGreedyModule, QValueModule
from torchrl.objectives import DQNLoss, SoftUpdate

# Set random seed for reproducibility
torch.manual_seed(0)

# Define custom environment specs with proper action space definition
observation_spec = torch.zeros(
    4
)  # Example: 4-dimensional observation space like CartPole
# Define action spec as a BoundedTensorSpec with proper shape and bounds
action_spec = BoundedTensorSpec(
    shape=(2,),  # 2 possible actions
    dtype=torch.float32,
    low=torch.zeros(2),
    high=torch.ones(2),
    device="cpu",
)


# %% Create a custom data generator function
def custom_data_generator(batch_size=10):
    while True:
        # Generate random observations, actions, rewards, next observations, and dones
        batch = TensorDict(
            {
                "observation": torch.randn(batch_size, 4),
                "action": torch.randint(0, 2, (batch_size, 1)),
                "reward": torch.randn(batch_size, 1),
                "next": {
                    "observation": torch.randn(batch_size, 4),
                    "done": torch.randint(0, 2, (batch_size, 1)).bool(),
                    "terminated": torch.randint(0, 2, (batch_size, 1)).bool(),
                },
            },
            batch_size=batch_size,
        )
        yield batch


# %%
# Create policy network
value_mlp = MLP(in_features=4, out_features=2, num_cells=[64, 64])
value_net = Mod(value_mlp, in_keys=["observation"], out_keys=["action_value"])
policy = Seq(value_net, QValueModule(spec=action_spec))

exploration_module = EGreedyModule(
    action_spec, annealing_num_steps=10_000, eps_init=0.5
)
policy_explore = Seq(policy, exploration_module)

# Initialize replay buffer
rb = ReplayBuffer(storage=LazyTensorStorage(10_000))

# Create data collector with custom data
init_rand_steps = 100
frames_per_batch = 10
collector = SyncDataCollector(
    custom_data_generator(),
    policy_explore,
    frames_per_batch=frames_per_batch,
    total_frames=-1,
    init_random_frames=init_rand_steps,
)

# Setup loss, optimizer and target network updater
# Pass the properly defined action_spec to DQNLoss
loss = DQNLoss(value_network=policy, action_space=action_spec, delay_value=True)
optim = Adam(loss.parameters(), lr=0.01)
updater = SoftUpdate(loss, eps=0.99)

# %% Training loop
total_count = 0
total_episodes = 0
for i, data in enumerate(collector):
    # Write data in replay buffer
    rb.extend(data)

    if len(rb) > init_rand_steps:
        # Sample from replay buffer and perform optimization
        sample = rb.sample(32)
        loss_vals = loss(sample)
        loss_vals["loss"].backward()
        optim.step()
        optim.zero_grad()

        # Update exploration factor
        exploration_module.step(data.numel())

        # Update target params
        updater.step()

        if i % 10 == 0:
            torchrl_logger.info(
                f"Iteration {i}, Loss: {loss_vals['loss'].item():.4f}, RB size: {len(rb)}"
            )

        total_count += data.numel()
        total_episodes += data["next", "done"].sum().item()

    # Stop after 50 iterations for this example
    if i >= 50:
        break

torchrl_logger.info(
    f"Training completed after {total_count} steps and {total_episodes} episodes."
)


# %%
# Evaluate the trained policy
def evaluate_policy(policy, num_episodes=5):
    total_reward = 0
    for _ in range(num_episodes):
        # Generate a single episode of data
        obs = torch.randn(1, 4)  # Initial observation
        done = False
        episode_reward = 0

        while not done:
            # Get action from policy
            with torch.no_grad():
                td = TensorDict({"observation": obs}, batch_size=[1])
                action_td = policy(td)
                action = action_td["action"]

            # Simulate next step (in a real environment, you would use env.step())
            next_obs = torch.randn(1, 4)
            reward = torch.randn(1)
            done = torch.rand(1) > 0.9  # Random termination

            # Update total reward
            episode_reward += reward.item()
            obs = next_obs

            if done:
                break

        total_reward += episode_reward

    avg_reward = total_reward / num_episodes
    torchrl_logger.info(
        f"Evaluation: Average reward over {num_episodes} episodes: {avg_reward:.2f}"
    )


# Evaluate the trained policy
evaluate_policy(policy)

# %%
