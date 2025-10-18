# %%
import time

import torch
from tensordict.nn import TensorDictModule as Mod
from tensordict.nn import TensorDictSequential as Seq
from torch.optim import Adam
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.envs import GymEnv, StepCounter, TransformedEnv
from torchrl.modules import MLP, EGreedyModule, QValueModule
from torchrl.objectives import DQNLoss, SoftUpdate
from torchrl.record import CSVLogger, VideoRecorder

from logger import get_logger, setup_logging

# %%
# Setup custom logger
setup_logging(
    level="INFO",
    log_file="logs/minimal_torchrl/training.log",
    console_output=True,
    colored_output=True,
)
logger = get_logger("minimal_torchrl")
logger.info("Starting minimal TorchRL DQN training example")

env = TransformedEnv(GymEnv("CartPole-v1"), StepCounter())
env.set_seed(0)
torch.manual_seed(0)
logger.info("Environment and seeds initialized")

# %%
# Designing a policy
# The next step is to build our policy.
# We'll be making a regular, deterministic version of the
# actor to be used within the loss module and during evaluation.
# Next, we will augment it with an exploration module for inference.


value_mlp = MLP(out_features=env.action_spec.shape[-1], num_cells=[64, 64])
value_net = Mod(value_mlp, in_keys=["observation"], out_keys=["action_value"])
policy = Seq(value_net, QValueModule(spec=env.action_spec))
exploration_module = EGreedyModule(
    env.action_spec, annealing_num_steps=100_000, eps_init=0.5
)
policy_explore = Seq(policy, exploration_module)
logger.info("Policy network and exploration module created")

# Data Collector and replay buffer
# Here comes the data part: we need a data collector to easily get batches of data and a replay buffer to store that data for training.


init_rand_steps = 1000
frames_per_batch = 100
optim_steps = 10
collector = SyncDataCollector(
    env,
    policy_explore,
    frames_per_batch=frames_per_batch,
    total_frames=-1,
    init_random_frames=init_rand_steps,
)
rb = ReplayBuffer(storage=LazyTensorStorage(100_000))
logger.info(
    f"Data collector and replay buffer initialized "
    f"(buffer_size=100k, frames_per_batch={frames_per_batch})"
)

# Loss module and optimizer
# We build our loss as indicated in the dedicated tutorial, with its optimizer
# and target parameter updater:

loss = DQNLoss(value_network=policy, action_space=env.action_spec, delay_value=True)
optim = Adam(loss.parameters(), lr=0.02)
updater = SoftUpdate(loss, eps=0.99)
logger.info("DQN loss, optimizer, and target updater created")
# CSV Logger for TorchRL metrics
# We'll be using a CSV logger to log our results, and save rendered videos.

path = "./runs"
csv_logger = CSVLogger(exp_name="dqn", log_dir=path, video_format="mp4")
video_recorder = VideoRecorder(csv_logger, tag="video")
record_env = TransformedEnv(
    GymEnv("CartPole-v1", from_pixels=True, pixels_only=False), video_recorder
)
logger.info(f"CSV logger and video recorder initialized (log_dir={path})")

# Training loop
# Instead of fixing a specific number of iterations to run, we will keep on training
# the network until it reaches a certain performance (arbitrarily defined as 200 steps
# in the environment - with CartPole, success is defined as having longer trajectories).

total_count = 0
total_episodes = 0
logger.info("Starting training loop (target: max_length > 200)")
t0 = time.time()
for i, data in enumerate(collector):
    # Write data in replay buffer
    rb.extend(data)
    max_length = rb[:]["next", "step_count"].max()
    if len(rb) > init_rand_steps:
        # Optim loop (we do several optim steps
        # per batch collected for efficiency)
        for _ in range(optim_steps):
            sample = rb.sample(128)
            loss_vals = loss(sample)
            loss_vals["loss"].backward()
            optim.step()
            optim.zero_grad()
            # Update exploration factor
            exploration_module.step(data.numel())
            # Update target params
            updater.step()
            if i % 10:
                logger.info(f"Max num steps: {max_length}, rb length {len(rb)}")
            total_count += data.numel()
            total_episodes += data["next", "done"].sum()
    if max_length > 200:
        break

t1 = time.time()

logger.info(
    f"Training completed! Solved after {total_count} steps, "
    f"{total_episodes} episodes in {t1 - t0:.2f}s"
)

# Rendering
# Finally, we run the environment for as many steps as we can and save the video
# locally (notice that we are not exploring).

logger.info("Recording final rollout video...")
record_env.rollout(max_steps=1000, policy=policy)
video_recorder.dump()
logger.info("Video saved successfully")

# %%
