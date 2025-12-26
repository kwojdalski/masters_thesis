"""Main training script for DDPG trading agent - refactored version.

This is a clean, modular version of the trading RL training script.
All configuration, data processing, models, and training logic have been
separated into reusable modules.
"""

# %%
from pathlib import Path

import gym_trading_env  # noqa: F401

from trading_rl import (
    DDPGTrainer,
    ExperimentConfig,
    create_actor,
    create_environment,
    create_value_network,
    evaluate_agent,
    prepare_data,
    run_multiple_experiments,
    set_seed,
    setup_logging,
    visualize_training,
)

# %%
"""Main training pipeline."""

# Load configuration
config = ExperimentConfig()

# Setup
logger = setup_logging(config)
set_seed(config.seed)

# Prepare data
logger.info("Preparing data...")
df = prepare_data(
    data_path=config.data.data_path,
    download_if_missing=config.data.download_data,
    exchange_names=config.data.exchange_names,
    symbols=config.data.symbols,
    timeframe=config.data.timeframe,
    data_dir=config.data.data_dir,
    since=config.data.download_since,
)

# %%
# Create environment
logger.info("Creating environment...")
env = create_environment(df, config)
# %%
# Get environment specs
n_obs = env.observation_spec["observation"].shape[-1]
n_act = env.action_spec.shape[-1]
logger.info(f"Environment: {n_obs} observations, {n_act} actions")

# %%
# Create models
logger.info("Creating models...")
actor = create_actor(
    n_obs,
    n_act,
    hidden_dims=config.network.actor_hidden_dims,
    spec=env.action_spec,
)
# %%
value_net = create_value_network(
    n_obs,
    n_act,
    hidden_dims=config.network.value_hidden_dims,
)
# %%
# Create trainer
logger.info("Initializing trainer...")
trainer = DDPGTrainer(
    actor=actor,
    value_net=value_net,
    env=env,
    config=config.training,
    checkpoint_dir=config.logging.log_dir,
    checkpoint_prefix=config.experiment_name,
)
# %%
# Train
logger.info("Starting training...")
logs = trainer.train()
# %%
# Save checkpoint
checkpoint_path = (
    Path(config.logging.log_dir) / f"{config.experiment_name}_checkpoint.pt"
)
trainer.save_checkpoint(str(checkpoint_path))

# %%
# Visualize results
logger.info("Creating visualizations...")
loss_plot = visualize_training(
    logs
    # save_path=str(
    #     Path(config.logging.log_dir) / f"{config.experiment_name}_losses.png"
    # ),
)
# %%
loss_plot
# %%
reward_plot, action_plot, action_probs_plot, final_reward, last_positions = evaluate_agent(
    env,
    actor,
    df,
    max_steps=1000,
    # save_path=str(Path(config.logging.log_dir) / f"{config.experiment_name}_eval"),
)
# %%
reward_plot
action_plot
final_reward
# %%
logger.info("Training complete!")
logger.info(f"Checkpoint saved to: {checkpoint_path}")

# %%
# Example usage for multiple experiments:
# study = run_multiple_experiments("trading_rl_experiments", n_trials=10)
# print(f"Best trial: {study.best_trial}")
# print(f"Best reward: {study.best_value}")

# To access stored metrics from all trials:
# for trial in study.trials:
#     print(f"Trial {trial.number}: reward={trial.user_attrs['final_reward']}")

# return {
#     "trainer": trainer,
#     "logs": logs,
#     "plots": {
#         "loss": loss_plot,
#         "reward": reward_plot,
#         "action": action_plot,
#     },
# }
run_multiple_experiments("trading_rl_experiments", n_trials=1)
