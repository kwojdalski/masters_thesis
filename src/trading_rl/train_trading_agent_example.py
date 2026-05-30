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
    PrepareDataConfig,
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
config.seed = set_seed(config.seed)

# Prepare data
logger.info("prepare data")
train_df, _val_df, _test_df = prepare_data(
    config.data.data_path,
    PrepareDataConfig.from_config(config.data),
)

# %%
# Create environment
logger.info("create environment")
env = create_environment(train_df, config)
# %%
# Get environment specs
n_obs = env.observation_spec["observation"].shape[-1]
n_act = env.action_spec.shape[-1]
logger.info("environment obs={} actions={}", n_obs, n_act)

# %%
# Create models
logger.info("create models")
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
logger.info("init trainer")
trainer = DDPGTrainer(
    actor=actor,
    value_net=value_net,
    env=env,
    config=config.training,
    n_obs=n_obs,
    n_act=n_act,
    actor_hidden_dims=config.network.actor_hidden_dims,
    value_hidden_dims=config.network.value_hidden_dims,
    checkpoint_dir=config.logging.log_dir,
    checkpoint_prefix=config.experiment_name,
)
# %%
# Train
logger.info("start training")
logs = trainer.train()
# %%
# Save checkpoint
checkpoint_path = (
    Path(config.logging.log_dir) / f"{config.experiment_name}_checkpoint.pt"
)
trainer.save_checkpoint(str(checkpoint_path))

# %%
# Visualize results
logger.info("create visualizations")
loss_plot = visualize_training(
    logs
)
# %%
loss_plot
# %%
reward_plot, action_plot, action_probs_plot, final_reward, last_positions = evaluate_agent(
    env,
    actor,
    train_df,
    max_steps=1000,
)
# %%
reward_plot
action_plot
final_reward
# %%
logger.info("training complete")
logger.info("checkpoint saved path={}", checkpoint_path)

# %%
run_multiple_experiments("trading_rl_experiments", n_trials=1)
