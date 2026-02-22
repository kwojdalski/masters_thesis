"""Test whether TorchRL ClipPPOLoss syncs parameters back to modules."""

import torch
from torch.optim import Adam
from torchrl.objectives import ClipPPOLoss

from trading_rl.models import create_ppo_actor, create_ppo_value_network
from torchrl.envs import GymEnv


def test_parameter_syncing():
    """Verify if optimizer updates sync back to original actor module."""

    # Create simple environment and networks
    env = GymEnv("CartPole-v1")
    n_obs = env.observation_spec["observation"].shape[-1]
    n_act = env.action_spec.space.n

    actor = create_ppo_actor(n_obs, n_act, hidden_dims=[64], spec=env.action_spec)
    value_net = create_ppo_value_network(n_obs, hidden_dims=[64])

    # Initialize PPO loss (like your code)
    ppo_loss = ClipPPOLoss(
        actor_network=actor,
        critic_network=value_net,
        clip_epsilon=0.2,
        entropy_bonus=0.01,
    )

    # Get initial parameter value from ORIGINAL actor module
    first_param_name = list(actor.named_parameters())[0][0]
    initial_value = list(actor.parameters())[0].clone()

    print(f"Testing parameter: {first_param_name}")
    print(f"Initial value (first 5 elements): {initial_value.flatten()[:5]}")

    # Create optimizer for FUNCTIONAL parameters (like your code)
    optimizer = Adam(
        list(ppo_loss.actor_network_params.values(True, True))
        + list(ppo_loss.critic_network_params.values(True, True)),
        lr=0.01
    )

    # Check if functional params are the SAME object or COPIES
    functional_param = list(ppo_loss.actor_network_params.values(True, True))[0]
    original_param = list(actor.parameters())[0]

    print(f"\nAre they the same object? {functional_param is original_param}")
    print(f"Do they share storage? {functional_param.data_ptr() == original_param.data_ptr()}")

    # Simulate one optimization step
    fake_sample = env.rollout(10)
    loss_dict = ppo_loss(fake_sample)
    total_loss = loss_dict["loss_objective"] + loss_dict["loss_critic"]
    total_loss.backward()
    optimizer.step()

    # Check if ORIGINAL actor module was updated
    updated_value = list(actor.parameters())[0]

    print(f"\nAfter optimizer.step():")
    print(f"Updated value (first 5 elements): {updated_value.flatten()[:5]}")
    print(f"Parameters changed? {not torch.allclose(initial_value, updated_value)}")

    if torch.allclose(initial_value, updated_value):
        print("\n⚠️  CRITICAL BUG: Original actor parameters NOT updated!")
        print("   Functional parameters and module parameters are SEPARATE")
        print("   You need to sync them manually!")
        return False
    else:
        print("\n✅ OK: Original actor parameters ARE updated")
        print("   Functional parameters share storage with module parameters")
        return True


if __name__ == "__main__":
    is_synced = test_parameter_syncing()
    exit(0 if is_synced else 1)
