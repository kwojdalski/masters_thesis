import time

import torch

# Training hyperparameters
init_rand_steps = 1000  # More exploration
frames_per_batch = 256  # Larger batches
optim_steps = 10  # Less aggressive optimization

# Value network with larger capacity
value_net = ValueOperator(
    MLP(
        in_features=n_obs,
        out_features=1,
        num_cells=[64, 64],  # Larger network
    ),
    in_keys=["observation"],
)

# Modified training loop
t0 = time.time()
for i, data in enumerate(collector):
    rb.extend(data)
    max_length = rb[:]["next", "step_count"].max()

    if len(rb) > init_rand_steps:
        for j in range(optim_steps):
            sample = rb.sample(256)  # Consistent batch size
            loss_vals = ddpg_loss(sample)

            # Compute total loss
            loss_value = loss_vals["loss_value"] + loss_vals["loss_actor"]

            # Proper backprop
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(
                ddpg_loss.parameters(), 1.0
            )  # Add gradient clipping
            optim.step()
            optim.zero_grad()
            updater.step()

            if j % 100 == 0:
                logger.info(f"Step {j}, Loss: {loss_value.item():.4f}")

        total_count += data.numel()
        total_episodes += data["next", "done"].sum()

    # Better stopping criterion
    if i >= 1000 or (
        max_length > 500 and total_episodes > 100
    ):  # More episodes and steps
        break

t1 = time.time()
