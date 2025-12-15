# PPO Training Flow

```mermaid
sequenceDiagram
    autonumber
    participant CLI as CLI (Typer)
    participant ExCmd as ExperimentCommand
    participant DataCmd as DataGeneratorCommand
    participant RME as RunMultipleExperiments
    participant RSE as RunSingleExperiment
    participant PPO as PPOTrainer / PPOTrainerContinuous
    participant Col as SyncDataCollector
    participant Buf as ReplayBuffer
    participant Loss as ClipPPOLoss
    participant Opt as AdamOptimizer
    participant Eval as Evaluator
    participant MLf as MLflow

    CLI->>ExCmd: experiment(--scenario, --generate-data)
    ExCmd->>ExCmd: load scenario config from src/configs/
    
    alt --generate-data
        ExCmd->>DataCmd: generate synthetic data
        DataCmd-->>ExCmd: data saved to standardized path
    end
    
    ExCmd->>RME: run_multiple_experiments(config)
    RME->>RSE: run_single_experiment(trial_config)
    RSE->>RSE: Select Trainer (Discrete vs Continuous)
    RSE->>Col: create_actor_critic + SyncDataCollector
    RSE->>PPO: trainer.train(callback)

    loop each batch frames_per_batch
        Col->>PPO: yield data TensorDict
        PPO->>Buf: replay_buffer.extend(data)
        Buf->>PPO: sample(batch_size)
        PPO->>Loss: compute ClipPPOLoss
        Loss-->>PPO: loss_objective + loss_critic + loss_entropy

        PPO->>Opt: backward_pass
        PPO->>Opt: optimization_step
        Opt-->>PPO: update actor and critic weights
        PPO->>MLf: callback.log_training_step()

        alt log_interval
            PPO->>Buf: gather buffer stats
            PPO->>MLf: log buffer metrics
        end

        alt eval_interval
            PPO->>Eval: env.rollout(eval_steps)
            Eval-->>PPO: rewards actions stats
            PPO->>MLf: log eval metrics
        end
    end

    PPO-->>RSE: logs and checkpoint saved
    RSE->>Eval: evaluate_agent deterministic rollout
    Eval-->>RSE: reward action plots final reward
    RSE->>MLf: log final metrics and artifacts
    RME->>MLf: comparison plots across trials

```

**Key components**

- **CLI (Typer)** with command classes: `ExperimentCommand` handles `--scenario` and `--generate-data` flags.
- **ExperimentCommand** loads scenario configs from standardized YAML files and coordinates data generation.
- **DataGeneratorCommand** creates synthetic data patterns (sine_wave, mean_reversion, etc.) with consistent naming.
- **run_multiple_experiments** fans out trial configs, seeds, and MLflow runs with scenario-based configuration.
- **run_single_experiment** automatically selects the appropriate trainer:
    - `PPOTrainer` for discrete environments (e.g., `gym_trading_env.discrete`).
    - `PPOTrainerContinuous` for continuous environments (e.g., `tradingenv`, `gym_trading_env.continuous`).
- **PPOTrainer.train** loops over batches from the collector, fills the replay buffer, samples mini-batches, computes the clipped PPO loss, and applies the shared Adam optimiser.
- **Evaluation** happens both periodically (during training) and once at the end via `evaluate_agent`, producing reward/action plots plus a PPO-only action-probability plot over the full requested horizon; MLflow also saves a combined evaluation figure when all three plots are present.
- **MLflow** captures every metric, config parameter, checkpoint, and generated plot with experiment names matching config files.

## Continuous Action Support

The system supports continuous action spaces (e.g., for portfolio optimization) through `PPOTrainerContinuous`.

### Architecture
- **Actor:** Uses `TanhNormal` distribution to output bounded continuous actions (e.g., portfolio weights in `[-1, 1]`).
- **Network:** The actor network outputs `loc` (mean) and `scale` (std) parameters for the distribution.
- **Trainer:** `PPOTrainerContinuous` inherits from `PPOTrainer` but overrides model creation and visualization logic.

### Visualization
- **Discrete PPO:** Visualizes action probabilities (Short/Hold/Long) over time as stacked areas.
- **Continuous PPO:** Visualizes the **Action Mean ± Std** over time as a line plot with a shaded ribbon, showing the policy's deterministic tendency and uncertainty.

### Evaluation Logic
- The evaluation process automatically handles different distribution types.
- For `TanhNormal` (which lacks an analytical mode), the system gracefully falls back to `InteractionType.DETERMINISTIC` (returning the `tanh(loc)` value) during deterministic rollouts.