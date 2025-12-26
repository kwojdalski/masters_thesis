# PPO Implementation Overview

## Summary
- On-policy actor-critic with clipped surrogate objective.
- Supports discrete and continuous action spaces via separate trainers.
- Training uses collector batches, mini-batch updates, and periodic evaluation/logging.

## Flow

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

## Components
- **CLI + configs**: `ExperimentCommand` loads YAML configs and optionally triggers data generation.
- **Trainer selection**: `PPOTrainer` for discrete envs, `PPOTrainerContinuous` for continuous envs.
- **Loss**: `ClipPPOLoss` with shared Adam optimizer for actor/critic.
- **Collector/buffer**: `SyncDataCollector` + replay buffer for minibatch sampling.

## Training Loop
- Collect batch → extend replay buffer → sample minibatches.
- Compute clipped PPO loss (objective + critic + entropy).
- Backprop and step optimizer; log per-step metrics.
- Periodic buffer stats and evaluation rollouts.

## Evaluation and Tracking
- Periodic eval during training plus final eval at the end.
- Reward/action plots, plus PPO-only action-probability plot.
- MLflow logs metrics, params, artifacts, and checkpoints.

## Continuous Action Notes
- **Actor** uses `TanhNormal` to keep actions in bounds (e.g., `[-1, 1]`).
- **Visualization**: discrete uses stacked action probabilities; continuous uses mean ± std ribbon.
- **Deterministic eval** falls back to `tanh(loc)` for `TanhNormal`.
