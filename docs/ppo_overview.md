# PPO Training Flow

```mermaid
sequenceDiagram
    autonumber
    participant CLI as CLI
    participant RME as RunMultipleExperiments
    participant RSE as RunSingleExperiment
    participant PPO as PPOTrainer
    participant Col as SyncDataCollector
    participant Buf as ReplayBuffer
    participant Loss as ClipPPOLoss
    participant Opt as AdamOptimizer
    participant Eval as Evaluator
    participant MLf as MLflow

    CLI->>RME: experiment()
    RME->>RSE: run_single_experiment(trial_config)
    RSE->>Col: create_ppo_actor_critic + SyncDataCollector
    RSE->>PPO: trainer.train(callback)

    loop each batch frames_per_batch
        Col->>PPO: yield data TensorDict
        PPO->>Buf: replay_buffer.extend(data)
        Buf->>PPO: sample(batch_size)
        PPO->>Loss: compute ClipPPOLoss
        Loss-->>PPO: loss_objective + loss_critic + loss_entropy

        PPO->>Opt: perform backward pass
        PPO->>Opt: perform optimization step
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

- **CLI (Typer)** invokes `run_multiple_experiments` when you call `python src/cli.py experiment`.
- **run_multiple_experiments** fans out trial configs, seeds, and MLflow runs.
- **run_single_experiment** builds the PPO actor/critic, the TorchRL `SyncDataCollector`, and the `PPOTrainer`.
- **PPOTrainer.train** loops over batches from the collector, fills the replay buffer, samples mini-batches, computes the clipped PPO loss, and applies the shared Adam optimiser.
- **Evaluation** happens both periodically (during training) and once at the end via `evaluate_agent`, producing reward/action plots.
- **MLflow** captures every metric, config parameter, checkpoint, and generated plot for later comparison.
