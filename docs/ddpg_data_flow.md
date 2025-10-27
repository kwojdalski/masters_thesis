# DDPG Data & Control Flow

```mermaid
flowchart TD
    subgraph CLI
        A[python src/cli.py train/experiment]
    end

    subgraph Config
        B[ExperimentConfig<br/>(YAML overrides)]
    end

    subgraph DataPrep
        C[prepare_data<br/>load_trading_data<br/>create_features]
    end

    subgraph EnvSetup
        D[create_environment<br/>Gym TradingEnv -> GymWrapper -> TransformedEnv(StepCounter)]
    end

    subgraph Models
        E[create_actor<br/>(DiscreteNet -> ProbabilisticActor)]
        F[create_value_network<br/>(ValueOperator over MLP)]
    end

    subgraph Trainer
        G[DDPGTrainer.__init__<br/>DDPGLoss + SoftUpdate<br/>ReplayBuffer(LazyTensorStorage)<br/>SyncDataCollector]
        H[DDPGTrainer.train<br/>for each collector batch]
        I[ReplayBuffer.extend]
        J[ReplayBuffer.sample]
        K[DDPGLoss(sample)]
        L[optimizer_actor.step()]
        M[optimizer_value.step()]
        N[SoftUpdate.step()]
        O[callback.log_training_step<br/>log_episode_stats -> MLflow]
    end

    subgraph Evaluation
        P[evaluate_agent<br/>Deterministic rollout<br/>Random rollout<br/>compare_rollouts]
        Q[Reward/Action plots]
    end

    subgraph Tracking
        R[MLflow<br/>Metrics + Params + Artifacts]
        S[Checkpoint Save<br/>logs/..._checkpoint.pt]
    end

    A --> B --> C --> D
    D --> E
    D --> F
    E --> G
    F --> G
    G --> H
    H --> I --> J --> K
    K --> L
    K --> M
    K --> N
    H --> O --> R
    H --> S
    H --> P --> Q --> R
    Q --> S
```

**Highlights**

- **Data**: `prepare_data` fetches parquet data, optional feature engineering, and returns the DataFrame slice defined by `train_size`.
- **Environment**: The Gym trading environment is wrapped for TorchRL and augmented with a `StepCounter` transform.
- **Models**: `create_actor` builds a `ProbabilisticActor` over your discrete policy network; `create_value_network` constructs the Q-value estimator.
- **Trainer**: `DDPGTrainer` orchestrates the TorchRL `SyncDataCollector`, replay buffer, `DDPGLoss`, and SoftUpdate target sync; training logs stream to the MLflow callback.
- **Evaluation**: `evaluate_agent` runs both deterministic and random rollouts, then `compare_rollouts` produces the reward/action plots that get logged.
- **Tracking**: MLflow holds metrics, params, and artifacts; checkpoints land in `logs/<experiment>/`.
